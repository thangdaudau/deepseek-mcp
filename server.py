#!/usr/bin/env python3
"""
DeepSeek MCP Server (ReAct Tool Loop & TDD Runner)
Production-grade integration of DeepSeek Flash (Thinking/Non-thinking),
automated TDD test runner, Doubt-Driven adversarial review, and safety guardrails.
"""

import os
import sys
import json
import re
import time
import stat
import fcntl
import signal
import select
import threading
import functools
import subprocess
import hashlib
import zlib
import copy
import tempfile
import shutil
import shlex
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MAX_READ_BYTES = 24 * 1024            # 24KB raw content per page
MAX_READ_LINES = 2000                 # 2000 lines max per read
CHUNK_SIZE = 64 * 1024                # 64KB chunk streaming
MAX_SEARCH_FILE_BYTES = 1024 * 1024   # 1MB max file scan in search
SEARCH_TIMEOUT_SECONDS = 10.0         # 10s search timeout
SEARCH_MAX_LINE_BYTES = 8192          # 8KB max line in search
MAX_SHELL_CAPTURE = 512 * 1024        # 512KB buffer for subprocess stdout/stderr
DEFAULT_CODER_SHELL_TIMEOUT = 30      # 30s timeout for coder run_shell (Gate G9)
OFFLOAD_THRESHOLD = 32_000            # 32,000 chars (~8,000 tokens)
OFFLOAD_PREVIEW_CHARS = 1_000         # 1,000 chars head + 1,000 chars tail
MAX_OFFLOAD_DIR_BYTES = 512 * 1024 * 1024 # 512MB max .deepseek/offload/ dir
OFFLOAD_TTL_SECONDS = 14_400          # 4 hours
MAX_ENVELOPE_CONTENT_CHARS = 28_000   # Max envelope chars before json
MAX_TOOL_RESULT = 64_000              # 64,000 chars per-tool-call cap in messages
MAX_CONTEXT_BYTES = 3_500_000         # 3.5MB safety context cap
MAX_SESSION_BYTES = 8 * 1024 * 1024   # 8MB max session JSON on disk
MAX_SHELL_OUT = MAX_SHELL_CAPTURE     # Đồng bộ cho tương thích ngược
WAL_COMPACT_FRAMES = 30               # Snapshot compaction every 30 turns
MAX_HOT_SESSIONS = 50                 # 50 hot sessions in RAM LRU
SESSION_TTL_SECONDS = 14_400          # 4 hours
SESSION_LOCK_TIMEOUT = 15.0           # 15s bounded wait retry for session_lock

# Khóa chặt bất biến runtime an toàn
if MAX_READ_BYTES > MAX_TOOL_RESULT - 4096:
    raise ValueError("MAX_READ_BYTES must leave room for JSON envelope in MAX_TOOL_RESULT")

DANGEROUS_COMMAND_PATTERNS = [
    # Chặn đứng 100% mọi lệnh xóa file/thư mục bất kể đối số
    r"\brm(\s+|$)",
    r"\brmdir(\s+|$)",
    r"\bunlink(\s+|$)",
    r"\bshred(\s+|$)",
    r"\bgit\s+(clean|rm)(\s+|$)",
    r"\bmkfs(\.[a-z0-9]+)?(\s|$)",
    r"\bdd\s+.*(if=|of=/dev/)",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bgit\s+push\s+.*(-f\b|--force\b)",
    r"\bfind\s+.*-delete\b",
    r"\bshutdown(\s|$)",
    r"\breboot(\s|$)",
    r"\bpoweroff(\s|$)",
    r"\b(chmod|chown)\s+-[rR]\s+.*(/|~|\$HOME)(\s|$)",
    r"\bchmod\s+[0-7]{3,4}\s+.*(/|~|\$HOME)(\s|$)"
]

server = MCPServer("DeepSeek-MCP")

class SessionBusyError(Exception):
    """Ném ra khi session file đang bị khóa bởi một tiến trình / turn khác."""
    pass

def tool_guard(fn):
    """Decorator bảo đảm 100% các tool của MCPServer luôn trả về chuỗi JSON hợp lệ, không ném exception thô ra stdio."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SessionBusyError as e:
            return json.dumps({
                "status": "ERROR",
                "error": {
                    "code": "E_SESSION_BUSY",
                    "message": str(e)
                }
            }, ensure_ascii=False)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            return json.dumps({
                "status": "ERROR",
                "error": {
                    "code": "E_INTERNAL",
                    "message": f"{type(e).__name__}: {str(e)}"
                }
            }, ensure_ascii=False)
    return wrapper

def get_api_key() -> str:
    # 1. Direct env var
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    # 2. Check ~/.local/share/deepseek-mcp/.env
    env_file = os.path.expanduser("~/.local/share/deepseek-mcp/.env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEEPSEEK_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass
            
    return ""

def get_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("Lỗi: Chưa tìm thấy DEEPSEEK_API_KEY trong env hoặc ~/.local/share/deepseek-mcp/.env!")
    return OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=120.0, max_retries=2)

SENSITIVE_ENV_KEYWORDS = ("KEY", "TOKEN", "SECRET", "PASS", "CREDENTIAL", "AUTH")

def build_scrubbed_env() -> Dict[str, str]:
    """Tẩy sạch các biến môi trường chứa secret và SSH credentials khỏi env của child process."""
    scrubbed = {
        k: v for k, v in os.environ.items()
        if k not in ("SSH_AUTH_SOCK", "SSH_AGENT_PID") and not any(kw in k.upper() for kw in SENSITIVE_ENV_KEYWORDS)
    }
    venv_bin = os.path.dirname(sys.executable)
    scrubbed["PATH"] = f"{venv_bin}{os.pathsep}{scrubbed.get('PATH', '')}"
    return scrubbed

def is_safe_path(base_dir: str, target_path: str, allow_offload_read: bool = False) -> Tuple[bool, str]:
    """
    Kiểm tra chặn Path Traversal ra ngoài workspace và denylist các đường dẫn nhạy cảm.
    Bộ lọc chống phá hoại vô tình (accidental footguns), chuẩn hóa lowercase trên dir parts
    để tương thích an toàn trên macOS (APFS) và Windows (NTFS).
    allow_offload_read=True: Cho phép đọc file trong thư mục .deepseek/offload/.
    """
    base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base, target_path))
    if not (target == base or target.startswith(base + os.sep)):
        return False, target

    rel = os.path.relpath(target, base)
    parts = Path(rel).parts

    # Chặn truy cập .git (chống RCE qua hooks/config) và .venv (chống can thiệp virtualenv binary)
    if any(p.lower() in (".git", ".venv") for p in parts):
        return False, target

    # Khóa chặt quyền ghi/sửa vào thư mục .deepseek (Gate G2)
    # Chỉ cho phép đọc file offload khi cờ allow_offload_read=True
    if any(p.lower() == ".deepseek" for p in parts):
        if not allow_offload_read:
            return False, target

    filename = parts[-1].lower() if parts else ""
    # Chặn file .env nhạy cảm (cho phép .env.example, .env.template, .env.sample)
    if filename == ".env" or (filename.startswith(".env.") and not filename.endswith((".example", ".template", ".sample"))):
        return False, target

    # Chặn private keys, certificates và netrc
    if filename in ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".netrc") or filename.endswith((".pem", ".key", ".p12", ".pfx")):
        return False, target

    return True, target

def check_command_safety(command: str) -> Tuple[bool, Optional[str]]:
    """
    Bộ lọc FOOTGUN chống phá bậy vô tình (rm -rf /, mkfs, git push -f).
    Dùng để bảo vệ môi trường dev của user khỏi các lệnh tự sát.
    """
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Lệnh bị chặn bởi Guardrail an toàn (trùng khớp pattern nguy hiểm: {pattern})"
    return True, None

def extract_telemetry(response_usage: Any) -> Dict[str, Any]:
    """Trích xuất viễn trắc token, KVCache hit/miss và reasoning tokens."""
    if not response_usage:
        return {}
    
    prompt_tokens = getattr(response_usage, "prompt_tokens", 0)
    prompt_cache_hit_tokens = getattr(response_usage, "prompt_cache_hit_tokens", 0)
    prompt_cache_miss_tokens = getattr(response_usage, "prompt_cache_miss_tokens", 0)
    completion_tokens = getattr(response_usage, "completion_tokens", 0)
    
    reasoning_tokens = 0
    details = getattr(response_usage, "completion_tokens_details", None)
    if details:
        reasoning_tokens = getattr(details, "reasoning_tokens", 0)
        
    cache_hit_rate = 0.0
    if prompt_tokens > 0:
        cache_hit_rate = round((prompt_cache_hit_tokens / prompt_tokens) * 100, 2)
        
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
        "cache_hit_rate_pct": cache_hit_rate,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": getattr(response_usage, "total_tokens", 0)
    }

# ---------------------------------------------------------------------------
# Session Manager for Stateful Multi-Turn Conversations (File-based Persistence)
# ---------------------------------------------------------------------------
SESSIONS_DIR = os.path.expanduser("~/.local/share/deepseek-mcp/sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
SESSION_TTL_SECONDS = 14400  # 4 hours (4 tiếng)
_HOT_SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_session_file(session_id: str) -> str:
    """Tạo đường dẫn an toàn cho session WAL file (.wal.jsonl), chặn path traversal và chống va chạm namespace."""
    safe_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:32]
    id_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return os.path.join(SESSIONS_DIR, f"{safe_prefix}_{id_hash}.wal.jsonl")

def get_legacy_session_file(session_id: str) -> str:
    """Tạo đường dẫn legacy file session .json phục vụ tương thích ngược."""
    safe_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)[:32]
    id_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return os.path.join(SESSIONS_DIR, f"{safe_prefix}_{id_hash}.json")

def session_lock(session_id: str, timeout: float = SESSION_LOCK_TIMEOUT):
    """Context manager khóa file session bằng fcntl.flock với bounded-wait retry (15s deadline). Fail-fast nếu timeout."""
    class _LockContext:
        def __init__(self, sid: str, tout: float):
            self.sid = sid
            self.tout = tout
            self.fh = None

        def __enter__(self):
            lock_file = get_session_file(self.sid) + ".lock"
            self.fh = open(lock_file, "a+")
            deadline = time.monotonic() + self.tout
            while True:
                try:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return self
                except (BlockingIOError, IOError):
                    if time.monotonic() >= deadline:
                        self.fh.close()
                        self.fh = None
                        raise SessionBusyError(f"Session '{self.sid}' đang được xử lý bởi một tiến trình khác (chờ vượt quá {self.tout}s).")
                    time.sleep(0.05)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.fh:
                try:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                self.fh.close()
                self.fh = None

    return _LockContext(session_id, timeout)

def _normalize_session(session_id: str, data: Optional[Dict[str, Any]], system_prompt: str) -> Dict[str, Any]:
    """Đảm bảo cấu trúc session luôn hợp lệ và đầy đủ các trường, loại bỏ hoàn toàn nguy cơ KeyError."""
    if not isinstance(data, dict):
        data = {}
    msgs = data.get("messages")
    if not isinstance(msgs, list) or not msgs:
        msgs = [{"role": "system", "content": system_prompt}]
    
    turn_count = data.get("turn_count", 0)
    try:
        turn_count = int(turn_count)
    except (ValueError, TypeError):
        turn_count = 0

    return {
        "session_id": session_id,
        "messages": msgs,
        "system_prompt": data.get("system_prompt") or system_prompt,
        "turn_count": turn_count,
        "created_at": data.get("created_at") or time.time(),
        "updated_at": time.time(),
        "schema_version": 2
    }

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Tải session từ Hot Plane (RAM) hoặc phục hồi tuần tự từ Cold Plane (True Delta WAL với CRC32) trên đĩa."""
    now = time.time()

    def _apply_baseline(sess: Dict[str, Any]) -> Dict[str, Any]:
        msgs = sess.get("messages", [])
        sess["_baseline_len"] = len(msgs)
        head_c = msgs[0].get("content", "") if msgs else ""
        sess["_baseline_head"] = head_c[:64] if isinstance(head_c, str) else ""
        return sess

    # 1. Hot Plane (RAM)
    if session_id in _HOT_SESSIONS:
        cached = _HOT_SESSIONS[session_id]
        if now - cached.get("updated_at", 0) > SESSION_TTL_SECONDS:
            delete_session(session_id)
            return None
        return _apply_baseline(copy.deepcopy(cached))

    # 2. Cold Plane (WAL: .wal.jsonl)
    wal_path = get_session_file(session_id)
    if os.path.exists(wal_path):
        current_session = None
        try:
            with open(wal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        crc = record.get("crc")
                        payload_str = record.get("payload")
                        if payload_str is None or crc is None:
                            continue
                        calc_crc = zlib.crc32(payload_str.encode("utf-8")) & 0xFFFFFFFF
                        if calc_crc != crc:
                            sys.stderr.write(f"Warning: Corrupt WAL frame in session {session_id} (CRC mismatch)\n")
                            continue

                        p = json.loads(payload_str)
                        rec_type = p.get("type")

                        if rec_type == "init":
                            current_session = {
                                "session_id": p.get("session_id", session_id),
                                "system_prompt": p.get("system_prompt", ""),
                                "messages": list(p.get("messages", [])),
                                "turn_count": p.get("turn", 0),
                                "created_at": p.get("created_at", record.get("ts", now)),
                                "updated_at": record.get("ts", now),
                                "schema_version": 2
                            }
                        elif rec_type == "turn":
                            if current_session is None:
                                current_session = {
                                    "session_id": session_id,
                                    "system_prompt": "",
                                    "messages": [],
                                    "turn_count": 0,
                                    "created_at": record.get("ts", now),
                                    "updated_at": record.get("ts", now),
                                    "schema_version": 2
                                }
                            delta = p.get("delta", [])
                            if isinstance(delta, list):
                                current_session["messages"].extend(delta)
                            current_session["turn_count"] = p.get("turn", current_session["turn_count"] + 1)
                            current_session["updated_at"] = p.get("ts", record.get("ts", now))
                        else:
                            # rec_type == "snapshot" hoặc legacy full payload
                            current_session = {
                                "session_id": p.get("session_id", session_id),
                                "system_prompt": p.get("system_prompt", ""),
                                "messages": list(p.get("messages", [])),
                                "turn_count": p.get("turn", p.get("turn_count", 0)),
                                "created_at": p.get("created_at", now),
                                "updated_at": p.get("updated_at", record.get("ts", now)),
                                "schema_version": 2
                            }
                    except Exception as parse_err:
                        sys.stderr.write(f"Warning: Invalid WAL line in session {session_id}: {parse_err}\n")
                        continue

            if current_session is not None:
                if now - current_session.get("updated_at", 0) > SESSION_TTL_SECONDS:
                    delete_session(session_id)
                    return None
                current_session = _apply_baseline(current_session)
                _HOT_SESSIONS[session_id] = copy.deepcopy(current_session)
                return current_session
        except Exception as e:
            sys.stderr.write(f"Error reading WAL for session {session_id}: {e}\n")

    # 3. Fallback: Legacy file .json
    legacy_path = get_legacy_session_file(session_id)
    if os.path.exists(legacy_path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            if now - legacy_data.get("updated_at", 0) > SESSION_TTL_SECONDS:
                delete_session(session_id)
                return None
            legacy_data = _apply_baseline(legacy_data)
            _HOT_SESSIONS[session_id] = copy.deepcopy(legacy_data)
            if save_session(session_id, legacy_data):
                try:
                    os.remove(legacy_path)
                except Exception:
                    pass
            return legacy_data
        except Exception:
            return None

    return None

def _prune_session_messages(messages: List[Dict[str, Any]], max_bytes: int = MAX_SESSION_BYTES) -> List[Dict[str, Any]]:
    """
    Tỉa tin nhắn lịch sử đảm bảo:
    1. Giữ nguyên messages[0] nếu là system prompt.
    2. Cắt theo ranh giới lượt hội thoại hoàn chỉnh (User Turn Boundary, role == 'user')
       để tuyệt đối không bao giờ để lại orphan 'tool' message gây lỗi 400 Bad Request trên API.
    3. Giảm kích thước UTF-8 xuống dưới max_bytes.
    4. Nếu ngay cả turn gần nhất vẫn vượt quá max_bytes, truncate nội dung text của message dài nhất.
    """
    if not messages:
        return []
    if len(messages) <= 1:
        return messages

    # Ngân sách an toàn trừ hao 2KB cho metadata khác của session
    budget = max(1024, max_bytes - 2048)

    # Kiểm tra kích thước hiện tại
    current_json = json.dumps({"messages": messages}, ensure_ascii=False, indent=2)
    if len(current_json.encode("utf-8")) <= budget:
        return messages

    has_system = messages[0].get("role") == "system"
    system_prefix = [messages[0]] if has_system else []
    start_search = 1 if has_system else 0

    user_indices = [i for i in range(start_search, len(messages)) if messages[i].get("role") == "user"]
    if not user_indices:
        pruned = messages
    else:
        # Tối ưu O(1) full serialization bằng ước lượng kích thước theo byte UTF-8 và tổng hậu tố (suffix sum)
        msg_lens = [len(json.dumps(m, ensure_ascii=False).encode("utf-8")) for m in messages]
        suffix_sum = [0] * (len(messages) + 1)
        for i in range(len(messages) - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + msg_lens[i]

        sys_len = msg_lens[0] if has_system else 0
        target_idx = None
        for idx in user_indices:
            # Dự trù thêm 50% overhead cho indentation và wrapper
            if (sys_len + suffix_sum[idx]) * 1.5 <= budget:
                target_idx = idx
                break

        if target_idx is not None:
            candidate = system_prefix + messages[target_idx:]
            if len(json.dumps({"messages": candidate}, ensure_ascii=False, indent=2).encode("utf-8")) <= budget:
                return candidate
            # Nếu do indent vượt nhẹ budget, duyệt tiếp các user turn kế tiếp
            start_pos = user_indices.index(target_idx) + 1
            for next_idx in user_indices[start_pos:]:
                cand = system_prefix + messages[next_idx:]
                if len(json.dumps({"messages": cand}, ensure_ascii=False, indent=2).encode("utf-8")) <= budget:
                    return cand
            pruned = system_prefix + messages[user_indices[-1]:]
        else:
            pruned = system_prefix + messages[user_indices[-1]:]

    # Nếu đã cắt hết các lượt cũ mà vẫn vượt budget, truncate content của các tin nhắn quá dài
    c_bytes = len(json.dumps({"messages": pruned}, ensure_ascii=False, indent=2).encode("utf-8"))
    if c_bytes > budget:
        pruned_copy = [dict(m) for m in pruned]
        pruned = pruned_copy
        for msg in reversed(pruned):
            content = msg.get("content")
            if isinstance(content, str) and len(content) > 10000:
                msg["content"] = content[:10000] + "\n<<MESSAGE CONTENT TRUNCATED DUE TO SIZE LIMIT>>"
                c_bytes = len(json.dumps({"messages": pruned}, ensure_ascii=False, indent=2).encode("utf-8"))
                if c_bytes <= budget:
                    break

    return pruned

def save_session(
    session_id: str,
    data: Dict[str, Any],
    delta_messages: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    Ghi session theo chuẩn True Delta Append-Only (O(n)):
    - Khởi tạo (Turn 1): Ghi 1 dòng 'init' (system prompt + tin nhắn ban đầu).
    - Các lượt tiếp theo (Turn 2..N): CHỈ append đúng các message mới sinh ra (delta), O(1) mỗi turn.
    - Compaction: Ép ghi snapshot khi:
      1. Messages bị cắt tỉa (pruning) do vượt MAX_SESSION_BYTES.
      2. Kích thước file WAL > 16MB hoặc số dòng WAL >= WAL_COMPACT_FRAMES (30).
      3. Phát hiện lệch baseline head hoặc số message nhỏ hơn _baseline_len.
    """
    try:
        wal_path = get_session_file(session_id)
        dir_name = os.path.dirname(wal_path)
        os.makedirs(dir_name, exist_ok=True)

        is_new_file = not os.path.exists(wal_path) or os.path.getsize(wal_path) == 0

        orig_len = len(data.get("messages", []))
        if "messages" in data and isinstance(data["messages"], list):
            data["messages"] = _prune_session_messages(data["messages"], MAX_SESSION_BYTES)

        curr_msgs = data.get("messages", [])
        curr_len = len(curr_msgs)
        curr_head = curr_msgs[0].get("content", "")[:64] if curr_msgs and isinstance(curr_msgs[0].get("content"), str) else ""

        base_len = data.get("_baseline_len", 0)
        base_head = data.get("_baseline_head", "")

        should_compact = False
        if not is_new_file:
            # Kiểm tra pruning hoặc lệch baseline
            if curr_len < orig_len or (base_len > 0 and curr_len < base_len) or (base_head and curr_head != base_head):
                should_compact = True
            else:
                try:
                    st = os.stat(wal_path)
                    if st.st_size > 16 * 1024 * 1024:
                        should_compact = True
                    else:
                        with open(wal_path, "r", encoding="utf-8") as f:
                            line_count = sum(1 for _ in f)
                        if line_count >= WAL_COMPACT_FRAMES:
                            should_compact = True
                except Exception:
                    pass

        now = time.time()

        if should_compact:
            # Ghi snapshot toàn vẹn thu gọn
            snap_payload = {
                "type": "snapshot",
                "session_id": session_id,
                "system_prompt": data.get("system_prompt", ""),
                "turn": data.get("turn_count", 0),
                "messages": curr_msgs,
                "created_at": data.get("created_at", now),
                "updated_at": now,
                "schema_version": 2
            }
            snap_str = json.dumps(snap_payload, ensure_ascii=False, separators=(',', ':'))
            crc = zlib.crc32(snap_str.encode("utf-8")) & 0xFFFFFFFF
            wal_record = json.dumps({"crc": crc, "ts": now, "payload": snap_str}, ensure_ascii=False) + "\n"

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                    temp_path = tf.name
                    tf.write(wal_record)
                    tf.flush()
                    os.fsync(tf.fileno())
                os.replace(temp_path, wal_path)
                temp_path = None
                dfd = os.open(dir_name, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
            data["_baseline_len"] = curr_len
            data["_baseline_head"] = curr_head
        else:
            if is_new_file:
                # Dòng đầu tiên của file: Khởi tạo session (init frame)
                init_payload = {
                    "type": "init",
                    "session_id": session_id,
                    "system_prompt": data.get("system_prompt", ""),
                    "turn": data.get("turn_count", 1),
                    "messages": curr_msgs,
                    "created_at": data.get("created_at", now),
                    "schema_version": 2
                }
                payload_str = json.dumps(init_payload, ensure_ascii=False, separators=(',', ':'))
            else:
                # Dòng tiếp theo: Chỉ ghi Delta messages mới sinh ra trong turn này
                actual_delta = delta_messages
                if actual_delta is None:
                    if base_len > 0 and curr_len >= base_len:
                        actual_delta = curr_msgs[base_len:]
                    elif session_id in _HOT_SESSIONS:
                        prev_msgs = _HOT_SESSIONS[session_id].get("messages", [])
                        if curr_len >= len(prev_msgs) and curr_msgs[:len(prev_msgs)] == prev_msgs:
                            actual_delta = curr_msgs[len(prev_msgs):]
                        else:
                            actual_delta = curr_msgs
                    else:
                        actual_delta = curr_msgs

                turn_payload = {
                    "type": "turn",
                    "turn": data.get("turn_count", 1),
                    "delta": actual_delta,
                    "ts": now
                }
                payload_str = json.dumps(turn_payload, ensure_ascii=False, separators=(',', ':'))

            crc = zlib.crc32(payload_str.encode("utf-8")) & 0xFFFFFFFF
            wal_record = json.dumps({"crc": crc, "ts": now, "payload": payload_str}, ensure_ascii=False) + "\n"

            with open(wal_path, "a", encoding="utf-8") as f:
                f.write(wal_record)
                f.flush()
                os.fsync(f.fileno())

            if is_new_file:
                dfd = os.open(dir_name, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)

            data["_baseline_len"] = curr_len
            data["_baseline_head"] = curr_head

        # Cập nhật Hot Plane (RAM) sau khi ghi đĩa thành công
        _HOT_SESSIONS[session_id] = copy.deepcopy(data)
        return True
    except Exception as e:
        sys.stderr.write(f"Error saving session {session_id}: {str(e)}\n")
        return False

def cleanup_expired_offload_files(workspace: str) -> None:
    """Tự động dọn dẹp các file offload cũ quá TTL (4 tiếng) hoặc khi tổng dung lượng > 512MB."""
    offload_dir = os.path.join(workspace, ".deepseek", "offload")
    if not os.path.exists(offload_dir):
        return
    try:
        now = time.time()
        files = []
        total_size = 0
        for entry in os.scandir(offload_dir):
            if entry.is_file() and entry.name.endswith(".txt"):
                st = entry.stat()
                total_size += st.st_size
                files.append((entry.path, st.st_mtime, st.st_size))

        remaining = []
        for path, mtime, sz in files:
            if now - mtime > OFFLOAD_TTL_SECONDS:
                try:
                    os.remove(path)
                    total_size -= sz
                except OSError:
                    pass
            else:
                remaining.append((path, mtime, sz))

        if total_size > MAX_OFFLOAD_DIR_BYTES:
            remaining.sort(key=lambda x: x[1])
            for path, _, sz in remaining:
                try:
                    os.remove(path)
                    total_size -= sz
                except OSError:
                    pass
                if total_size <= MAX_OFFLOAD_DIR_BYTES:
                    break
    except Exception:
        pass

def pack_tool_result(
    content: str,
    tool: str,
    workspace: str,
    session_id: Optional[str] = None,
    turn: int = 1
) -> str:
    """
    Nếu len(content) > OFFLOAD_THRESHOLD (32_000 chars), lưu RAW 100% xuống .deepseek/offload/<sha256>.txt.
    Tạo .deepseek/.gitignore nếu chưa có.
    Trả về JSON envelope tóm tắt preview_head (1000 chars) và preview_tail (1000 chars).
    """
    if len(content) <= OFFLOAD_THRESHOLD:
        return content

    try:
        cleanup_expired_offload_files(workspace)
        deepseek_dir = os.path.join(workspace, ".deepseek")
        offload_dir = os.path.join(deepseek_dir, "offload")
        os.makedirs(offload_dir, exist_ok=True)

        gi_path = os.path.join(deepseek_dir, ".gitignore")
        if not os.path.exists(gi_path):
            try:
                with open(gi_path, "w", encoding="utf-8") as f:
                    f.write("*\n!.gitignore\n")
            except OSError:
                pass

        raw_bytes = content.encode("utf-8")
        h = hashlib.sha256(raw_bytes).hexdigest()[:16]
        offload_file = os.path.join(offload_dir, f"{h}.txt")

        if not os.path.exists(offload_file):
            temp_p = None
            try:
                with tempfile.NamedTemporaryFile("wb", dir=offload_dir, delete=False) as tf:
                    temp_p = tf.name
                    tf.write(raw_bytes)
                    tf.flush()
                    os.fsync(tf.fileno())
                os.replace(temp_p, offload_file)
                temp_p = None
            finally:
                if temp_p and os.path.exists(temp_p):
                    try:
                        os.remove(temp_p)
                    except OSError:
                        pass

        rel_path = os.path.relpath(offload_file, workspace)
        envelope = {
            "status": "OFFLOADED",
            "tool": tool,
            "handle": h,
            "rel_path": rel_path,
            "preview_head": content[:OFFLOAD_PREVIEW_CHARS],
            "preview_tail": content[-OFFLOAD_PREVIEW_CHARS:],
            "total_chars": len(content),
            "guidance": (
                f"Tool output ({len(content)} chars) vượt ngưỡng 32KB đã được lưu RAW tại '{rel_path}'. "
                f"Hãy dùng tool 'read_file' với tham số 'byte_offset' và path='{rel_path}' để đọc phân trang an toàn."
            )
        }
        return json.dumps(envelope, ensure_ascii=False, indent=2)
    except Exception as e:
        return content[:OFFLOAD_THRESHOLD] + f"\n<<OUTPUT TRUNCATED DUE TO OFFLOAD ERROR: {str(e)}>>"

def delete_session(session_id: str) -> bool:
    """Xóa session khỏi RAM và xóa file WAL (.wal.jsonl) cùng file legacy trên đĩa."""
    _HOT_SESSIONS.pop(session_id, None)
    wal_path = get_session_file(session_id)
    legacy_path = get_legacy_session_file(session_id)
    deleted = False
    for p in (wal_path, legacy_path):
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted = True
            except Exception:
                pass
    return deleted

def delete_all_sessions() -> int:
    """Xóa sạch toàn bộ session trong RAM và file WAL/legacy trên đĩa (không xóa file .lock)."""
    _HOT_SESSIONS.clear()
    if not os.path.exists(SESSIONS_DIR):
        return 0
    count = 0
    now = time.time()
    for fname in os.listdir(SESSIONS_DIR):
        fpath = os.path.join(SESSIONS_DIR, fname)
        if fname.endswith((".wal.jsonl", ".json")):
            try:
                os.remove(fpath)
                count += 1
            except Exception:
                pass
        elif fname.startswith("tmp"):
            try:
                if now - os.path.getmtime(fpath) > 60:
                    os.remove(fpath)
            except Exception:
                pass
    return count

def cleanup_expired_sessions() -> None:
    """Tự động dọn dẹp các session quá hạn TTL (4h) trong RAM và trên đĩa."""
    now = time.time()
    expired_keys = [sid for sid, d in _HOT_SESSIONS.items() if now - d.get("updated_at", 0) > SESSION_TTL_SECONDS]
    for sid in expired_keys:
        _HOT_SESSIONS.pop(sid, None)

    if not os.path.exists(SESSIONS_DIR):
        return
    try:
        for fname in os.listdir(SESSIONS_DIR):
            fpath = os.path.join(SESSIONS_DIR, fname)
            if fname.endswith((".wal.jsonl", ".json")):
                try:
                    if now - os.path.getmtime(fpath) > SESSION_TTL_SECONDS:
                        os.remove(fpath)
                except Exception:
                    pass
            elif fname.startswith("tmp"):
                try:
                    if now - os.path.getmtime(fpath) > 3600:
                        os.remove(fpath)
                except Exception:
                    pass
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Local Workspace Tools & Schemas (Read-only vs Full Execution)
# ---------------------------------------------------------------------------
READ_ONLY_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Đọc nội dung file trong workspace (hỗ trợ phân trang theo dòng hoặc theo byte offset chuẩn Google view_file).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối trong workspace"},
                    "start_line": {"type": "integer", "description": "Dòng bắt đầu đọc (1-indexed, mặc định: 1)"},
                    "line_count": {"type": "integer", "description": "Số dòng cần đọc tối đa (mặc định: đọc tối đa 2000 dòng hoặc tới 24KB)"},
                    "line_numbers": {"type": "boolean", "description": "Nếu True, thêm tiền tố số dòng '<line_number>: ' vào đầu mỗi dòng trong content (mặc định: True)."},
                    "byte_offset": {"type": "integer", "description": "Byte offset để bắt đầu đọc nội dung (phân trang theo byte chuẩn Google view_file khi đọc file/offload lớn)."}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Liệt kê danh sách các file trong workspace để khảo sát cấu trúc dự án",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn thư mục cần xem trong workspace (để trống hoặc '.' để xem từ root workspace)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": "Tìm kiếm chuỗi hoặc biểu thức chính quy (regex) trong các file của workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Thư mục hoặc file tương đối cần tìm kiếm (để trống hoặc '.' để tìm từ root)"},
                    "pattern": {"type": "string", "description": "Chuỗi hoặc biểu thức regex cần tìm"},
                    "max_results": {"type": "integer", "description": "Số lượng kết quả tối đa cần trả về (mặc định: 100)"}
                },
                "required": ["pattern"]
            }
        }
    }
]

LOCAL_TOOLS_SCHEMA = READ_ONLY_TOOLS_SCHEMA + [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Ghi đè hoặc tạo mới một file trong workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối trong workspace"},
                    "content": {"type": "string", "description": "Nội dung đầy đủ của file"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file_replace",
            "description": "Thay thế chính xác một khối text trong file bằng nội dung mới (hỗ trợ khoanh vùng dải dòng start_line..end_line chuẩn Google replace_file_content).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối"},
                    "search_block": {"type": "string", "description": "Đoạn text hiện có cần thay thế"},
                    "replace_block": {"type": "string", "description": "Đoạn text mới thay thế vào"},
                    "start_line": {"type": "integer", "description": "Dòng bắt đầu phạm vi thay thế (1-indexed, chuẩn Google replace_file_content)."},
                    "end_line": {"type": "integer", "description": "Dòng kết thúc phạm vi thay thế (1-indexed, chuẩn Google replace_file_content)."}
                },
                "required": ["path", "search_block", "replace_block"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Thực thi lệnh shell trong workspace (linter, test runner, build). Bị chặn nếu lệnh nguy hiểm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Lệnh shell cần chạy"}
                },
                "required": ["command"]
            }
        }
    }
]

TOOL_SYSTEM_DIRECTIVE = (
    "\n\n[CRITICAL TOOL PROTOCOL & EXECUTION RULES]\n"
    "1. ZERO-PREAMBLE CONSTRAINT: When you need to inspect files, you MUST invoke the tool immediately. "
    "NEVER output conversational filler, introductory remarks, or preamble text before a tool call.\n"
    "2. STRICT SCHEMA ADHERENCE: Use ONLY the exact tool names and parameter keys "
    "defined in the schema provided to you. Do not invent aliases or extra parameters.\n"
    "3. NO RAW MARKUP: Rely strictly on native tool calls. Never emit raw DSML or XML tags into conversation text.\n"
    "4. SYNTHESIS: Provide your comprehensive analysis or verdict only after all necessary tools have returned their results.\n"
    "5. PAGINATION PROTOCOL: If 'read_file' returns 'truncated': true with 'next_start_line' or 'next_byte_offset', "
    "invoke 'read_file' again with the specified pagination parameter to read subsequent content. Never guess missing file lines.\n"
    "6. LINE NUMBERS PROTOCOL: 'read_file' prefixes each line with '<line_number>: ' for exact navigation. "
    "When applying edits via 'edit_file_replace', you MUST STRIP these line number prefixes from 'search_block' and 'replace_block'. "
    "To insert code before/after existing code, preserve the anchor search_block in your replace_block."
)

def serialize_assistant_message(msg: Any) -> Dict[str, Any]:
    """Chuyển đổi assistant message thành dictionary tương thích JSON và DeepSeek API."""
    d: Dict[str, Any] = {
        "role": "assistant",
        "content": getattr(msg, "content", "") or ""
    }
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        d["reasoning_content"] = reasoning
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [{
            "id": getattr(tc, "id", f"call_{i}"),
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        } for i, tc in enumerate(tool_calls)]
    return d

def sanitize_messages_for_api(messages: List[Dict[str, Any]], turn_start_idx: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loại bỏ reasoning_content khỏi messages trước khi gửi lên API.
    Nếu turn_start_idx is None, loại bỏ reasoning_content khỏi 100% messages.
    Nếu turn_start_idx được chỉ định, chỉ loại bỏ ở các message cũ (idx < turn_start_idx).
    """
    cutoff = len(messages) if turn_start_idx is None else turn_start_idx
    cleaned = []
    for idx, m in enumerate(messages):
        if m.get("role") == "assistant" and "reasoning_content" in m:
            if idx < cutoff:
                cleaned.append({k: v for k, v in m.items() if k != "reasoning_content"})
            else:
                cleaned.append(dict(m))
        else:
            cleaned.append(m)
    return cleaned

DANGEROUS_ENV_PREFIX_KEYS = ("PATH", "SHELL", "BASH_ENV", "ENV", "IFS")
DANGEROUS_ENV_PREFIX_STARTS = ("LD_", "DYLD_", "GIT_CONFIG")

def parse_env_prefix(tokens: List[str]) -> Tuple[Dict[str, str], List[str], Optional[str]]:
    """
    Tách các token 'KEY=VALUE' ở đầu danh sách lệnh argv thành extra_env dictionary.
    Chặn đứng các biến môi trường nhạy cảm/nguy hiểm (LD_*, DYLD_*, PATH, GIT_CONFIG*...).
    """
    extra_env: Dict[str, str] = {}
    cmd_tokens: List[str] = []
    i = 0
    # Xử lý nếu binary 'env' hoặc đường dẫn kết thúc bằng '/env' được gọi ở đầu
    if tokens and (tokens[0] == "env" or tokens[0].endswith("/env")):
        i = 1
        while i < len(tokens) and tokens[i] in ("-i", "--ignore-environment"):
            i += 1

    while i < len(tokens):
        tok = tokens[i]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", tok)
        if m:
            k = m.group(1)
            v = m.group(2)
            if k in DANGEROUS_ENV_PREFIX_KEYS or any(k.startswith(p) for p in DANGEROUS_ENV_PREFIX_STARTS):
                return {}, [], f"Biến môi trường nguy hiểm '{k}' bị chặn bởi Guardrail an toàn!"
            extra_env[k] = v
            i += 1
        else:
            cmd_tokens = tokens[i:]
            break

    if not cmd_tokens:
        return {}, [], "Không tìm thấy lệnh thực thi sau tiền tố env!"
    return extra_env, cmd_tokens, None

def _run_capped(
    cmd: Union[str, List[str]],
    cwd: str,
    env: dict,
    timeout: int = 90,
    cap: int = MAX_SHELL_OUT,
    kill_on_cap: bool = True
) -> Tuple[str, int, str]:
    """
    Thực thi lệnh với process group riêng biệt, shell=False an toàn, streaming output có trần và timeout.
    Trả về (output_text, exit_code, status_flag) với status_flag in ("ok", "capped", "timeout", "error").
    """
    if isinstance(cmd, str):
        try:
            tokens = shlex.split(cmd)
        except Exception as e:
            return f"Command syntax error (shlex): {str(e)}", -1, "error"
    elif isinstance(cmd, list):
        tokens = list(cmd)
    else:
        return "Invalid command type (must be str or list)", -1, "error"

    if not tokens:
        return "Empty command", -1, "error"

    extra_env, parsed_argv, env_err = parse_env_prefix(tokens)
    if env_err:
        return env_err, -1, "error"

    effective_env = dict(env) if env is not None else build_scrubbed_env()
    effective_env.update(extra_env)

    effective_cwd = cwd if cwd is not None else os.getcwd()

    # Resolve executable path an toàn
    executable = parsed_argv[0]
    resolved_path = shutil.which(executable, path=effective_env.get("PATH"))
    if not resolved_path:
        if os.path.isabs(executable) or os.path.dirname(executable):
            candidate = os.path.realpath(os.path.join(effective_cwd, executable))
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                resolved_path = candidate
        if not resolved_path:
            return f"Command not found: {executable}", -1, "error"

    argv = [resolved_path] + parsed_argv[1:]

    try:
        proc = subprocess.Popen(
            argv,
            shell=False,
            cwd=effective_cwd,
            env=effective_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Tạo process group riêng để kill toàn bộ cây tiến trình
        )
        pgid = proc.pid
    except Exception as e:
        return f"Process spawn error: {str(e)}", -1, "error"

    rfd = proc.stdout.fileno()
    try:
        os.set_blocking(rfd, False)
    except Exception:
        pass

    head_cap = cap // 2
    tail_cap = cap - head_cap
    head_buf = bytearray()
    tail_buf = bytearray()
    total_bytes_read = 0
    capped = False
    flag = "ok"
    deadline = time.monotonic() + timeout
    drain_deadline = None

    while True:
        now = time.monotonic()
        if drain_deadline is not None and now > drain_deadline:
            break

        poll_res = proc.poll()
        if poll_res is not None and drain_deadline is None:
            drain_deadline = now + 0.25

        try:
            r_ready, _, _ = select.select([rfd], [], [], 0.02)
            if r_ready:
                chunk = os.read(rfd, 64 * 1024)
                if not chunk:
                    break
                total_bytes_read += len(chunk)
                if len(head_buf) < head_cap:
                    needed = head_cap - len(head_buf)
                    take = min(len(chunk), needed)
                    head_buf.extend(chunk[:take])
                    chunk_tail = chunk[take:]
                else:
                    chunk_tail = chunk

                if chunk_tail:
                    tail_buf.extend(chunk_tail)
                    if len(tail_buf) > tail_cap:
                        tail_buf = tail_buf[-tail_cap:]

                if total_bytes_read >= cap:
                    capped = True
                    if kill_on_cap:
                        flag = "capped"
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                        drain_deadline = time.monotonic() + 0.10
        except (BlockingIOError, InterruptedError):
            pass
        except (OSError, ValueError):
            break

        if poll_res is None and now > deadline:
            flag = "timeout"
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            drain_deadline = time.monotonic() + 0.10

        if poll_res is not None and drain_deadline is not None and time.monotonic() > drain_deadline:
            break

    if flag == "ok":
        if capped:
            flag = "capped"
        elif proc.poll() is None and time.monotonic() > deadline:
            flag = "timeout"

    if proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    try:
        proc.stdout.close()
    except Exception:
        pass

    if total_bytes_read <= cap:
        text = (head_buf + tail_buf).decode("utf-8", errors="replace")
    else:
        dropped = total_bytes_read - len(head_buf) - len(tail_buf)
        head_txt = head_buf.decode("utf-8", errors="replace")
        tail_txt = tail_buf.decode("utf-8", errors="replace")
        marker = (
            f"\n\n... [OUTPUT TRUNCATED: DROPPED {dropped} BYTES IN THE MIDDLE | "
            f"KEPT FIRST {len(head_buf)} BYTES HEAD + LAST {len(tail_buf)} BYTES TAIL (CAP={cap} BYTES)] ...\n\n"
        )
        text = head_txt + marker + tail_txt

    if flag == "capped":
        text += f"\n<<OUTPUT TRUNCATED AT {cap} BYTES>>"
    elif flag == "timeout":
        text += f"\n<<COMMAND TIMED OUT AFTER {timeout} SECONDS>>"

    rc = proc.returncode if proc.returncode is not None else -9
    return text, rc, flag

def _read_file_envelope(
    workspace: str,
    rel_path: str,
    start_line: Any = 1,
    line_count: Any = None,
    line_numbers: Any = True,
    byte_offset: Any = None
) -> str:
    """Đọc file hỗ trợ phân trang Byte Offset (chuẩn Google view_file) và Line-based, chống OOM, trả JSON envelope."""
    safe, full_path = is_safe_path(workspace, rel_path, allow_offload_read=True)
    if not safe:
        return json.dumps({"status": "ERROR", "code": "E_PATH_DENIED", "message": f"Path '{rel_path}' nằm ngoài workspace!"}, ensure_ascii=False)

    try:
        st = os.stat(full_path)
    except OSError as e:
        return json.dumps({"status": "ERROR", "code": "E_NOT_FOUND", "message": f"File '{rel_path}' không tồn tại: {str(e)}"}, ensure_ascii=False)

    if not stat.S_ISREG(st.st_mode):
        return json.dumps({"status": "ERROR", "code": "E_NOT_REGULAR", "message": f"'{rel_path}' không phải regular file (chặn FIFO / device file)."}, ensure_ascii=False)

    def _as_int(v, default):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    offset = max(0, _as_int(byte_offset, 0))

    if st.st_size == 0:
        return json.dumps({
            "status": "OK",
            "path": rel_path,
            "byte_offset": offset,
            "start_line": 1,
            "end_line": 0,
            "next_start_line": None,
            "next_byte_offset": None,
            "truncated": False,
            "eof": True,
            "line_numbers": False,
            "file_size": 0,
            "bytes_returned": 0,
            "next_step": None,
            "content": ""
        }, ensure_ascii=False)

    # 1. Nếu có byte_offset được chỉ định: Dùng Byte Offset Slicing chuẩn Google view_file
    if byte_offset is not None and (str(byte_offset) != "" and _as_int(byte_offset, -1) >= 0):
        try:
            with open(full_path, "rb") as f:
                head = f.read(8192)
                if b"\x00" in head:
                    return json.dumps({"status": "ERROR", "code": "E_BINARY", "message": f"'{rel_path}' là file nhị phân, không hỗ trợ xem text."}, ensure_ascii=False)
                f.seek(offset)
                raw_chunk = f.read(MAX_READ_BYTES)

            while raw_chunk:
                try:
                    valid_text = raw_chunk.decode("utf-8")
                    break
                except UnicodeDecodeError as ue:
                    if ue.start + 1 >= len(raw_chunk) or ue.end == len(raw_chunk):
                        raw_chunk = raw_chunk[:ue.start]
                    else:
                        valid_text = raw_chunk.decode("utf-8", errors="replace")
                        break
            else:
                valid_text = ""

            bytes_read = len(raw_chunk)
            next_offset = offset + bytes_read
            truncated = (next_offset < st.st_size and bytes_read > 0)
            eof = (not truncated)

            return json.dumps({
                "status": "OK",
                "path": rel_path,
                "byte_offset": offset,
                "bytes_returned": bytes_read,
                "next_byte_offset": next_offset if truncated else None,
                "truncated": truncated,
                "file_size": st.st_size,
                "eof": eof,
                "next_step": f"call read_file with path='{rel_path}' byte_offset={next_offset}" if truncated else None,
                "content": valid_text
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "ERROR", "code": "E_READ_FAILED", "message": str(e)}, ensure_ascii=False)

    # 2. Ngược lại: Đọc theo dòng (Line-based)
    start = max(1, _as_int(start_line, 1))
    want = MAX_READ_LINES if line_count is None else max(1, min(_as_int(line_count, MAX_READ_LINES), MAX_READ_LINES))
    use_line_numbers = True if line_numbers is None else bool(line_numbers)

    out = []
    tail = b""
    line_no = 0
    emitted_bytes = 0
    emitted_chars = 0
    next_start = None
    truncated = False
    fallback_to_byte_slice = False

    try:
        with open(full_path, "rb") as f:
            head = f.read(8192)
            if b"\x00" in head:
                return json.dumps({"status": "ERROR", "code": "E_BINARY", "message": f"'{rel_path}' là file nhị phân, không hỗ trợ xem text."}, ensure_ascii=False)
            f.seek(0)

            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                tail += chunk
                parts = tail.split(b"\n")
                tail = parts.pop()

                stop = False
                for raw in parts:
                    line_no += 1
                    if line_no < start:
                        continue
                    line_text = raw.decode("utf-8", errors="replace") + "\n"
                    line_chars = len(line_text) + (len(str(line_no)) + 2 if use_line_numbers else 0)
                    if len(out) >= want or (out and (emitted_bytes + len(raw) > MAX_READ_BYTES or emitted_chars + line_chars > MAX_ENVELOPE_CONTENT_CHARS)):
                        next_start = line_no
                        truncated = True
                        stop = True
                        break
                    out.append(line_text)
                    emitted_bytes += len(raw) + 1
                    emitted_chars += line_chars
                if stop:
                    break

                if len(tail) >= MAX_READ_BYTES:
                    if line_no + 1 >= start:
                        truncated = True
                        if not out:
                            fallback_to_byte_slice = True
                        break
                    else:
                        tail = b""

            if not truncated and tail:
                line_no += 1
                if line_no >= start:
                    tail_text = tail.decode("utf-8", errors="replace")
                    tail_chars = len(tail_text) + (len(str(line_no)) + 2 if use_line_numbers else 0)
                    if len(out) >= want or (out and (emitted_bytes + len(tail) > MAX_READ_BYTES or emitted_chars + tail_chars > MAX_ENVELOPE_CONTENT_CHARS)):
                        next_start = line_no
                        truncated = True
                    else:
                        out.append(tail_text)
                        emitted_bytes += len(tail)
                        emitted_chars += tail_chars
    except Exception as e:
        return json.dumps({"status": "ERROR", "code": "E_READ_FAILED", "message": str(e)}, ensure_ascii=False)

    # Chuyển đổi mượt mà sang byte_offset khi dòng đơn lẻ > 24KB (Gate G1 - Xóa bỏ vĩnh viễn E_LINE_TOO_LONG)
    if fallback_to_byte_slice or (truncated and not out and tail):
        return _read_file_envelope(workspace, rel_path, byte_offset=0)

    if truncated and not out and not tail:
        truncated = False

    if truncated and next_start is None and out:
        next_start = start + len(out)

    eof = (next_start is None and not truncated)
    end_line = start + len(out) - 1 if out else max(0, line_no)
    if use_line_numbers:
        body = "".join(f"{start + idx}: {line_str}" for idx, line_str in enumerate(out))
    else:
        body = "".join(out)

    return json.dumps({
        "status": "OK",
        "path": rel_path,
        "start_line": start,
        "end_line": end_line,
        "next_start_line": next_start,
        "truncated": truncated,
        "eof": eof,
        "line_numbers": use_line_numbers,
        "file_size": st.st_size,
        "bytes_returned": emitted_bytes,
        "next_step": f"call read_file with path='{rel_path}' start_line={next_start}" if next_start else None,
        "content": body
    }, ensure_ascii=False)

def _search_file_helper(workspace: str, rel_path: str, pattern: str, max_results: int = 100) -> str:
    """Tìm kiếm regex an toàn trong workspace."""
    safe, full_path = is_safe_path(workspace, rel_path or ".", allow_offload_read=False)
    if not safe:
        return "Error: Path ngoài workspace."
    if not os.path.exists(full_path):
        return f"Error: Path '{rel_path}' không tồn tại."

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

    matches = []
    deadline = time.monotonic() + SEARCH_TIMEOUT_SECONDS

    target_files = []
    if os.path.isfile(full_path):
        target_files.append((full_path, os.path.relpath(full_path, workspace)))
    else:
        for root, dirs, files in os.walk(full_path):
            dirs[:] = [d for d in dirs if d.lower() not in (".git", ".venv", ".deepseek", "__pycache__", "node_modules")]
            for f in sorted(files):
                f_full = os.path.join(root, f)
                target_files.append((f_full, os.path.relpath(f_full, workspace)))

    for f_full, r_path in target_files:
        if time.monotonic() > deadline or len(matches) >= max_results:
            break
        try:
            st = os.stat(f_full)
            if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_SEARCH_FILE_BYTES:
                continue
            with open(f_full, "rb") as bf:
                head = bf.read(4096)
                if b"\x00" in head:
                    continue
            with open(f_full, "r", encoding="utf-8", errors="replace") as tf:
                for lno, line in enumerate(tf, 1):
                    if len(line) > SEARCH_MAX_LINE_BYTES:
                        continue
                    if regex.search(line):
                        matches.append(f"{r_path}:{lno}: {line.rstrip()}")
                        if len(matches) >= max_results:
                            break
        except Exception:
            continue

    res_text = "\n".join(matches) if matches else f"No matches found for pattern '{pattern}'."
    return pack_tool_result(res_text, tool="search_file", workspace=workspace)

def execute_local_tool(workspace: str, name: str, args: Dict[str, Any]) -> str:
    """Thực thi tool local với chốt chặn an toàn Sandbox."""
    if name == "read_file":
        rel_path = args.get("path", "")
        start_line = args.get("start_line", 1)
        line_count = args.get("line_count", None)
        line_numbers = args.get("line_numbers", True)
        byte_offset = args.get("byte_offset", None)
        return _read_file_envelope(workspace, rel_path, start_line, line_count, line_numbers, byte_offset)
            
    elif name == "search_file":
        rel_path = args.get("path", "")
        pattern = args.get("pattern", "")
        max_results = args.get("max_results", 100)
        return _search_file_helper(workspace, rel_path, pattern, max_results)

    elif name == "write_file":
        rel_path = args.get("path", "")
        content = args.get("content", "")
        safe, full_path = is_safe_path(workspace, rel_path, allow_offload_read=False)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace hoặc bị từ chối truy cập!"
        if os.path.exists(full_path):
            st = os.stat(full_path)
            if not stat.S_ISREG(st.st_mode):
                return f"Error: '{rel_path}' không phải regular file (chặn FIFO / device file)."
        temp_p = None
        try:
            dir_name = os.path.dirname(full_path)
            os.makedirs(dir_name, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8", newline="") as tf:
                temp_p = tf.name
                tf.write(content)
                tf.flush()
                os.fsync(tf.fileno())
            if os.path.exists(full_path):
                try:
                    os.chmod(temp_p, stat.S_IMODE(os.stat(full_path).st_mode))
                except OSError:
                    pass
            else:
                try:
                    os.chmod(temp_p, 0o644)
                except OSError:
                    pass
            os.replace(temp_p, full_path)
            temp_p = None
            try:
                dfd = os.open(dir_name, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
            return f"Successfully wrote {len(content)} characters to '{rel_path}'."
        except Exception as e:
            return f"Error writing file: {str(e)}"
        finally:
            if temp_p and os.path.exists(temp_p):
                try:
                    os.remove(temp_p)
                except Exception:
                    pass
            
    elif name == "edit_file_replace":
        rel_path = args.get("path", "")
        search_block = args.get("search_block", "")
        replace_block = args.get("replace_block", "")
        start_line_arg = args.get("start_line")
        end_line_arg = args.get("end_line")

        safe, full_path = is_safe_path(workspace, rel_path, allow_offload_read=False)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace hoặc bị từ chối truy cập!"
        if not os.path.exists(full_path):
            return f"Error: File '{rel_path}' không tồn tại."
        st = os.stat(full_path)
        if not stat.S_ISREG(st.st_mode):
            return f"Error: '{rel_path}' không phải regular file (chặn FIFO / device file)."
        if st.st_size > 5 * 1024 * 1024:
            return f"Error: File '{rel_path}' quá lớn ({st.st_size} bytes > 5MB), không hỗ trợ replace trực tiếp."
        temp_p = None
        try:
            try:
                with open(full_path, "r", encoding="utf-8", newline="") as f:
                    current_text = f.read()
            except UnicodeDecodeError:
                return f"Error: File '{rel_path}' không phải định dạng UTF-8 hợp lệ, từ chối chỉnh sửa để chống hỏng dữ liệu."

            # Xử lý phạm vi dòng start_line & end_line (Line-Bounded Scope)
            is_scoped = (start_line_arg is not None and end_line_arg is not None)
            if is_scoped:
                try:
                    s_line = int(start_line_arg)
                    e_line = int(end_line_arg)
                except (ValueError, TypeError):
                    return "Error: start_line và end_line phải là số nguyên hợp lệ."

                lines = current_text.splitlines(keepends=True)
                total_lines = len(lines)
                if not (1 <= s_line <= e_line <= total_lines):
                    return f"Error: Dải dòng [{s_line}, {e_line}] không hợp lệ (tổng số dòng file: {total_lines})."

                target_scope = "".join(lines[s_line - 1 : e_line])
                prefix_scope = "".join(lines[: s_line - 1])
                suffix_scope = "".join(lines[e_line :])
                active_text = target_scope
            else:
                active_text = current_text

            if search_block not in active_text:
                if "\r\n" in active_text and "\r\n" not in search_block:
                    crlf_search = search_block.replace("\r\n", "\n").replace("\n", "\r\n")
                    if crlf_search in active_text:
                        search_block = crlf_search
                        replace_block = replace_block.replace("\r\n", "\n").replace("\n", "\r\n")
                elif "\r\n" not in active_text and "\r\n" in search_block:
                    lf_search = search_block.replace("\r\n", "\n")
                    if lf_search in active_text:
                        search_block = lf_search
                        replace_block = replace_block.replace("\r\n", "\n")

            if search_block not in active_text:
                s_lines = search_block.splitlines(keepends=True)
                s_nums = []
                s_rests = []
                is_pure_numbered = True

                for line in s_lines:
                    le = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                    content_part = line[:-len(le)] if le else line
                    m = re.match(r"^(?P<n>[1-9][0-9]{0,8}): (?P<rest>.*)$", content_part)
                    if m:
                        s_nums.append(int(m.group("n")))
                        s_rests.append(m.group("rest") + le)
                    else:
                        is_pure_numbered = False
                        break

                if is_pure_numbered and s_nums:
                    expected_run = list(range(s_nums[0], s_nums[0] + len(s_nums)))
                    if s_nums == expected_run:
                        candidate = "".join(s_rests)
                        if candidate not in active_text:
                            if "\r\n" in active_text and "\r\n" not in candidate:
                                cand_crlf = candidate.replace("\r\n", "\n").replace("\n", "\r\n")
                                if cand_crlf in active_text:
                                    candidate = cand_crlf
                            elif "\r\n" not in active_text and "\r\n" in candidate:
                                cand_lf = candidate.replace("\r\n", "\n")
                                if cand_lf in active_text:
                                    candidate = cand_lf

                        if candidate in active_text:
                            search_block = candidate
                            # Kiểm tra an toàn cho replace_block: CHỈ strip nếu 100% dòng cũng khớp numbering hợp lệ
                            r_lines = replace_block.splitlines(keepends=True)
                            r_nums = []
                            r_rests = []
                            r_pure_numbered = True
                            for r_line in r_lines:
                                r_le = "\r\n" if r_line.endswith("\r\n") else ("\n" if r_line.endswith("\n") else "")
                                r_content = r_line[:-len(r_le)] if r_le else r_line
                                rm = re.match(r"^(?P<n>[1-9][0-9]{0,8}): (?P<rest>.*)$", r_content)
                                if rm:
                                    r_nums.append(int(rm.group("n")))
                                    r_rests.append(rm.group("rest") + r_le)
                                else:
                                    r_pure_numbered = False
                                    break
                            if r_pure_numbered and r_nums:
                                r_expected = list(range(r_nums[0], r_nums[0] + len(r_nums)))
                                if r_nums == r_expected:
                                    replace_block = "".join(r_rests)

            if search_block not in active_text:
                scope_msg = f" trong dải dòng [{start_line_arg}, {end_line_arg}]" if is_scoped else ""
                return f"Error: search_block không tìm thấy{scope_msg} trong file '{rel_path}'."

            # Đảm bảo line ending của replace_block khớp với active_text
            if "\r\n" in active_text and "\r\n" not in replace_block:
                replace_block = replace_block.replace("\r\n", "\n").replace("\n", "\r\n")
            elif "\r\n" not in active_text and "\r\n" in replace_block:
                replace_block = replace_block.replace("\r\n", "\n")

            count = active_text.count(search_block)
            if count > 1:
                return f"Error: search_block xuất hiện {count} lần trong file. Phải cung cấp thêm ngữ cảnh xung quanh để xác định duy nhất."

            if is_scoped:
                new_scoped = active_text.replace(search_block, replace_block, 1)
                new_text = prefix_scope + new_scoped + suffix_scope
            else:
                new_text = active_text.replace(search_block, replace_block, 1)

            dir_name = os.path.dirname(full_path)
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8", newline="") as tf:
                temp_p = tf.name
                tf.write(new_text)
                tf.flush()
                os.fsync(tf.fileno())
            if os.path.exists(full_path):
                try:
                    os.chmod(temp_p, stat.S_IMODE(os.stat(full_path).st_mode))
                except OSError:
                    pass
            os.replace(temp_p, full_path)
            temp_p = None
            try:
                dfd = os.open(dir_name, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
            return f"Successfully replaced search block in '{rel_path}'."
        except Exception as e:
            return f"Error editing file: {str(e)}"
        finally:
            if temp_p and os.path.exists(temp_p):
                try:
                    os.remove(temp_p)
                except Exception:
                    pass
            
    elif name == "run_shell":
        cmd = args.get("command", "")
        safe, reason = check_command_safety(cmd)
        if not safe:
            return f"Command Rejected by Safety Guardrail: {reason}"
        out, rc, flag = _run_capped(cmd, workspace, build_scrubbed_env(), timeout=DEFAULT_CODER_SHELL_TIMEOUT)
        res_str = f"Exit code: {rc}\n{out}"
        return pack_tool_result(res_str, tool="run_shell", workspace=workspace)
            
    elif name in ["list_files", "list_directory"]:
        rel_path = args.get("path") or args.get("sub_dir") or ""
        if rel_path == ".":
            rel_path = ""
        safe, full_path = is_safe_path(workspace, rel_path, allow_offload_read=False)
        if not safe:
            return "Error: Path ngoài workspace."
        if not os.path.exists(full_path):
            return "Error: Directory does not exist."
        try:
            files_list = []
            for root, dirs, files in os.walk(full_path):
                dirs[:] = sorted([d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv", ".deepseek"]])
                for file in sorted(files):
                    rel = os.path.relpath(os.path.join(root, file), workspace)
                    files_list.append(rel)
                    if len(files_list) >= 200:
                        break
                if len(files_list) >= 200:
                    break
            files_list.sort()
            return json.dumps(files_list[:100], indent=2)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# Tool 1: deepseek_ask (Trực tiếp suy luận hoặc truy vấn DeepSeek Flash)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_ask(
    prompt: str,
    session_id: Optional[str] = None,
    reset_session: bool = False,
    workspace_dir: Optional[str] = None,
    max_tool_turns: int = 5,
    thinking: bool = True,
    effort: str = "high",
    system_prompt: Optional[str] = None
) -> str:
    """
    Truy vấn DeepSeek-Flash, hỗ trợ:
    1. Stateful Multi-turn bền vững (lưu trữ JSON trên đĩa qua session_id, có file lock fail-fast).
    2. Chế độ điều tra tự động (Autonomous Investigation): Nếu truyền workspace_dir, DeepSeek được cấp tool READ-ONLY
       (read_file, list_files) để tự đi đọc file và khảo sát codebase mà không cần Antigravity mớm code thủ công.
    3. Tận dụng tối đa KV Cache Prefix Matching trên GPU cluster.
    
    Args:
        prompt: Nội dung câu hỏi / lập luận phản biện / yêu cầu điều tra codebase.
        session_id: Định danh phiên làm việc để duy trì ngữ cảnh multi-turn (ví dụ: Conversation ID).
        reset_session: Nếu True, xóa sạch ngữ cảnh cũ của session_id và bắt đầu hội thoại mới.
        workspace_dir: Đường dẫn thư mục dự án cần điều tra (bật tính năng read-only tool calls).
        max_tool_turns: Số lượt đọc file tối đa trong chế độ điều tra (mặc định: 5).
        thinking: Bật (True) hoặc tắt (False) Thinking Mode. Mặc định: True.
        effort: Mức độ suy luận CoT ('low', 'high', 'max'). Mặc định: 'high'.
        system_prompt: Chỉ dẫn hệ thống tĩnh (được đặt ở đầu phiên để tối ưu KV Cache).
    """
    client = get_client()
    cleanup_expired_sessions()
    
    workspace = None
    if workspace_dir:
        workspace = os.path.realpath(workspace_dir)
        if not os.path.isdir(workspace):
            return json.dumps({"error": f"Workspace '{workspace_dir}' không tồn tại!"}, ensure_ascii=False)
            
    default_sys = (
        "You are an elite software architect and senior code engineer assisting on mission-critical production systems. "
        "Be precise, rigorous, and truthful."
    )
    base_sys = system_prompt or default_sys
    if workspace:
        initial_sys = (
            base_sys +
            "\n\nYou have access to local read-only tools: 'read_file', 'list_files', and 'search_file'. "
            "Use them autonomously to inspect files in the workspace whenever you need code context before providing your verdict."
            + TOOL_SYSTEM_DIRECTIVE
        )
    else:
        initial_sys = base_sys
    
    def _execute_ask_turn():
        session = None
        turn_start_msg_count = 0
        if session_id:
            raw_loaded = None
            if not reset_session:
                raw_loaded = load_session(session_id)
            session = _normalize_session(session_id, raw_loaded, initial_sys)
            
            turn_start_msg_count = len(session["messages"])
            session["messages"].append({"role": "user", "content": prompt})
            session["updated_at"] = time.time()
            session["turn_count"] += 1
            messages = session["messages"]
            current_turn = session["turn_count"]
        else:
            messages = [
                {"role": "system", "content": initial_sys},
                {"role": "user", "content": prompt}
            ]
            current_turn = 1
            
        kwargs: Dict[str, Any] = {
            "model": DEFAULT_MODEL,
            "messages": sanitize_messages_for_api(messages),
        }
        
        if thinking:
            valid_effort = effort if effort in ["low", "high", "max"] else "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = valid_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            kwargs["reasoning_effort"] = "none"
            
        start_time = time.time()
        tools_used: List[Dict[str, Any]] = []
        tool_turns_executed = 0
        all_telemetry: List[Dict[str, Any]] = []
        final_content = ""
        final_reasoning = ""
        
        try:
            if workspace:
                kwargs["tools"] = READ_ONLY_TOOLS_SCHEMA
                for _ in range(max_tool_turns):
                    tool_turns_executed += 1
                    kwargs["messages"] = sanitize_messages_for_api(messages)
                    response = client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    all_telemetry.append(extract_telemetry(response.usage))
                    
                    messages.append(serialize_assistant_message(msg))
                    
                    if not getattr(msg, "tool_calls", None):
                        final_content = msg.content or ""
                        final_reasoning = getattr(msg, "reasoning_content", None) or ""
                        break
                        
                    for i, tc in enumerate(msg.tool_calls):
                        fn_name = tc.function.name
                        try:
                            fn_args = json.loads(tc.function.arguments)
                        except Exception:
                            fn_args = {}
                        if not isinstance(fn_args, dict):
                            fn_args = {}
                        tc_id = getattr(tc, "id", f"call_{i}")
                        try:
                            if fn_name in ["read_file", "list_files", "list_directory", "search_file"]:
                                tool_out = execute_local_tool(workspace, fn_name, fn_args)
                                tools_used.append({"tool": fn_name, "args": fn_args})
                            else:
                                tool_out = f"Error: Tool '{fn_name}' không được phép trong chế độ read-only!"
                        except Exception as te:
                            tool_out = f"Tool execution error: {str(te)}"
                            
                        # Cap 32.000 ký tự cho từng lệnh gọi tool đơn lẻ (per-tool-call)
                        str_out = str(tool_out)
                        if len(str_out) > MAX_TOOL_RESULT:
                            str_out = str_out[:MAX_TOOL_RESULT] + f"\n<<OUTPUT TRUNCATED AT {MAX_TOOL_RESULT} CHARS>>"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": str_out
                        })
                        
                if not final_content:
                    messages.append({
                        "role": "user",
                        "content": "Synthesize your final answer now based on the information read."
                    })
                    kwargs_final = {
                        "model": DEFAULT_MODEL,
                        "messages": sanitize_messages_for_api(messages),
                        "extra_body": kwargs.get("extra_body", {}),
                        "reasoning_effort": kwargs.get("reasoning_effort", "none")
                    }
                    res_final = client.chat.completions.create(**kwargs_final)
                    msg_final = res_final.choices[0].message
                    final_content = msg_final.content or ""
                    final_reasoning = getattr(msg_final, "reasoning_content", None) or ""
                    messages.append(serialize_assistant_message(msg_final))
                    all_telemetry.append(extract_telemetry(res_final.usage))
                    
                elapsed = round(time.time() - start_time, 2)
                
                total_prompt = sum(t.get("prompt_tokens", 0) for t in all_telemetry)
                total_completion = sum(t.get("completion_tokens", 0) for t in all_telemetry)
                total_hit = sum(t.get("prompt_cache_hit_tokens", 0) for t in all_telemetry)
                total_miss = sum(t.get("prompt_cache_miss_tokens", 0) for t in all_telemetry)
                total_reasoning = sum(t.get("reasoning_tokens", 0) for t in all_telemetry)
                
                telemetry = {
                    "prompt_tokens": total_prompt,
                    "prompt_cache_hit_tokens": total_hit,
                    "prompt_cache_miss_tokens": total_miss,
                    "cache_hit_rate_pct": round((total_hit / total_prompt * 100), 2) if total_prompt > 0 else 0.0,
                    "completion_tokens": total_completion,
                    "reasoning_tokens": total_reasoning,
                    "total_tokens": total_prompt + total_completion,
                    "elapsed_seconds": elapsed,
                    "turns_count": tool_turns_executed
                }
            else:
                response = client.chat.completions.create(**kwargs)
                elapsed = round(time.time() - start_time, 2)
                
                msg = response.choices[0].message
                final_content = msg.content or ""
                final_reasoning = getattr(msg, "reasoning_content", None) or ""
                telemetry = extract_telemetry(response.usage)
                telemetry["elapsed_seconds"] = elapsed
                messages.append(serialize_assistant_message(msg))
                
            if session_id and session:
                session["messages"] = messages
                session["updated_at"] = time.time()
                save_session(session_id, session, delta_messages=messages[turn_start_msg_count:])
                
            result = {
                "session_id": session_id,
                "turn": current_turn,
                "model": DEFAULT_MODEL,
                "mode": "thinking" if thinking else "non-thinking",
                "effort": effort if thinking else "none",
                "workspace_dir": workspace,
                "investigation_tool_turns": tool_turns_executed if workspace else 0,
                "tools_used": tools_used if workspace else [],
                "content": final_content,
                "reasoning_content": final_reasoning if thinking else None,
                "telemetry": telemetry
            }
            if session_id:
                result["session_active_turns"] = current_turn
                
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            if session_id and session:
                session["messages"] = session["messages"][:turn_start_msg_count]
                session["turn_count"] -= 1
                _HOT_SESSIONS[session_id] = copy.deepcopy(session)
            return json.dumps({"error": f"Lỗi gọi DeepSeek API: {str(e)}"}, ensure_ascii=False)

    if session_id:
        with session_lock(session_id):
            return _execute_ask_turn()
    else:
        return _execute_ask_turn()


@server.tool()
@tool_guard
def deepseek_session_clear(session_id: Optional[str] = None) -> str:
    """
    Xóa bỏ một hoặc toàn bộ session lưu ngữ cảnh multi-turn của deepseek_ask trên đĩa.
    
    Args:
        session_id: ID của session cần xóa. Nếu 'all', xóa toàn bộ session trên đĩa. Không được để trống.
    """
    if session_id is None or not session_id.strip():
        return json.dumps({
            "status": "ERROR",
            "code": "E_BAD_ARG",
            "message": "Tham số session_id không được để trống. Để xóa toàn bộ session, truyền session_id='all'."
        }, ensure_ascii=False)

    sid = session_id.strip()
    if sid.lower() == "all":
        count = delete_all_sessions()
        return json.dumps({"status": "SUCCESS", "message": f"Đã xóa toàn bộ {count} sessions trên đĩa."}, ensure_ascii=False)
    else:
        with session_lock(sid):
            deleted = delete_session(sid)
        if deleted:
            return json.dumps({"status": "SUCCESS", "message": f"Session '{sid}' đã được xóa khỏi đĩa."}, ensure_ascii=False)
        return json.dumps({"status": "NOT_FOUND", "message": f"Session '{sid}' không tồn tại trên đĩa."}, ensure_ascii=False)



def materialize_adversarial_tests(workspace: str, review_text: str) -> List[Dict[str, Any]]:
    """
    Bóc tách các block ADVERSARIAL_TESTS từ review của DeepSeek,
    cô lập lưu vào thư mục <workspace>/.deepseek/adversarial/,
    áp dụng Collision Guard chống ghi đè và trả về metadata test (Gate G11).
    """
    if not workspace or not os.path.isdir(workspace) or not review_text:
        return []

    adv_dir = os.path.join(workspace, ".deepseek", "adversarial")
    try:
        os.makedirs(adv_dir, exist_ok=True)
    except Exception as e:
        return [{"status": "ERROR", "error": f"Không thể tạo thư mục {adv_dir}: {str(e)}"}]

    adv_section = review_text
    adv_match = re.search(r"(?:^|\n)##\s+ADVERSARIAL_TESTS.*?(?=\n##|\Z)", review_text, re.DOTALL | re.IGNORECASE)
    if adv_match:
        adv_section = adv_match.group(0)

    pattern = re.compile(
        r"(?:RUN_COMMAND:\s*(?P<cmd>[^\n]+)\s*)?"
        r"```(?P<lang>[a-zA-Z0-9_-]+):(?P<path>[^\n]+)\n(?P<code>.*?)```",
        re.DOTALL
    )

    written_tests = []
    for match in pattern.finditer(adv_section):
        raw_cmd = (match.group("cmd") or "").strip()
        lang = (match.group("lang") or "").strip()
        raw_path = match.group("path").strip()
        code = match.group("code")

        cleaned_path = raw_path.replace("\\", "/")
        if cleaned_path.startswith(".deepseek/adversarial/"):
            cleaned_path = cleaned_path[len(".deepseek/adversarial/"):]
        elif cleaned_path.startswith("./.deepseek/adversarial/"):
            cleaned_path = cleaned_path[len("./.deepseek/adversarial/"):]
        else:
            cleaned_path = os.path.basename(cleaned_path)

        if not cleaned_path or cleaned_path in (".", "..") or ".." in cleaned_path:
            cleaned_path = f"test_adv_{hashlib.sha256(code.encode('utf-8')).hexdigest()[:8]}.txt"

        target_file = os.path.join(adv_dir, cleaned_path)
        rel_to_workspace = os.path.relpath(target_file, workspace).replace("\\", "/")

        # Collision Guard: Nếu file đã tồn tại, tự động gắn suffix _<hash8>
        if os.path.exists(target_file):
            h = hashlib.sha256(code.encode("utf-8")).hexdigest()[:8]
            base, ext = os.path.splitext(cleaned_path)
            safe_filename = f"{base}_{h}{ext}"
            target_file = os.path.join(adv_dir, safe_filename)
            new_rel = os.path.relpath(target_file, workspace).replace("\\", "/")
            if raw_cmd and rel_to_workspace in raw_cmd:
                raw_cmd = raw_cmd.replace(rel_to_workspace, new_rel)
            elif raw_cmd and raw_path in raw_cmd:
                raw_cmd = raw_cmd.replace(raw_path, new_rel)
            rel_to_workspace = new_rel

        try:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(code)

            final_cmd = raw_cmd
            if not final_cmd:
                if lang in ("python", "py") or rel_to_workspace.endswith(".py"):
                    final_cmd = f"pytest {rel_to_workspace}"
                elif lang in ("typescript", "javascript", "ts", "js"):
                    final_cmd = f"npx vitest run {rel_to_workspace}"
                elif lang in ("go", "golang") or rel_to_workspace.endswith(".go"):
                    final_cmd = f"go test {rel_to_workspace}"

            # Chuẩn hóa python -> python3 nếu hệ thống không có binary 'python'
            if final_cmd and final_cmd.startswith("python ") and not shutil.which("python"):
                if shutil.which("python3"):
                    final_cmd = "python3 " + final_cmd[7:]

            written_tests.append({
                "path": rel_to_workspace,
                "absolute_path": os.path.realpath(target_file),
                "run_cmd": final_cmd,
                "language": lang,
                "status": "WRITTEN"
            })
        except Exception as write_err:
            written_tests.append({
                "path": rel_to_workspace,
                "error": str(write_err),
                "status": "WRITE_FAILED"
            })

    return written_tests


# ---------------------------------------------------------------------------
# Tool 2: deepseek_adversarial_review (Doubt-Driven Adversarial Code Review)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_adversarial_review(
    code_or_diff: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    context_spec: Optional[str] = None,
    effort: str = "max"
) -> str:
    """
    Thực hiện phản biện gắt (Doubt-Driven Development) trên đoạn code hoặc Git Diff độc lập.
    Hỗ trợ tự động thu thập git diff an toàn (kèm untracked files), cấp read-only tools tra cứu codebase,
    và xuất các test case đối kháng có thể chạy ngay (ADVERSARIAL_TESTS) theo đa ngôn ngữ.
    
    Args:
        code_or_diff: Nội dung mã nguồn hoặc git diff cần thẩm định.
        workspace_dir: Đường dẫn workspace cần tự động bóc tách git diff và tra cứu context.
        context_spec: Tài liệu đặc tả hoặc yêu cầu của bài toán (nếu có).
        effort: Mức độ suy luận phản biện ('high' hoặc 'max'). Mặc định: 'max'.
    """
    client = get_client()
    workspace = None
    diff_text = code_or_diff or ""

    if workspace_dir:
        workspace = os.path.realpath(workspace_dir)
        if not os.path.isdir(workspace):
            return json.dumps({"error": f"Workspace '{workspace_dir}' không tồn tại!"}, ensure_ascii=False)

        exec_env = build_scrubbed_env()
        collected_diff = ""
        is_git = os.path.exists(os.path.join(workspace, ".git"))
        if is_git:
            try:
                # 1. Thu thập git diff tracked an toàn (Gate G5: vô hiệu hóa fsmonitor và ext-diff)
                cmd_tracked = [
                    "git", "-c", "core.fsmonitor=false", "diff",
                    "--no-ext-diff", "--no-textconv", "HEAD"
                ]
                d_out, d_rc, _ = _run_capped(cmd_tracked, workspace, exec_env, timeout=15)
                if d_rc == 0 and d_out.strip():
                    collected_diff += d_out.strip() + "\n"
                elif d_rc != 0:
                    cmd_fallback = [
                        "git", "-c", "core.fsmonitor=false", "diff",
                        "--no-ext-diff", "--no-textconv"
                    ]
                    f_out, f_rc, _ = _run_capped(cmd_fallback, workspace, exec_env, timeout=15)
                    if f_rc == 0 and f_out.strip():
                        collected_diff += f_out.strip() + "\n"

                # 2. Thu thập untracked files an toàn (Gate G6)
                cmd_untracked = ["git", "ls-files", "--others", "--exclude-standard"]
                u_out, u_rc, _ = _run_capped(cmd_untracked, workspace, exec_env, timeout=10)
                if u_rc == 0 and u_out.strip():
                    untracked_files = [f.strip() for f in u_out.splitlines() if f.strip()]
                    for u_file in untracked_files[:30]:
                        safe_u, full_u = is_safe_path(workspace, u_file, allow_offload_read=False)
                        if safe_u and os.path.isfile(full_u):
                            try:
                                u_sz = os.path.getsize(full_u)
                                if u_sz <= 256 * 1024:
                                    with open(full_u, "r", encoding="utf-8", errors="replace") as uf:
                                        u_content = uf.read()
                                    collected_diff += f"\n--- /dev/null\n+++ b/{u_file}\n@@ -0,0 +1,{len(u_content.splitlines())} @@\n"
                                    for line in u_content.splitlines():
                                        collected_diff += f"+{line}\n"
                            except Exception:
                                pass
            except Exception as ge:
                collected_diff += f"\n[Git diff extraction warning: {str(ge)}]\n"

        if collected_diff.strip():
            if diff_text:
                diff_text = f"{diff_text}\n\n=== REPO GIT DIFF & UNTRACKED FILES ===\n{collected_diff}"
            else:
                diff_text = collected_diff

    if not diff_text.strip():
        diff_text = "No code or diff provided."

    system_prompt = (
        "You are an Elite Principal Security & Reliability Engineer executing an Adversarial Doubt-Driven Code Review.\n"
        "Your explicit bias is to DISPROVE correctness, find security bugs, race conditions, memory growth, and unhandled failure modes.\n\n"
        "Review Protocol:\n"
        "1. STRICT DIFF-BOUNDED AUDIT: Focus primarily on the provided changes and their direct callers/dependencies.\n"
        "2. READ-ONLY TOOL ACCESS: If workspace tools are available, use 'read_file', 'list_files', and 'search_file' "
        "autonomously to inspect caller sites, function signatures, types, or existing tests before rendering your verdict.\n"
        "3. MULTI-LANGUAGE ADVERSARIAL TESTS (CRITICAL): If you identify defect(s) or edge case(s), you MUST provide "
        "concrete, executable adversarial test cases in the language of the target project (Python, TypeScript, Go, Rust, Java, etc.).\n\n"
        "Format your output strictly as follows:\n"
        "- VERDICT: [CLEAN / MINOR_WARNINGS / CRITICAL_DEFECTS]\n"
        "- SUMMARY: 2-3 sentences\n"
        "- SPECIFIC_VULNERABILITIES_OR_DEFECTS: List defects with file, line, explanation, impact, and reproduction steps.\n"
        "- ADVERSARIAL_TESTS:\n"
        "  Provide reproduction tests in the '## ADVERSARIAL_TESTS' section.\n"
        "  All adversarial tests are isolated in '.deepseek/adversarial/<filename>' to prevent workspace pollution.\n"
        "  Resolve imports relative from '.deepseek/adversarial/' back to project source (e.g. '../../src/foo' or '@/...').\n"
        "  Provide the exact test code and run command in this exact format:\n"
        "  RUN_COMMAND: <exact shell command to execute the test, e.g. pytest .deepseek/adversarial/test_adv.py or npx vitest run .deepseek/adversarial/adv.test.ts>\n"
        "  ```<language>:.deepseek/adversarial/<filename>\n"
        "  <test source code>\n"
        "  ```"
        + (TOOL_SYSTEM_DIRECTIVE if workspace else "")
    )

    user_prompt = f"### CONTEXT SPEC:\n{context_spec or 'Not specified'}\n\n### CODE / DIFF TO AUDIT:\n{diff_text}"
    valid_effort = effort if effort in ["low", "high", "max"] else "max"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        tools_used = []
        all_telemetry = []
        final_content = ""
        final_reasoning = ""

        if workspace:
            kwargs = {
                "model": DEFAULT_MODEL,
                "messages": sanitize_messages_for_api(messages),
                "tools": READ_ONLY_TOOLS_SCHEMA,
                "extra_body": {"thinking": {"type": "enabled"}},
                "reasoning_effort": valid_effort
            }
            for _ in range(5):
                kwargs["messages"] = sanitize_messages_for_api(messages)
                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                all_telemetry.append(extract_telemetry(response.usage))
                messages.append(serialize_assistant_message(msg))

                if not getattr(msg, "tool_calls", None):
                    final_content = msg.content or ""
                    final_reasoning = getattr(msg, "reasoning_content", None) or ""
                    break

                for i, tc in enumerate(msg.tool_calls):
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except Exception:
                        fn_args = {}
                    if not isinstance(fn_args, dict):
                        fn_args = {}
                    tc_id = getattr(tc, "id", f"call_{i}")
                    try:
                        if fn_name in ["read_file", "list_files", "list_directory", "search_file"]:
                            tool_out = execute_local_tool(workspace, fn_name, fn_args)
                            tools_used.append({"tool": fn_name, "args": fn_args})
                        else:
                            tool_out = f"Error: Tool '{fn_name}' không được phép trong adversarial review!"
                    except Exception as te:
                        tool_out = f"Tool execution error: {str(te)}"

                    str_out = str(tool_out)
                    if len(str_out) > MAX_TOOL_RESULT:
                        str_out = str_out[:MAX_TOOL_RESULT] + f"\n<<OUTPUT TRUNCATED AT {MAX_TOOL_RESULT} CHARS>>"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": str_out
                    })

            if not final_content:
                messages.append({
                    "role": "user",
                    "content": "Synthesize your final adversarial review verdict now based on the information read."
                })
                kwargs_final = {
                    "model": DEFAULT_MODEL,
                    "messages": sanitize_messages_for_api(messages),
                    "extra_body": {"thinking": {"type": "enabled"}},
                    "reasoning_effort": valid_effort
                }
                res_final = client.chat.completions.create(**kwargs_final)
                msg_final = res_final.choices[0].message
                final_content = msg_final.content or ""
                final_reasoning = getattr(msg_final, "reasoning_content", None) or ""
                all_telemetry.append(extract_telemetry(res_final.usage))

            telemetry = {
                "prompt_tokens": sum(t.get("prompt_tokens", 0) for t in all_telemetry),
                "completion_tokens": sum(t.get("completion_tokens", 0) for t in all_telemetry),
                "total_tokens": sum(t.get("total_tokens", 0) for t in all_telemetry)
            }
        else:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=sanitize_messages_for_api(messages),
                extra_body={"thinking": {"type": "enabled"}},
                reasoning_effort=valid_effort
            )
            msg = response.choices[0].message
            final_content = msg.content or ""
            final_reasoning = getattr(msg, "reasoning_content", None) or ""
            telemetry = extract_telemetry(response.usage)

        written_tests = []
        if workspace:
            written_tests = materialize_adversarial_tests(workspace, final_content)

        return json.dumps({
            "audit_verdict": final_content,
            "reasoning_review": final_reasoning,
            "written_tests": written_tests,
            "tools_used": tools_used,
            "telemetry": telemetry
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Lỗi phản biện: {str(e)}"}, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Tool 3: deepseek_code_task (ReAct Loop, TDD Runner & Tri-State Verdict)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_code_task(
    workspace_dir: str,
    task: str,
    test_cmd: Optional[str] = None,
    thinking: bool = True,
    effort: str = "high",
    max_turns: int = 12,
    run_adversarial_review: bool = False
) -> str:
    """
    Thực thi tác vụ code tự động bằng DeepSeek TDD Loop:
    1. Thiết lập Git Safety Net.
    2. Architect Agent lập Spec & Blueprint chấp nhận.
    3. Coder Agent chạy ReAct Loop gọi tool sửa code và chạy test tự động.
    4. Trả về báo cáo Tri-State Verdict (PASSED, FAILED, INCONCLUSIVE) kèm Git Diff sạch.
    
    Args:
        workspace_dir: Đường dẫn tuyệt đối của thư mục dự án cần sửa.
        task: Mô tả yêu cầu bài toán cần giải quyết.
        test_cmd: Lệnh chạy test tự động (ví dụ: 'pytest tests/test_auth.py').
        thinking: Bật Thinking Mode cho giai đoạn Architect.
        effort: Mức độ suy luận CoT ('low', 'high', 'max').
        max_turns: Số lượt tối đa của vòng lặp ReAct (tránh vô hạn).
    """
    workspace = os.path.realpath(workspace_dir)
    if not os.path.isdir(workspace):
        return json.dumps({"status": "ERROR", "error": {"code": "E_NOT_FOUND", "message": f"Workspace '{workspace_dir}' không tồn tại!"}}, ensure_ascii=False)

    if test_cmd:
        safe_cmd, cmd_reason = check_command_safety(test_cmd)
        if not safe_cmd:
            return json.dumps({
                "status": "ERROR",
                "error": {
                    "code": "E_CMD_REJECTED",
                    "message": f"test_cmd bị chặn bởi Safety Guardrail: {cmd_reason}"
                }
            }, ensure_ascii=False)

    client = get_client()
    task_id = int(time.time())
    safety_branch = f"deepseek/task-{task_id}"
    exec_env = build_scrubbed_env()
    git_errors = []

    # 1. Thiết lập Git Safety Net
    is_git = os.path.exists(os.path.join(workspace, ".git"))
    if is_git:
        try:
            cb_res = subprocess.run(["git", "checkout", "-b", safety_branch], cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env)
            if cb_res.returncode != 0:
                git_errors.append(f"git checkout -b failed: {cb_res.stderr.strip()}")
        except Exception as ge:
            git_errors.append(f"git safety net error: {str(ge)}")

    # 2. Giai đoạn 1: Architect Specification
    existing_files = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = sorted([d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv", ".deepseek"]])
        for f in sorted(files):
            existing_files.append(os.path.relpath(os.path.join(root, f), workspace))
            if len(existing_files) >= 50:
                break
        if len(existing_files) >= 50:
            break

    arch_system = (
        "You are an Elite Software Architect. Your sole mission is to analyze requirements and lay down an exact, concrete execution blueprint for the Coder.\n"
        "The Coder is an execution worker who will follow your instructions step-by-step. Do not produce vague hand-waving advice.\n\n"
        "Your specification MUST strictly provide:\n"
        "1. TARGET FILES: Explicitly list which files to create or modify. Instruct Coder to touch ONLY these files and preserve existing style.\n"
        "2. TEST SPECIFICATION: Define the exact test file to create/modify, the test cases/assertions, and the exact verification shell/bash command.\n"
        "3. MINIMAL IMPLEMENTATION: Design the simplest, most direct solution that solves the problem. Explicitly forbid unnecessary abstractions.\n"
        "4. PRE-MORTEM & EDGE CASES: Identify subtle failure modes, boundary values, and race conditions so the Coder handles them directly.\n\n"
        "DO NOT use tool calls or DSML syntax. Output pure markdown blueprint only."
    )
    arch_user = f"Task: {task}\nWorkspace: {workspace}\nExisting files: {json.dumps(existing_files)}\nAutomated test command: {test_cmd or 'None specified'}"

    arch_res = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": arch_system},
            {"role": "user", "content": arch_user}
        ],
        extra_body={"thinking": {"type": "enabled"}},
        reasoning_effort=effort if effort in ["low", "high", "max"] else "high"
    )
    spec_plan = arch_res.choices[0].message.content or ""
    arch_telemetry = extract_telemetry(arch_res.usage)

    # 3. Giai đoạn 2: Coder ReAct Loop
    coder_system = (
        "You are an Autonomous Senior Software Engineer. You have access to local tools: read_file, write_file, edit_file_replace, run_shell, list_files, search_file. "
        "Follow the Architect's Spec strictly. Write or modify code in the workspace, and verify using the shell. "
        "When finished, summarize your changes and state 'TASK_COMPLETE'."
        + TOOL_SYSTEM_DIRECTIVE
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": coder_system},
        {"role": "user", "content": f"SPECIFICATION & PLAN:\n{spec_plan}\n\nExecute the code changes in the workspace now. Use your tools."}
    ]

    turns_executed = 0
    test_passed = False
    last_test_output = ""
    last_test_flag = "ok"
    loop_telemetry = []

    for turn in range(max_turns):
        turns_executed += 1
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=sanitize_messages_for_api(messages),
            tools=LOCAL_TOOLS_SCHEMA,
            extra_body={"thinking": {"type": "disabled"}},
            reasoning_effort="none"
        )
        msg = res.choices[0].message
        messages.append(serialize_assistant_message(msg))
        loop_telemetry.append(extract_telemetry(res.usage))

        if not msg.tool_calls:
            if test_cmd:
                test_out, test_rc, test_flag = _run_capped(test_cmd, workspace, exec_env, timeout=DEFAULT_CODER_SHELL_TIMEOUT, kill_on_cap=False)
                last_test_output = test_out
                last_test_flag = test_flag
                if test_rc == 0:
                    test_passed = True
                    break
                elif test_flag == "timeout":
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Automated test execution TIMED OUT after {DEFAULT_CODER_SHELL_TIMEOUT}s (killed with SIGKILL):\n"
                            f"{last_test_output}\n"
                            "The tests did not fail on assertion, they ran out of time. Do NOT break existing correct logic. "
                            "Check for infinite loops, blocking operations, or excessive work."
                        )
                    })
                    continue
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Automated test verification FAILED with exit code {test_rc}:\n{last_test_output}\nFix the code to make tests pass."
                    })
                    continue
            else:
                break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except Exception:
                fn_args = {}
            try:
                tool_output = execute_local_tool(workspace, fn_name, fn_args)
            except Exception as te:
                tool_output = f"Tool execution error: {str(te)}"

            str_out = str(tool_output)
            if len(str_out) > MAX_TOOL_RESULT:
                str_out = str_out[:MAX_TOOL_RESULT] + f"\n<<OUTPUT TRUNCATED AT {MAX_TOOL_RESULT} CHARS>>"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str_out
            })

    if not test_passed and test_cmd:
        test_out, test_rc, test_flag = _run_capped(test_cmd, workspace, exec_env, timeout=DEFAULT_CODER_SHELL_TIMEOUT, kill_on_cap=False)
        last_test_output = test_out
        last_test_flag = test_flag
        if test_rc == 0:
            test_passed = True

    # 4. Thu thập Git Diff an toàn
    git_diff = ""
    if is_git:
        try:
            subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env)
            diff_res = subprocess.run(
                ["git", "-c", "core.fsmonitor=false", "--no-pager", "diff", "--cached", "--no-ext-diff", "--no-textconv"],
                cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env, errors="replace"
            )
            git_diff = diff_res.stdout
        except Exception as de:
            git_errors.append(f"git diff error: {str(de)}")

    # 5. Tri-State Verdict (PASSED, FAILED, INCONCLUSIVE)
    if test_cmd:
        if test_passed:
            verdict = "PASSED"
        elif last_test_flag == "timeout":
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAILED"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "status": verdict,
        "task": task,
        "workspace": workspace,
        "safety_branch": safety_branch if is_git else "N/A (Not a git repo)",
        "turns_executed": turns_executed,
        "architect_spec": spec_plan,
        "test_verification": {
            "test_cmd": test_cmd,
            "passed": test_passed if test_cmd else None,
            "status": "PASSED" if test_passed else ("TIMEOUT" if last_test_flag == "timeout" else ("FAILED" if test_cmd else "NONE")),
            "output_snippet": last_test_output[:2000] if last_test_output else ("No test command run" if not test_cmd else "")
        },
        "git_diff_summary": git_diff[:3000] if git_diff else "No uncommitted git diff",
        "git_errors": git_errors if git_errors else None,
        "telemetry_summary": {
            "architect_tokens": arch_telemetry,
            "coder_loop_total_turns": len(loop_telemetry)
        }
    }
    return json.dumps(report, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Tool 4: deepseek_health (Kiểm tra kết nối và token)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_health() -> str:
    """Kiểm tra tình trạng kết nối tới DeepSeek API, độ trễ và tính hợp lệ của API Key."""
    client = get_client()
    start = time.time()
    try:
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            extra_body={"thinking": {"type": "disabled"}},
            reasoning_effort="none"
        )
        latency = round(time.time() - start, 2)
        telemetry = extract_telemetry(res.usage)
        return json.dumps({
            "status": "HEALTHY",
            "model": DEFAULT_MODEL,
            "latency_seconds": latency,
            "telemetry": telemetry
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "UNHEALTHY",
            "error": str(e)
        }, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    server.run(transport="stdio")
