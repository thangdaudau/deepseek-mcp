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
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MAX_READ_BYTES = 24 * 1024        # 24KB raw content per page
MAX_READ_LINES = 2000             # 2000 lines max per read
CHUNK_SIZE = 64 * 1024            # 64KB chunk streaming
MAX_ENVELOPE_CONTENT_CHARS = 28_000 # Trần ký tự an toàn trước khi serialize JSON envelope
MAX_TOOL_RESULT = 64_000          # 64,000 chars per-tool-call cap in messages
MAX_SESSION_BYTES = 8 * 1024 * 1024 # 8MB max session JSON on disk
MAX_SHELL_OUT = 128 * 1024        # 128KB output cap for shell commands

# Khóa chặt bất biến runtime an toàn
if MAX_READ_BYTES > MAX_TOOL_RESULT - 4096:
    raise ValueError("MAX_READ_BYTES must leave room for JSON envelope in MAX_TOOL_RESULT")

DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+.*(-[a-zA-Z0-9]*[rfRF]|--recursive|--force).*\s+(/|~|\$HOME|\*|/\*)(\s|$)",
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
    """Tẩy sạch các biến môi trường chứa secret trực tiếp khỏi env của child process nhưng giữ nguyên HOME, SSH credentials và toolchain."""
    scrubbed = {
        k: v for k, v in os.environ.items()
        if k in ("SSH_AUTH_SOCK", "SSH_AGENT_PID") or not any(kw in k.upper() for kw in SENSITIVE_ENV_KEYWORDS)
    }
    venv_bin = os.path.dirname(sys.executable)
    scrubbed["PATH"] = f"{venv_bin}{os.pathsep}{scrubbed.get('PATH', '')}"
    return scrubbed

def is_safe_path(base_dir: str, target_path: str) -> Tuple[bool, str]:
    """
    Kiểm tra chặn Path Traversal ra ngoài workspace và denylist các đường dẫn nhạy cảm.
    Bộ lọc chống phá hoại vô tình (accidental footguns), chuẩn hóa lowercase trên dir parts
    để tương thích an toàn trên macOS (APFS) và Windows (NTFS).
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
SESSION_TTL_SECONDS = 86400  # 24 hours (1 ngày)
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

def session_lock(session_id: str):
    """Context manager khóa file session bằng fcntl.flock (LOCK_EX | LOCK_NB). Fail-fast nếu đang bận."""
    class _LockContext:
        def __init__(self, sid: str):
            self.sid = sid
            self.fh = None

        def __enter__(self):
            lock_file = get_session_file(self.sid) + ".lock"
            self.fh = open(lock_file, "a+")
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, IOError):
                self.fh.close()
                self.fh = None
                raise SessionBusyError(f"Session '{self.sid}' đang được xử lý bởi một tiến trình khác.")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.fh:
                try:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                self.fh.close()
                self.fh = None

    return _LockContext(session_id)

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
    """Tải session từ Hot Plane (RAM) hoặc phục hồi từ Cold Plane (WAL với CRC32) trên đĩa."""
    now = time.time()

    # 1. Hot Plane (RAM)
    if session_id in _HOT_SESSIONS:
        cached = _HOT_SESSIONS[session_id]
        if now - cached.get("updated_at", 0) > SESSION_TTL_SECONDS:
            delete_session(session_id)
            return None
        return copy.deepcopy(cached)

    # 2. Cold Plane (WAL: .wal.jsonl)
    wal_path = get_session_file(session_id)
    if os.path.exists(wal_path):
        last_valid_data = None
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
                        if payload_str is not None and crc is not None:
                            calc_crc = zlib.crc32(payload_str.encode("utf-8")) & 0xFFFFFFFF
                            if calc_crc == crc:
                                last_valid_data = json.loads(payload_str)
                            else:
                                sys.stderr.write(f"Warning: Corrupt WAL frame in session {session_id} (CRC mismatch)\n")
                    except Exception as parse_err:
                        sys.stderr.write(f"Warning: Invalid WAL line in session {session_id}: {parse_err}\n")
                        continue

            if last_valid_data is not None:
                if now - last_valid_data.get("updated_at", 0) > SESSION_TTL_SECONDS:
                    delete_session(session_id)
                    return None
                _HOT_SESSIONS[session_id] = copy.deepcopy(last_valid_data)
                return last_valid_data
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

def save_session(session_id: str, data: Dict[str, Any]) -> bool:
    """Ghi session: cập nhật Hot Plane RAM và ghi nối tiếp (append-only WAL) kèm CRC-32."""
    try:
        # Nếu session có messages, tỉa theo ranh giới turn boundary hợp lệ trước khi serialize
        if "messages" in data and isinstance(data["messages"], list):
            data["messages"] = _prune_session_messages(data["messages"], MAX_SESSION_BYTES)

        # Serialize compact payload và tính CRC-32
        payload_str = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        crc = zlib.crc32(payload_str.encode("utf-8")) & 0xFFFFFFFF
        wal_record = json.dumps({
            "crc": crc,
            "ts": time.time(),
            "payload": payload_str
        }, ensure_ascii=False) + "\n"

        wal_path = get_session_file(session_id)
        dir_name = os.path.dirname(wal_path)
        os.makedirs(dir_name, exist_ok=True)

        # Compaction check: nếu file đã vượt quá 30 frames hoặc > 512KB, nén về snapshot duy nhất
        should_compact = False
        if os.path.exists(wal_path):
            try:
                st = os.stat(wal_path)
                if st.st_size > 512 * 1024:
                    should_compact = True
                else:
                    with open(wal_path, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                    if line_count >= 30:
                        should_compact = True
            except Exception:
                pass

        if should_compact:
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
        else:
            is_new = not os.path.exists(wal_path)
            with open(wal_path, "a", encoding="utf-8") as f:
                f.write(wal_record)
                f.flush()
                os.fsync(f.fileno())
            if is_new:
                dfd = os.open(dir_name, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)

        # Cập nhật Hot Plane (RAM) sau khi ghi đĩa thành công
        _HOT_SESSIONS[session_id] = copy.deepcopy(data)
        return True
    except Exception as e:
        sys.stderr.write(f"Error saving session {session_id}: {str(e)}\n")
        return False

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
    """Tự động dọn dẹp các session quá hạn TTL (24h) trong RAM và trên đĩa."""
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
            "description": "Đọc nội dung file trong workspace (hỗ trợ phân trang an toàn). Mặc định mỗi dòng trong 'content' được đánh số '<line_number>: <mã nguồn>' để định vị chuẩn xác. Khi dùng edit_file_replace, KHÔNG đưa tiền tố số dòng vào search_block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối trong workspace"},
                    "start_line": {"type": "integer", "description": "Dòng bắt đầu đọc (1-indexed, mặc định: 1)"},
                    "line_count": {"type": "integer", "description": "Số dòng cần đọc tối đa (mặc định: đọc tối đa 2000 dòng hoặc tới 24KB)"},
                    "line_numbers": {"type": "boolean", "description": "Nếu True, thêm tiền tố số dòng '<line_number>: ' vào đầu mỗi dòng trong content (mặc định: True)."}
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
            "description": "Thay thế chính xác một khối text trong file bằng nội dung mới",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối"},
                    "search_block": {"type": "string", "description": "Đoạn text hiện có cần thay thế"},
                    "replace_block": {"type": "string", "description": "Đoạn text mới thay thế vào"}
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
    "2. STRICT SCHEMA ADHERENCE: Use ONLY the exact tool names ('read_file', 'list_files') and exact parameter keys "
    "defined in the schema. Do not invent aliases or extra parameters.\n"
    "3. NO RAW MARKUP: Rely strictly on native tool calls. Never emit raw DSML or XML tags into conversation text.\n"
    "4. SYNTHESIS: Provide your comprehensive analysis or verdict only after all necessary tools have returned their results.\n"
    "5. PAGINATION PROTOCOL: If 'read_file' returns 'truncated': true, invoke 'read_file' again with 'start_line'=next_start_line "
    "to read subsequent content. Never guess or hallucinate missing file lines.\n"
    "6. LINE NUMBERS PROTOCOL: 'read_file' prefixes each line with '<line_number>: ' for exact navigation. "
    "When applying edits via 'edit_file_replace', you MUST STRIP these line number prefixes from 'search_block' and 'replace_block'."
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

def sanitize_messages_for_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Loại bỏ reasoning_content khỏi messages trước khi gửi lên API để tiết kiệm input tokens."""
    cleaned = []
    for m in messages:
        if m.get("role") == "assistant" and "reasoning_content" in m:
            cleaned.append({k: v for k, v in m.items() if k != "reasoning_content"})
        else:
            cleaned.append(m)
    return cleaned

def _run_capped(cmd: str, cwd: str, env: dict, timeout: int = 90, cap: int = MAX_SHELL_OUT, kill_on_cap: bool = True) -> Tuple[str, int, str]:
    """
    Thực thi shell command với process group riêng biệt, streaming output có trần và kill tức thời khi chạm trần (nếu kill_on_cap=True).
    Đối với test runner (kill_on_cap=False), buffer chỉ nhận tối đa cap bytes nhưng tiến trình vẫn được phép chạy hoàn tất để lấy exit code thật.
    Trả về (output_text, exit_code, status_flag) với status_flag in ("ok", "capped", "timeout", "error").
    """
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=cwd,
            env=env,
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

    buf = bytearray()
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
            # Cho thời hạn ân hạn 250ms để đọc nốt buffer còn sót của grandchildren
            drain_deadline = now + 0.25

        # Đọc non-blocking từ stdout pipe
        try:
            r_ready, _, _ = select.select([rfd], [], [], 0.02)
            if r_ready:
                chunk = os.read(rfd, 64 * 1024)
                if not chunk:
                    # Pipe EOF thật sự
                    break
                if len(buf) < cap:
                    buf.extend(chunk[:cap - len(buf)])
                if len(buf) >= cap:
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

        # Kiểm tra timeout
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

    # Dọn dẹp tiến trình con và nhóm tiến trình
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

    text = bytes(buf).decode("utf-8", errors="replace")
    if flag == "capped":
        text += f"\n<<OUTPUT TRUNCATED AT {cap} BYTES>>"
    elif flag == "timeout":
        text += f"\n<<COMMAND TIMED OUT AFTER {timeout} SECONDS>>"

    rc = proc.returncode if proc.returncode is not None else -9
    return text, rc, flag

def _read_file_envelope(workspace: str, rel_path: str, start_line: Any = 1, line_count: Any = None, line_numbers: Any = True) -> str:
    """Đọc file theo chunking nhị phân, chống OOM, chống vỡ tiếng Việt, trả JSON envelope."""
    safe, full_path = is_safe_path(workspace, rel_path)
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

    start = max(1, _as_int(start_line, 1))
    want = MAX_READ_LINES if line_count is None else max(1, min(_as_int(line_count, MAX_READ_LINES), MAX_READ_LINES))
    use_line_numbers = True if line_numbers is None else bool(line_numbers)

    if st.st_size == 0:
        return json.dumps({
            "path": rel_path,
            "start_line": start,
            "end_line": 0,
            "next_start_line": None,
            "truncated": False,
            "eof": True,
            "line_numbers": use_line_numbers,
            "file_size": 0,
            "bytes_returned": 0,
            "next_step": None,
            "content": ""
        }, ensure_ascii=False)

    out = []
    tail = b""
    line_no = 0
    emitted_bytes = 0
    emitted_chars = 0
    next_start = None
    truncated = False

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

                # Chặn trường hợp 1 dòng đơn lẻ dài vượt trần MAX_READ_BYTES không có dấu \n
                if len(tail) >= MAX_READ_BYTES:
                    if line_no + 1 >= start:
                        truncated = True
                        break
                    else:
                        # Đang ở giai đoạn skip trước start_line, xả bớt tail để tiếp tục đếm dòng
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

    # Chống mâu thuẫn envelope khi chạm EOF đúng biên
    if truncated and not out and not tail:
        truncated = False

    # Nếu chạm trần byte MAX_READ_BYTES mà không đọc được dòng hoàn chỉnh nào (dòng đơn lẻ > 24KB)
    if truncated and not out and tail:
        sample = tail[:2000].decode("utf-8", errors="replace")
        return json.dumps({
            "status": "ERROR",
            "code": "E_LINE_TOO_LONG",
            "message": f"Dòng thứ {max(start, line_no + 1)} của '{rel_path}' dài vượt quá trần an toàn ({MAX_READ_BYTES} bytes, minified code hoặc JSON đơn dòng). Hãy dùng run_shell với lệnh 'head -c {MAX_READ_BYTES} ...' để bóc tách.",
            "path": rel_path,
            "start_line": start,
            "file_size": st.st_size,
            "partial_content": sample
        }, ensure_ascii=False)

    # Nếu bị truncated do chạm byte cap nhưng next_start chưa được gán
    if truncated and next_start is None and out:
        next_start = start + len(out)

    eof = (next_start is None and not truncated)
    end_line = start + len(out) - 1 if out else max(0, line_no)
    if use_line_numbers:
        body = "".join(f"{start + idx}: {line_str}" for idx, line_str in enumerate(out))
    else:
        body = "".join(out)

    return json.dumps({
        "path": rel_path,
        "start_line": start,
        "end_line": end_line,
        "next_start_line": next_start,
        "truncated": truncated,
        "eof": eof,
        "line_numbers": use_line_numbers,
        "file_size": st.st_size,
        "bytes_returned": emitted_bytes,
        "next_step": f"call read_file with start_line={next_start}" if next_start else None,
        "content": body
    }, ensure_ascii=False)

def execute_local_tool(workspace: str, name: str, args: Dict[str, Any]) -> str:
    """Thực thi tool local với chốt chặn an toàn Sandbox."""
    if name == "read_file":
        rel_path = args.get("path", "")
        start_line = args.get("start_line", 1)
        line_count = args.get("line_count", None)
        line_numbers = args.get("line_numbers", True)
        return _read_file_envelope(workspace, rel_path, start_line, line_count, line_numbers)
            
    elif name == "write_file":
        rel_path = args.get("path", "")
        content = args.get("content", "")
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace!"
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
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace!"
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

            if search_block not in current_text:
                # Fallback 1: Chuẩn hóa line-ending nếu có lệch CRLF (\r\n) vs LF (\n)
                if "\r\n" in current_text and "\r\n" not in search_block:
                    crlf_search = search_block.replace("\r\n", "\n").replace("\n", "\r\n")
                    if crlf_search in current_text:
                        search_block = crlf_search
                        replace_block = replace_block.replace("\r\n", "\n").replace("\n", "\r\n")
                elif "\r\n" not in current_text and "\r\n" in search_block:
                    lf_search = search_block.replace("\r\n", "\n")
                    if lf_search in current_text:
                        search_block = lf_search
                        replace_block = replace_block.replace("\r\n", "\n")

            if search_block not in current_text:
                # Fallback 2: Chỉ bóc tách tiền tố số dòng khi 100% dòng bắt đầu ở Column 0 và tăng dần liên tục (Contiguity Gate)
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
                        if candidate not in current_text:
                            if "\r\n" in current_text and "\r\n" not in candidate:
                                cand_crlf = candidate.replace("\r\n", "\n").replace("\n", "\r\n")
                                if cand_crlf in current_text:
                                    candidate = cand_crlf
                            elif "\r\n" not in current_text and "\r\n" in candidate:
                                cand_lf = candidate.replace("\r\n", "\n")
                                if cand_lf in current_text:
                                    candidate = cand_lf

                        if candidate in current_text:
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
                                    if "\r\n" in current_text and "\r\n" not in replace_block:
                                        replace_block = replace_block.replace("\r\n", "\n").replace("\n", "\r\n")
                                    elif "\r\n" not in current_text and "\r\n" in replace_block:
                                        replace_block = replace_block.replace("\r\n", "\n")

            if search_block not in current_text:
                return f"Error: search_block không tìm thấy trong file '{rel_path}'."
            count = current_text.count(search_block)
            if count > 1:
                return f"Error: search_block xuất hiện {count} lần trong file. Phải cung cấp thêm ngữ cảnh xung quanh để xác định duy nhất."
            new_text = current_text.replace(search_block, replace_block, 1)
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
        out, rc, flag = _run_capped(cmd, workspace, build_scrubbed_env(), timeout=90)
        return f"Exit code: {rc}\n{out}"
            
    elif name in ["list_files", "list_directory"]:
        rel_path = args.get("path") or args.get("sub_dir") or ""
        if rel_path == ".":
            rel_path = ""
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return "Error: Path ngoài workspace."
        if not os.path.exists(full_path):
            return "Error: Directory does not exist."
        try:
            files_list = []
            for root, dirs, files in os.walk(full_path):
                dirs[:] = sorted([d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv"]])
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
            "\n\nYou have access to local read-only tools: 'read_file' and 'list_files'. "
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
                            if fn_name in ["read_file", "list_files", "list_directory"]:
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
                save_session(session_id, session)
                
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
                save_session(session_id, session)
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
        deleted = delete_session(sid)
        if deleted:
            return json.dumps({"status": "SUCCESS", "message": f"Session '{sid}' đã được xóa khỏi đĩa."}, ensure_ascii=False)
        return json.dumps({"status": "NOT_FOUND", "message": f"Session '{sid}' không tồn tại trên đĩa."}, ensure_ascii=False)



# ---------------------------------------------------------------------------
# Tool 2: deepseek_adversarial_review (Doubt-Driven Adversarial Code Review)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_adversarial_review(
    code_or_diff: str,
    context_spec: Optional[str] = None,
    effort: str = "high"
) -> str:
    """
    Thực hiện phản biện gắt (Doubt-Driven Development) trên đoạn code hoặc Git Diff.
    Persona độc lập có thiên kiến 'Bác bỏ, tìm cách phá hỏng code' để phát hiện edge cases, race conditions, memory leaks, bảo mật.
    
    Args:
        code_or_diff: Nội dung mã nguồn hoặc git diff cần thẩm định.
        context_spec: Tài liệu đặc tả hoặc yêu cầu của bài toán (nếu có).
        effort: Mức độ suy luận phản biện ('high' hoặc 'max'). Mặc định: 'high'.
    """
    client = get_client()
    system_prompt = (
        "You are a ruthless, cynical Principal Security & Reliability Engineer performing an Adversarial Doubt-Driven Review. "
        "Your explicit bias is to DISPROVE correctness, NOT to approve. "
        "Actively search for: "
        "1. Race conditions, deadlocks, lack of thread safety. "
        "2. Silent failure modes, swallowed exceptions, unhandled None/null. "
        "3. Boundary off-by-one errors, overflow, memory growth. "
        "4. Security vulnerabilities (injection, auth bypass, path traversal). "
        "5. Regression risks in production. "
        "Format your output strictly: "
        "- VERDICT: [CLEAN / MINOR_WARNINGS / CRITICAL_DEFECTS] "
        "- SUMMARY: 2-3 sentences "
        "- SPECIFIC_VULNERABILITIES_OR_DEFECTS: List with file, line, explanation, and how it breaks."
    )
    
    user_prompt = f"### CONTEXT SPEC:\n{context_spec or 'Not specified'}\n\n### CODE / DIFF TO AUDIT:\n{code_or_diff}"
    valid_effort = effort if effort in ["high", "max"] else "high"
    
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={"thinking": {"type": "enabled"}},
            reasoning_effort=valid_effort
        )
        msg = response.choices[0].message
        telemetry = extract_telemetry(response.usage)
        
        return json.dumps({
            "audit_verdict": msg.content,
            "reasoning_review": getattr(msg, "reasoning_content", None),
            "telemetry": telemetry
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Lỗi phản biện: {str(e)}"}, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Tool 3: deepseek_code_task (ReAct Loop, TDD Runner & Review)
# ---------------------------------------------------------------------------
@server.tool()
@tool_guard
def deepseek_code_task(
    workspace_dir: str,
    task: str,
    test_cmd: Optional[str] = None,
    thinking: bool = True,
    effort: str = "high",
    run_adversarial_review: bool = True,
    max_turns: int = 12
) -> str:
    """
    Thực thi tác vụ code tự động bằng DeepSeek TDD Loop:
    1. Tạo Safety Branch trong git.
    2. Architect Agent làm rõ yêu cầu & lập Spec (Thinking CoT Pre-Mortem).
    3. Coder Agent chạy ReAct Loop gọi tool sửa code & chạy test tự động.
    4. Tự phản biện Adversarial Review gắt gao trên Git Diff.
    5. Xuất báo cáo viễn trắc trung thực.
    
    Args:
        workspace_dir: Đường dẫn tuyệt đối của thư mục dự án cần sửa.
        task: Mô tả yêu cầu bài toán cần giải quyết.
        test_cmd: Lệnh chạy test tự động (ví dụ: 'pytest tests/test_auth.py').
        thinking: Bật Thinking Mode cho giai đoạn Architect và Reviewer.
        effort: Mức độ suy luận CoT ('low', 'high', 'max').
        run_adversarial_review: Chạy vòng thẩm định phản biện gắt sau khi test xanh.
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
    
    # 1. Thiết lập Git Safety Net nếu workspace là git repo
    is_git = os.path.exists(os.path.join(workspace, ".git"))
    original_branch = ""
    if is_git:
        try:
            curr_b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env)
            original_branch = curr_b.stdout.strip()
            cb_res = subprocess.run(["git", "checkout", "-b", safety_branch], cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env)
            if cb_res.returncode != 0:
                git_errors.append(f"git checkout -b failed: {cb_res.stderr.strip()}")
        except Exception as ge:
            git_errors.append(f"git safety net error: {str(ge)}")

    # 2. Giai đoạn 1: Architect / Requirement Clarification & Acceptance Criteria
    existing_files = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = sorted([d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv"]])
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
        "2. TEST SPECIFICATION: Define the exact test file to create/modify, the test cases/assertions, and the exact verification shell/bash command (e.g. test runner or execution command).\n"
        "3. MINIMAL IMPLEMENTATION: Design the simplest, most direct solution that solves the problem. Explicitly forbid unnecessary abstractions, design patterns, or unrequested flexibility.\n"
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

    # 3. Giai đoạn 2: Coder ReAct Loop với Local Tools
    coder_system = (
        "You are an Autonomous Senior Software Engineer. You have access to local tools: read_file, write_file, edit_file_replace, run_shell, list_files. "
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
    loop_telemetry = []

    for turn in range(max_turns):
        turns_executed += 1
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=sanitize_messages_for_api(messages),
            tools=LOCAL_TOOLS_SCHEMA,
            extra_body={"thinking": {"type": "disabled"}}, # Non-thinking mode for fast, precise tool calls
            reasoning_effort="none"
        )
        msg = res.choices[0].message
        messages.append(serialize_assistant_message(msg))
        loop_telemetry.append(extract_telemetry(res.usage))

        if not msg.tool_calls:
            # Model declares completion or gives text
            if test_cmd:
                test_out, test_rc, test_flag = _run_capped(test_cmd, workspace, exec_env, timeout=90, kill_on_cap=False)
                last_test_output = test_out
                if test_rc == 0:
                    test_passed = True
                    break
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Automated test verification FAILED with exit code {test_rc}:\n{last_test_output}\nFix the code to make tests pass."
                    })
                    continue
            else:
                test_passed = True
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

    # Nghiệm thu tự động cuối cùng với test_cmd nếu chưa được verify ở turn cuối
    if not test_passed and test_cmd:
        test_out, test_rc, test_flag = _run_capped(test_cmd, workspace, exec_env, timeout=90, kill_on_cap=False)
        last_test_output = test_out
        if test_rc == 0:
            test_passed = True

    # 4. Thu thập Git Diff an toàn
    git_diff = ""
    if is_git:
        try:
            add_res = subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env)
            if add_res.returncode != 0:
                git_errors.append(f"git add failed: {add_res.stderr.strip()}")
            diff_res = subprocess.run(
                ["git", "--no-pager", "diff", "--cached", "--no-ext-diff", "--no-textconv"],
                cwd=workspace, capture_output=True, text=True, timeout=30, env=exec_env, errors="replace"
            )
            git_diff = diff_res.stdout
        except Exception as de:
            git_errors.append(f"git diff error: {str(de)}")

    # 5. Giai đoạn 3: Adversarial Review trên Git Diff (nếu có diff và bật option)
    adversarial_report = None
    if run_adversarial_review and git_diff:
        adversarial_report = deepseek_adversarial_review(
            code_or_diff=git_diff[:200_000],
            context_spec=spec_plan,
            effort="max"
        )

    # Tổng hợp báo cáo viễn trắc trung thực
    report = {
        "status": "SUCCESS" if test_passed else "TESTS_FAILED_OR_INCOMPLETE",
        "task": task,
        "workspace": workspace,
        "safety_branch": safety_branch if is_git else "N/A (Not a git repo)",
        "turns_executed": turns_executed,
        "architect_spec": spec_plan,
        "test_verification": {
            "test_cmd": test_cmd,
            "passed": test_passed,
            "output_snippet": last_test_output[:2000] if last_test_output else "No test command run"
        },
        "git_diff_summary": git_diff[:3000] if git_diff else "No uncommitted git diff",
        "git_errors": git_errors if git_errors else None,
        "adversarial_review": json.loads(adversarial_report) if adversarial_report else "Skipped",
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
