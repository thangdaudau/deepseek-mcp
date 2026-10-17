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
import subprocess
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from openai import OpenAI
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+-[a-zA-Z0-9]*[rfRF][a-zA-Z0-9]*\s+(/|~|\$HOME)(\s|$)",
    r"\brm\s+-[a-zA-Z0-9]*[rfRF][a-zA-Z0-9]*\s+\*(\s|$)",
    r"\bmkfs(\.[a-z0-9]+)?(\s|$)",
    r"\bdd\s+if=",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bgit\s+push\s+.*--force\b",
    r"\bshutdown(\s|$)",
    r"\breboot(\s|$)",
    r"\bpoweroff(\s|$)",
    r"\b(chmod|chown)\s+-[rR]\s+777\s+(/|~)(\s|$)"
]

server = MCPServer("DeepSeek-MCP")

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
    return OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)

def is_safe_path(base_dir: str, target_path: str) -> Tuple[bool, str]:
    """Kiểm tra chặn Path Traversal ra ngoài workspace."""
    base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base, target_path))
    if target == base or target.startswith(base + os.sep):
        return True, target
    return False, target

def check_command_safety(command: str) -> Tuple[bool, Optional[str]]:
    """Kiểm tra lệnh nguy hiểm trong blacklist."""
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
# ---------------------------------------------------------------------------
# Session Manager for Stateful Multi-Turn Conversations (File-based Persistence)
# ---------------------------------------------------------------------------
SESSIONS_DIR = os.path.expanduser("~/.local/share/deepseek-mcp/sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
SESSION_TTL_SECONDS = 86400  # 24 hours (1 ngày)

def get_session_file(session_id: str) -> str:
    """Tạo đường dẫn an toàn cho session file, chặn path traversal."""
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
    return os.path.join(SESSIONS_DIR, f"{safe_id}.json")

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Tải session từ đĩa, tự hủy nếu quá hạn TTL."""
    file_path = get_session_file(session_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if time.time() - data.get("updated_at", 0) > SESSION_TTL_SECONDS:
                os.remove(file_path)
                return None
            return data
        except Exception:
            return None
    return None

def save_session(session_id: str, data: Dict[str, Any]) -> None:
    """Ghi bền vững session xuống file JSON trên đĩa."""
    file_path = get_session_file(session_id)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def delete_session(session_id: str) -> bool:
    """Xóa file session chỉ định trên đĩa."""
    file_path = get_session_file(session_id)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception:
            pass
    return False

def delete_all_sessions() -> int:
    """Xóa sạch toàn bộ session trên đĩa."""
    if not os.path.exists(SESSIONS_DIR):
        return 0
    count = 0
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".json"):
            try:
                os.remove(os.path.join(SESSIONS_DIR, fname))
                count += 1
            except Exception:
                pass
    return count

def cleanup_expired_sessions() -> None:
    """Tự động dọn dẹp các session quá hạn TTL (24h) trên đĩa."""
    if not os.path.exists(SESSIONS_DIR):
        return
    now = time.time()
    try:
        for fname in os.listdir(SESSIONS_DIR):
            if fname.endswith(".json"):
                fpath = os.path.join(SESSIONS_DIR, fname)
                try:
                    if now - os.path.getmtime(fpath) > SESSION_TTL_SECONDS:
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
            "description": "Đọc nội dung một file trong workspace để điều tra/phân tích mã nguồn",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Đường dẫn file tương đối trong workspace"}
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
                    "sub_dir": {"type": "string", "description": "Thư mục con cần xem (để trống nếu xem từ root workspace)"}
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

def serialize_assistant_message(msg: Any) -> Dict[str, Any]:
    """Chuyển đổi assistant message thành dictionary tương thích JSON và DeepSeek API."""
    d: Dict[str, Any] = {
        "role": "assistant",
        "content": msg.content or ""
    }
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        d["reasoning_content"] = reasoning
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        } for tc in msg.tool_calls]
    return d

def execute_local_tool(workspace: str, name: str, args: Dict[str, Any]) -> str:
    """Thực thi tool local với chốt chặn an toàn Sandbox."""
    if name == "read_file":
        rel_path = args.get("path", "")
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace!"
        if not os.path.exists(full_path):
            return f"Error: File '{rel_path}' không tồn tại."
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
            
    elif name == "write_file":
        rel_path = args.get("path", "")
        content = args.get("content", "")
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace!"
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to '{rel_path}'."
        except Exception as e:
            return f"Error writing file: {str(e)}"
            
    elif name == "edit_file_replace":
        rel_path = args.get("path", "")
        search_block = args.get("search_block", "")
        replace_block = args.get("replace_block", "")
        safe, full_path = is_safe_path(workspace, rel_path)
        if not safe:
            return f"Error: Path '{rel_path}' nằm ngoài workspace!"
        if not os.path.exists(full_path):
            return f"Error: File '{rel_path}' không tồn tại."
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                current_text = f.read()
            if search_block not in current_text:
                return f"Error: search_block không tìm thấy trong file '{rel_path}'."
            count = current_text.count(search_block)
            if count > 1:
                return f"Error: search_block xuất hiện {count} lần trong file. Phải cung cấp thêm ngữ cảnh xung quanh để xác định duy nhất."
            new_text = current_text.replace(search_block, replace_block, 1)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_text)
            return f"Successfully replaced search block in '{rel_path}'."
        except Exception as e:
            return f"Error editing file: {str(e)}"
            
    elif name == "run_shell":
        cmd = args.get("command", "")
        safe, reason = check_command_safety(cmd)
        if not safe:
            return f"Command Rejected by Safety Guardrail: {reason}"
        try:
            venv_bin = os.path.dirname(sys.executable)
            env = os.environ.copy()
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=90,
                env=env
            )
            out = res.stdout + ("\n--- STDERR ---\n" + res.stderr if res.stderr else "")
            return f"Exit code: {res.returncode}\n{out[:4000]}"
        except subprocess.TimeoutExpired:
            return "Command timed out after 90 seconds."
        except Exception as e:
            return f"Command execution error: {str(e)}"
            
    elif name == "list_files":
        sub_dir = args.get("sub_dir", "")
        safe, full_path = is_safe_path(workspace, sub_dir)
        if not safe:
            return "Error: Path ngoài workspace."
        if not os.path.exists(full_path):
            return "Error: Directory does not exist."
        try:
            files_list = []
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv"]]
                for file in files:
                    rel = os.path.relpath(os.path.join(root, file), workspace)
                    files_list.append(rel)
                    if len(files_list) >= 100:
                        break
                if len(files_list) >= 100:
                    break
            return json.dumps(files_list[:100], indent=2)
        except Exception as e:
            return f"Error listing files: {str(e)}"

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# Tool 1: deepseek_ask (Trực tiếp suy luận hoặc truy vấn DeepSeek Flash)
# ---------------------------------------------------------------------------
@server.tool()
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
    1. Stateful Multi-turn bền vững (lưu trữ JSON trên đĩa qua session_id).
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
    if workspace:
        default_sys += (
            " You have access to local read-only tools: 'read_file' and 'list_files'. "
            "Use them autonomously to inspect files in the workspace whenever you need code context before providing your verdict."
        )
    
    session = None
    turn_start_msg_count = 0
    if session_id:
        if not reset_session:
            session = load_session(session_id)
            
        if not session:
            initial_sys = system_prompt or default_sys
            session = {
                "session_id": session_id,
                "messages": [{"role": "system", "content": initial_sys}],
                "system_prompt": initial_sys,
                "updated_at": time.time(),
                "created_at": time.time(),
                "turn_count": 0
            }
        
        turn_start_msg_count = len(session["messages"])
        session["messages"].append({"role": "user", "content": prompt})
        session["updated_at"] = time.time()
        session["turn_count"] += 1
        messages = session["messages"]
        current_turn = session["turn_count"]
    else:
        messages = [
            {"role": "system", "content": system_prompt or default_sys},
            {"role": "user", "content": prompt}
        ]
        current_turn = 1
        
    kwargs: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": messages,
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
                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                all_telemetry.append(extract_telemetry(response.usage))
                messages.append(serialize_assistant_message(msg))
                
                if not getattr(msg, "tool_calls", None):
                    final_content = msg.content or ""
                    final_reasoning = getattr(msg, "reasoning_content", None) or ""
                    break
                    
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except Exception:
                        fn_args = {}
                    if fn_name in ["read_file", "list_files"]:
                        tool_out = execute_local_tool(workspace, fn_name, fn_args)
                        tools_used.append({"tool": fn_name, "args": fn_args})
                    else:
                        tool_out = f"Error: Tool '{fn_name}' không được phép trong chế độ read-only!"
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(tool_out)
                    })
                    
            if not final_content:
                messages.append({
                    "role": "user",
                    "content": "Synthesize your final answer now based on the information read."
                })
                kwargs_final = {
                    "model": DEFAULT_MODEL,
                    "messages": messages,
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


@server.tool()
def deepseek_session_clear(session_id: Optional[str] = None) -> str:
    """
    Xóa bỏ một hoặc toàn bộ session lưu ngữ cảnh multi-turn của deepseek_ask trên đĩa.
    
    Args:
        session_id: ID của session cần xóa. Nếu None, xóa toàn bộ session trên đĩa.
    """
    if session_id:
        deleted = delete_session(session_id)
        if deleted:
            return json.dumps({"status": "SUCCESS", "message": f"Session '{session_id}' đã được xóa khỏi đĩa."}, ensure_ascii=False)
        return json.dumps({"status": "NOT_FOUND", "message": f"Session '{session_id}' không tồn tại trên đĩa."}, ensure_ascii=False)
    else:
        count = delete_all_sessions()
        return json.dumps({"status": "SUCCESS", "message": f"Đã xóa toàn bộ {count} sessions trên đĩa."}, ensure_ascii=False)



# ---------------------------------------------------------------------------
# Tool 2: deepseek_adversarial_review (Doubt-Driven Adversarial Code Review)
# ---------------------------------------------------------------------------
@server.tool()
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
        return json.dumps({"error": f"Workspace '{workspace_dir}' không tồn tại!"}, ensure_ascii=False)
        
    client = get_client()
    task_id = int(time.time())
    safety_branch = f"deepseek/task-{task_id}"
    
    # 1. Thiết lập Git Safety Net nếu workspace là git repo
    is_git = os.path.exists(os.path.join(workspace, ".git"))
    original_branch = ""
    if is_git:
        try:
            curr_b = subprocess.run("git rev-parse --abbrev-ref HEAD", shell=True, cwd=workspace, capture_output=True, text=True)
            original_branch = curr_b.stdout.strip()
            subprocess.run(f"git checkout -b {safety_branch}", shell=True, cwd=workspace, capture_output=True)
        except Exception:
            pass

    # 2. Giai đoạn 1: Architect / Requirement Clarification & Acceptance Criteria
    existing_files = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", "node_modules", "venv", ".venv"]]
        for f in files:
            existing_files.append(os.path.relpath(os.path.join(root, f), workspace))
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
    )
    
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": coder_system},
        {"role": "user", "content": f"SPECIFICATION & PLAN:\n{spec_plan}\n\nExecute the code changes in the workspace now. Use your tools."}
    ]
    
    turns_executed = 0
    test_passed = False
    last_test_output = ""
    loop_telemetry = []
    
    venv_bin = os.path.dirname(sys.executable)
    exec_env = os.environ.copy()
    exec_env["PATH"] = f"{venv_bin}:{exec_env.get('PATH', '')}"

    for turn in range(max_turns):
        turns_executed += 1
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=LOCAL_TOOLS_SCHEMA,
            extra_body={"thinking": {"type": "disabled"}}, # Non-thinking mode for fast, precise tool calls
            reasoning_effort="none"
        )
        msg = res.choices[0].message
        messages.append(msg)
        loop_telemetry.append(extract_telemetry(res.usage))

        if not msg.tool_calls:
            # Model declares completion or gives text
            if test_cmd:
                test_exec = subprocess.run(test_cmd, shell=True, cwd=workspace, capture_output=True, text=True, timeout=90, env=exec_env)
                last_test_output = test_exec.stdout + test_exec.stderr
                if test_exec.returncode == 0:
                    test_passed = True
                    break
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Automated test verification FAILED with exit code {test_exec.returncode}:\n{last_test_output}\nFix the code to make tests pass."
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
            tool_output = execute_local_tool(workspace, fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_output)
            })

    # 4. Thu thập Git Diff (Bao gồm cả file mới tạo thông qua git add -A)
    git_diff = ""
    if is_git:
        try:
            subprocess.run("git add -A", shell=True, cwd=workspace, capture_output=True)
            diff_res = subprocess.run("git diff --cached", shell=True, cwd=workspace, capture_output=True, text=True)
            git_diff = diff_res.stdout
        except Exception:
            pass

    # 5. Giai đoạn 3: Adversarial Review trên Git Diff (nếu có diff và bật option)
    adversarial_report = None
    if run_adversarial_review and git_diff:
        adversarial_report = deepseek_adversarial_review(
            code_or_diff=git_diff,
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
