import os
import sys
import time
import json
import tempfile
import pytest

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import (
    is_safe_path,
    check_command_safety,
    extract_telemetry,
    execute_local_tool,
    server
)

def test_path_safety_sandbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Safe paths
        safe, target = is_safe_path(tmpdir, "subdir/file.txt")
        assert safe is True
        assert target.startswith(tmpdir)
        
        # Path traversal attack
        safe, target = is_safe_path(tmpdir, "../../etc/passwd")
        assert safe is False

def test_command_blacklist_guardrail():
    # Dangerous commands
    safe, reason = check_command_safety("rm -rf /")
    assert safe is False
    assert "Guardrail" in reason

    safe, reason = check_command_safety("rm -rf *")
    assert safe is False

    safe, reason = check_command_safety("dd if=/dev/zero of=/dev/sda")
    assert safe is False

    safe, reason = check_command_safety("git push origin main --force")
    assert safe is False

    # Safe commands
    safe, reason = check_command_safety("pytest tests/test_code.py")
    assert safe is True
    assert reason is None

    safe, reason = check_command_safety("npm run build")
    assert safe is True

def test_extract_telemetry():
    class DummyUsage:
        prompt_tokens = 1000
        prompt_cache_hit_tokens = 850
        prompt_cache_miss_tokens = 150
        completion_tokens = 200
        total_tokens = 1200
        class DummyDetails:
            reasoning_tokens = 120
        completion_tokens_details = DummyDetails()

    res = extract_telemetry(DummyUsage())
    assert res["prompt_tokens"] == 1000
    assert res["prompt_cache_hit_tokens"] == 850
    assert res["cache_hit_rate_pct"] == 85.0
    assert res["reasoning_tokens"] == 120
    assert res["total_tokens"] == 1200

def test_local_tool_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write file
        res = execute_local_tool(tmpdir, "write_file", {"path": "hello.py", "content": "def foo():\n    return 42\n"})
        assert "Successfully wrote" in res
        
        # 2. Read file
        content = execute_local_tool(tmpdir, "read_file", {"path": "hello.py"})
        assert "return 42" in content
        
        # 3. Edit file replace
        res_edit = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "hello.py",
            "search_block": "return 42",
            "replace_block": "return 100"
        })
        assert "Successfully replaced" in res_edit
        
        content_after = execute_local_tool(tmpdir, "read_file", {"path": "hello.py"})
        assert "return 100" in content_after
        
        # 4. List files
        files_json = execute_local_tool(tmpdir, "list_files", {})
        assert "hello.py" in files_json

def test_path_safety_denylist():
    with tempfile.TemporaryDirectory() as tmpdir:
        # .git protection (RCE prevention)
        safe, _ = is_safe_path(tmpdir, ".git/hooks/pre-commit")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, ".git/config")
        assert safe is False

        # .venv protection
        safe, _ = is_safe_path(tmpdir, ".venv/bin/python")
        assert safe is False

        # .env protection (sensitive configs)
        safe, _ = is_safe_path(tmpdir, ".env")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, ".env.production")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, ".env.local")
        assert safe is False

        # Allowed templates
        safe, _ = is_safe_path(tmpdir, ".env.example")
        assert safe is True
        safe, _ = is_safe_path(tmpdir, ".env.template")
        assert safe is True

        # Private keys and certificates
        safe, _ = is_safe_path(tmpdir, "id_rsa.key")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, "certificate.pem")
        assert safe is False

def test_read_file_single_large_line():
    """Kiểm tra đọc file 1 dòng duy nhất có kích thước 2MB mà không bị OOM hay treo."""
    with tempfile.TemporaryDirectory() as tmpdir:
        large_line_file = os.path.join(tmpdir, "oneline.txt")
        # Tạo file 1 dòng duy nhất dài 1MB
        with open(large_line_file, "w", encoding="utf-8") as f:
            f.write("A" * (1024 * 1024))

        start = time.time()
        res = execute_local_tool(tmpdir, "read_file", {"path": "oneline.txt"})
        elapsed = time.time() - start

        assert elapsed < 2.0
        data = json.loads(res)
        assert data["truncated"] is True
        assert len(data["content"]) <= 256 * 1024

def test_read_file_vietnamese_paging():
    """Kiểm tra phân trang qua next_start_line trên file tiếng Việt không bị lỗi font mojibake."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "vietnamese.txt")
        lines = [
            "Dòng 1: Xin chào thế giới từ DeepSeek MCP\n",
            "Dòng 2: Phản biện kiến trúc Doubt-Driven Development\n",
            "Dòng 3: Tối ưu hoá bộ đệm KV Cache và viễn trắc trung thực\n",
            "Dòng 4: Không over-engineer, giữ vững nguyên tắc Karpathy\n"
        ]
        with open(test_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Đọc 2 dòng đầu
        res1 = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "vietnamese.txt", "start_line": 1, "line_count": 2}))
        assert res1["start_line"] == 1
        assert res1["end_line"] == 2
        assert res1["next_start_line"] == 3
        assert res1["truncated"] is True
        assert "Xin chào thế giới" in res1["content"]

        # Đọc tiếp 2 dòng sau qua next_start_line
        res2 = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "vietnamese.txt", "start_line": res1["next_start_line"], "line_count": 2}))
        assert res2["start_line"] == 3
        assert res2["end_line"] == 4
        assert res2["eof"] is True
        assert "Không over-engineer" in res2["content"]
        assert "\ufffd" not in res1["content"] and "\ufffd" not in res2["content"]

def test_read_file_fifo_pipe():
    """Kiểm tra chặn FIFO pipe ngay lập tức mà không bao giờ mở file gây treo socket MCP."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fifo_path = os.path.join(tmpdir, "test_fifo")
        os.mkfifo(fifo_path)

        start = time.time()
        res = execute_local_tool(tmpdir, "read_file", {"path": "test_fifo"})
        elapsed = time.time() - start

        assert elapsed < 1.0  # Không bị block/treo
        data = json.loads(res)
        assert data["code"] == "E_NOT_REGULAR"

def test_read_file_bad_args():
    """Kiểm tra xử lý đối số dị dạng (start_line là string hoặc âm, line_count là 0)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "dummy.txt")
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("line 1\nline 2\n")

        res1 = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "dummy.txt", "start_line": "1", "line_count": 0}))
        assert res1["start_line"] == 1

        res2 = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "dummy.txt", "start_line": -5, "line_count": "invalid"}))
        assert res2["start_line"] == 1

def test_path_safety_case_insensitive():
    """Kiểm tra chặn bypass chữ hoa/thường trên hệ thống file như macOS APFS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        safe, _ = is_safe_path(tmpdir, ".GIT/hooks/pre-commit")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, ".Git/config")
        assert safe is False
        safe, _ = is_safe_path(tmpdir, ".VENV/bin/python")
        assert safe is False

def test_run_shell_yes_kill_fast():
    """Kiểm tra lệnh nhả output vô tận (yes) bị kill tức thì khi chạm 128KB, không chờ 90s."""
    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.time()
        res = execute_local_tool(tmpdir, "run_shell", {"command": "yes"})
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert "OUTPUT TRUNCATED" in res

def test_run_shell_non_utf8_bytes():
    """Kiểm tra lệnh in byte không hợp lệ UTF-8 không làm crash UnicodeDecodeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        res = execute_local_tool(
            tmpdir,
            "run_shell",
            {"command": "printf '\\xff\\xfe\\xfd'"}
        )
        assert "Exit code:" in res

def test_run_shell_env_scrubbing():
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["TEST_SECRET_KEY"] = "super-secret-123"
        os.environ["DATABASE_AUTH_PASSWORD"] = "pass-456"
        try:
            res = execute_local_tool(
                tmpdir,
                "run_shell",
                {"command": 'python3 -c "import os; print(\'TEST_SECRET_KEY\' in os.environ, \'DATABASE_AUTH_PASSWORD\' in os.environ)"'}
            )
            assert "False False" in res
        finally:
            os.environ.pop("TEST_SECRET_KEY", None)
            os.environ.pop("DATABASE_AUTH_PASSWORD", None)
