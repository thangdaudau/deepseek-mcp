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
    """Kiểm tra đọc file 1 dòng duy nhất có kích thước 1MB trả về E_LINE_TOO_LONG, không OOM, không loop vô tận."""
    with tempfile.TemporaryDirectory() as tmpdir:
        large_line_file = os.path.join(tmpdir, "oneline.txt")
        # Tạo file 1 dòng duy nhất dài 1MB (không chứa dấu xuống dòng)
        with open(large_line_file, "w", encoding="utf-8") as f:
            f.write("A" * (1024 * 1024))

        start = time.time()
        res = execute_local_tool(tmpdir, "read_file", {"path": "oneline.txt"})
        elapsed = time.time() - start

        assert elapsed < 2.0
        data = json.loads(res)
        assert data["status"] == "ERROR"
        assert data["code"] == "E_LINE_TOO_LONG"
        assert "vượt quá trần an toàn" in data["message"]
        assert len(data["partial_content"]) > 0

def test_read_file_multi_large_lines_paging():
    """Kiểm tra file nhiều dòng dài chạm byte cap: next_start_line trỏ đúng dòng kế tiếp, không bị None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        large_file = os.path.join(tmpdir, "multilarge.txt")
        # 3 dòng, mỗi dòng 100KB: dòng 1 và 2 cộng lại là 200KB, dòng 3 sẽ chạm trần 256KB
        with open(large_file, "w", encoding="utf-8") as f:
            f.write("A" * (100 * 1024) + "\n")
            f.write("B" * (100 * 1024) + "\n")
            f.write("C" * (100 * 1024) + "\n")

        res = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "multilarge.txt", "start_line": 1}))
        assert res["truncated"] is True
        assert res["next_start_line"] == 3
        assert res["eof"] is False
        assert len(res["content"]) > 150 * 1024

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

def test_run_capped_non_killing_test_runner():
    """Kiểm tra _run_capped với kill_on_cap=False (dành cho TDD test runner) cho phép tiến trình chạy hết và trả exit code 0."""
    from server import _run_capped
    with tempfile.TemporaryDirectory() as tmpdir:
        # Lệnh in 200KB output (> 128KB cap) rồi exit 0
        cmd = 'python3 -c "import sys; sys.stdout.write(\'X\' * 200000); sys.exit(0)"'
        out, rc, flag = _run_capped(cmd, tmpdir, os.environ.copy(), timeout=10, cap=128 * 1024, kill_on_cap=False)
        assert rc == 0
        assert flag == "capped"
        assert "OUTPUT TRUNCATED" in out
        assert len(out) >= 128 * 1024

def test_read_file_deep_offset_beyond_256kb():
    """Kiểm tra đọc file tại offset sâu (dòng 7000 trên file 800KB) không bị false E_LINE_TOO_LONG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "deep_file.txt")
        lines = [f"Line {str(i).zfill(6)}: " + "A" * 30 + "\n" for i in range(1, 20001)]
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)

        res_raw = execute_local_tool(tmpdir, "read_file", {"path": "deep_file.txt", "start_line": 7000, "line_count": 100})
        res = json.loads(res_raw)
        assert res.get("status") != "ERROR"
        assert res["start_line"] == 7000
        assert "Line 007000" in res["content"]
        assert len(res["content"]) > 0

def test_read_file_full_file_reconstruction_property():
    """Property test: Đọc lặp qua nhiều trang tới EOF trên file 1MB, ghép lại phải khớp 100% bản gốc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "large_source.txt")
        # Tạo file ~1MB gồm 15.000 dòng
        original_lines = [f"Dòng {i}: Test nội dung tiếng Việt có dấu và mã nguồn Python def func_{i}(): return {i}\n" for i in range(1, 15001)]
        original_text = "".join(original_lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(original_text)

        reconstructed_parts = []
        cur_start = 1
        turns = 0

        while cur_start is not None and turns < 50:
            turns += 1
            res_raw = execute_local_tool(tmpdir, "read_file", {"path": "large_source.txt", "start_line": cur_start})
            data = json.loads(res_raw)
            reconstructed_parts.append(data["content"])
            if data["eof"]:
                break
            cur_start = data["next_start_line"]

        reconstructed_text = "".join(reconstructed_parts)
        assert reconstructed_text == original_text
        assert turns > 1  # Đảm bảo đã thực sự phân qua nhiều trang


