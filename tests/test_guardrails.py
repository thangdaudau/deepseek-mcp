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
    """Kiểm tra file nhiều dòng dài chạm byte cap (24KB): next_start_line trỏ đúng dòng kế tiếp, không bị None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        large_file = os.path.join(tmpdir, "multilarge.txt")
        # 3 dòng, mỗi dòng 10KB: dòng 1 và 2 cộng lại là 20KB, dòng 3 sẽ chạm trần 24KB
        with open(large_file, "w", encoding="utf-8") as f:
            f.write("A" * (10 * 1024) + "\n")
            f.write("B" * (10 * 1024) + "\n")
            f.write("C" * (10 * 1024) + "\n")

        res = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "multilarge.txt", "start_line": 1}))
        assert res["truncated"] is True
        assert res["next_start_line"] == 3
        assert res["eof"] is False
        assert len(res["content"]) > 18 * 1024

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

        # 1. Kiểm tra raw mode (line_numbers=False): tái tạo 100% byte gốc
        reconstructed_parts = []
        cur_start = 1
        turns = 0

        while cur_start is not None and turns < 100:
            turns += 1
            res_raw = execute_local_tool(tmpdir, "read_file", {"path": "large_source.txt", "start_line": cur_start, "line_numbers": False})
            data = json.loads(res_raw)
            reconstructed_parts.append(data["content"])
            if data["eof"]:
                break
            cur_start = data["next_start_line"]

        reconstructed_text = "".join(reconstructed_parts)
        assert reconstructed_text == original_text
        assert turns > 1  # Đảm bảo đã thực sự phân qua nhiều trang

        # 2. Kiểm tra default mode (line_numbers=True): đánh số dòng chính xác cả khi nhảy cóc
        page1 = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "large_source.txt", "start_line": 1, "line_count": 3}))
        assert "1: Dòng 1:" in page1["content"]
        assert "2: Dòng 2:" in page1["content"]
        assert "3: Dòng 3:" in page1["content"]
        assert page1["line_numbers"] is True

        # Nhảy cóc đến dòng 500
        page_jump = json.loads(execute_local_tool(tmpdir, "read_file", {"path": "large_source.txt", "start_line": 500, "line_count": 3}))
        assert "500: Dòng 500:" in page_jump["content"]
        assert "501: Dòng 501:" in page_jump["content"]
        assert "502: Dòng 502:" in page_jump["content"]

def test_edit_file_replace_auto_strip_line_numbers():
    """Kiểm tra edit_file_replace tự động bóc tách số dòng nếu LLM copy từ read_file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "sample.py")
        code = "def hello():\n    print('Hello')\n    return 42\n"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        # Mô phỏng LLM đọc file và copy nguyên đoạn có tiền tố số dòng vào search_block
        search_block_with_numbers = "2:     print('Hello')\n3:     return 42\n"
        replace_block_with_numbers = "2:     print('World')\n3:     return 100\n"

        res = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "sample.py",
            "search_block": search_block_with_numbers,
            "replace_block": replace_block_with_numbers
        })

        assert "Successfully replaced search block" in res
        with open(filepath, "r", encoding="utf-8") as f:
            new_code = f.read()
        assert "print('World')" in new_code
        assert "return 100" in new_code
        # Chắc chắn số dòng '2: ' hay '3: ' không bị chèn vào mã nguồn file thật
        assert "2:" not in new_code

def test_run_capped_hang_with_capped_output():
    """Kiểm tra C1: Lệnh vừa nhả > 128KB vừa bị treo/sleep thì timeout vẫn kích hoạt và kill sạch, không bị nuốt thành capped."""
    from server import _run_capped
    with tempfile.TemporaryDirectory() as tmpdir:
        # Lệnh in 150KB rồi sleep 10s (với timeout=1s)
        cmd = 'python3 -c "import sys, time; sys.stdout.write(\'X\' * 150000); sys.stdout.flush(); time.sleep(10)"'
        start = time.time()
        out, rc, flag = _run_capped(cmd, tmpdir, os.environ.copy(), timeout=1, cap=128 * 1024, kill_on_cap=False)
        elapsed = time.time() - start
        assert elapsed < 4.0  # Không bị treo 10s
        assert flag == "timeout"
        assert "COMMAND TIMED OUT" in out

def test_env_scrubbing_preserves_ssh_auth_sock():
    """Kiểm tra C5: build_scrubbed_env tẩy sạch secret token nhưng bảo toàn SSH_AUTH_SOCK cho git-over-ssh."""
    from server import build_scrubbed_env
    old_val = os.environ.get("SSH_AUTH_SOCK")
    old_secret = os.environ.get("MY_API_KEY")
    try:
        os.environ["SSH_AUTH_SOCK"] = "/tmp/ssh-agent.sock"
        os.environ["MY_API_KEY"] = "sk-secret-123456"
        env = build_scrubbed_env()
        assert env.get("SSH_AUTH_SOCK") == "/tmp/ssh-agent.sock"
        assert "MY_API_KEY" not in env
    finally:
        if old_val:
            os.environ["SSH_AUTH_SOCK"] = old_val
        else:
            os.environ.pop("SSH_AUTH_SOCK", None)
        if old_secret:
            os.environ["MY_API_KEY"] = old_secret
        else:
            os.environ.pop("MY_API_KEY", None)

def test_edit_file_replace_rejects_non_utf8():
    """Kiểm tra N1: edit_file_replace từ chối file non-UTF8 bằng strict encoding, chống hỏng dữ liệu."""
    from server import execute_local_tool
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = os.path.join(tmpdir, "latin1.txt")
        # Byte 0xe9 trong ISO-8859-1 (ký tự é), không hợp lệ trong UTF-8 đơn lẻ
        with open(bad_file, "wb") as f:
            f.write(b"caf\xe9 resume\n")
            
        res = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "latin1.txt",
            "search_block": "caf",
            "replace_block": "tea",
        })
        assert "không phải định dạng UTF-8 hợp lệ" in res
        # File gốc phải còn nguyên vẹn, không bị đè bởi \ufffd
        with open(bad_file, "rb") as f:
            assert f.read() == b"caf\xe9 resume\n"

def test_write_and_edit_tempfile_cleanup_on_error(monkeypatch):
    """Kiểm tra N4: tempfile tự dọn dẹp sạch sẽ trong khối finally khi os.replace gặp lỗi."""
    from server import execute_local_tool
    with tempfile.TemporaryDirectory() as tmpdir:
        # Giả lập lỗi ở bước os.replace khi gọi write_file
        def fail_replace(src, dst):
            raise OSError("Simulated disk failure during replace")
            
        monkeypatch.setattr(os, "replace", fail_replace)
        
        res = execute_local_tool(tmpdir, "write_file", {
            "path": "target.txt",
            "content": "hello atomic write",
        })
        assert "Simulated disk failure during replace" in res
        # Bắt buộc không có bất kỳ file tạm (NamedTemporaryFile) nào còn sót lại
        assert os.listdir(tmpdir) == []

def test_file_permissions_preserved_on_write_and_edit():
    """Kiểm tra R1: atomic write bảo toàn chmod (như executable 0o755) của file gốc."""
    import stat
    from server import execute_local_tool
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "deploy.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\necho deploy\n")
        os.chmod(script_path, 0o755)
        
        # 1. Test edit_file_replace
        res = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "deploy.sh",
            "search_block": "echo deploy",
            "replace_block": "echo release",
        })
        assert "Successfully replaced" in res
        mode = stat.S_IMODE(os.stat(script_path).st_mode)
        assert mode == 0o755

        # 2. Test write_file overwriting existing file
        res = execute_local_tool(tmpdir, "write_file", {
            "path": "deploy.sh",
            "content": "#!/bin/bash\necho v2\n",
        })
        assert "Successfully wrote" in res
        mode = stat.S_IMODE(os.stat(script_path).st_mode)
        assert mode == 0o755

        # 3. Test write_file creating a brand-new file gets standard 0o644 (R1b)
        res = execute_local_tool(tmpdir, "write_file", {
            "path": "new_file.txt",
            "content": "brand new file",
        })
        assert "Successfully wrote" in res
        new_mode = stat.S_IMODE(os.stat(os.path.join(tmpdir, "new_file.txt")).st_mode)
        assert new_mode == 0o644

def test_edit_file_replace_preserves_crlf():
    """Kiểm tra R2: edit_file_replace giữ nguyên định dạng CRLF (\\r\\n), không bị ép thành LF."""
    from server import execute_local_tool
    with tempfile.TemporaryDirectory() as tmpdir:
        crlf_file = os.path.join(tmpdir, "windows.txt")
        with open(crlf_file, "wb") as f:
            f.write(b"line1\r\nline2\r\nline3\r\n")
            
        res = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "windows.txt",
            "search_block": "line2",
            "replace_block": "line2_modified",
        })
        assert "Successfully replaced" in res
        with open(crlf_file, "rb") as f:
            content = f.read()
            assert content == b"line1\r\nline2_modified\r\nline3\r\n"

def test_session_pruning_turn_boundary():
    """Kiểm tra N2/N3: _prune_session_messages bảo tồn turn boundary (bắt đầu bằng 'user'), không để orphan 'tool' message."""
    from server import _prune_session_messages
    
    # Giả lập lịch sử với 2 turn hoàn chỉnh chứa tool calls
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # Turn 1 (dài 10KB để dễ vượt budget)
        {"role": "user", "content": "Query 1: " + ("A" * 10000)},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents 1"},
        {"role": "assistant", "content": "Answer 1."},
        # Turn 2
        {"role": "user", "content": "Query 2."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": "file contents 2"},
        {"role": "assistant", "content": "Answer 2."},
    ]
    
    # Thiết lập max_bytes nhỏ (ví dụ 8KB) để buộc phải prune Turn 1
    pruned = _prune_session_messages(messages, max_bytes=8192)
    
    # 1. messages[0] luôn là system prompt
    assert pruned[0]["role"] == "system"
    # 2. Tin nhắn tiếp theo PHẢI là 'user' (Turn 2), TUYỆT ĐỐI không phải orphan 'tool' hay assistant tool call
    assert pruned[1]["role"] == "user"
    assert pruned[1]["content"] == "Query 2."
    # 3. Các tin nhắn tool trong Turn 2 được giữ nguyên vẹn
    roles = [m["role"] for m in pruned]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert pruned[3]["role"] == "tool"
    assert pruned[3]["tool_call_id"] == "call_2"
    assert pruned[2]["tool_calls"][0]["id"] == "call_2"



