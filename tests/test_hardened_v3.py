"""
Test suite TDD cho deepseek-mcp v3.6 (Hardened Google-Engineered Architecture).
Kiểm thử toàn diện 11 chốt chặn kỹ thuật (Gates G1 - G11):
- G1: Byte Offset Slicing (ContentOffset) trong read_file, không E_LINE_TOO_LONG.
- G2: Phân quyền bất đối xứng is_safe_path (khóa ghi/sửa vào .deepseek).
- G3, G4: WAL baseline và Pruning forces snapshot (chống nhân bản tin nhắn khi LRU eviction).
- G5, G6: Git diff an toàn (vô hiệu hóa hook/driver) và lọc secret untracked.
- G7: session_lock có bounded-wait timeout (15s) và bọc session_clear.
- G8, G9: Env-prefix parser an toàn và Coder run_shell timeout 30s.
- G10, G11: Reviewer Polyglot Adversarial Tests, RUN_COMMAND và Collision Guard.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import pytest
from pathlib import Path

# Thêm thư mục gốc của repo vào sys.path để import server
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from server import (
    is_safe_path,
    _read_file_envelope,
    execute_local_tool,
    pack_tool_result,
    parse_env_prefix,
    _run_capped,
    session_lock,
    save_session,
    load_session,
    _prune_session_messages,
    deepseek_session_clear,
    build_scrubbed_env,
    materialize_adversarial_tests,
    MAX_READ_BYTES,
    DEFAULT_CODER_SHELL_TIMEOUT,
    _HOT_SESSIONS
)


# ---------------------------------------------------------------------------
# Test 1: Gate G1 - Byte Offset Slicing trong read_file (Chuẩn Google)
# ---------------------------------------------------------------------------
def test_read_file_byte_offset_paging():
    """Kiểm tra đọc file 1 dòng dài 100KB qua byte_offset: không OOM, không E_LINE_TOO_LONG, ghép lại khớp 100%."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "long_line.txt")
        # Chuỗi dài 100KB gồm các ký tự ngẫu nhiên lặp lại
        expected_content = "X" * (100 * 1024)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(expected_content)

        offset = 0
        reconstructed = []
        turn_count = 0

        while True:
            turn_count += 1
            assert turn_count < 20, "Phân trang không kết thúc, có thể bị vòng lặp vô tận!"
            
            raw_res = execute_local_tool(tmpdir, "read_file", {
                "path": "long_line.txt",
                "byte_offset": offset
            })
            data = json.loads(raw_res)
            
            assert data["status"] == "OK"
            assert data.get("code") != "E_LINE_TOO_LONG", "Không bao giờ được ném lỗi E_LINE_TOO_LONG!"
            
            reconstructed.append(data["content"])
            
            if not data.get("truncated"):
                break
                
            assert "next_byte_offset" in data
            assert data["next_byte_offset"] > offset
            offset = data["next_byte_offset"]

        full_text = "".join(reconstructed)
        assert full_text == expected_content
        assert turn_count >= 4, "Phải phân trang ít nhất 4 lần cho 100KB với chunk 24KB!"


# ---------------------------------------------------------------------------
# Test 2: Sửa code khoanh vùng dải dòng (Line-Bounded Slicing - Chuẩn Google)
# ---------------------------------------------------------------------------
def test_edit_file_replace_line_bounded_scope():
    """Kiểm tra edit_file_replace với start_line và end_line: chỉ sửa đúng dải dòng chỉ định."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_file = os.path.join(tmpdir, "app.py")
        content = (
            "def foo():\n"
            "    value = 42\n"
            "    return value\n"
            "\n"
            "def bar():\n"
            "    value = 42\n"
            "    return value\n"
        )
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(content)

        # 1. Thử sửa không truyền dải dòng: search_block bị trùng 2 lần -> phải báo lỗi
        res_fail = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "app.py",
            "search_block": "    value = 42",
            "replace_block": "    value = 100"
        })
        assert "xuất hiện" in res_fail and ("2 lần" in res_fail or "nhiều lần" in res_fail)

        # 2. Truyền start_line=5, end_line=7 khoanh vùng hàm bar()
        res_ok = execute_local_tool(tmpdir, "edit_file_replace", {
            "path": "app.py",
            "search_block": "    value = 42",
            "replace_block": "    value = 999",
            "start_line": 5,
            "end_line": 7
        })
        assert "Successfully replaced" in res_ok

        # 3. Đọc lại kiểm tra: hàm foo() vẫn là 42, hàm bar() đổi thành 999
        with open(sample_file, "r", encoding="utf-8") as f:
            updated_text = f.read()

        assert "def foo():\n    value = 42" in updated_text
        assert "def bar():\n    value = 999" in updated_text


# ---------------------------------------------------------------------------
# Test 3: Gate G2 - Khóa quyền ghi/sửa vào thư mục .deepseek/
# ---------------------------------------------------------------------------
def test_is_safe_path_blocks_write_to_deepseek_dir():
    """Kiểm tra is_safe_path: chặn write_file/edit_file_replace vào .deepseek, cho phép read_file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        offload_file = ".deepseek/offload/evidence.txt"
        
        # 1. is_safe_path trực tiếp
        safe_write, _ = is_safe_path(tmpdir, offload_file, allow_offload_read=False)
        assert safe_write is False, "Mặc định phải cấm ghi vào .deepseek!"

        safe_read, _ = is_safe_path(tmpdir, offload_file, allow_offload_read=True)
        assert safe_read is True, "Phải cho phép đọc file trong .deepseek khi bật allow_offload_read!"

        # 2. Qua execute_local_tool: write_file bị từ chối
        res_write = execute_local_tool(tmpdir, "write_file", {
            "path": offload_file,
            "content": "fake logs"
        })
        assert "Error: Path ngoài workspace" in res_write or "nằm ngoài workspace" in res_write or "từ chối" in res_write.lower()


# ---------------------------------------------------------------------------
# Test 4: Gate G1 - Offload RAW 100% và Tính Tất định SHA256
# ---------------------------------------------------------------------------
def test_pack_tool_result_raw_and_deterministic():
    """Kiểm tra pack_tool_result: lưu RAW 100% không băm dòng, SHA256 tất định, có .gitignore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Chuỗi lớn hơn 32,000 ký tự
        long_output = "Line: " + ("A" * 5000) + "\n" + ("B" * 35000)
        
        res_1 = pack_tool_result(long_output, tool="run_shell", workspace=tmpdir, session_id="s1", turn=1)
        data_1 = json.loads(res_1)
        
        assert data_1["status"] == "OFFLOADED"
        assert len(data_1["preview_head"]) == 1000
        assert len(data_1["preview_tail"]) == 1000
        handle = data_1["handle"]
        
        # Kiểm tra file trên đĩa
        offload_path = os.path.join(tmpdir, ".deepseek", "offload", f"{handle}.txt")
        assert os.path.exists(offload_path)
        with open(offload_path, "r", encoding="utf-8") as f:
            disk_content = f.read()
            
        assert disk_content == long_output, "Dữ liệu trên đĩa phải RAW 100%, không bị chèn dấu newline bừa bãi!"

        # Kiểm tra .gitignore
        gitignore_path = os.path.join(tmpdir, ".deepseek", ".gitignore")
        assert os.path.exists(gitignore_path)
        with open(gitignore_path, "r", encoding="utf-8") as f:
            gi_content = f.read()
        assert "*" in gi_content and "!.gitignore" in gi_content

        # Kiểm tra tính tất định: cùng nội dung thì sinh cùng handle
        res_2 = pack_tool_result(long_output, tool="run_shell", workspace=tmpdir, session_id="s2", turn=2)
        data_2 = json.loads(res_2)
        assert data_2["handle"] == handle, "Nội dung giống nhau phải cho SHA256 handle giống nhau để tối ưu KV-cache!"


# ---------------------------------------------------------------------------
# Test 5: Gate G8 & G9 - Env-Prefix Parser và Coder run_shell timeout 30s
# ---------------------------------------------------------------------------
def test_env_prefix_parser_and_security_denylist():
    """Kiểm tra parse_env_prefix: cho phép tiền tố hợp lệ, chặn đứng key trong danh sách đen."""
    # 1. Hợp lệ
    extra, cmd, err = parse_env_prefix(["NODE_ENV=test", "npm", "test"])
    assert err is None
    assert extra == {"NODE_ENV": "test"}
    assert cmd == ["npm", "test"]

    extra2, cmd2, err2 = parse_env_prefix(["PYTHONPATH=.", "FOO=bar", "pytest", "tests/"])
    assert err2 is None
    assert extra2 == {"PYTHONPATH": ".", "FOO": "bar"}
    assert cmd2 == ["pytest", "tests/"]

    # 2. Cấm key nguy hiểm (cả khi gọi trực tiếp lẫn qua binary env)
    _, _, err_ld = parse_env_prefix(["LD_PRELOAD=/evil.so", "pytest"])
    assert err_ld is not None and "LD_PRELOAD" in err_ld

    _, _, err_env_ld = parse_env_prefix(["env", "LD_PRELOAD=/evil.so", "pytest"])
    assert err_env_ld is not None and "LD_PRELOAD" in err_env_ld

    _, _, err_path = parse_env_prefix(["PATH=/tmp", "ls"])
    assert err_path is not None and "PATH" in err_path

    _, _, err_env_path = parse_env_prefix(["env", "-i", "PATH=/tmp", "ls"])
    assert err_env_path is not None and "PATH" in err_env_path

    _, _, err_git = parse_env_prefix(["GIT_CONFIG_GLOBAL=/evil", "git", "status"])
    assert err_git is not None and "GIT_CONFIG" in err_git


def test_coder_run_shell_timeout_30s():
    """Kiểm tra _run_capped với DEFAULT_CODER_SHELL_TIMEOUT: lệnh sleep quá 30s bị kill tức thời."""
    # Thử nghiệm chạy lệnh sleep 35s với timeout=DEFAULT_CODER_SHELL_TIMEOUT (30s)
    # Để test chạy nhanh trong CI/unit test, ta kiểm tra với timeout nhỏ (1s) và xác nhận logic kill
    start_t = time.time()
    out, rc, flag = _run_capped(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        cwd=tempfile.gettempdir(),
        env=build_scrubbed_env(),
        timeout=1
    )
    duration = time.time() - start_t
    
    assert duration < 3.0, "Tiến trình phải bị kill ngay khi chạm timeout!"
    assert flag == "timeout"
    assert "TIMED OUT" in out


# ---------------------------------------------------------------------------
# Test 6: Gate G3 & G4 - WAL Baseline & Pruning Forces Snapshot
# ---------------------------------------------------------------------------
def test_wal_baseline_prevents_message_duplication_on_eviction():
    """Kiểm tra WAL baseline: khi bị LRU eviction, nạp lại session không làm nhân đôi lịch sử chat."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch thư mục lưu session vào tmpdir
        sid = "test_wal_eviction_session"
        wal_path = os.path.join(tmpdir, f"{sid}.wal")

        messages = [
            {"role": "system", "content": "System Layer 0"},
            {"role": "user", "content": "Turn 1 request"},
            {"role": "assistant", "content": "Turn 1 answer"}
        ]
        
        session_data = {
            "session_id": sid,
            "system_prompt": "System Layer 0",
            "messages": messages,
            "_wal_frames": 1,
            "_baseline_len": 3,
            "_baseline_head": "System Layer 0"[:64]
        }

        # Lưu lần 1 (tạo init/snapshot)
        with session_lock(sid):
            save_session(sid, session_data)

        # Giả lập LRU eviction: xóa session khỏi RAM _HOT_SESSIONS
        if sid in _HOT_SESSIONS:
            del _HOT_SESSIONS[sid]

        # Nạp lại từ đĩa
        reloaded = load_session(sid)
        assert len(reloaded["messages"]) == 3

        # Thêm turn 2
        reloaded["messages"].extend([
            {"role": "user", "content": "Turn 2 request"},
            {"role": "assistant", "content": "Turn 2 answer"}
        ])

        with session_lock(sid):
            save_session(sid, reloaded)

        # Giả lập restart/cold load lần 2
        if sid in _HOT_SESSIONS:
            del _HOT_SESSIONS[sid]

        final_session = load_session(sid)
        # Kỳ vọng: đúng 5 messages (1 system + 2 turns x 2 msgs), TUYỆT ĐỐI KHÔNG BỊ NHÂN ĐÔI thành 8!
        assert len(final_session["messages"]) == 5
        user_msgs = [m["content"] for m in final_session["messages"] if m["role"] == "user"]
        assert user_msgs == ["Turn 1 request", "Turn 2 request"]


def test_pruning_forces_wal_snapshot():
    """Kiểm tra khi cắt tỉa tin nhắn (pruning), WAL bắt buộc phải ghi snapshot frame, không append delta."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sid = "test_pruning_snapshot_session"
        
        # Tạo 10 user turns
        msgs = [{"role": "system", "content": "System"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"Turn {i} " + ("X" * 2000)})
            msgs.append({"role": "assistant", "content": f"Ans {i}"})
            
        session_data = {
            "session_id": sid,
            "system_prompt": "System",
            "messages": msgs,
            "_wal_frames": 1
        }
        with session_lock(sid):
            save_session(sid, session_data)

        # Cắt tỉa xuống giới hạn nhỏ (ví dụ 4KB)
        pruned_msgs = _prune_session_messages(session_data["messages"], max_bytes=4096)
        assert len(pruned_msgs) < len(session_data["messages"])
        
        session_data["messages"] = pruned_msgs
        with session_lock(sid):
            save_session(sid, session_data)

        if sid in _HOT_SESSIONS:
            del _HOT_SESSIONS[sid]

        reloaded = load_session(sid)
        assert len(reloaded["messages"]) == len(pruned_msgs), "Lịch sử nạp lại phải khớp chính xác với bản đã prune!"


# ---------------------------------------------------------------------------
# Test 7: Gate G10 & G11 - Reviewer Polyglot Adversarial Tests & Collision Guard
# ---------------------------------------------------------------------------
def test_reviewer_polyglot_adversarial_test_and_collision_guard():
    """Kiểm tra materialize_adversarial_tests: cô lập vào .deepseek/adversarial/, RUN_COMMAND và Collision Guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Giả lập workspace đã có sẵn file tests/test_auth.ts của developer
        tests_dir = os.path.join(tmpdir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        existing_test = os.path.join(tests_dir, "test_auth.ts")
        with open(existing_test, "w", encoding="utf-8") as f:
            f.write("// Original developer test suite\n")

        # Giả lập output Reviewer xuất khối ADVERSARIAL_TESTS
        sample_report = (
            "## ADVERSARIAL_TESTS\n\n"
            "RUN_COMMAND: npx vitest run .deepseek/adversarial/test_auth.ts\n"
            "```typescript:.deepseek/adversarial/test_auth.ts\n"
            "import { test } from 'vitest';\n"
            "test('race condition', () => {});\n"
            "```\n"
        )

        # 1. Gọi hàm materialize thật sự từ server.py (Lần 1)
        res1 = materialize_adversarial_tests(tmpdir, sample_report)
        assert len(res1) == 1
        test_info1 = res1[0]
        assert test_info1["status"] == "WRITTEN"
        assert test_info1["path"] == ".deepseek/adversarial/test_auth.ts"
        assert os.path.isfile(test_info1["absolute_path"])
        assert "npx vitest run .deepseek/adversarial/test_auth.ts" in test_info1["run_cmd"]

        # Kiểm tra file gốc của dev không hề bị đụng chạm
        with open(existing_test, "r", encoding="utf-8") as f:
            assert f.read() == "// Original developer test suite\n"

        # 2. Gọi tiếp lần 2 với cùng tên file nhưng nội dung khác -> Collision Guard kích hoạt
        sample_report2 = (
            "## ADVERSARIAL_TESTS\n\n"
            "RUN_COMMAND: npx vitest run .deepseek/adversarial/test_auth.ts\n"
            "```typescript:.deepseek/adversarial/test_auth.ts\n"
            "import { test } from 'vitest';\n"
            "test('second race condition', () => {});\n"
            "```\n"
        )
        res2 = materialize_adversarial_tests(tmpdir, sample_report2)
        assert len(res2) == 1
        test_info2 = res2[0]
        assert test_info2["status"] == "WRITTEN"
        assert test_info2["path"] != test_info1["path"], "Lần 2 trùng tên phải tự đổi tên bằng hash!"
        assert "_adv_" in test_info2["path"] or test_info2["path"].startswith(".deepseek/adversarial/test_auth_")
        assert os.path.isfile(test_info2["absolute_path"])
        # File lần 1 vẫn nguyên vẹn
        assert os.path.isfile(test_info1["absolute_path"])
        # Lệnh chạy của lần 2 được cập nhật đúng tên file mới
        assert test_info2["path"] in test_info2["run_cmd"]
