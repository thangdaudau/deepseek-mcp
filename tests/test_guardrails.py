import os
import sys
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

def test_read_file_size_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        large_file = os.path.join(tmpdir, "large.txt")
        # Create a file slightly larger than 2MB
        with open(large_file, "wb") as f:
            f.write(b"0" * (2 * 1024 * 1024 + 1024))

        res = execute_local_tool(tmpdir, "read_file", {"path": "large.txt"})
        assert "vượt quá giới hạn dung lượng 2MB" in res

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
