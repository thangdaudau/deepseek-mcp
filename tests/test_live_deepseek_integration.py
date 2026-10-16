import os
import sys
import json
import time
import tempfile
import subprocess
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import (
    deepseek_ask,
    deepseek_adversarial_review,
    deepseek_code_task,
    deepseek_health
)

def test_live_health():
    res = json.loads(deepseek_health())
    assert res["status"] == "HEALTHY"
    assert res["model"] == "deepseek-flash"
    assert res["latency_seconds"] > 0

def test_live_ask_non_thinking():
    raw = deepseek_ask(prompt="Trả lời ngắn gọn: 2 + 2 bằng mấy?", thinking=False)
    res = json.loads(raw)
    assert res["mode"] == "non-thinking"
    assert res["reasoning_content"] is None
    assert "4" in res["content"]
    assert res["telemetry"]["completion_tokens"] > 0

def test_live_ask_thinking_mode():
    raw = deepseek_ask(
        prompt="9.11 và 9.8 số nào lớn hơn? Giải thích ngắn gọn.",
        thinking=True,
        effort="high"
    )
    res = json.loads(raw)
    assert res["mode"] == "thinking"
    assert res["reasoning_content"] is not None
    assert len(res["reasoning_content"]) > 10
    assert "9.8" in res["content"]
    # Reasoning tokens should be counted
    assert res["telemetry"]["reasoning_tokens"] > 0

def test_live_kv_cache_prefix():
    # Large static prefix (> 250 tokens) to trigger server cluster block indexing
    static_prefix = (
        "System Policy Contract Alpha: You are an internal microservice evaluating system invariants. "
        "Strict rule: Every numeric calculation must be formatted in JSON schema with keys: calculation, result, valid. "
        "Domain constraints: All integers are signed 64-bit. Memory alignment is 8-byte aligned. "
        + "The cluster maintains Paxos consensus across 5 geo-distributed replicas. " * 15
    )
    
    # First call: Cache Miss / Primer
    raw1 = deepseek_ask(prompt="Evaluate: 10 + 20", thinking=False, system_prompt=static_prefix)
    res1 = json.loads(raw1)
    
    time.sleep(3) # Cluster indexing window
    
    # Second call: Exact same prefix, different question -> Should Hit KV Cache
    raw2 = deepseek_ask(prompt="Evaluate: 30 + 40", thinking=False, system_prompt=static_prefix)
    res2 = json.loads(raw2)
    
    # Verify cache hit on second call
    hit_tokens_2 = res2["telemetry"]["prompt_cache_hit_tokens"]
    print(f"\nKV Cache Test: Call 1 hit={res1['telemetry']['prompt_cache_hit_tokens']} miss={res1['telemetry']['prompt_cache_miss_tokens']} | Call 2 hit={hit_tokens_2} miss={res2['telemetry']['prompt_cache_miss_tokens']} hit_rate={res2['telemetry']['cache_hit_rate_pct']}%")
    assert hit_tokens_2 > 0

def test_live_adversarial_review():
    buggy_code = """
def withdraw(account_id: str, amount: float):
    # Intentional race condition & missing balance check
    balance = get_balance_from_db(account_id)
    time.sleep(0.01) # Simulated network delay
    new_balance = balance - amount
    update_balance_in_db(account_id, new_balance)
    return new_balance
"""
    raw = deepseek_adversarial_review(
        code_or_diff=buggy_code,
        context_spec="Banking withdrawal function that must be thread-safe and prevent overdraft.",
        effort="high"
    )
    res = json.loads(raw)
    verdict_text = res["audit_verdict"]
    
    # Check that adversarial reviewer detected race condition or overdraft
    assert any(keyword in verdict_text.upper() for keyword in ["RACE", "CONCURRENCY", "OVERDRAFT", "THREAD", "NEGATIVE", "LOCK"])
    assert res["telemetry"]["reasoning_tokens"] > 0

def test_live_code_task_harness_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize a temporary git repository
        subprocess.run("git init -b main", shell=True, cwd=tmpdir, check=True, capture_output=True)
        subprocess.run('git config user.email "test@example.com"', shell=True, cwd=tmpdir, check=True)
        subprocess.run('git config user.name "Test Dev"', shell=True, cwd=tmpdir, check=True)

        # Create an initial test file that tests a function not yet implemented
        test_content = (
            "import pytest\n"
            "from math_lib import fibonacci\n\n"
            "def test_fib():\n"
            "    assert fibonacci(0) == 0\n"
            "    assert fibonacci(1) == 1\n"
            "    assert fibonacci(5) == 5\n"
            "    assert fibonacci(6) == 8\n"
            "    with pytest.raises(ValueError):\n"
            "        fibonacci(-1)\n"
        )
        with open(os.path.join(tmpdir, "test_math.py"), "w") as f:
            f.write(test_content)

        subprocess.run("git add . && git commit -m 'initial commit with test'", shell=True, cwd=tmpdir, check=True, capture_output=True)

        # Now run deepseek_code_task to implement math_lib.py and make test pass
        task_desc = "Tạo file math_lib.py chứa hàm fibonacci(n: int) -> int. Xử lý n < 0 thì raise ValueError. Làm cho test trong test_math.py pass."
        test_cmd = f"{sys.executable} -m pytest test_math.py"

        raw_report = deepseek_code_task(
            workspace_dir=tmpdir,
            task=task_desc,
            test_cmd=test_cmd,
            thinking=True,
            effort="high",
            run_adversarial_review=True,
            max_turns=6
        )

        report = json.loads(raw_report)
        print(f"\nHarness E2E Report:\nStatus: {report.get('status')}\nTurns: {report.get('turns_executed')}")
        print(f"Git Diff:\n{report.get('git_diff_summary')}")

        assert report["status"] == "SUCCESS"
        assert report["test_verification"]["passed"] is True
        assert os.path.exists(os.path.join(tmpdir, "math_lib.py"))
        
        # Verify the tests actually pass
        verify_run = subprocess.run(test_cmd, shell=True, cwd=tmpdir, capture_output=True, text=True)
        assert verify_run.returncode == 0
