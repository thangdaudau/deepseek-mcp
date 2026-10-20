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

def test_live_code_task_e2e():
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
            max_turns=8
        )

        report = json.loads(raw_report)
        print(f"\nCode Task E2E Report:\nStatus: {report.get('status')}\nTurns: {report.get('turns_executed')}")
        print(f"Git Diff:\n{report.get('git_diff_summary')}")

        assert report["status"] in ("PASSED", "SUCCESS")
        assert report["test_verification"]["passed"] is True
        assert os.path.exists(os.path.join(tmpdir, "math_lib.py"))
        
        # Verify the tests actually pass
        verify_run = subprocess.run(test_cmd, shell=True, cwd=tmpdir, capture_output=True, text=True)
        assert verify_run.returncode == 0


def test_live_coder_and_reviewer_adversarial_e2e():
    """
    Test E2E thực chiến kép:
    1. Coder (deepseek_code_task) tự đọc và sửa file account.py có sẵn để implement hàm withdraw, chạy test pass.
    2. Reviewer (deepseek_adversarial_review) soi Git Diff, phát hiện lỗi Race Condition, tự viết test đối kháng.
    3. Server tự động bóc tách và ghi file test đối kháng vào .deepseek/adversarial/.
    4. Chạy file test đối kháng để chứng minh bug được tái hiện.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Khởi tạo Git repo
        subprocess.run("git init -b main", shell=True, cwd=tmpdir, check=True, capture_output=True)
        subprocess.run('git config user.email "dev@example.com"', shell=True, cwd=tmpdir, check=True)
        subprocess.run('git config user.name "Dev"', shell=True, cwd=tmpdir, check=True)

        # 1. Tạo file account.py ban đầu (chưa có withdraw)
        init_code = (
            "class Account:\n"
            "    def __init__(self, balance: int = 100):\n"
            "        self.balance = balance\n"
        )
        with open(os.path.join(tmpdir, "account.py"), "w", encoding="utf-8") as f:
            f.write(init_code)

        # Tạo file test đơn luồng cơ bản
        init_test = (
            "from account import Account\n\n"
            "def test_withdraw_basic():\n"
            "    acc = Account(100)\n"
            "    assert acc.withdraw(40) is True\n"
            "    assert acc.balance == 60\n"
            "    assert acc.withdraw(70) is False\n"
        )
        with open(os.path.join(tmpdir, "test_account.py"), "w", encoding="utf-8") as f:
            f.write(init_test)

        subprocess.run("git add . && git commit -m 'Initial commit with Account class and test'", shell=True, cwd=tmpdir, check=True, capture_output=True)

        # 2. Giao việc cho Coder: Sửa file account.py để implement withdraw
        task_desc = (
            "Sửa file account.py, bổ sung phương thức withdraw(self, amount: int) -> bool vào class Account. "
            "Nếu self.balance >= amount thì dùng time.sleep(0.05) mô phỏng độ trễ I/O, sau đó trừ balance và trả về True. "
            "Nếu balance < amount thì trả về False. Làm cho test_account.py pass."
        )
        test_cmd = f"{sys.executable} -m pytest test_account.py"

        print("\n--- GIAI ĐOẠN 1: CODER SỬA CODE & CHẠY TEST ---")
        coder_raw = deepseek_code_task(
            workspace_dir=tmpdir,
            task=task_desc,
            test_cmd=test_cmd,
            thinking=True,
            effort="high",
            max_turns=6
        )
        coder_report = json.loads(coder_raw)
        print(f"Coder Status: {coder_report.get('status')}")
        print(f"Coder Turns: {coder_report.get('turns_executed')}")
        print(f"Git Diff Summary:\n{coder_report.get('git_diff_summary')}")

        assert coder_report["status"] in ("PASSED", "SUCCESS")
        assert coder_report["test_verification"]["passed"] is True

        # Kiểm tra file account.py đã được Coder sửa và có hàm withdraw
        with open(os.path.join(tmpdir, "account.py"), "r", encoding="utf-8") as f:
            account_code = f.read()
        assert "def withdraw" in account_code
        assert "sleep" in account_code

        # 3. Giai đoạn 2: Reviewer nhảy vào audit Git Diff và tự viết test đối kháng
        print("\n--- GIAI ĐOẠN 2: REVIEWER PHẢN BIỆN & SINH TEST ĐỐI KHÁNG ---")
        review_raw = deepseek_adversarial_review(
            workspace_dir=tmpdir,
            context_spec=(
                "SPECIFICATION REQUIREMENT: The Account class must be strictly thread-safe under concurrent access. "
                "Multiple concurrent threads withdrawing simultaneously must NEVER cause an overdraft (balance < 0) "
                "or inconsistent balance updates. Any race condition is a CRITICAL defect."
            ),
            effort="high"
        )
        review_res = json.loads(review_raw)
        print(f"Reviewer Verdict:\n{review_res.get('audit_verdict')[:800]}...")
        written_tests = review_res.get("written_tests", [])
        print(f"Written Adversarial Tests Count: {len(written_tests)}")
        for wt in written_tests:
            print(f" -> Path: {wt.get('path')} | Run cmd: {wt.get('run_cmd')}")

        # Kiểm tra Reviewer đã bắt được lỗi Race Condition và sinh test vật lý
        assert len(written_tests) >= 1, "Reviewer phải tự động sinh ít nhất 1 test case đối kháng vật lý!"
        first_test = written_tests[0]
        assert first_test["status"] == "WRITTEN"
        assert os.path.isfile(first_test["absolute_path"]), "File test đối kháng phải tồn tại thật sự trên đĩa!"
        assert ".deepseek/adversarial" in first_test["path"], "Test đối kháng phải được cô lập trong .deepseek/adversarial/!"

        # 4. Giai đoạn 3: Thực thi test đối kháng để chứng minh bug được phát hiện
        print("\n--- GIAI ĐOẠN 3: THỰC THI TEST ĐỐI KHÁNG ---")
        adv_cmd = first_test.get("run_cmd") or f"{sys.executable} -m pytest {first_test['path']}"
        # Đảm bảo dùng đúng sys.executable của venv hiện tại
        if adv_cmd.startswith("pytest "):
            adv_cmd = f"{sys.executable} -m " + adv_cmd
        elif adv_cmd.startswith("python "):
            adv_cmd = f"{sys.executable} " + adv_cmd[7:]
        elif adv_cmd.startswith("python3 "):
            adv_cmd = f"{sys.executable} " + adv_cmd[8:]

        # Thêm cờ -x để dừng ngay khi test đối kháng bắt được bug đầu tiên (tiết kiệm thời gian)
        if "pytest" in adv_cmd and " -x" not in adv_cmd:
            adv_cmd += " -x"

        adv_proc = subprocess.run(
            adv_cmd,
            shell=True,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60
        )
        print(f"Adversarial Test Exit Code: {adv_proc.returncode}")
        print(f"Adversarial Test Output:\n{adv_proc.stdout}\n{adv_proc.stderr}")

        # Vì code withdraw của Coder không có lock (dính race condition),
        # nên test đối kháng đa luồng của Reviewer phải phát hiện được lỗi (exit code != 0)!
        assert adv_proc.returncode != 0, "Test đối kháng phải phát hiện được lỗi (test fail) trên code chưa an toàn!"
        print(f"\n[E2E VERIFIED] Coder đã sửa code thành công, và Reviewer đã tự sinh test đối kháng bắt thóp bug tại {first_test['path']}!")

