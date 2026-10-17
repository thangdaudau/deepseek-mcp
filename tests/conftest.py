import os
import pytest
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server

@pytest.fixture(autouse=True)
def isolate_sessions_dir(monkeypatch, tmp_path):
    """Tự động monkeypatch SESSIONS_DIR sang tmp_path trong mỗi test để bảo vệ session thật của user."""
    test_session_dir = str(tmp_path / "sessions")
    os.makedirs(test_session_dir, exist_ok=True)
    monkeypatch.setattr(server, "SESSIONS_DIR", test_session_dir)
