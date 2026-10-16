import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import server

@pytest.mark.anyio
async def test_server_tools_registered():
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "deepseek_ask" in tool_names
    assert "deepseek_adversarial_review" in tool_names
    assert "deepseek_code_task" in tool_names
    assert "deepseek_health" in tool_names
    assert "deepseek_session_clear" in tool_names

    # Verify tool schemas have required descriptions and parameters
    ask_tool = next(t for t in tools if t.name == "deepseek_ask")
    assert "prompt" in ask_tool.input_schema.get("properties", {})
    assert "session_id" in ask_tool.input_schema.get("properties", {})
    assert "reset_session" in ask_tool.input_schema.get("properties", {})
    assert "thinking" in ask_tool.input_schema.get("properties", {})
    assert "effort" in ask_tool.input_schema.get("properties", {})

    code_task_tool = next(t for t in tools if t.name == "deepseek_code_task")
    assert "workspace_dir" in code_task_tool.input_schema.get("properties", {})
    assert "task" in code_task_tool.input_schema.get("properties", {})
    assert "test_cmd" in code_task_tool.input_schema.get("properties", {})

def test_ask_session_lifecycle():
    import json
    from server import ASK_SESSIONS, deepseek_session_clear, cleanup_expired_sessions
    
    # Clean slate
    ASK_SESSIONS.clear()
    
    # Emulate session creation
    session_id = "test-session-1"
    ASK_SESSIONS[session_id] = {
        "messages": [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "turn 1 question"},
            {"role": "assistant", "content": "turn 1 answer"}
        ],
        "system_prompt": "system instruction",
        "updated_at": 1000.0,
        "created_at": 1000.0,
        "turn_count": 1
    }
    
    assert session_id in ASK_SESSIONS
    assert len(ASK_SESSIONS[session_id]["messages"]) == 3
    
    # Test specific session clear
    clear_res = json.loads(deepseek_session_clear(session_id))
    assert clear_res["status"] == "SUCCESS"
    assert session_id not in ASK_SESSIONS
    
    # Test clearing all
    ASK_SESSIONS["s1"] = {"updated_at": 1000.0}
    ASK_SESSIONS["s2"] = {"updated_at": 1000.0}
    clear_all_res = json.loads(deepseek_session_clear(None))
    assert clear_all_res["status"] == "SUCCESS"
    assert len(ASK_SESSIONS) == 0

