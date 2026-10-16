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
    import os
    from server import (
        save_session,
        load_session,
        delete_session,
        delete_all_sessions,
        deepseek_session_clear,
        get_session_file
    )
    
    session_id = "test-disk-session-1"
    file_path = get_session_file(session_id)
    
    # 1. Clean slate
    delete_session(session_id)
    assert not os.path.exists(file_path)
    
    # 2. Save session to disk
    session_data = {
        "session_id": session_id,
        "messages": [
            {"role": "system", "content": "system instruction"},
            {"role": "user", "content": "turn 1 question"},
            {"role": "assistant", "content": "turn 1 answer"}
        ],
        "system_prompt": "system instruction",
        "updated_at": 2000000000.0,
        "created_at": 2000000000.0,
        "turn_count": 1
    }
    save_session(session_id, session_data)
    assert os.path.exists(file_path)
    
    # 3. Load session from disk (survives restart)
    loaded = load_session(session_id)
    assert loaded is not None
    assert loaded["session_id"] == session_id
    assert len(loaded["messages"]) == 3
    
    # 4. Clear single session tool
    clear_res = json.loads(deepseek_session_clear(session_id))
    assert clear_res["status"] == "SUCCESS"
    assert not os.path.exists(file_path)
    assert load_session(session_id) is None
    
    # 5. Clear all sessions tool
    save_session("s1", {"messages": [], "updated_at": 2000000000.0})
    save_session("s2", {"messages": [], "updated_at": 2000000000.0})
    clear_all_res = json.loads(deepseek_session_clear(None))
    assert clear_all_res["status"] == "SUCCESS"
    assert load_session("s1") is None
    assert load_session("s2") is None


