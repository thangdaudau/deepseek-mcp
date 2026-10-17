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
    assert "workspace_dir" in ask_tool.input_schema.get("properties", {})
    assert "max_tool_turns" in ask_tool.input_schema.get("properties", {})

    code_task_tool = next(t for t in tools if t.name == "deepseek_code_task")
    assert "workspace_dir" in code_task_tool.input_schema.get("properties", {})
    assert "task" in code_task_tool.input_schema.get("properties", {})
    assert "test_cmd" in code_task_tool.input_schema.get("properties", {})

def test_read_only_tools_schema_isolation():
    from server import READ_ONLY_TOOLS_SCHEMA, LOCAL_TOOLS_SCHEMA
    
    ro_tool_names = [t["function"]["name"] for t in READ_ONLY_TOOLS_SCHEMA]
    assert "read_file" in ro_tool_names
    assert "list_files" in ro_tool_names

    read_tool = next(t for t in READ_ONLY_TOOLS_SCHEMA if t["function"]["name"] == "read_file")
    props = read_tool["function"]["parameters"]["properties"]
    assert "path" in props
    assert "start_line" in props
    assert "line_count" in props

    # Ensure mutating tools are strictly forbidden in read-only schema
    assert "write_file" not in ro_tool_names
    assert "edit_file_replace" not in ro_tool_names
    assert "run_shell" not in ro_tool_names
    
    all_tool_names = [t["function"]["name"] for t in LOCAL_TOOLS_SCHEMA]
    assert "write_file" in all_tool_names
    assert "run_shell" in all_tool_names


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
    
    # 5. Clear all sessions tool (with explicit 'all' and rejection on None)
    save_session("s1", {"messages": [], "updated_at": 2000000000.0})
    save_session("s2", {"messages": [], "updated_at": 2000000000.0})
    clear_all_res = json.loads(deepseek_session_clear("all"))
    assert clear_all_res["status"] == "SUCCESS"
    assert load_session("s1") is None
    assert load_session("s2") is None

    save_session("s3", {"messages": [], "updated_at": 2000000000.0})
    # None or empty must be rejected with E_BAD_ARG to prevent accidental total wipeout
    clear_none_res = json.loads(deepseek_session_clear(None))
    assert clear_none_res["status"] == "ERROR"
    assert clear_none_res["code"] == "E_BAD_ARG"
    assert load_session("s3") is not None

    clear_empty_res = json.loads(deepseek_session_clear("  "))
    assert clear_empty_res["status"] == "ERROR"
    assert clear_empty_res["code"] == "E_BAD_ARG"
    assert load_session("s3") is not None

    clear_all_res2 = json.loads(deepseek_session_clear("all"))
    assert clear_all_res2["status"] == "SUCCESS"
    assert load_session("s3") is None

    # Test namespace collision prevention
    f1 = get_session_file("conv/1")
    f2 = get_session_file("conv_1")
    assert f1 != f2, "Collision: conv/1 and conv_1 must resolve to distinct session files"

def test_list_files_schema_and_execution_alias():
    from server import READ_ONLY_TOOLS_SCHEMA, execute_local_tool
    import json
    
    # Check schema has path parameter
    list_tool = next(t for t in READ_ONLY_TOOLS_SCHEMA if t["function"]["name"] == "list_files")
    assert "path" in list_tool["function"]["parameters"]["properties"]
    
    # Test execute_local_tool with path="." and alias "list_directory"
    cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = execute_local_tool(cur_dir, "list_directory", {"path": "."})
    files = json.loads(out)
    assert isinstance(files, list)
    assert "server.py" in files or "README.md" in files

def test_session_lock_concurrency():
    import json
    from server import session_lock, SessionBusyError, deepseek_ask
    
    session_id = "test-lock-concurrency"
    
    with session_lock(session_id):
        # Trying to acquire the same session lock must immediately raise SessionBusyError
        with pytest.raises(SessionBusyError):
            with session_lock(session_id):
                pass
        
        # Calling deepseek_ask with the busy session must return E_SESSION_BUSY envelope
        res_raw = deepseek_ask(prompt="hello", session_id=session_id)
        res = json.loads(res_raw)
        assert res["status"] == "ERROR"
        assert res["error"]["code"] == "E_SESSION_BUSY"
        assert "đang được xử lý" in res["error"]["message"]

def test_test_cmd_guardrail_rejected(tmp_path):
    import json
    from server import deepseek_code_task
    
    ws = str(tmp_path)
    res_raw = deepseek_code_task(
        workspace_dir=ws,
        task="Testing safety rejection",
        test_cmd="rm -rf /"
    )
    res = json.loads(res_raw)
    assert res["status"] == "ERROR"
    assert res["error"]["code"] == "E_CMD_REJECTED"
    assert "Safety Guardrail" in res["error"]["message"]




