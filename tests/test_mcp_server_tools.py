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

    # Verify tool schemas have required descriptions and parameters
    ask_tool = next(t for t in tools if t.name == "deepseek_ask")
    assert "prompt" in ask_tool.input_schema.get("properties", {})
    assert "thinking" in ask_tool.input_schema.get("properties", {})
    assert "effort" in ask_tool.input_schema.get("properties", {})

    code_task_tool = next(t for t in tools if t.name == "deepseek_code_task")
    assert "workspace_dir" in code_task_tool.input_schema.get("properties", {})
    assert "task" in code_task_tool.input_schema.get("properties", {})
    assert "test_cmd" in code_task_tool.input_schema.get("properties", {})
