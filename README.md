# DeepSeek MCP Server

Hệ thống MCP Server tích hợp DeepSeek-Flash (V4.1), hỗ trợ Thinking Mode (CoT) linh hoạt, TDD test runner tự động và cơ chế phản biện gắt Doubt-Driven Development.

## Vị trí cài đặt
- **Mã nguồn Server:** `~/.local/share/deepseek-mcp/server.py`
- **Môi trường ảo (venv):** `~/.local/share/deepseek-mcp/venv/`
- **Cấu hình API Key:** `~/.local/share/deepseek-mcp/.env`
- **Cấu hình Antigravity MCP:** `~/.gemini/config/mcp_config.json`

## Danh sách công cụ (MCP Tools)

### 1. `deepseek_ask`
Truy vấn DeepSeek-Flash hỗ trợ suy luận logic, phản biện đa hiệp (Multi-turn) và tự động khảo sát codebase:
- `prompt` (string, required): Nội dung câu hỏi, lập luận phản biện hoặc yêu cầu điều tra.
- `session_id` (string, optional): ID phiên làm việc để duy trì ngữ cảnh multi-turn bền vững trên đĩa (`~/.local/share/deepseek-mcp/sessions/<session_id>.json`). Khuyến nghị ghim Conversation ID của agent.
- `reset_session` (bool, default `False`): Xóa sạch ngữ cảnh cũ của session và bắt đầu hội thoại mới.
- `workspace_dir` (string, optional): Đường dẫn thư mục dự án / tài liệu. Khi cung cấp, kích hoạt vòng lặp Read-Only ReAct giúp DeepSeek tự hành gọi các tool `read_file` và `list_files` để bóc tách mã nguồn trực tiếp mà không cần copy paste nội dung file vào prompt.
- `max_tool_turns` (int, default `5`): Số lượt gọi tool tối đa trong 1 truy vấn khảo sát.
- `thinking` (bool, default `True`): Bật/tắt suy luận CoT.
- `effort` (string: `'low'`, `'high'`, `'max'`, default `'high'`): Mức độ suy luận CoT.
- `system_prompt` (string, optional): Chỉ dẫn hệ thống tĩnh để tối ưu KV Cache Prefix Matching.

### 2. `deepseek_code_task`
Tự động thực hiện chu trình: **Lập Spec (Architect CoT)** -> **Sửa file & chạy test (Coder ReAct)** -> **Tự phản biện (Doubt Gate)**:
- `workspace_dir` (string, required): Đường dẫn tuyệt đối thư mục dự án.
- `task` (string, required): Mô tả bài toán cần làm.
- `test_cmd` (string, optional): Lệnh chạy test tự động (ví dụ: `pytest tests/test_perf.py`).
- `thinking` (bool, default `True`): Dùng thinking mode cho Architect và Reviewer.
- `effort` (string, default `'high'`): Mức độ suy luận CoT.
- `run_adversarial_review` (bool, default `True`): Tự động thẩm định git diff sau khi test xanh.
- `max_turns` (int, default `12`): Số lượt ReAct tối đa.

### 3. `deepseek_adversarial_review`
Persona phản biện độc lập (Doubt-Driven) mang tâm thế "bác bỏ, tìm cách phá hỏng code":
- `code_or_diff` (string, required): Đoạn code hoặc git diff cần audit.
- `context_spec` (string, optional): Tài liệu yêu cầu / hợp đồng chức năng.
- `effort` (string: `'high'` hoặc `'max'`, default `'high'`).

### 4. `deepseek_session_clear`
Quản lý và giải phóng session trên đĩa:
- `session_id` (string, optional): ID session cần xóa. Nếu truyền `"all"`, toàn bộ session cache sẽ được dọn sạch.

### 5. `deepseek_health`
Kiểm tra tình trạng kết nối tới DeepSeek API, đo latency và kiểm tra token.

## Chốt chặn an toàn (Guardrails)
1. **Workspace Boundary:** Chặn path traversal ra ngoài `workspace_dir`.
2. **Command Blacklist:** Chặn các lệnh nguy hiểm (`rm -rf /`, `rm -rf *`, `mkfs`, `dd`, `git push --force`...).
3. **Git Safety Net:** Tự động tạo branch tạm `deepseek/task-<timestamp>` trước khi sửa mã nguồn.

## Kiểm thử
Chạy toàn bộ 11 bài unit & live integration tests:
```bash
~/.local/share/deepseek-mcp/venv/bin/pytest ~/.local/share/deepseek-mcp/tests/ -v
```
