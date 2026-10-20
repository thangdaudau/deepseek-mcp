# DeepSeek MCP Server

MCP Server tích hợp mô hình **DeepSeek-Flash** (ngữ cảnh 1M tokens), cung cấp bộ công cụ mạnh mẽ cho AI Agent và lập trình viên: điều tra codebase tự hành, tự động lập trình theo chu trình TDD và phản biện độc lập Doubt-Driven Development.

---

## 1. Cấu hình nhanh

### Bước 1: Khai báo API Key
Tạo hoặc chỉnh sửa file `~/.local/share/deepseek-mcp/.env`:
```bash
DEEPSEEK_API_KEY="sk-your-deepseek-api-key"
DEEPSEEK_MODEL="deepseek-flash"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### Bước 2: Đăng ký MCP Server
Thêm cấu hình vào `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "deepseek": {
      "command": "/home/peapea/.local/share/deepseek-mcp/venv/bin/python3",
      "args": [
        "/home/peapea/.local/share/deepseek-mcp/server.py"
      ],
      "env": {
        "DEEPSEEK_MODEL": "deepseek-flash"
      }
    }
  }
}
```

---

## 2. Hướng dẫn sử dụng các MCP Tools

### 1. `deepseek_ask`
Dùng để suy luận logic sâu, hỏi đáp kỹ thuật, tranh luận nhiều hiệp hoặc tự động điều tra codebase.

- **Các tham số:**
  - `prompt` *(string, bắt buộc)*: Câu hỏi, luận điểm hoặc yêu cầu khảo sát.
  - `workspace_dir` *(string, tùy chọn)*: Đường dẫn thư mục dự án. Khi cung cấp, DeepSeek được cấp quyền tự động gọi các công cụ đọc file (`read_file`, `list_files`, `search_file`) để tự tìm hiểu mã nguồn mà không cần copy-paste thủ công vào prompt.
  - `session_id` *(string, tùy chọn)*: ID phiên làm việc để duy trì ngữ cảnh multi-turn bền vững. Khuyến nghị dùng Conversation ID của agent.
  - `reset_session` *(bool, mặc định `False`)*: Xóa ngữ cảnh cũ của session và bắt đầu hội thoại mới.
  - `max_tool_turns` *(int, mặc định `5`)*: Số lượt tự đọc file tối đa trong một lần điều tra.
  - `thinking` *(bool, mặc định `True`)*: Bật/tắt chế độ suy luận CoT (Thinking mode).
  - `effort` *(string: `'low'`, `'high'`, `'max'`, mặc định `'high'`)*: Mức độ đào sâu suy luận CoT.

- **Ví dụ sử dụng:**
  ```json
  {
    "prompt": "Find where authentication tokens are verified and check for potential expiration bypasses.",
    "workspace_dir": "/path/to/my-project",
    "session_id": "my-chat-session-01",
    "effort": "high"
  }
  ```

---

### 2. `deepseek_code_task`
Ủy thác cho DeepSeek tự động viết code và chạy test khép kín (TDD ReAct loop).

- **Các tham số:**
  - `workspace_dir` *(string, bắt buộc)*: Đường dẫn tuyệt đối của thư mục dự án.
  - `task` *(string, bắt buộc)*: Mô tả chi tiết yêu cầu bài toán hoặc tính năng cần triển khai/sửa lỗi.
  - `test_cmd` *(string, tùy chọn)*: Lệnh chạy test tự động (ví dụ: `pytest tests/test_auth.py` hoặc `npm test`).
  - `thinking` *(bool, mặc định `True`)*: Bật suy luận cho giai đoạn Architect lập Spec.
  - `effort` *(string, mặc định `'high'`)*: Mức độ suy luận.
  - `max_turns` *(int, mặc định `12`)*: Giới hạn số lượt Coder ReAct sửa file và chạy test.

- **Ví dụ sử dụng:**
  ```json
  {
    "workspace_dir": "/path/to/my-project",
    "task": "Thêm method validate_token(token) vào auth.py và đảm bảo test trong tests/test_auth.py pass.",
    "test_cmd": "pytest tests/test_auth.py"
  }
  ```

---

### 3. `deepseek_adversarial_review`
Phản biện gắt mã nguồn hoặc Git Diff độc lập theo tinh thần Doubt-Driven Development (tìm lỗi race condition, memory leak, bypass logic, vỡ hợp đồng dữ liệu).

- **Các tham số:**
  - `workspace_dir` *(string, tùy chọn)*: Đường dẫn thư mục dự án. Server sẽ tự động thu thập Git Diff sạch và các file untracked để thẩm định.
  - `context_spec` *(string, tùy chọn)*: Tài liệu đặc tả, yêu cầu kỹ thuật hoặc ràng buộc của bài toán.
  - `code_or_diff` *(string, tùy chọn)*: Đoạn code hoặc Git Diff dán trực tiếp nếu không chỉ định `workspace_dir`.
  - `effort` *(string: `'high'` hoặc `'max'`, mặc định `'max'`)*: Mức độ phản biện.

- **Kết quả trả về:**
  - Báo cáo lỗi chi tiết: Danh sách lỗ hổng tiềm ẩn kèm mức độ nghiêm trọng.
  - **Tự động sinh Test đối kháng vật lý:** Nếu phát hiện lỗi, Reviewer sẽ tự động viết test case đối kháng, server tự động lưu file vào `<workspace>/.deepseek/adversarial/` và trả về lệnh `run_cmd` để chạy thử ngay lập tức (có cơ chế chống ghi đè an toàn).

- **Ví dụ sử dụng:**
  ```json
  {
    "workspace_dir": "/path/to/my-project",
    "context_spec": "Hàm chuyển tiền phải đảm bảo thread-safe tuyệt đối, cấm race condition gây âm số dư.",
    "effort": "max"
  }
  ```

---

### 4. `deepseek_session_clear`
Quản lý và giải phóng các session hội thoại multi-turn lưu trên đĩa.

- **Các tham số:**
  - `session_id` *(string, bắt buộc)*: ID của session cần xóa. Nếu truyền `"all"`, toàn bộ session trên đĩa sẽ được dọn sạch.

---

### 5. `deepseek_health`
Kiểm tra tình trạng kết nối tới DeepSeek API, độ trễ mạng và tính hợp lệ của API Key.

- **Các tham số:** Không có tham số.

---

## 3. Chạy kiểm thử hệ thống

Để kiểm tra toàn bộ hoạt động của MCP Server và các chốt chặn an toàn:

```bash
# Chạy toàn bộ 47 unit & integration tests offline
/home/peapea/.local/share/deepseek-mcp/venv/bin/pytest tests/test_hardened_v3.py tests/test_guardrails.py tests/test_mcp_server_tools.py -v

# Chạy test thực chiến E2E gọi trực tiếp DeepSeek API (yêu cầu có mạng)
/home/peapea/.local/share/deepseek-mcp/venv/bin/pytest tests/test_live_deepseek_integration.py -k test_live_coder_and_reviewer_adversarial_e2e -s -v
```
