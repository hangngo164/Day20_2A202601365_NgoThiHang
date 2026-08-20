# Báo Cáo Phân Tích Failure Modes & Mitigations — Multi-Agent Research System

> **Học viên:** Ngô Thị Hằng (MSSV: 2A202601365)  
> **Repository:** [hangngo164/Day20_2A202601365_NgoThiHang](https://github.com/hangngo164/Day20_2A202601365_NgoThiHang)  
> **Khóa học:** Lab 20 - Multi-Agent Research System

---

## I. Tổng Quan

Trong kiến trúc Multi-Agent (`Supervisor` + `Researcher` + `Analyst` + `Writer` + `Critic`), hệ thống được chia nhỏ thành nhiều tác tử chuyên biệt giao tiếp qua **Shared State**.

Tuy mang lại chất lượng nghiên cứu vượt trội và có trích dẫn nguồn rõ ràng, hệ thống Multi-Agent phải đối mặt với nhiều rủi ro vận hành (**Failure Modes**) phức tạp hơn so với Single-Agent. Dưới đây là phân tích chi tiết 6 Failure Modes thực tế và các giải pháp phòng vệ (Guardrails & Mitigations) đã triển khai trong mã nguồn.

---

## II. Chi Tiết Các Failure Modes và Giải Pháp

### 1. LLM API Timeout & Rate Limit

* **Mô tả:** OpenAI API có thể bị timeout hoặc trả mã lỗi `429 Too Many Requests` (Rate Limit) do hệ thống multi-agent gửi liên tiếp nhiều request trong thời gian ngắn (Supervisor ➔ Researcher ➔ Analyst ➔ Writer ➔ Critic).
* **Cách phát hiện:** Nhật ký lỗi xuất hiện `APITimeoutError`, `RateLimitError` hoặc `APIConnectionError`.
* **Giải pháp đã triển khai:**
  * Bọc hàm gọi API bằng decorator `@retry` từ thư viện `tenacity` với chiến lược `wait_exponential` (thử lại tối đa 3 lần với thời gian chờ tăng dần `1s ➔ 2s ➔ 4s`).
  * Chỉ retry với các lỗi mạng và quá tải, bỏ qua lỗi sai cấu hình (`401/403`).
  * Cài đặt `timeout=45s` cho mỗi request.

```python
# src/multi_agent_research_lab/services/llm_client.py
@retry(
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
    ...
```

---

### 2. Vòng Lặp Định Tuyến Vô Hạn & Routing Không Hợp Lệ (Infinite Loop)

* **Mô tả:** Supervisor có thể trả về câu trả lời định tuyến không đúng định dạng (ví dụ: *"I think we should call researcher..."* thay vì chỉ trả 1 từ khóa) hoặc lặp vô hạn giữa `Researcher` và `Analyst` mà không chuyển sang bước `Writer`.
* **Cách phát hiện:** Biến `state.iteration` tăng liên tục hoặc `route` không nằm trong tập hợp hợp lệ `{"researcher", "analyst", "writer", "done"}`.
* **Giải pháp đã triển khai:**
  * **Guardrail cứng (Max Iterations):** Cài đặt `MAX_ITERATIONS = 6`. Nếu vượt ngưỡng, hệ thống tự động ép trạng thái `route = 'done'` và tổng hợp từ dữ liệu hiện có trong state.
  * **Heuristic Fallback (Luật xác định):** Khi Supervisor gặp lỗi hoặc trả route không hợp lệ, hệ thống tự động kích hoạt hàm `_heuristic_route(state)` để chọn agent tiếp theo dựa trên trạng thái state.

```python
# src/multi_agent_research_lab/agents/supervisor.py
if state.iteration >= self._settings.max_iterations:
    logger.warning("Max iterations reached, forcing done")
    state.record_route("done")
    return state

if route not in valid_routes:
    route = self._heuristic_route(state)  # Deterministic fallback
```

---

### 3. Lỗi Kết Nối / Hết Hạn Ngạch Dịch Vụ Tìm Kiếm (Search API Failure)

* **Mô tả:** Tavily Search API hết hạn ngạch (quota), mất mạng hoặc chưa cấu hình key khiến Researcher không thể thu thập tài liệu từ internet.
* **Cách phát hiện:** `SearchClient.search()` raise Exception, danh sách `state.sources` rỗng.
* **Giải pháp đã triển khai:**
  * **Mock Fallback Provider:** Tự động chuyển đổi sang tập dữ liệu nghiên cứu mẫu (Mock Documents) về GraphRAG khi không có `TAVILY_API_KEY` hoặc khi gọi API thất bại.
  * **Graceful Degradation:** Researcher ghi nhận cảnh báo vào `state.errors`, các agent Analyst và Writer phía sau vẫn tiếp tục hoạt động bình thường trên nguồn dữ liệu fallback.

```python
# src/multi_agent_research_lab/services/search_client.py
if self._api_key and self._api_key.strip():
    docs = self._tavily_search(query, max_results)
else:
    logger.warning("No TAVILY_API_KEY found, using mock search results")
    docs = self._mock_search(query, max_results)
```

---

### 4. Ảo Giác Nguồn Trích Dẫn & Sai Lệch Thông Tin (Hallucinated Citations)

* **Mô tả:** Writer agent có thể tự suy diễn ra các số trích dẫn giả mạo (ví dụ: `[6]`, `[7]`) không hề có trong danh sách 5 nguồn ban đầu, hoặc diễn giải sai lệch dữ kiện so với tài liệu gốc.
* **Cách phát hiện:** Số thứ tự trích dẫn trong `final_answer` lớn hơn độ dài của `state.sources`, hoặc chỉ số `citation_coverage` thấp.
* **Giải pháp đã triển khai:**
  * **Critic Agent:** Thêm node Critic hoạt động như một bên kiểm định độc lập sau khi Writer hoàn thành để đánh giá tính xác thực (Fact-checking).
  * **Đo lường Citation Coverage:** Đo tỷ lệ nguồn được trích dẫn thực tế trong benchmark:
    $$\text{Citation Coverage} = \frac{\text{Số nguồn được trích dẫn}}{\text{Tổng số nguồn}}$$
  * Kết quả thực nghiệm đạt **100% Citation Coverage** (5/5 nguồn được trích dẫn chính xác `[1]` đến `[5]`).

---

### 5. Sai Lệch Cấu Hình Máy Chủ Observability (Langfuse Host Mismatch)

* **Mô tả:** Tài khoản người dùng được tạo trên US Cloud (`https://us.cloud.langfuse.com`), nhưng SDK mặc định gọi sang EU Cloud (`https://cloud.langfuse.com`), dẫn tới lỗi `401 Unauthorized: Invalid credentials. Confirm that you've configured the correct host`.
* **Cách phát hiện:** Xuất hiện lỗi OTLP exporter HTTP 401 trong quá trình đẩy span batch.
* **Giải pháp đã triển khai:**
  * Cập nhật `Settings` trong `core/config.py` hỗ trợ linh hoạt cả `LANGFUSE_HOST` và `LANGFUSE_BASE_URL` thông qua `AliasChoices`.
  * Bổ sung hàm kiểm tra xác thực an toàn `client.auth_check()` trước khi kích hoạt tracing.
  * Tự động sinh và hiển thị link trực tiếp tới Trace trên Langfuse Cloud ngay sau mỗi lượt chạy CLI.

```python
# src/multi_agent_research_lab/core/config.py
langfuse_host: str = Field(
    default="https://cloud.langfuse.com",
    validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
)
```

---

### 6. Tràn Ngữ Cảnh Token (Context Window Overflow)

* **Mô tả:** Khi thu thập nhiều tài liệu web có nội dung dài, tổng độ dài prompt tích lũy qua các agent có thể vượt quá giới hạn token hoặc làm loãng ngữ cảnh (Context Pollution).
* **Cách phát hiện:** `output_tokens` bị ngắt quãng bất thường hoặc `finish_reason = "length"`.
* **Giải pháp đã triển khai:**
  * Cắt tỉa (truncate) mỗi snippet nguồn tối đa 500 ký tự trong `SearchClient`.
  * Sử dụng mô hình `gpt-4o-mini` với ngữ cảnh rộng (128K tokens).
  * Mỗi agent chỉ nhận đúng phần dữ liệu cần thiết từ Shared State thay vì toàn bộ lịch sử thô.

---

## III. Bảng Tổng Hợp Ma Trận Rủi Ro & Biện Pháp

| Failure Mode | Xác Suất | Mức Độ Ảnh Hưởng | Biện Pháp Phòng Vệ (Mitigation) |
| :--- | :---: | :---: | :--- |
| **LLM Timeout / Rate Limit** | Trung bình | Rất cao (Treo hệ thống) | `@retry` 3 lần với Exponential Backoff |
| **Invalid Routing / Infinite Loop** | Thấp | Rất cao (Đốt token vô hạn) | `MAX_ITERATIONS = 6` + Heuristic Fallback |
| **Search API Failure** | Trung bình | Trung bình (Thiếu nguồn mới) | Mock Data Fallback + Graceful Degradation |
| **Hallucinated Citations** | Trung bình | Cao (Sai lệch học thuật) | `CriticAgent` kiểm tra chéo độc lập |
| **Langfuse Host Mismatch** | Thấp | Trung bình (Mất trace) | Hỗ trợ `AliasChoices` cho biến môi trường |
| **Context Overflow** | Thấp | Trung bình (Bị cắt output) | Truncate snippet 500 ký tự + chọn lọc state |

---

## IV. Kết Luận & Bài Học Kinh Nghiệm

1. **Guardrails là bắt buộc trong Production:** Tuyệt đối không để agent chạy tự do mà không có giới hạn vòng lặp (`MAX_ITERATIONS`), thời gian chờ (`TIMEOUT`), và xử lý lỗi ngắt (`FALLBACK`).
2. **Tách bạch vai trò (Separation of Concerns):** Phân chia rõ ràng giữa Agent tìm kiếm (Researcher), Agent phân tích (Analyst), Agent viết bài (Writer) và Agent duyệt bài (Critic) giúp kiểm soát chất lượng đầu ra chặt chẽ.
3. **Observability là "Hộp Đen":** Tracing chi tiết từng bước gọi API (Tokens, Cost, Latency) giúp nhanh chóng phát hiện điểm nghẽn và tối ưu hóa hệ thống.
