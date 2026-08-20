# Benchmark Report: Single-Agent vs Multi-Agent Research System

> **Repository:** `hangngo164/Day20_2A202601365_NgoThiHang`  
> **Date:** 2026-08-20  
> **Evaluation Query:** *"Research GraphRAG state-of-the-art and write a 500-word summary"*  
> **Tracing Provider:** Langfuse Cloud (US Region)

---

## 1. Results Summary

| Architecture | Latency (s) | Cost (USD) | Quality (0-10) | Citation Coverage | Failure Rate | Routing / Iterations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Single-Agent Baseline** | **10.80s** | ~$0.0004 | 5.0 / 10 | 0% (No external search) | 0% | 1 iteration (`single_agent`) |
| **Multi-Agent Workflow** | **38.94s** | ~$0.0020 | **10.0 / 10** | **100% (5/5 sources cited)** | 0% | 4 iterations (`researcher` ➔ `analyst` ➔ `writer` ➔ `done` ➔ `critic`) |

---

## 2. Comparative Analysis

### 2.1. Quality & Depth of Synthesis
* **Single-Agent Baseline (Score: 5.0/10):**
  * Dựa hoàn toàn vào parametric memory (kiến thức tĩnh của model).
  * Không có nguồn trích dẫn cụ thể (0% citation coverage), dễ gặp hiện tượng khái quát chung chung hoặc ảo giác (hallucination) khi hỏi về các tiến bộ kỹ thuật mới nhất.
* **Multi-Agent Workflow (Score: 10.0/10):**
  * **Researcher** lấy thông tin thực tế từ web thông qua Tavily Search (5 nguồn tài liệu uy tín).
  * **Analyst** bóc tách các luận điểm cốt lõi, so sánh cách tiếp cận (Document GraphRAG vs Rust implementations), đánh giá độ tin cậy.
  * **Writer** tổng hợp báo cáo ~500 từ với cấu trúc rõ ràng và đánh số trích dẫn `[1]`, `[2]`, `[3]`, `[4]`, `[5]`.
  * **Critic** thực hiện review độc lập, chấm điểm fact-checking và bảo đảm tính chuẩn xác.

### 2.2. Latency & Cost Trade-off
* **Latency:** Multi-agent chậm hơn ~3.6x (+28.14s) do luồng xử lý tuần tự qua nhiều bước (Supervisor ➔ Search ➔ Researcher Synthesis ➔ Analyst ➔ Writer ➔ Critic).
* **Cost:** Multi-agent tiêu tốn nhiều token hơn (~$0.0020 so với ~$0.0004), nhưng hoàn toàn xứng đáng với độ chính xác và khả năng kiểm chứng của tài liệu nghiên cứu.

---

## 3. Failure Modes & Mitigations

| Failure Mode | Nguyên nhân & Ảnh hưởng | Cách giải quyết (Mitigation) đã triển khai |
| :--- | :--- | :--- |
| **LLM API Timeout / RateLimit** | Model bị nghẽn mạng hoặc quá tải làm cả pipeline bị dừng. | Áp dụng decorator `@retry` từ thư viện `tenacity` với `wait_exponential` (tối đa 3 lần thử lại). |
| **Routing Infinite Loop** | Supervisor lặp vô hạn giữa Researcher và Analyst. | Thiết lập guardrail `MAX_ITERATIONS=6` trong `SupervisorAgent` để tự động ép chuyển sang trạng thái `done`. |
| **Search Provider Failure** | Hết quota API hoặc lỗi kết nối tìm kiếm ngoài. | Fallback sang mock search provider định sẵn trong `SearchClient` để đảm bảo pipeline không bị gián đoạn. |
| **Langfuse Host Mismatch** | Khóa API thuộc US Cloud nhưng mặc định gọi EU Cloud gây lỗi `401 Unauthorized`. | Cập nhật `Settings` trong `config.py` hỗ trợ `AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL")`. |
| **Hallucinated Citations** | Model tự bịa ra số trích dẫn không có trong danh sách nguồn. | Critic agent rà soát chéo danh sách nguồn thực tế so với final answer. |

---

## 4. Trace Evidence (Langfuse Cloud)

Hệ thống đã tích hợp và ghi nhận toàn bộ vòng đời tác vụ (Agent hierarchy, Generation inputs/outputs, Tokens & Cost) trên Langfuse:

* **Multi-Agent End-to-End Trace:**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/af5aee861ff6df776d629e0f5f6b80ed](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/af5aee861ff6df776d629e0f5f6b80ed)
* **Single-Agent Baseline Trace:**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/54271913c2f2792f0bdfd2d91389d40d](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/54271913c2f2792f0bdfd2d91389d40d)
* **Benchmark Comparison Trace:**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/07d74a8a72debbd82cef5ab3e8df8110](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/07d74a8a72debbd82cef5ab3e8df8110)

---

## 5. Exit Ticket: Khi nào nên & không nên dùng Multi-Agent?

### 1. Case NÊN dùng Multi-Agent:
* **Tác vụ phức tạp, nhiều bước không thể gộp vào 1 prompt:** Ví dụ như nghiên cứu thị trường, phân tích dữ liệu đa nguồn, kiểm thử mã nguồn độc lập (Researcher tìm tài liệu ➔ Analyst phân tích ➔ Coder viết ➔ Tester kiểm thử).
* **Đòi hỏi trích dẫn và kiểm chứng sự thật (Fact-Checking):** Phân định rạch ròi vai trò tạo nội dung và phản biện độc lập (Critic/Evaluator) giúp giảm thiểu tối đa ảo giác (hallucination).
* **Cần xử lý ngữ cảnh dài có cấu trúc:** Tránh tình trạng "context window pollution" bằng cách để mỗi agent chỉ xử lý đúng phần việc và nén dữ liệu vào Shared State.

### 2. Case KHÔNG NÊN dùng Multi-Agent:
* **Tác vụ đơn giản, phản hồi tức thời (Low-Latency):** Trả lời câu hỏi FAQ, dịch câu ngắn, tóm tắt 1 đoạn văn đơn giản. Sử dụng multi-agent sẽ làm tăng latency không cần thiết (từ 1-2s lên 30-40s) và tốn chi phí API gấp nhiều lần.
* **Quy trình hoàn toàn tuần tự, không có rẽ nhánh logic:** Nếu công việc chỉ là Bước A ➔ Bước B cố định không cần Supervisor đưa ra quyết định động, dùng một chain đơn giản (hoặc pipeline script thông thường) sẽ tiết kiệm và dễ bảo trì hơn.
