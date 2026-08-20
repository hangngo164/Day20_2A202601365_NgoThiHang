# Lab 20: Multi-Agent Research System

> **Học viên:** Ngô Thị Hằng (MSSV: 2A202601365)  
> **Repository:** [hangngo164/Day20_2A202601365_NgoThiHang](https://github.com/hangngo164/Day20_2A202601365_NgoThiHang)  
> **Trạng thái:** Hoàn thành 100% (Pass 14/14 tests, 0 lint error, 0 type error)

Hệ thống nghiên cứu đa tác tử (**Multi-Agent Systems**) xây dựng trên nền tảng **LangGraph** gồm 5 tác tử chuyên biệt: **Supervisor + Researcher + Analyst + Writer + Critic**, tích hợp quan sát toàn diện (**Observability**) với **Langfuse Cloud**.

---

## 📌 Deliverables & Trace Evidence (Bằng chứng nộp bài)

### 1. Báo cáo đánh giá (Reports)
* 📊 **Benchmark Report (So sánh Single vs Multi-Agent):** [`reports/benchmark_report.md`](reports/benchmark_report.md)
* 🛡️ **Failure Modes & Mitigations (Phân tích lỗi & Cách phòng vệ):** [`reports/failure_modes.md`](reports/failure_modes.md)

### 2. Đường dẫn trực tiếp đến Trace trên Langfuse Cloud
* 🔗 **Multi-Agent Workflow Trace (End-to-End):**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/af5aee861ff6df776d629e0f5f6b80ed](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/af5aee861ff6df776d629e0f5f6b80ed)
* 🔗 **Single-Agent Baseline Trace:**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/54271913c2f2792f0bdfd2d91389d40d](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/54271913c2f2792f0bdfd2d91389d40d)
* 🔗 **Benchmark Comparison Trace:**  
  [https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/07d74a8a72debbd82cef5ab3e8df8110](https://us.cloud.langfuse.com/project/cmt13ogoa063pad0dcafai14e/traces/07d74a8a72debbd82cef5ab3e8df8110)

---

## 🏛️ Kiến trúc hệ thống

```text
User Query
   │
   ▼
Supervisor / Router (LangGraph StateGraph)
   ├──► Researcher Agent   ──► [Tool: Tavily Search] ──► sources & research_notes
   ├──► Analyst Agent      ──► Structured Insights   ──► analysis_notes
   ├──► Writer Agent       ──► Synthesis ~500 words  ──► final_answer with citations [1]..[5]
   └──► Critic Agent       ──► Fact-checking review  ──► Safety & Quality evaluation
   │
   ▼
Langfuse Tracing (Agent/Tool/Generation Spans) + Benchmark Report
```

---

## 🚀 Hướng dẫn chạy nhanh (Quickstart)

### 1. Cài đặt môi trường
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Chạy kiểm thử & Code Quality
```bash
pytest                  # Chạy 14 bài unit tests
ruff check src tests    # Kiểm tra linting
mypy src                # Kiểm tra kiểu dữ liệu (Type checking)
```

### 3. Chạy các lệnh CLI
* **Single-Agent Baseline:**
  ```bash
  python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art"
  ```
* **Multi-Agent Workflow:**
  ```bash
  python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art"
  ```
* **Chạy Benchmark & Xuất Báo cáo:**
  ```bash
  python -m multi_agent_research_lab.cli benchmark
  ```

---

## 📈 Kết quả Benchmark tóm tắt

| Chỉ số | Single-Agent Baseline | Multi-Agent Workflow | Nhận xét Trade-off |
| :--- | :---: | :---: | :--- |
| **Thời gian (Latency)** | **10.80s** | 38.94s | Single-Agent nhanh hơn ~3.6x. |
| **Chi phí (Cost)** | ~$0.0004 | ~$0.0020 | Multi-Agent dùng nhiều token hơn do phân rã tác vụ. |
| **Chất lượng (Quality)** | 5.0 / 10 | **10.0 / 10** | Multi-Agent vượt trội về độ sâu, cấu trúc và tính khách quan. |
| **Độ phủ trích dẫn** | 0% | **100% (5/5)** | Multi-Agent trích dẫn đầy đủ nguồn tài liệu thực tế. |
| **Tỷ lệ lỗi (Failure Rate)** | 0% | 0% | Hệ thống ổn định nhờ cơ chế Guardrails (Retry, Fallback, Max Iterations). |
