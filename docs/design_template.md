# Design Document: Multi-Agent Research System

## 1. Task Description

Hệ thống cần xử lý **research queries phức tạp** bằng cách phân chia công việc cho nhiều agent chuyên biệt:
- Tìm kiếm và thu thập thông tin từ web
- Phân tích và đánh giá chất lượng thông tin
- Tổng hợp thành bài viết có trích dẫn
- Kiểm tra sự chính xác (fact-check)

## 2. Why Multi-Agent?

Single-agent gặp hạn chế khi:
- **Context window overload**: Một agent phải giữ tất cả context (search, analyze, write) → giảm chất lượng
- **No separation of concerns**: Khó debug khi không biết agent fail ở bước nào
- **No iterative refinement**: Single-agent viết 1 lần, không có review/feedback loop
- **Limited specialization**: Mỗi task cần system prompt và temperature khác nhau

Multi-agent giải quyết bằng cách:
- Mỗi agent có **role rõ ràng** với prompt tối ưu
- **Shared state** cho phép truyền kết quả giữa agents
- **Supervisor** điều phối và enforce guardrails
- **Critic** cung cấp quality assurance

## 3. Agent Roles

| Agent | Input | Output | Model Config |
|---|---|---|---|
| Supervisor | Full state | Route decision | gpt-4o-mini, temp=0.0 |
| Researcher | Query | sources, research_notes | gpt-4o-mini, temp=0.2 |
| Analyst | research_notes | analysis_notes | gpt-4o-mini, temp=0.1 |
| Writer | research_notes + analysis_notes | final_answer | gpt-4o-mini, temp=0.4 |
| Critic | final_answer + sources | quality review | gpt-4o-mini, temp=0.1 |

## 4. Shared State Fields

| Field | Type | Lý do cần |
|---|---|---|
| `request` | ResearchQuery | Giữ query gốc và config |
| `iteration` | int | Track loop count cho max_iterations |
| `route_history` | list[str] | Debug routing decisions |
| `sources` | list[SourceDocument] | Researcher tạo, Writer/Critic dùng |
| `research_notes` | str | Researcher output → Analyst input |
| `analysis_notes` | str | Analyst output → Writer input |
| `final_answer` | str | Writer output → Critic review |
| `agent_results` | list[AgentResult] | Audit trail mỗi agent |
| `trace` | list[dict] | Observability events |
| `errors` | list[str] | Error tracking |

## 5. Graph Flow

```text
START → Supervisor → [researcher] → Supervisor → [analyst] → Supervisor → [writer] → Supervisor → [critic] → END
```

- Supervisor dùng LLM để quyết định route, với heuristic fallback
- Max 6 iterations, timeout 60s
- Conditional edges từ supervisor → worker nodes
- Critic chạy 1 lần sau khi writer hoàn thành

## 6. Benchmark Plan

| Query | Metric | Expected |
|---|---|---|
| "Research GraphRAG" | Latency | Multi 3-5x slower |
| "Research GraphRAG" | Quality (0-10) | Multi 1-3 points higher |
| "Research GraphRAG" | Cost | Multi 3-5x more |
| "Compare single vs multi" | Citation coverage | Multi >60% |
| "Summarize guardrails" | Failure rate | Both <10% |
