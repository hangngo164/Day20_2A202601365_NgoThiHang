"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily implementation and mock fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily API when key is available, otherwise returns mock results.
        """
        with trace_span(
            name="search_documents",
            as_type="tool",
            input={"query": query, "max_results": max_results},
        ) as span:
            if self._api_key and self._api_key.strip():
                docs = self._tavily_search(query, max_results)
            else:
                logger.warning("No TAVILY_API_KEY found, using mock search results")
                docs = self._mock_search(query, max_results)

            span["output"] = {
                "num_results": len(docs),
                "titles": [d.title for d in docs],
            }
            return docs

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Search using Tavily API."""
        from tavily import TavilyClient

        logger.info("Tavily search | query=%s | max_results=%d", query[:80], max_results)
        client = TavilyClient(api_key=self._api_key)
        response = client.search(query=query, max_results=max_results)

        docs: list[SourceDocument] = []
        for result in response.get("results", []):
            docs.append(
                SourceDocument(
                    title=result.get("title", "Untitled"),
                    url=result.get("url"),
                    snippet=result.get("content", "")[:500],
                    metadata={"score": result.get("score", 0.0)},
                )
            )

        logger.info("Tavily returned %d results", len(docs))
        return docs

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Return mock search results for testing without API key."""
        mock_data = [
            SourceDocument(
                title="Multi-Agent Systems: A Modern Approach",
                url="https://example.com/multi-agent-systems",
                snippet="Multi-agent systems (MAS) involve multiple autonomous agents interacting "
                        "in a shared environment. Key challenges include coordination, "
                        "communication, and conflict resolution between agents.",
            ),
            SourceDocument(
                title="LangGraph: Building Stateful Multi-Agent Applications",
                url="https://example.com/langgraph-guide",
                snippet="LangGraph enables building stateful, multi-actor applications with LLMs. "
                        "It extends LangChain with cyclic graphs, persistence, and human-in-the-loop.",
            ),
            SourceDocument(
                title="GraphRAG: Retrieval-Augmented Generation with Knowledge Graphs",
                url="https://example.com/graphrag",
                snippet="GraphRAG combines knowledge graphs with RAG pipelines to improve factual "
                        "accuracy. Microsoft's implementation uses community summaries for "
                        "global queries and entity-based retrieval for local queries.",
            ),
            SourceDocument(
                title="Production Guardrails for LLM Agents",
                url="https://example.com/guardrails",
                snippet="Production LLM agents require guardrails including max iteration limits, "
                        "timeout controls, input/output validation, retry with backoff, "
                        "and comprehensive logging and tracing.",
            ),
            SourceDocument(
                title="Benchmarking Agent Systems: Metrics and Methods",
                url="https://example.com/benchmarking",
                snippet="Effective agent benchmarking measures latency, cost, quality, "
                        "citation coverage, and failure rate. Comparing single-agent vs "
                        "multi-agent architectures reveals tradeoffs.",
            ),
        ]
        return mock_data[:max_results]
