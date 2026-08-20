"""Researcher agent — collects sources and creates research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are a research assistant. Given search results about a topic, create concise, well-organized research notes.

Your notes should:
1. Summarize key findings from each source
2. Include source references [1], [2], etc.
3. Highlight important facts, numbers, and claims
4. Note any conflicting information between sources
5. Be comprehensive but concise (aim for 300-500 words)

Format your response as structured research notes with clear headings."""


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Searches for relevant documents, then uses LLM to synthesize research notes.
        """

        logger.info("Researcher starting | query=%s", state.request.query[:80])

        # Step 1: Search for sources
        try:
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources.extend(sources)
            logger.info("Researcher found %d sources", len(sources))
        except Exception as e:
            error_msg = f"Search failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Continue with any existing sources

        # Step 2: Build context from sources
        if not state.sources:
            state.research_notes = f"No sources found for query: {state.request.query}"
            state.agent_results.append(
                AgentResult(agent=AgentName.RESEARCHER, content=state.research_notes)
            )
            return state

        source_text = ""
        for i, src in enumerate(state.sources, 1):
            source_text += f"\n[{i}] {src.title}\n    URL: {src.url or 'N/A'}\n    {src.snippet}\n"

        # Step 3: Synthesize research notes via LLM
        user_prompt = (
            f"Research query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Search results:\n{source_text}\n\n"
            f"Please create comprehensive research notes from these sources."
        )

        try:
            response = self._llm.complete(RESEARCH_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content
            logger.info("Research notes generated | length=%d", len(response.content))
        except Exception as e:
            error_msg = f"LLM synthesis failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Fallback: use raw source snippets
            state.research_notes = f"Raw sources (LLM unavailable):\n{source_text}"

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes or "",
                metadata={"source_count": len(state.sources)},
            )
        )
        state.add_trace_event("researcher", {"source_count": len(state.sources)})
        return state
