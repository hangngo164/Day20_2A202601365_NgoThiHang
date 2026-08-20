"""Writer agent — produces final answer from research and analysis notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

WRITER_SYSTEM_PROMPT = """You are a professional research writer.
Synthesize research notes and analysis into a clear, well-structured final response.

Your response must:
1. Be well-organized with clear sections and headings
2. Include citations referencing the source numbers [1], [2], etc.
3. Present findings in a logical flow
4. Be appropriate for the target audience
5. Include a brief conclusion with key takeaways
6. Be approximately 500 words unless otherwise specified

Write in a professional yet accessible tone."""


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Synthesizes a clear response with citations or source references.
        """

        logger.info("Writer starting")

        # Build source reference list
        source_refs = ""
        for i, src in enumerate(state.sources, 1):
            source_refs += f"[{i}] {src.title} — {src.url or 'N/A'}\n"

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"--- Research Notes ---\n{state.research_notes or 'None'}\n\n"
            f"--- Analysis Notes ---\n{state.analysis_notes or 'None'}\n\n"
            f"--- Sources ---\n{source_refs or 'No sources'}\n\n"
            f"Write a comprehensive, well-cited final response."
        )

        try:
            response = self._llm.complete(WRITER_SYSTEM_PROMPT, user_prompt)
            state.final_answer = response.content
            logger.info("Final answer generated | length=%d", len(response.content))
        except Exception as e:
            error_msg = f"Writer LLM failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            # Fallback: combine existing notes
            state.final_answer = (
                f"# {state.request.query}\n\n"
                f"## Research Notes\n{state.research_notes or 'N/A'}\n\n"
                f"## Analysis\n{state.analysis_notes or 'N/A'}\n\n"
                f"*Note: Auto-generated fallback due to writer error.*"
            )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer or "",
            )
        )
        state.add_trace_event("writer", {"answer_length": len(state.final_answer or "")})
        return state
