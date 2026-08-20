"""Analyst agent — turns research notes into structured insights."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are an expert research analyst. Given research notes, produce a structured analysis.

Your analysis must include:
1. **Key Claims**: List the 3-5 most important claims or findings
2. **Viewpoint Comparison**: Compare different perspectives or approaches mentioned
3. **Evidence Strength**: Rate the strength of evidence (strong/moderate/weak) for each claim
4. **Gaps & Weaknesses**: Identify what's missing or poorly supported
5. **Synthesis**: A brief paragraph connecting the key themes

Be critical and objective. Flag any unsupported claims or potential biases."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Extracts key claims, compares viewpoints, and flags weak evidence.
        """

        logger.info("Analyst starting")

        if not state.research_notes:
            state.analysis_notes = "No research notes available for analysis."
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=state.analysis_notes)
            )
            return state

        user_prompt = (
            f"Original query: {state.request.query}\n"
            f"Target audience: {state.request.audience}\n\n"
            f"Research notes to analyze:\n{state.research_notes}\n\n"
            f"Number of sources referenced: {len(state.sources)}\n\n"
            f"Please provide a structured analysis."
        )

        try:
            response = self._llm.complete(ANALYST_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content
            logger.info("Analysis notes generated | length=%d", len(response.content))
        except Exception as e:
            error_msg = f"Analyst LLM failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.analysis_notes = (
                f"Analysis unavailable (LLM error). "
                f"Research notes summary: {state.research_notes[:200]}..."
            )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=state.analysis_notes or "",
            )
        )
        state.add_trace_event("analyst", {"analysis_length": len(state.analysis_notes or "")})
        return state
