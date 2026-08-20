"""Critic agent — optional fact-checking and quality review."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a fact-checking critic. Review the final answer against the source material.

Evaluate:
1. **Citation Coverage**: Are all major claims supported by cited sources? (percentage estimate)
2. **Hallucination Check**: Are there any claims not supported by the provided sources?
3. **Completeness**: Does the answer address the original query fully?
4. **Accuracy**: Are the source references correct and not misrepresented?
5. **Quality Score**: Rate overall quality from 1-10

Provide a brief structured review with specific examples of issues found.
End with: QUALITY_SCORE: X/10 and CITATION_COVERAGE: X%"""


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Checks fact accuracy, citation coverage, and potential hallucinations.
        """

        logger.info("Critic starting review")

        if not state.final_answer:
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.CRITIC,
                    content="No final answer to review.",
                )
            )
            return state

        # Build source context for verification
        source_text = ""
        for i, src in enumerate(state.sources, 1):
            source_text += f"[{i}] {src.title}: {src.snippet}\n"

        user_prompt = (
            f"Original query: {state.request.query}\n\n"
            f"--- Sources ---\n{source_text or 'No sources provided'}\n\n"
            f"--- Research Notes ---\n{state.research_notes or 'None'}\n\n"
            f"--- Final Answer to Review ---\n{state.final_answer}\n\n"
            f"Please provide your critical review."
        )

        try:
            response = self._llm.complete(CRITIC_SYSTEM_PROMPT, user_prompt)
            review = response.content
            logger.info("Critic review complete | length=%d", len(review))
        except Exception as e:
            error_msg = f"Critic LLM failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            review = "Critic review unavailable due to LLM error."

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=review,
                metadata={"review_type": "fact_check"},
            )
        )
        state.add_trace_event("critic", {"review_length": len(review)})
        return state
