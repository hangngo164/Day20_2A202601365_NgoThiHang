"""Supervisor / router agent."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

ROUTING_SYSTEM_PROMPT = """You are a research supervisor. Your job is to decide which agent should run next.

Available agents:
- researcher: Searches the web and collects research notes. Use when research_notes is empty.
- analyst: Analyzes research notes and extracts key insights. Use when research_notes exist but analysis_notes is empty.
- writer: Writes the final answer from research and analysis. Use when both research_notes and analysis_notes exist but final_answer is empty.
- done: The task is complete. Use when final_answer exists.

Rules:
1. Follow the order: researcher -> analyst -> writer -> done
2. If notes are already filled, skip to the next step.
3. Never pick the same agent twice in a row unless justified.

Respond with ONLY one word: researcher, analyst, writer, or done."""


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Implements a routing policy with LLM-based decisions, max iteration
        enforcement, and failure fallback.
        """

        # Guard: max iterations
        if state.iteration >= self._settings.max_iterations:
            logger.warning("Max iterations (%d) reached, forcing done", self._settings.max_iterations)
            state.record_route("done")
            if not state.final_answer:
                state.final_answer = (
                    "Research was terminated due to max iteration limit. "
                    "Partial results may be available in research_notes and analysis_notes."
                )
            state.add_trace_event("supervisor", {"action": "force_done_max_iter"})
            return state

        # Build context for LLM routing decision
        context = (
            f"Query: {state.request.query}\n"
            f"Iteration: {state.iteration}\n"
            f"Route history: {state.route_history}\n"
            f"Has research_notes: {state.research_notes is not None}\n"
            f"Has analysis_notes: {state.analysis_notes is not None}\n"
            f"Has final_answer: {state.final_answer is not None}\n"
            f"Sources count: {len(state.sources)}\n"
            f"Errors: {state.errors}"
        )

        try:
            response = self._llm.complete(ROUTING_SYSTEM_PROMPT, context)
            route = response.content.strip().lower()

            # Validate route
            valid_routes = {"researcher", "analyst", "writer", "done"}
            if route not in valid_routes:
                logger.warning("LLM returned invalid route '%s', using heuristic fallback", route)
                route = self._heuristic_route(state)

        except Exception as e:
            logger.error("Supervisor LLM call failed: %s, using heuristic", e)
            state.errors.append(f"Supervisor LLM error: {e}")
            route = self._heuristic_route(state)

        logger.info("Supervisor routed to: %s (iteration %d)", route, state.iteration)
        state.record_route(route)
        state.agent_results.append(
            AgentResult(agent=AgentName.SUPERVISOR, content=f"Routed to: {route}")
        )
        state.add_trace_event("supervisor", {"route": route, "iteration": state.iteration})
        return state

    @staticmethod
    def _heuristic_route(state: ResearchState) -> str:
        """Deterministic fallback routing when LLM fails."""
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        return "done"
