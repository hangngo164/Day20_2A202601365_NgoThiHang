"""LangGraph workflow — orchestrates the multi-agent research pipeline."""

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()

    # ---- Node functions (LangGraph nodes operate on dicts) ---- #

    def _supervisor_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Supervisor decides the next route."""
        state = ResearchState(**state_dict)
        with trace_span(
            name="supervisor",
            attributes={"iteration": state.iteration},
            as_type="agent",
            input={"route_history": state.route_history, "iteration": state.iteration},
        ):
            result = self._supervisor.run(state)
        return result.model_dump()

    def _researcher_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Researcher searches and creates notes."""
        state = ResearchState(**state_dict)
        with trace_span(
            name="researcher",
            attributes={"query": state.request.query[:50]},
            as_type="agent",
            input={"query": state.request.query},
        ):
            result = self._researcher.run(state)
        return result.model_dump()

    def _analyst_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Analyst extracts insights from research notes."""
        state = ResearchState(**state_dict)
        with trace_span(
            name="analyst",
            as_type="agent",
            input={"research_notes_length": len(state.research_notes) if state.research_notes else 0},
        ):
            result = self._analyst.run(state)
        return result.model_dump()

    def _writer_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Writer synthesizes the final answer."""
        state = ResearchState(**state_dict)
        with trace_span(
            name="writer",
            as_type="agent",
            input={"analysis_notes_length": len(state.analysis_notes) if state.analysis_notes else 0},
        ):
            result = self._writer.run(state)
        return result.model_dump()

    def _critic_node(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Critic reviews the final answer."""
        state = ResearchState(**state_dict)
        with trace_span(
            name="critic",
            as_type="agent",
            input={"final_answer_length": len(state.final_answer) if state.final_answer else 0},
        ):
            result = self._critic.run(state)
        return result.model_dump()

    @staticmethod
    def _route_after_supervisor(state_dict: dict[str, Any]) -> str:
        """Conditional edge: pick the next node based on the last route."""
        route_history = state_dict.get("route_history", [])
        if not route_history:
            return "researcher"

        last_route = route_history[-1]
        if last_route == "done":
            return "critic"  # Run critic before finishing
        return last_route  # researcher / analyst / writer

    def build(self) -> StateGraph:
        """Create a LangGraph StateGraph with conditional routing.

        Graph flow:
            supervisor -> (researcher | analyst | writer | critic) -> supervisor (loop)
            supervisor -> done -> critic -> END
        """

        graph = StateGraph(dict)

        # Add nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("researcher", self._researcher_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("critic", self._critic_node)

        # Set entry point
        graph.set_entry_point("supervisor")

        # Conditional routing from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
            },
        )

        # After each worker, go back to supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        # After critic, end
        graph.add_edge("critic", END)

        logger.info("Multi-agent graph built successfully")
        return graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state.

        Compiles the graph, invokes it, and converts the result back to ResearchState.
        """

        logger.info("Starting multi-agent workflow | query=%s", state.request.query[:80])

        with trace_span(
            name="multi_agent_workflow",
            attributes={"query": state.request.query},
            as_type="span",
            input={"query": state.request.query},
        ) as root_span:
            graph = self.build()
            compiled = graph.compile()

            # Convert state to dict for LangGraph
            initial_state = state.model_dump()

            # Run the graph
            final_state_dict = compiled.invoke(initial_state)

        # Convert back to ResearchState
        result = ResearchState(**final_state_dict)
        logger.info(
            "Workflow complete | iterations=%d | routes=%s",
            result.iteration,
            result.route_history,
        )
        return result
