"""Unit tests for implemented agents.

Replaces the skeleton guard test. These tests verify that agents produce
expected outputs when given proper state, using mocked LLM/search clients.
"""

from unittest.mock import patch

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMResponse


def _make_state(**kwargs) -> ResearchState:
    """Helper to create a ResearchState with defaults."""
    defaults = {"request": ResearchQuery(query="Explain multi-agent systems")}
    defaults.update(kwargs)
    return ResearchState(**defaults)


def _mock_llm_response(content: str = "Mock LLM response") -> LLMResponse:
    return LLMResponse(content=content, input_tokens=100, output_tokens=50, cost_usd=0.001)


# --- Supervisor Tests ---


class TestSupervisorAgent:
    @patch("multi_agent_research_lab.agents.supervisor.LLMClient")
    def test_routes_to_researcher_when_no_notes(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response("researcher")
        agent = SupervisorAgent()
        state = _make_state()
        result = agent.run(state)
        assert result.route_history[-1] == "researcher"

    @patch("multi_agent_research_lab.agents.supervisor.LLMClient")
    def test_routes_to_analyst_when_has_research(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response("analyst")
        agent = SupervisorAgent()
        state = _make_state(research_notes="Some research")
        result = agent.run(state)
        assert result.route_history[-1] == "analyst"

    @patch("multi_agent_research_lab.agents.supervisor.LLMClient")
    def test_forces_done_at_max_iterations(self, mock_llm_cls):
        agent = SupervisorAgent()
        state = _make_state(iteration=10)  # > default max_iterations=6
        result = agent.run(state)
        assert result.route_history[-1] == "done"

    @patch("multi_agent_research_lab.agents.supervisor.LLMClient")
    def test_heuristic_fallback_on_llm_error(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.side_effect = Exception("API Error")
        agent = SupervisorAgent()
        state = _make_state()
        result = agent.run(state)
        assert result.route_history[-1] == "researcher"  # heuristic: no notes -> researcher


# --- Researcher Tests ---


class TestResearcherAgent:
    @patch("multi_agent_research_lab.agents.researcher.LLMClient")
    @patch("multi_agent_research_lab.agents.researcher.SearchClient")
    def test_populates_sources_and_notes(self, mock_search_cls, mock_llm_cls):
        mock_search_cls.return_value.search.return_value = [
            SourceDocument(title="Test Source", snippet="Test content", url="https://test.com"),
        ]
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response("Research findings...")
        agent = ResearcherAgent()
        state = _make_state()
        result = agent.run(state)
        assert len(result.sources) == 1
        assert result.research_notes == "Research findings..."

    @patch("multi_agent_research_lab.agents.researcher.LLMClient")
    @patch("multi_agent_research_lab.agents.researcher.SearchClient")
    def test_handles_search_failure(self, mock_search_cls, mock_llm_cls):
        mock_search_cls.return_value.search.side_effect = Exception("Search down")
        agent = ResearcherAgent()
        state = _make_state()
        result = agent.run(state)
        assert len(result.errors) > 0


# --- Analyst Tests ---


class TestAnalystAgent:
    @patch("multi_agent_research_lab.agents.analyst.LLMClient")
    def test_populates_analysis_notes(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response("Key claims: ...")
        agent = AnalystAgent()
        state = _make_state(research_notes="Some research notes")
        result = agent.run(state)
        assert result.analysis_notes == "Key claims: ..."

    @patch("multi_agent_research_lab.agents.analyst.LLMClient")
    def test_handles_no_research_notes(self, mock_llm_cls):
        agent = AnalystAgent()
        state = _make_state()
        result = agent.run(state)
        assert "No research notes" in (result.analysis_notes or "")


# --- Writer Tests ---


class TestWriterAgent:
    @patch("multi_agent_research_lab.agents.writer.LLMClient")
    def test_populates_final_answer(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response("Final answer here")
        agent = WriterAgent()
        state = _make_state(
            research_notes="Research notes",
            analysis_notes="Analysis notes",
        )
        result = agent.run(state)
        assert result.final_answer == "Final answer here"


# --- Critic Tests ---


class TestCriticAgent:
    @patch("multi_agent_research_lab.agents.critic.LLMClient")
    def test_reviews_final_answer(self, mock_llm_cls):
        mock_llm_cls.return_value.complete.return_value = _mock_llm_response(
            "Review: QUALITY_SCORE: 8/10 CITATION_COVERAGE: 80%"
        )
        agent = CriticAgent()
        state = _make_state(final_answer="Some final answer")
        result = agent.run(state)
        assert any(r.agent.value == "critic" for r in result.agent_results)

    @patch("multi_agent_research_lab.agents.critic.LLMClient")
    def test_skips_when_no_answer(self, mock_llm_cls):
        agent = CriticAgent()
        state = _make_state()
        result = agent.run(state)
        assert any("No final answer" in r.content for r in result.agent_results)
