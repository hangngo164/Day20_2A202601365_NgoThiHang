"""Benchmark for single-agent vs multi-agent comparison."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, quality, cost, citation coverage, and failure rate.

    Enhanced benchmark with comprehensive metrics collection.
    """

    started = perf_counter()

    try:
        state = runner(query)
        latency = perf_counter() - started
    except Exception as e:
        latency = perf_counter() - started
        logger.error("Benchmark run '%s' failed: %s", run_name, e)
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Run failed: {e}",
        )
        state = ResearchState(
            request={"query": query},  # type: ignore[arg-type]
            errors=[str(e)],
        )
        return state, metrics

    # --- Compute quality metrics ---

    # Estimated token cost from agent results
    estimated_cost = _estimate_cost(state)

    # Quality score heuristic (0-10)
    quality_score = _compute_quality_score(state)

    # Citation coverage
    citation_coverage = _compute_citation_coverage(state)

    # Failure rate
    failure_rate = len(state.errors) / max(state.iteration, 1) if state.errors else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=min(failure_rate, 1.0),
        notes=f"iterations={state.iteration}, routes={state.route_history}",
    )

    logger.info(
        "Benchmark '%s' complete | latency=%.2fs | quality=%.1f | cost=$%s",
        run_name,
        latency,
        quality_score or 0,
        f"{estimated_cost:.6f}" if estimated_cost else "N/A",
    )

    return state, metrics


def _estimate_cost(state: ResearchState) -> float | None:
    """Estimate total cost from agent results metadata."""
    # Count LLM calls made (approximate from agent_results)
    num_llm_calls = len(state.agent_results)
    if num_llm_calls == 0:
        return None

    # Rough estimate: gpt-4o-mini ~$0.00015/1K input, $0.0006/1K output
    # Average ~500 input tokens, ~300 output tokens per call
    estimated_per_call = (500 / 1000 * 0.00015) + (300 / 1000 * 0.0006)
    return estimated_per_call * num_llm_calls


def _compute_quality_score(state: ResearchState) -> float | None:
    """Compute a heuristic quality score (0-10)."""
    score = 0.0

    # Has final answer? (+3)
    if state.final_answer:
        score += 3.0
        # Answer length quality (500+ words is ideal)
        word_count = len(state.final_answer.split())
        if word_count >= 400:
            score += 2.0
        elif word_count >= 200:
            score += 1.0

    # Has research notes? (+1.5)
    if state.research_notes:
        score += 1.5

    # Has analysis? (+1.5)
    if state.analysis_notes:
        score += 1.5

    # Has sources? (+1)
    if state.sources:
        score += min(len(state.sources) / 3, 1.0)

    # Penalty for errors (-0.5 each, max -2)
    score -= min(len(state.errors) * 0.5, 2.0)

    # Has citations in answer? (+1)
    if state.final_answer and "[" in state.final_answer:
        score += 1.0

    return min(max(score, 0.0), 10.0)


def _compute_citation_coverage(state: ResearchState) -> float | None:
    """Compute what fraction of sources are referenced in the final answer."""
    if not state.sources or not state.final_answer:
        return None

    referenced = 0
    for i in range(1, len(state.sources) + 1):
        if f"[{i}]" in state.final_answer:
            referenced += 1

    return referenced / len(state.sources)
