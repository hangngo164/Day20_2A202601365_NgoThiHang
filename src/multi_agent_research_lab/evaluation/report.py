"""Benchmark report rendering — markdown output with comparison analysis."""

import datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a comprehensive markdown report.

    Includes comparison table, analysis, and recommendations.
    """

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"> Generated: {now}",
        "",
        "## Results Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}/10"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    # Add comparative analysis if we have both baseline and multi-agent
    if len(metrics) >= 2:
        baseline = metrics[0]
        multi = metrics[1]

        lines.extend(
            [
                "",
                "## Comparative Analysis",
                "",
            ]
        )

        # Latency comparison
        latency_diff = multi.latency_seconds - baseline.latency_seconds
        latency_ratio = multi.latency_seconds / max(baseline.latency_seconds, 0.01)
        lines.append(
            f"- **Latency**: Multi-agent is {latency_ratio:.1f}x "
            f"{'slower' if latency_diff > 0 else 'faster'} "
            f"({latency_diff:+.2f}s)"
        )

        # Quality comparison
        if baseline.quality_score is not None and multi.quality_score is not None:
            quality_diff = multi.quality_score - baseline.quality_score
            lines.append(
                f"- **Quality**: Multi-agent scored {quality_diff:+.1f} points "
                f"{'higher' if quality_diff > 0 else 'lower'}"
            )

        # Cost comparison
        if baseline.estimated_cost_usd is not None and multi.estimated_cost_usd is not None:
            cost_diff = multi.estimated_cost_usd - baseline.estimated_cost_usd
            lines.append(
                f"- **Cost**: Multi-agent costs ${cost_diff:+.4f} "
                f"{'more' if cost_diff > 0 else 'less'}"
            )

        lines.extend(
            [
                "",
                "## Key Findings",
                "",
                "1. **Multi-agent systems**: Structured decomposition of complex tasks",
                "2. **Trade-off**: Higher quality at the cost of higher latency and API calls",
                "3. **Reliability**: Supervisor routing ensures graceful degradation",
                "",
                "## Failure Modes & Mitigations",
                "",
                "| Failure Mode | Impact | Mitigation |",
                "|---|---|---|",
                "| LLM API timeout | Agent hangs | tenacity retry with exponential backoff |",
                "| Invalid routing decision | Infinite loop | max_iterations guard + fallback |",
                "| Search API failure | No sources | Mock fallback data |",
                "| Hallucinated citations | Inaccurate output | Critic agent fact-checking |",
                "| Token limit exceeded | Truncated output | Chunk research notes |",
            ]
        )

    lines.append("")
    return "\n".join(lines)
