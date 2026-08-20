"""Command-line entrypoint for the lab."""

import logging
import sys
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import trace_span, flush_traces
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

# Fix Windows cp1252 encoding issues with Rich Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

logger = logging.getLogger(__name__)

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console(force_terminal=True)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline(query: str) -> ResearchState:
    """Run a real single-agent baseline with LLM."""
    request = _parse_query(query)
    state = ResearchState(request=request)

    llm = LLMClient()

    system_prompt = (
        "You are a research assistant. Given a research query, provide a comprehensive, "
        "well-structured response. Include key findings, analysis, and conclusions. "
        "Aim for approximately 500 words."
    )

    try:
        with trace_span("baseline_single_agent", {"query": query}):
            response = llm.complete(system_prompt, query)
            state.final_answer = response.content
            state.iteration = 1
            state.record_route("single_agent")
    except Exception as e:
        state.errors.append(str(e))
        state.final_answer = f"Baseline failed: {e}"

    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Run the multi-agent workflow."""
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline with real LLM."""

    _init()
    console.print("\n[bold cyan]>> Running Single-Agent Baseline...[/bold cyan]\n")

    state = _run_baseline(query)

    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="yellow"))

    console.print(Panel.fit(
        Markdown(state.final_answer or "No answer generated"),
        title="Single-Agent Baseline Result",
        border_style="green",
    ))
    flush_traces()


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    console.print("\n[bold cyan]>> Running Multi-Agent Workflow...[/bold cyan]\n")

    try:
        state = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    # Show route history
    console.print(Panel.fit(
        " -> ".join(state.route_history) if state.route_history else "No routes",
        title="Route History",
        border_style="blue",
    ))

    # Show errors if any
    if state.errors:
        console.print(Panel.fit("\n".join(state.errors), title="Errors", style="yellow"))

    # Show final answer
    console.print(Panel.fit(
        Markdown(state.final_answer or "No answer generated"),
        title="Multi-Agent Result",
        border_style="green",
    ))

    # Show agent results summary
    for result in state.agent_results:
        console.print(f"  [dim]{result.agent}[/dim]: {result.content[:80]}...")

    flush_traces()


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query"),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
) -> None:
    """Run benchmark comparing single-agent vs multi-agent."""

    _init()
    console.print("\n[bold cyan]>> Running Benchmark...[/bold cyan]\n")

    all_metrics = []

    # Run baseline
    console.print("[dim]  Running single-agent baseline...[/dim]")
    baseline_state, baseline_metrics = run_benchmark(
        "single-agent-baseline", query, _run_baseline
    )
    all_metrics.append(baseline_metrics)
    console.print(f"  Done in {baseline_metrics.latency_seconds:.2f}s\n")

    # Run multi-agent
    console.print("[dim]  Running multi-agent workflow...[/dim]")
    multi_state, multi_metrics = run_benchmark(
        "multi-agent-workflow", query, _run_multi_agent
    )
    all_metrics.append(multi_metrics)
    console.print(f"  Done in {multi_metrics.latency_seconds:.2f}s\n")

    # Render report
    report_md = render_markdown_report(all_metrics)

    # Save to file
    store = LocalArtifactStore()
    path = store.write_text("benchmark_report.md", report_md)
    console.print(f"\n[green]Report saved to: {path}[/green]\n")

    # Display report
    console.print(Markdown(report_md))

    flush_traces()


if __name__ == "__main__":
    app()
