"""Tracing hooks with Langfuse integration following best practices.

Provides structured tracing with observation types (agent, tool, generation),
token & cost tracking, metadata enrichment, and graceful local logging fallback.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# Global Langfuse client (lazy-initialized)
_langfuse_client = None
_langfuse_initialized = False
_last_trace_url: str | None = None


def _get_langfuse():
    """Lazily initialize and return the Langfuse client."""
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True
    try:
        settings = get_settings()

        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            if _langfuse_client.auth_check():
                logger.info("Langfuse tracing initialized and authenticated successfully")
            else:
                logger.warning("Langfuse authentication check failed. Check your API keys.")
        else:
            logger.info("Langfuse keys not configured, using local tracing only")
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s. Using local tracing.", e)

    return _langfuse_client


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    as_type: str = "span",
    input: Any = None,
    output: Any = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for tracing a span or agent/tool of work.

    Sends observations to Langfuse with appropriate type (agent, tool, retriever, span),
    maintaining OpenTelemetry context hierarchy, and logs locally.
    """
    global _last_trace_url
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "as_type": as_type,
        "attributes": attributes or {},
        "duration_seconds": None,
    }

    langfuse = _get_langfuse()
    obs_cm = None
    active_obs = None

    try:
        if langfuse:
            try:
                obs_cm = langfuse.start_as_current_observation(
                    name=name,
                    as_type=as_type,  # type: ignore[arg-type]
                    input=input,
                    metadata=attributes or {},
                )
                active_obs = obs_cm.__enter__()
                url = langfuse.get_trace_url()
                if url:
                    _last_trace_url = url
            except Exception as e:
                logger.debug("Langfuse observation creation failed: %s", e)
                obs_cm = None

        logger.debug("SPAN START | %s (%s) | attrs=%s", name, as_type, attributes)
        yield span

    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration

        logger.info(
            "SPAN END | %s | type=%s | duration=%.3fs | attrs=%s",
            name,
            as_type,
            duration,
            attributes,
        )

        if obs_cm and active_obs:
            try:
                if output is not None and hasattr(active_obs, "update"):
                    active_obs.update(output=output)
                obs_cm.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Langfuse observation exit failed: %s", e)


@contextmanager
def trace_generation(
    name: str,
    model: str,
    input_messages: Any,
    metadata: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for tracing LLM generations with token counts and costs."""
    global _last_trace_url
    started = perf_counter()
    gen_data: dict[str, Any] = {
        "name": name,
        "model": model,
        "duration_seconds": None,
        "output": None,
        "usage_details": None,
        "cost_details": None,
    }

    langfuse = _get_langfuse()
    obs_cm = None
    active_obs = None

    try:
        if langfuse:
            try:
                obs_cm = langfuse.start_as_current_observation(
                    name=name,
                    as_type="generation",
                    model=model,
                    input=input_messages,
                    metadata=metadata or {},
                )
                active_obs = obs_cm.__enter__()
                url = langfuse.get_trace_url()
                if url:
                    _last_trace_url = url
            except Exception as e:
                logger.debug("Langfuse generation creation failed: %s", e)
                obs_cm = None

        yield gen_data

    finally:
        duration = perf_counter() - started
        gen_data["duration_seconds"] = duration

        if obs_cm and active_obs:
            try:
                update_kwargs: dict[str, Any] = {}
                if gen_data.get("output") is not None:
                    update_kwargs["output"] = gen_data["output"]
                if gen_data.get("usage_details") is not None:
                    update_kwargs["usage_details"] = gen_data["usage_details"]
                if gen_data.get("cost_details") is not None:
                    update_kwargs["cost_details"] = gen_data["cost_details"]

                if update_kwargs and hasattr(active_obs, "update"):
                    active_obs.update(**update_kwargs)
                obs_cm.__exit__(None, None, None)
            except Exception as e:
                logger.debug("Langfuse generation exit failed: %s", e)


def get_current_trace_url() -> str | None:
    """Return the Langfuse trace URL for the active/latest trace."""
    global _last_trace_url
    langfuse = _get_langfuse()
    if langfuse:
        try:
            url = langfuse.get_trace_url()
            if url:
                _last_trace_url = url
        except Exception:
            pass
    return _last_trace_url


def flush_traces() -> str | None:
    """Flush any pending traces to Langfuse and return the trace URL."""
    global _last_trace_url
    langfuse = _get_langfuse()
    trace_url = _last_trace_url
    if langfuse:
        try:
            url = langfuse.get_trace_url()
            if url:
                trace_url = url
                _last_trace_url = url
            langfuse.flush()
            logger.info("Langfuse traces flushed")
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
    return trace_url
