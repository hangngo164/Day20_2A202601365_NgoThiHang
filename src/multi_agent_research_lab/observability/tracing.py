"""Tracing hooks.

Provides both a minimal local tracer and Langfuse integration.
Students can also plug in LangSmith or OpenTelemetry.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

# Global Langfuse client (lazy-initialized)
_langfuse_client = None
_langfuse_initialized = False


def _get_langfuse():
    """Lazily initialize and return the Langfuse client."""
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    _langfuse_initialized = True
    try:
        from multi_agent_research_lab.core.config import get_settings
        settings = get_settings()

        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse
            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracing initialized successfully")
        else:
            logger.info("Langfuse keys not configured, using local tracing only")
    except Exception as e:
        logger.warning("Failed to initialize Langfuse: %s. Using local tracing.", e)

    return _langfuse_client


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Context manager for tracing a span of work.

    Sends spans to Langfuse when configured, always logs locally.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    langfuse = _get_langfuse()
    langfuse_span = None

    try:
        # Start Langfuse span if available
        if langfuse:
            try:
                langfuse_span = langfuse.trace(
                    name=name,
                    metadata=attributes or {},
                )
            except Exception as e:
                logger.debug("Langfuse trace creation failed: %s", e)

        logger.debug("SPAN START | %s | attrs=%s", name, attributes)
        yield span

    finally:
        duration = perf_counter() - started
        span["duration_seconds"] = duration

        logger.info(
            "SPAN END | %s | duration=%.3fs | attrs=%s",
            name, duration, attributes,
        )

        # End Langfuse span
        if langfuse_span:
            try:
                langfuse_span.update(
                    metadata={
                        **(attributes or {}),
                        "duration_seconds": round(duration, 4),
                    }
                )
            except Exception as e:
                logger.debug("Langfuse span update failed: %s", e)


def flush_traces() -> None:
    """Flush any pending traces to the backend."""
    langfuse = _get_langfuse()
    if langfuse:
        try:
            langfuse.flush()
            logger.info("Langfuse traces flushed")
        except Exception as e:
            logger.warning("Langfuse flush failed: %s", e)
