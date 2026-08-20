"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import logging
from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APITimeoutError, RateLimitError

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import trace_generation

logger = logging.getLogger(__name__)

# Approximate pricing per 1K tokens for gpt-4o-mini (as of mid-2025)
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenAI implementation."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    @retry(
        retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Uses OpenAI ChatCompletion with retry, timeout, token logging, and Langfuse tracing.
        """

        logger.info("LLM request | model=%s | system_len=%d | user_len=%d",
                     self._model, len(system_prompt), len(user_prompt))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        with trace_generation(
            name=f"openai_completion ({self._model})",
            model=self._model,
            input_messages=messages,
        ) as gen_data:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.3,
                timeout=45,
            )

            choice = response.choices[0]
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None

            # Estimate cost
            cost_usd: float | None = None
            if input_tokens is not None and output_tokens is not None:
                pricing = _PRICING.get(self._model, _PRICING["gpt-4o-mini"])
                cost_usd = (
                    input_tokens / 1000 * pricing["input"]
                    + output_tokens / 1000 * pricing["output"]
                )

            logger.info(
                "LLM response | tokens_in=%s | tokens_out=%s | cost=$%s",
                input_tokens, output_tokens,
                f"{cost_usd:.6f}" if cost_usd else "N/A",
            )

            gen_data["output"] = choice.message.content or ""
            if input_tokens is not None and output_tokens is not None:
                gen_data["usage_details"] = {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                }
            if cost_usd is not None:
                gen_data["cost_details"] = {"total": cost_usd}

            return LLMResponse(
                content=choice.message.content or "",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
