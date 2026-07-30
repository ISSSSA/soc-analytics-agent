from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from soc_agent.schemas import LLMUsage

logger = logging.getLogger(__name__)

litellm.drop_params = True
litellm.set_verbose = False  # type: ignore[attr-defined]

T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMClientError(RuntimeError):
    pass


class LLMPermanentError(LLMClientError):
    pass


_RETRYABLE: tuple[type[BaseException], ...] = (
    litellm.Timeout,  # type: ignore[attr-defined]
    litellm.RateLimitError,  # type: ignore[attr-defined]
    litellm.APIConnectionError,  # type: ignore[attr-defined]
    litellm.InternalServerError,  # type: ignore[attr-defined]
    litellm.ServiceUnavailableError,  # type: ignore[attr-defined]
)


class LLMClient:
    """Async Cloud-LLM client with strict-JSON structured output via LiteLLM."""

    def __init__(
        self,
        model: str = "anthropic/claude-haiku-4-5",
        *,
        max_concurrent: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if max_concurrent <= 0:
            raise ValueError(f"max_concurrent must be > 0, got {max_concurrent}")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        self._extra_headers = extra_headers or {}

        self._total_cost = 0.0
        self._total_tokens_in = 0
        self._total_tokens_out = 0

    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
    ) -> T:
        """Call the LLM and validate the response against *schema*.

        Three-level fallback:
          1. ``json_schema`` (strict) — best for OpenAI / Claude / Gemini
          2. ``json_object`` — for providers that reject strict schemas
          3. plain text + JSON extraction — universal, works with any model
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return await self._call_strict(messages, schema)
        except litellm.BadRequestError as e:  # type: ignore[attr-defined]
            logger.warning(
                "Strict JSON schema rejected (%s); falling back to json_object mode",
                e.__class__.__name__,
            )
        try:
            return await self._call_json_object(messages, user, schema)
        except litellm.BadRequestError as e:  # type: ignore[attr-defined]
            logger.warning(
                "json_object mode rejected (%s); falling back to plain text mode",
                e.__class__.__name__,
            )
        return await self._call_plain_text(messages, user, schema)

    def get_usage_stats(self) -> LLMUsage:
        return LLMUsage(
            model=self.model,
            total_cost_usd=round(self._total_cost, 6),
            total_tokens_in=self._total_tokens_in,
            total_tokens_out=self._total_tokens_out,
        )

    async def _call_strict(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
    ) -> T:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": True,
            },
        }
        content = await self._run_with_retry(messages, response_format)
        try:
            return schema.model_validate_json(content)
        except ValidationError:
            pass
        extracted = _extract_json(content)
        try:
            return schema.model_validate_json(extracted)
        except ValidationError as e:
            raise LLMPermanentError(
                f"LLM returned JSON that doesn't match {schema.__name__}: {e}"
            ) from e

    async def _call_json_object(
        self,
        messages: list[dict[str, str]],
        user: str,
        schema: type[T],
    ) -> T:
        """Fallback path: embed schema in user prompt and validate JSON manually with one retry."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        augmented_user = (
            f"{user}\n\n"
            f"Respond with ONLY a JSON object matching this schema — no prose, "
            f"no markdown fences:\n```json\n{schema_json}\n```"
        )
        messages = [
            messages[0],
            {"role": "user", "content": augmented_user},
        ]
        response_format = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(2):
            content = await self._run_with_retry(messages, response_format)
            for raw in (content, _extract_json(content)):
                try:
                    return schema.model_validate_json(raw)
                except ValidationError as e:
                    last_err = e
            logger.warning(
                "json_object mode validation failed on attempt %d: %s",
                attempt + 1,
                last_err,
            )
        raise LLMPermanentError(
            f"LLM returned JSON that doesn't match {schema.__name__} "
            f"after retry: {last_err}"
        ) from last_err

    async def _call_plain_text(
        self,
        messages: list[dict[str, str]],
        user: str,
        schema: type[T],
    ) -> T:
        """Last-resort fallback: no response_format, extract JSON from free text."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        augmented_user = (
            f"{user}\n\n"
            f"You MUST respond with a JSON object matching this schema.\n"
            f"Do NOT include any text before or after the JSON.\n"
            f"```json\n{schema_json}\n```"
        )
        messages = [
            messages[0],
            {"role": "user", "content": augmented_user},
        ]

        last_err: Exception | None = None
        for attempt in range(2):
            content = await self._run_with_retry(messages, response_format=None)
            extracted = _extract_json(content)
            try:
                return schema.model_validate_json(extracted)
            except ValidationError as e:
                last_err = e
                logger.warning(
                    "Plain text JSON extraction failed on attempt %d: %s",
                    attempt + 1,
                    e,
                )
        raise LLMPermanentError(
            f"LLM plain-text response doesn't contain valid "
            f"{schema.__name__} after retry: {last_err}"
        ) from last_err

    async def _run_with_retry(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "timeout": self._timeout,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers

        async with self._semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=2, min=1, max=30),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
                    response = await litellm.acompletion(**kwargs)
                    self._record_usage(response)
                    content = _extract_content(response)
                    return content
        raise LLMClientError("unreachable")  # pragma: no cover

    def _record_usage(self, response: Any) -> None:
        try:
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._total_tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
                self._total_tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            if cost:
                self._total_cost += float(cost)
        except Exception:
            logger.debug("usage accounting failed", exc_info=True)


def _extract_content(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError) as e:
        raise LLMClientError(f"unexpected LLM response shape: {response!r}") from e


def _extract_json(text: str) -> str:
    """Pull the first JSON object from free-form LLM text.

    Tries fenced code block first, then outermost ``{...}``.
    For thinking-mode models (e.g. Qwen3) the code block or raw text
    may contain reasoning before the actual JSON — always fall through
    to brace extraction as a second pass.
    """
    candidates: list[str] = []
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text)
    for candidate in candidates:
        m2 = _JSON_BRACE_RE.search(candidate)
        if m2:
            return m2.group(0)
    return text.strip()


def build_openrouter_headers(
    *,
    app_name: str | None = None,
    site_url: str | None = None,
) -> dict[str, str]:
    """Build extra_headers dict for OpenRouter identification."""
    headers: dict[str, str] = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers
