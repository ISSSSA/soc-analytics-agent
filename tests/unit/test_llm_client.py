from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel, ConfigDict, Field

from soc_agent.recommendations.llm_client import (
    LLMClient,
    LLMPermanentError,
)


class SampleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


def _fake_response(content: str, pt: int = 120, ct: int = 40) -> MagicMock:
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = pt
    usage.completion_tokens = ct
    resp.usage = usage
    return resp


@pytest.fixture
def tenacity_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from tenacity import nap

    monkeypatch.setattr(nap, "sleep", lambda *_a, **_kw: None)

    async def _nosleep(*_a: Any, **_kw: Any) -> None:
        return None

    for attr in ("async_sleep", "asyncio_sleep"):
        if hasattr(nap, attr):
            monkeypatch.setattr(nap, attr, _nosleep)


class TestGenerateStructured:
    @pytest.mark.asyncio
    async def test_success_strict(self) -> None:
        valid = json.dumps({"answer": "hi", "confidence": 0.9})
        mock_acomp = AsyncMock(return_value=_fake_response(valid))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0023),
        ):
            client = LLMClient(model="openai/gpt-4o-mini")
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert isinstance(result, SampleSchema)
        assert result.answer == "hi"
        assert result.confidence == 0.9
        mock_acomp.assert_called_once()
        call_kwargs = mock_acomp.call_args.kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True
        assert call_kwargs["model"] == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_invalid_json_shape_raises_permanent(self) -> None:
        bad = json.dumps({"answer": "ok"})
        mock_acomp = AsyncMock(return_value=_fake_response(bad))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            with pytest.raises(LLMPermanentError, match="SampleSchema"):
                await client.generate_structured("sys", "user", SampleSchema)

    @pytest.mark.asyncio
    async def test_system_and_user_in_messages(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        mock_acomp = AsyncMock(return_value=_fake_response(valid))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            await client.generate_structured("SYSPROMPT", "USERPROMPT", SampleSchema)

        messages = mock_acomp.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "SYSPROMPT"}
        assert messages[1] == {"role": "user", "content": "USERPROMPT"}


class TestFallback:
    @pytest.mark.asyncio
    async def test_falls_back_on_bad_request_to_json_object(self) -> None:
        valid = json.dumps({"answer": "fallback-ok", "confidence": 0.6})
        bad_req = litellm.BadRequestError(
            message="response_format not supported",
            model="test",
            llm_provider="test",
        )
        mock_acomp = AsyncMock(
            side_effect=[bad_req, _fake_response(valid)],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert result.answer == "fallback-ok"
        assert mock_acomp.call_count == 2
        second_kwargs = mock_acomp.call_args_list[1].kwargs
        assert second_kwargs["response_format"] == {"type": "json_object"}
        second_user = second_kwargs["messages"][1]["content"]
        assert "confidence" in second_user
        assert "json" in second_user.lower()

    @pytest.mark.asyncio
    async def test_json_object_validation_retry(self) -> None:
        bad_req = litellm.BadRequestError(
            message="no", model="test", llm_provider="test"
        )
        invalid1 = json.dumps({"answer": "x"})
        valid2 = json.dumps({"answer": "y", "confidence": 0.7})
        mock_acomp = AsyncMock(
            side_effect=[
                bad_req,
                _fake_response(invalid1),
                _fake_response(valid2),
            ],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert result.answer == "y"
        assert mock_acomp.call_count == 3

    @pytest.mark.asyncio
    async def test_json_object_permanent_after_retries(self) -> None:
        bad_req = litellm.BadRequestError(
            message="no", model="test", llm_provider="test"
        )
        invalid = json.dumps({"answer": "x"})
        mock_acomp = AsyncMock(
            side_effect=[
                bad_req,
                _fake_response(invalid),
                _fake_response(invalid),
            ],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            with pytest.raises(LLMPermanentError, match="after retry"):
                await client.generate_structured("sys", "user", SampleSchema)


class TestRetries:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self, tenacity_fast: None) -> None:
        valid = json.dumps({"answer": "ok", "confidence": 0.5})
        rl = litellm.RateLimitError(
            message="slow down", model="test", llm_provider="test"
        )
        mock_acomp = AsyncMock(side_effect=[rl, _fake_response(valid)])
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(max_retries=3)
            result = await client.generate_structured("s", "u", SampleSchema)
        assert result.answer == "ok"
        assert mock_acomp.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self, tenacity_fast: None) -> None:
        valid = json.dumps({"answer": "ok", "confidence": 0.5})
        to = litellm.Timeout(message="slow", model="test", llm_provider="test")
        mock_acomp = AsyncMock(side_effect=[to, _fake_response(valid)])
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(max_retries=3)
            await client.generate_structured("s", "u", SampleSchema)
        assert mock_acomp.call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_auth_error(self) -> None:
        err = litellm.AuthenticationError(
            message="bad key", model="test", llm_provider="test"
        )
        mock_acomp = AsyncMock(side_effect=err)
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(max_retries=3)
            with pytest.raises(litellm.AuthenticationError):
                await client.generate_structured("s", "u", SampleSchema)
        assert mock_acomp.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_after_retries_exhausted(self, tenacity_fast: None) -> None:
        rl = litellm.RateLimitError(
            message="slow", model="test", llm_provider="test"
        )
        mock_acomp = AsyncMock(side_effect=[rl, rl, rl])
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(max_retries=3)
            with pytest.raises(litellm.RateLimitError):
                await client.generate_structured("s", "u", SampleSchema)
        assert mock_acomp.call_count == 3


class TestUsageStats:
    @pytest.mark.asyncio
    async def test_accumulates_across_calls(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        resp = _fake_response(valid, pt=100, ct=50)
        mock_acomp = AsyncMock(return_value=resp)
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.001),
        ):
            client = LLMClient()
            await client.generate_structured("s", "u", SampleSchema)
            await client.generate_structured("s", "u", SampleSchema)

        stats = client.get_usage_stats()
        assert stats.total_tokens_in == 200
        assert stats.total_tokens_out == 100
        assert stats.total_cost_usd == pytest.approx(0.002)
        assert stats.model == "anthropic/claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_cost_failure_is_non_fatal(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        mock_acomp = AsyncMock(return_value=_fake_response(valid))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch(
                "litellm.completion_cost",
                side_effect=Exception("not in cost table"),
            ),
        ):
            client = LLMClient()
            await client.generate_structured("s", "u", SampleSchema)
        stats = client.get_usage_stats()
        assert stats.total_cost_usd == 0.0
        assert stats.total_tokens_in == 120


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_semaphore_limits_in_flight(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        in_flight = 0
        max_seen = 0
        event = asyncio.Event()

        async def _slow(*_a: Any, **_kw: Any) -> MagicMock:
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.02)
            event.set()
            in_flight -= 1
            return _fake_response(valid)

        with (
            patch("litellm.acompletion", side_effect=_slow),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(max_concurrent=2)
            await asyncio.gather(
                *(client.generate_structured("s", "u", SampleSchema) for _ in range(6)),
            )

        assert max_seen <= 2
        assert max_seen > 0


class TestConstruction:
    def test_rejects_non_positive_concurrency(self) -> None:
        with pytest.raises(ValueError):
            LLMClient(max_concurrent=0)

    def test_rejects_zero_retries(self) -> None:
        with pytest.raises(ValueError):
            LLMClient(max_retries=0)
