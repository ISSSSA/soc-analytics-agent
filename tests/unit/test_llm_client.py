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
    _extract_json,
    build_openrouter_headers,
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


class TestPlainTextFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_plain_text_on_double_bad_request(self) -> None:
        valid = json.dumps({"answer": "plain-ok", "confidence": 0.8})
        bad_req = litellm.BadRequestError(
            message="not supported", model="test", llm_provider="test",
        )
        mock_acomp = AsyncMock(
            side_effect=[bad_req, bad_req, _fake_response(valid)],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert result.answer == "plain-ok"
        assert mock_acomp.call_count == 3
        third_kwargs = mock_acomp.call_args_list[2].kwargs
        assert "response_format" not in third_kwargs

    @pytest.mark.asyncio
    async def test_plain_text_extracts_json_from_markdown_fence(self) -> None:
        wrapped = '```json\n{"answer": "fenced", "confidence": 0.5}\n```'
        bad_req = litellm.BadRequestError(
            message="no", model="test", llm_provider="test",
        )
        mock_acomp = AsyncMock(
            side_effect=[bad_req, bad_req, _fake_response(wrapped)],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert result.answer == "fenced"

    @pytest.mark.asyncio
    async def test_plain_text_extracts_json_from_prose(self) -> None:
        prose = 'Here is the answer:\n{"answer": "embedded", "confidence": 0.3}\nHope this helps!'
        bad_req = litellm.BadRequestError(
            message="no", model="test", llm_provider="test",
        )
        mock_acomp = AsyncMock(
            side_effect=[bad_req, bad_req, _fake_response(prose)],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            result = await client.generate_structured("sys", "user", SampleSchema)

        assert result.answer == "embedded"

    @pytest.mark.asyncio
    async def test_plain_text_permanent_after_retries(self) -> None:
        bad_req = litellm.BadRequestError(
            message="no", model="test", llm_provider="test",
        )
        garbage = "I cannot produce JSON."
        mock_acomp = AsyncMock(
            side_effect=[bad_req, bad_req, _fake_response(garbage), _fake_response(garbage)],
        )
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            with pytest.raises(LLMPermanentError, match="plain-text"):
                await client.generate_structured("sys", "user", SampleSchema)


class TestExtractJson:
    def test_fenced_code_block(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == '{"a": 1}'

    def test_fenced_without_lang(self) -> None:
        text = '```\n{"a": 1}\n```'
        assert _extract_json(text) == '{"a": 1}'

    def test_bare_json(self) -> None:
        text = '{"a": 1}'
        assert _extract_json(text) == '{"a": 1}'

    def test_json_with_surrounding_prose(self) -> None:
        text = 'Here:\n{"a": 1}\nDone.'
        assert _extract_json(text) == '{"a": 1}'

    def test_no_json_returns_stripped(self) -> None:
        text = "  no json here  "
        assert _extract_json(text) == "no json here"


class TestBuildOpenRouterHeaders:
    def test_both_set(self) -> None:
        h = build_openrouter_headers(app_name="soc-agent", site_url="https://example.com")
        assert h == {"HTTP-Referer": "https://example.com", "X-Title": "soc-agent"}

    def test_only_app_name(self) -> None:
        h = build_openrouter_headers(app_name="soc-agent")
        assert h == {"X-Title": "soc-agent"}

    def test_none_returns_empty(self) -> None:
        assert build_openrouter_headers() == {}


class TestExtraHeaders:
    @pytest.mark.asyncio
    async def test_headers_passed_to_litellm(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        mock_acomp = AsyncMock(return_value=_fake_response(valid))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient(extra_headers={"X-Title": "test-app"})
            await client.generate_structured("s", "u", SampleSchema)

        call_kwargs = mock_acomp.call_args.kwargs
        assert call_kwargs["extra_headers"] == {"X-Title": "test-app"}

    @pytest.mark.asyncio
    async def test_no_headers_when_empty(self) -> None:
        valid = json.dumps({"answer": "x", "confidence": 0.5})
        mock_acomp = AsyncMock(return_value=_fake_response(valid))
        with (
            patch("litellm.acompletion", mock_acomp),
            patch("litellm.completion_cost", return_value=0.0),
        ):
            client = LLMClient()
            await client.generate_structured("s", "u", SampleSchema)

        call_kwargs = mock_acomp.call_args.kwargs
        assert "extra_headers" not in call_kwargs


class TestConstruction:
    def test_rejects_non_positive_concurrency(self) -> None:
        with pytest.raises(ValueError):
            LLMClient(max_concurrent=0)

    def test_rejects_zero_retries(self) -> None:
        with pytest.raises(ValueError):
            LLMClient(max_retries=0)
