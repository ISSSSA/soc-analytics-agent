from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from soc_agent.classification.classifier import (
    ClassificationUnsupportedError,
    ClassifierClient,
    ClassifierError,
)

BASE_URL = "http://inference"
CLASSIFY_URL = f"{BASE_URL}/classify"


def _mock_response(
    n: int,
    *,
    label: str = "credential_stuffing",
    score: float = 0.9,
    model_id: str = "test-classifier-v1",
) -> dict[str, Any]:
    return {
        "predictions": [
            {
                "label": label,
                "score": score,
                "top_k": [[label, score], ["brute_force", 1 - score]],
            }
            for _ in range(n)
        ],
        "model_id": model_id,
        "processing_time_ms": 2.0,
    }


@pytest.fixture
def httpx_tenacity_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    from tenacity import nap

    monkeypatch.setattr(nap, "sleep", lambda *_a, **_kw: None)

    async def _nosleep(*_a: Any, **_kw: Any) -> None:
        return None

    for attr in ("async_sleep", "asyncio_sleep"):
        if hasattr(nap, attr):
            monkeypatch.setattr(nap, attr, _nosleep)


class TestBasic:
    @pytest.mark.asyncio
    async def test_classify_two_texts(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(2))
        async with ClassifierClient(BASE_URL) as clf:
            preds = await clf.classify_texts(["log A", "log B"])
        assert len(preds) == 2
        assert preds[0].label == "credential_stuffing"
        assert preds[0].score == 0.9
        assert len(preds[0].top_k) == 2

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        async with ClassifierClient(BASE_URL) as clf:
            preds = await clf.classify_texts([])
        assert preds == []

    @pytest.mark.asyncio
    async def test_payload_contains_top_k(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL) as clf:
            await clf.classify_texts(["log"], top_k=7)
        req = httpx_mock.get_request()
        assert req is not None
        body = json.loads(req.read())
        assert body["texts"] == ["log"]
        assert body["top_k"] == 7
        assert body["return_all_scores"] is False

    @pytest.mark.asyncio
    async def test_order_preserved(self, httpx_mock: HTTPXMock) -> None:
        distinct = {
            "predictions": [
                {"label": f"L{i}", "score": 0.5 + i * 0.01, "top_k": []}
                for i in range(5)
            ],
            "model_id": "x",
            "processing_time_ms": 1.0,
        }
        httpx_mock.add_response(url=CLASSIFY_URL, json=distinct)
        async with ClassifierClient(BASE_URL) as clf:
            preds = await clf.classify_texts([f"t{i}" for i in range(5)])
        assert [p.label for p in preds] == ["L0", "L1", "L2", "L3", "L4"]

    @pytest.mark.asyncio
    async def test_rejects_bad_response_length(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL) as clf:
            with pytest.raises(ClassifierError, match="1 predictions for 2 texts"):
                await clf.classify_texts(["a", "b"])


class TestBatching:
    @pytest.mark.asyncio
    async def test_splits_into_multiple_requests(self, httpx_mock: HTTPXMock) -> None:
        for sz in (3, 3, 1):
            httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(sz))
        async with ClassifierClient(BASE_URL, batch_size=3) as clf:
            preds = await clf.classify_texts([f"log {i}" for i in range(7)])
        assert len(preds) == 7
        assert len(httpx_mock.get_requests()) == 3


class TestRetries:
    @pytest.mark.asyncio
    async def test_retries_5xx_then_succeeds(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, status_code=503, text="overloaded")
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL, max_retries=3) as clf:
            preds = await clf.classify_texts(["x"])
        assert len(preds) == 1
        assert len(httpx_mock.get_requests()) == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_4xx(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, status_code=401, text="unauth")
        async with ClassifierClient(BASE_URL, max_retries=3) as clf:
            with pytest.raises(ClassifierError, match="401"):
                await clf.classify_texts(["x"])
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_501_raises_unsupported_and_does_not_retry(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url=CLASSIFY_URL, status_code=501, text="no classifier head"
        )
        async with ClassifierClient(BASE_URL, max_retries=3) as clf:
            with pytest.raises(ClassificationUnsupportedError):
                await clf.classify_texts(["x"])
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL, max_retries=3) as clf:
            preds = await clf.classify_texts(["x"])
        assert len(preds) == 1


class TestAuth:
    @pytest.mark.asyncio
    async def test_bearer_sent(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL, api_key="tok-xyz") as clf:
            await clf.classify_texts(["x"])
        req = httpx_mock.get_request()
        assert req is not None
        assert req.headers["Authorization"] == "Bearer tok-xyz"

    @pytest.mark.asyncio
    async def test_no_auth_no_header(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=CLASSIFY_URL, json=_mock_response(1))
        async with ClassifierClient(BASE_URL, api_key=None) as clf:
            await clf.classify_texts(["x"])
        req = httpx_mock.get_request()
        assert req is not None
        assert "Authorization" not in req.headers


class TestConstruction:
    def test_rejects_zero_batch_size(self) -> None:
        with pytest.raises(ValueError):
            ClassifierClient(BASE_URL, batch_size=0)

    def test_rejects_zero_max_retries(self) -> None:
        with pytest.raises(ValueError):
            ClassifierClient(BASE_URL, max_retries=0)

    def test_trailing_slash_stripped(self) -> None:
        clf = ClassifierClient("http://h:8001///")
        assert clf._base_url == "http://h:8001"
