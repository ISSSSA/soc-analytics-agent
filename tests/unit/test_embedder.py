from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from pytest_httpx import HTTPXMock

from soc_agent.clustering.embedder import EmbedderClient, EmbedderError

BASE_URL = "http://inference"
EMBED_URL = f"{BASE_URL}/embeddings"


def _mock_response(
    texts_count: int, dim: int = 4, model_id: str = "test-model-v1"
) -> dict[str, Any]:
    return {
        "embeddings": [[float(i + 1) / 10] * dim for i in range(texts_count)],
        "model_id": model_id,
        "pooling": "mean",
        "normalized": True,
        "processing_time_ms": 1.2,
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
    async def test_embed_two_texts(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(2))
        async with EmbedderClient(BASE_URL) as emb:
            vectors = await emb.embed_texts(["a", "b"])
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (2, 4)
        assert vectors.dtype == np.float32

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        async with EmbedderClient(BASE_URL) as emb:
            vectors = await emb.embed_texts([])
        assert vectors.shape == (0, 0)

    @pytest.mark.asyncio
    async def test_payload_matches_params(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL) as emb:
            await emb.embed_texts(["hello"], pooling="cls", normalize=False)

        request = httpx_mock.get_request()
        assert request is not None
        body = request.read()
        import json as _json

        parsed = _json.loads(body)
        assert parsed["texts"] == ["hello"]
        assert parsed["pooling"] == "cls"
        assert parsed["normalize"] is False

    @pytest.mark.asyncio
    async def test_rejects_bad_server_response(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL) as emb:
            with pytest.raises(EmbedderError, match="1 embeddings for 2 texts"):
                await emb.embed_texts(["a", "b"])


class TestCaching:
    @pytest.mark.asyncio
    async def test_second_call_hits_cache(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(2))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            v1 = await emb.embed_texts(["a", "b"])
            assert emb.stats.misses == 2
            assert emb.stats.hits == 0

        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            v2 = await emb.embed_texts(["a", "b"])
            assert emb.stats.hits == 2
            assert emb.stats.misses == 0

        np.testing.assert_array_equal(v1, v2)

    @pytest.mark.asyncio
    async def test_cache_different_pooling_is_different_key(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a"], pooling="mean")

        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a"], pooling="cls")
            assert emb.stats.misses == 1

    @pytest.mark.asyncio
    async def test_cache_different_normalize_is_different_key(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a"], normalize=True)

        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a"], normalize=False)
            assert emb.stats.misses == 1

    @pytest.mark.asyncio
    async def test_mixed_hit_and_miss(
        self, httpx_mock: HTTPXMock, tmp_path: Path
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a"])

        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(2))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            result = await emb.embed_texts(["a", "b", "c"])
            assert emb.stats.hits == 1
            assert emb.stats.misses == 2
            assert result.shape == (3, 4)

    @pytest.mark.asyncio
    async def test_no_cache_mode_always_fetches(
        self, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, cache_dir=None) as emb:
            await emb.embed_texts(["a"])
            await emb.embed_texts(["a"])
            assert emb.stats.hits == 0
            assert emb.stats.misses == 2

    @pytest.mark.asyncio
    async def test_hit_rate(self, httpx_mock: HTTPXMock, tmp_path: Path) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(3))
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a", "b", "c"])
        async with EmbedderClient(BASE_URL, cache_dir=tmp_path) as emb:
            await emb.embed_texts(["a", "b"])
            assert emb.stats.hit_rate == 1.0


class TestBatching:
    @pytest.mark.asyncio
    async def test_splits_large_input(self, httpx_mock: HTTPXMock) -> None:
        for sz in (3, 3, 3, 1):
            httpx_mock.add_response(url=EMBED_URL, json=_mock_response(sz))
        async with EmbedderClient(BASE_URL, batch_size=3) as emb:
            vectors = await emb.embed_texts([f"log {i}" for i in range(10)])
        assert vectors.shape == (10, 4)
        assert len(httpx_mock.get_requests()) == 4


class TestRetries:
    @pytest.mark.asyncio
    async def test_retries_5xx_then_succeeds(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        httpx_mock.add_response(url=EMBED_URL, status_code=503, text="overloaded")
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, max_retries=3) as emb:
            vectors = await emb.embed_texts(["x"])
        assert vectors.shape == (1, 4)
        assert len(httpx_mock.get_requests()) == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_5xx(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        from soc_agent.clustering.embedder import _TransientError

        for _ in range(3):
            httpx_mock.add_response(url=EMBED_URL, status_code=502, text="bad gateway")
        async with EmbedderClient(BASE_URL, max_retries=3) as emb:
            with pytest.raises(_TransientError):
                await emb.embed_texts(["x"])
        assert len(httpx_mock.get_requests()) == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_4xx(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, status_code=401, text="unauthorized")
        async with EmbedderClient(BASE_URL, max_retries=3) as emb:
            with pytest.raises(EmbedderError, match="401"):
                await emb.embed_texts(["x"])
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        httpx_mock.add_exception(httpx.ConnectError("refused"))
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, max_retries=3) as emb:
            vectors = await emb.embed_texts(["x"])
        assert vectors.shape == (1, 4)

    @pytest.mark.asyncio
    async def test_retries_on_read_timeout(
        self, httpx_mock: HTTPXMock, httpx_tenacity_fast: None
    ) -> None:
        httpx_mock.add_exception(httpx.ReadTimeout("slow"))
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, max_retries=3) as emb:
            vectors = await emb.embed_texts(["x"])
        assert vectors.shape == (1, 4)


class TestAuth:
    @pytest.mark.asyncio
    async def test_bearer_header_sent(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, api_key="tok-123") as emb:
            await emb.embed_texts(["x"])
        req = httpx_mock.get_request()
        assert req is not None
        assert req.headers["Authorization"] == "Bearer tok-123"

    @pytest.mark.asyncio
    async def test_no_api_key_no_header(self, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(url=EMBED_URL, json=_mock_response(1))
        async with EmbedderClient(BASE_URL, api_key=None) as emb:
            await emb.embed_texts(["x"])
        req = httpx_mock.get_request()
        assert req is not None
        assert "Authorization" not in req.headers


class TestConstruction:
    def test_rejects_non_positive_batch_size(self) -> None:
        with pytest.raises(ValueError):
            EmbedderClient(BASE_URL, batch_size=0)

    def test_rejects_non_positive_max_retries(self) -> None:
        with pytest.raises(ValueError):
            EmbedderClient(BASE_URL, max_retries=0)

    def test_trailing_slash_stripped(self) -> None:
        emb = EmbedderClient("http://host:8001///")
        assert emb._base_url == "http://host:8001"
