from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import httpx
import numpy as np
from diskcache import Cache
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from soc_agent.schemas import EmbeddingsRequest, EmbeddingsResponse

logger = logging.getLogger(__name__)

Pooling = Literal["mean", "cls"]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class EmbedderError(RuntimeError):
    pass


class _TransientError(Exception):
    pass


_RETRYABLE: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.RemoteProtocolError,
    _TransientError,
)


class EmbedderClient:
    """Async client to the inference service /embeddings endpoint with on-disk caching."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        cache_dir: Path | str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._batch_size = batch_size
        self._timeout = timeout
        self._max_retries = max_retries
        self._cache: Cache | None = Cache(str(cache_dir)) if cache_dir is not None else None
        self._client: httpx.AsyncClient | None = None
        self.stats = CacheStats()

    async def __aenter__(self) -> Self:
        await self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers=headers,
            )
        return self._client

    @staticmethod
    def _cache_key(text: str, pooling: Pooling, normalize: bool) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{pooling}:{int(normalize)}:{h}"

    async def embed_texts(
        self,
        texts: list[str],
        *,
        pooling: Pooling = "mean",
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute (N, D) float32 embeddings for `texts`, using on-disk cache when available."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        client = await self._ensure_client()
        n = len(texts)
        vectors: list[list[float] | None] = [None] * n
        uncached_idx: list[int] = []
        uncached_texts: list[str] = []

        if self._cache is not None:
            for i, t in enumerate(texts):
                key = self._cache_key(t, pooling, normalize)
                hit = self._cache.get(key)
                if hit is not None:
                    vectors[i] = hit
                    self.stats.hits += 1
                else:
                    uncached_idx.append(i)
                    uncached_texts.append(t)
                    self.stats.misses += 1
        else:
            uncached_idx = list(range(n))
            uncached_texts = list(texts)
            self.stats.misses += n

        if uncached_texts:
            fresh = await self._embed_uncached(client, uncached_texts, pooling, normalize)
            for idx, vec in zip(uncached_idx, fresh, strict=True):
                vectors[idx] = vec
                if self._cache is not None:
                    self._cache.set(
                        self._cache_key(texts[idx], pooling, normalize), vec
                    )

        return np.asarray(vectors, dtype=np.float32)

    async def _embed_uncached(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
        pooling: Pooling,
        normalize: bool,
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = await self._post_embeddings(client, batch, pooling, normalize)
            if len(resp.embeddings) != len(batch):
                raise EmbedderError(
                    f"Server returned {len(resp.embeddings)} embeddings for {len(batch)} texts"
                )
            out.extend(resp.embeddings)
        return out

    async def _post_embeddings(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
        pooling: Pooling,
        normalize: bool,
    ) -> EmbeddingsResponse:
        req = EmbeddingsRequest(texts=texts, pooling=pooling, normalize=normalize)
        payload = req.model_dump()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                r = await client.post("/embeddings", json=payload)
                if r.status_code >= 500:
                    raise _TransientError(
                        f"{r.status_code} from /embeddings: {r.text[:200]}"
                    )
                if r.status_code >= 400:
                    raise EmbedderError(
                        f"HTTP {r.status_code} from /embeddings: {r.text[:500]}"
                    )
                return EmbeddingsResponse.model_validate(r.json())
        raise EmbedderError("unreachable")  # pragma: no cover
