from __future__ import annotations

import logging
from types import TracebackType
from typing import Self

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from soc_agent.schemas import ClassifyRequest, ClassifyResponse, ClassPrediction

logger = logging.getLogger(__name__)


class ClassifierError(RuntimeError):
    pass


class ClassificationUnsupportedError(ClassifierError):
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


class ClassifierClient:
    """Async client to the inference service /classify endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
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
        self._client: httpx.AsyncClient | None = None

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

    async def classify_texts(
        self,
        texts: list[str],
        *,
        top_k: int = 5,
        return_all_scores: bool = False,
    ) -> list[ClassPrediction]:
        """Classify `texts`; raises ClassificationUnsupportedError on HTTP 501."""
        if not texts:
            return []
        client = await self._ensure_client()

        out: list[ClassPrediction] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = await self._post_classify(client, batch, top_k, return_all_scores)
            if len(resp.predictions) != len(batch):
                raise ClassifierError(
                    f"Server returned {len(resp.predictions)} predictions "
                    f"for {len(batch)} texts"
                )
            out.extend(resp.predictions)
        return out

    async def _post_classify(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
        top_k: int,
        return_all_scores: bool,
    ) -> ClassifyResponse:
        req = ClassifyRequest(texts=texts, top_k=top_k, return_all_scores=return_all_scores)
        payload = req.model_dump()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        ):
            with attempt:
                r = await client.post("/classify", json=payload)
                if r.status_code == 501:
                    raise ClassificationUnsupportedError(
                        f"/classify: {r.text[:200]}"
                    )
                if r.status_code >= 500:
                    raise _TransientError(
                        f"{r.status_code} from /classify: {r.text[:200]}"
                    )
                if r.status_code >= 400:
                    raise ClassifierError(
                        f"HTTP {r.status_code} from /classify: {r.text[:500]}"
                    )
                return ClassifyResponse.model_validate(r.json())
        raise ClassifierError("unreachable")  # pragma: no cover
