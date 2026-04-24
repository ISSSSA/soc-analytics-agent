from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from inference_service import metrics
from inference_service.auth import BearerAuth
from inference_service.config import InferenceSettings, get_inference_settings
from inference_service.model_loader import (
    ClassificationUnsupportedError,
    ModelProvider,
    ModelUnavailableError,
    OOMError,
    SecBERTProvider,
)
from inference_service.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassPrediction,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


def _build_limiter(cfg: InferenceSettings) -> Limiter:
    return Limiter(
        key_func=get_remote_address,
        default_limits=[cfg.rate_limit],
        headers_enabled=True,
    )


def _configure_logging(cfg: InferenceSettings) -> None:
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg: InferenceSettings = app.state.config
    _configure_logging(cfg)
    provider: ModelProvider | None = getattr(app.state, "provider", None)
    if provider is None:
        logger.info(
            "Loading model %s (device=%s fp16=%s)",
            cfg.secbert_model_path,
            cfg.device,
            cfg.use_fp16,
        )
        provider = SecBERTProvider(
            cfg.secbert_model_path,
            device=cfg.device,
            use_fp16=cfg.use_fp16,
            max_batch_size=cfg.max_batch_size,
            hf_token=(cfg.hf_token.get_secret_value() if cfg.hf_token else None),
        )
        app.state.provider = provider
        if cfg.warmup_on_startup:
            try:
                provider.warmup()
            except Exception:
                logger.exception("Warmup failed — continuing, first request will be slow")
    app.state.gpu_lock = asyncio.Lock()
    yield


def create_app(
    *,
    cfg: InferenceSettings | None = None,
    provider: ModelProvider | None = None,
) -> FastAPI:
    """Build the FastAPI app; tests pass a pre-built `provider` to skip model load."""
    cfg = cfg or get_inference_settings()
    limiter = _build_limiter(cfg)
    bearer = BearerAuth(cfg.inference_api_key.get_secret_value() if cfg.inference_api_key else None)

    app = FastAPI(
        title="SOC Agent Inference Service",
        description="SecBERT embeddings & classification over HTTP.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.limiter = limiter
    app.state.bearer = bearer
    if provider is not None:
        app.state.provider = provider

    _register_routes(app)
    _register_error_handlers(app)
    return app


async def _auth_dep(request: Request) -> None:
    bearer: BearerAuth = request.app.state.bearer
    await bearer(request)


def _get_provider(request: Request) -> ModelProvider:
    provider: ModelProvider | None = getattr(request.app.state, "provider", None)
    if provider is None or not provider.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return provider


async def _run_on_gpu(
    request: Request,
    fn: Callable[[], Any],
) -> Any:
    lock: asyncio.Lock = request.app.state.gpu_lock
    async with lock:
        return await asyncio.to_thread(fn)


def _register_routes(app: FastAPI) -> None:  # noqa: PLR0915
    cfg: InferenceSettings = app.state.config

    def _check_request_limits(texts: list[str], endpoint: str) -> None:
        if len(texts) > cfg.max_texts_per_request:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Too many texts: {len(texts)} > {cfg.max_texts_per_request}",
            )
        for i, t in enumerate(texts):
            if len(t) > cfg.max_text_length:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"texts[{i}] too long: {len(t)} > {cfg.max_text_length}",
                )
        metrics.BATCH_SIZE.labels(endpoint=endpoint).observe(len(texts))

    @app.post("/embeddings", response_model=EmbeddingsResponse)
    async def embeddings(
        request: Request,
        body: EmbeddingsRequest,
        _: None = Depends(_auth_dep),
        provider: ModelProvider = Depends(_get_provider),
    ) -> EmbeddingsResponse:
        _check_request_limits(body.texts, "embeddings")
        started = time.perf_counter()
        try:
            vectors = await _run_on_gpu(
                request, lambda: provider.embed(body.texts, body.pooling, body.normalize)
            )
        except OOMError as e:
            metrics.REQUEST_COUNT.labels(endpoint="embeddings", status="503").inc()
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ModelUnavailableError as e:
            metrics.REQUEST_COUNT.labels(endpoint="embeddings", status="503").inc()
            raise HTTPException(status_code=503, detail=str(e)) from e
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.LATENCY.labels(endpoint="embeddings").observe(elapsed_ms / 1000)
        metrics.REQUEST_COUNT.labels(endpoint="embeddings", status="200").inc()
        return EmbeddingsResponse(
            embeddings=vectors,
            model_id=provider.model_id,
            pooling=body.pooling,
            normalized=body.normalize,
            processing_time_ms=elapsed_ms,
        )

    @app.post("/classify", response_model=ClassifyResponse)
    async def classify(
        request: Request,
        body: ClassifyRequest,
        _: None = Depends(_auth_dep),
        provider: ModelProvider = Depends(_get_provider),
    ) -> ClassifyResponse:
        _check_request_limits(body.texts, "classify")
        if not provider.supports_classification:
            metrics.REQUEST_COUNT.labels(endpoint="classify", status="501").inc()
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="This model has no classifier head",
            )
        started = time.perf_counter()
        try:
            preds = await _run_on_gpu(
                request, lambda: provider.classify(body.texts, body.top_k)
            )
        except ClassificationUnsupportedError as e:
            metrics.REQUEST_COUNT.labels(endpoint="classify", status="501").inc()
            raise HTTPException(status_code=501, detail=str(e)) from e
        except OOMError as e:
            metrics.REQUEST_COUNT.labels(endpoint="classify", status="503").inc()
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ModelUnavailableError as e:
            metrics.REQUEST_COUNT.labels(endpoint="classify", status="503").inc()
            raise HTTPException(status_code=503, detail=str(e)) from e
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.LATENCY.labels(endpoint="classify").observe(elapsed_ms / 1000)
        metrics.REQUEST_COUNT.labels(endpoint="classify", status="200").inc()
        return ClassifyResponse(
            predictions=[
                ClassPrediction(label=p.label, score=p.score, top_k=p.top_k) for p in preds
            ],
            model_id=provider.model_id,
            processing_time_ms=elapsed_ms,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        provider: ModelProvider | None = getattr(request.app.state, "provider", None)
        if provider is None or not provider.is_ready:
            return HealthResponse(
                status="error",
                model_loaded=False,
                gpu_available=False,
                supports_classification=False,
            )
        info = provider.device_info()
        used_mb = info.get("gpu_memory_used_mb")
        if isinstance(used_mb, int):
            metrics.GPU_MEMORY_USED_MB.set(used_mb)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            gpu_available=bool(info.get("gpu_available")),
            gpu_name=info.get("gpu_name"),
            gpu_memory_total_mb=info.get("gpu_memory_total_mb"),
            gpu_memory_used_mb=info.get("gpu_memory_used_mb"),
            supports_classification=provider.supports_classification,
            num_classes=provider.num_classes,
            class_labels=provider.class_labels,
        )

    @app.get("/metrics")
    async def get_metrics() -> Response:
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(_request: Request, exc: RateLimitExceeded) -> Response:
        return Response(
            content=f"Rate limit exceeded: {exc.detail}",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="text/plain",
        )


app = create_app()
