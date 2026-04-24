from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient

from inference_service.config import InferenceSettings
from inference_service.model_loader import (
    ClassificationUnsupportedError,
    ClassifyPred,
    ModelUnavailableError,
    OOMError,
)
from inference_service.server import create_app


@dataclass
class FakeProvider:
    """Deterministic provider for wiring tests with hash-derived 384-dim embeddings."""

    model_id_: str = "fake/mini-v1"
    ready: bool = True
    classifier: bool = True
    labels: list[str] = field(
        default_factory=lambda: ["credential_stuffing", "brute_force", "reconnaissance"]
    )
    dim: int = 384

    embed_calls: int = 0
    classify_calls: int = 0
    last_embed_batch: list[str] = field(default_factory=list)
    last_classify_batch: list[str] = field(default_factory=list)

    oom_on_embed: bool = False
    oom_on_classify: bool = False
    fail_model_unavailable: bool = False

    @property
    def model_id(self) -> str:
        return self.model_id_

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def supports_classification(self) -> bool:
        return self.classifier

    @property
    def num_classes(self) -> int | None:
        return len(self.labels) if self.classifier else None

    @property
    def class_labels(self) -> list[str] | None:
        return list(self.labels) if self.classifier else None

    def device_info(self) -> dict[str, Any]:
        return {
            "device": "cpu",
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_used_mb": None,
        }

    def embed(
        self, texts: list[str], pooling: Literal["mean", "cls"], normalize: bool
    ) -> list[list[float]]:
        self.embed_calls += 1
        self.last_embed_batch = list(texts)
        if self.fail_model_unavailable:
            raise ModelUnavailableError("fake: not ready")
        if self.oom_on_embed:
            raise OOMError("fake: CUDA OOM even at batch_size=1")

        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vals = [(b / 255.0) for b in (h * ((self.dim // len(h)) + 1))[: self.dim]]
            if normalize:
                norm = sum(v * v for v in vals) ** 0.5 or 1.0
                vals = [v / norm for v in vals]
            out.append(vals)
        return out

    def classify(self, texts: list[str], top_k: int) -> list[ClassifyPred]:
        self.classify_calls += 1
        self.last_classify_batch = list(texts)
        if self.fail_model_unavailable:
            raise ModelUnavailableError("fake: not ready")
        if not self.classifier:
            raise ClassificationUnsupportedError("fake: no classifier head")
        if self.oom_on_classify:
            raise OOMError("fake: CUDA OOM even at batch_size=1")

        preds: list[ClassifyPred] = []
        k = min(top_k, len(self.labels))
        for t in texts:
            idx = int.from_bytes(hashlib.sha256(t.encode()).digest()[:2], "big") % len(self.labels)
            scores = [0.1] * len(self.labels)
            scores[idx] = 0.9
            total = sum(scores)
            scores = [s / total for s in scores]
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
            top = [(self.labels[i], float(s)) for i, s in ranked]
            preds.append(ClassifyPred(label=top[0][0], score=top[0][1], top_k=top))
        return preds

    def warmup(self) -> None:
        return


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def settings_no_auth(monkeypatch: pytest.MonkeyPatch) -> InferenceSettings:
    for var in ("INFERENCE_API_KEY", "SECBERT_MODEL_PATH", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return InferenceSettings(inference_api_key=None, rate_limit="100000/minute")


@pytest.fixture
def settings_with_auth(monkeypatch: pytest.MonkeyPatch) -> InferenceSettings:
    monkeypatch.setenv("INFERENCE_API_KEY", "secret-token")
    return InferenceSettings(rate_limit="100000/minute")


@pytest.fixture
def client(fake_provider: FakeProvider, settings_no_auth: InferenceSettings):  # type: ignore[no-untyped-def]
    app = create_app(cfg=settings_no_auth, provider=fake_provider)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_auth(fake_provider: FakeProvider, settings_with_auth: InferenceSettings):  # type: ignore[no-untyped-def]
    app = create_app(cfg=settings_with_auth, provider=fake_provider)
    with TestClient(app) as c:
        yield c
