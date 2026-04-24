from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def real_provider():  # type: ignore[no-untyped-def]
    from inference_service.model_loader import SecBERTProvider

    p = SecBERTProvider(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        use_fp16=False,
        max_batch_size=8,
    )
    return p


class TestRealMiniModel:
    def test_ready(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        assert real_provider.is_ready is True
        assert real_provider.model_id == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embeddings_shape(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        vectors = real_provider.embed(["hello world", "another sentence"], "mean", True)
        assert len(vectors) == 2
        assert len(vectors[0]) == 384

    def test_normalization_produces_unit_vectors(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        vectors = real_provider.embed(["hello"], "mean", True)
        norm = sum(v * v for v in vectors[0]) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

    def test_cls_pooling_differs_from_mean(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        mean_vec = real_provider.embed(["hello world"], "mean", False)[0]
        cls_vec = real_provider.embed(["hello world"], "cls", False)[0]
        assert not all(abs(a - b) < 1e-6 for a, b in zip(mean_vec, cls_vec, strict=True))

    def test_classifier_not_supported_for_mini(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        assert real_provider.supports_classification is False
        assert real_provider.class_labels is None

    def test_end_to_end_via_http(self, real_provider) -> None:  # type: ignore[no-untyped-def]
        from fastapi.testclient import TestClient

        from inference_service.config import InferenceSettings
        from inference_service.server import create_app

        cfg = InferenceSettings(inference_api_key=None, rate_limit="100000/minute")
        app = create_app(cfg=cfg, provider=real_provider)
        with TestClient(app) as c:
            r = c.post("/embeddings", json={"texts": ["hello world"]})
            assert r.status_code == 200
            body = r.json()
            assert len(body["embeddings"][0]) == 384

            r = c.post("/classify", json={"texts": ["hello"]})
            assert r.status_code == 501

            r = c.get("/health")
            assert r.status_code == 200
            assert r.json()["supports_classification"] is False
