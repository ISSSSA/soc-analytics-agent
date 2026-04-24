from __future__ import annotations

from fastapi.testclient import TestClient

from tests.inference.conftest import FakeProvider


class TestEmbeddings:
    def test_basic(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": ["hello", "world"]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["embeddings"]) == 2
        assert len(body["embeddings"][0]) == 384
        assert body["pooling"] == "mean"
        assert body["normalized"] is True
        assert body["processing_time_ms"] >= 0

    def test_deterministic(self, client: TestClient) -> None:
        r1 = client.post("/embeddings", json={"texts": ["hello"]})
        r2 = client.post("/embeddings", json={"texts": ["hello"]})
        assert r1.json()["embeddings"] == r2.json()["embeddings"]

    def test_respects_normalize_false(
        self, client: TestClient, fake_provider: FakeProvider
    ) -> None:
        r = client.post(
            "/embeddings",
            json={"texts": ["hello"], "pooling": "cls", "normalize": False},
        )
        assert r.status_code == 200
        assert r.json()["normalized"] is False
        assert r.json()["pooling"] == "cls"

    def test_empty_texts_rejected(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": []})
        assert r.status_code == 422

    def test_too_many_texts(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": ["x"] * 513})
        assert r.status_code == 422

    def test_text_too_long(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": ["x" * 2049]})
        assert r.status_code == 422

    def test_extra_field_rejected(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": ["x"], "bogus": 1})
        assert r.status_code == 422


class TestClassify:
    def test_basic(self, client: TestClient) -> None:
        r = client.post("/classify", json={"texts": ["hello"], "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert "label" in pred and "score" in pred and "top_k" in pred
        assert len(pred["top_k"]) == 3

    def test_top_k_bounds(self, client: TestClient) -> None:
        assert client.post("/classify", json={"texts": ["x"], "top_k": 0}).status_code == 422

    def test_unsupported_when_no_classifier(
        self, fake_provider: FakeProvider, settings_no_auth, monkeypatch
    ) -> None:
        fake_provider.classifier = False
        from inference_service.server import create_app

        app = create_app(cfg=settings_no_auth, provider=fake_provider)
        with TestClient(app) as c:
            r = c.post("/classify", json={"texts": ["x"]})
            assert r.status_code == 501


class TestHealth:
    def test_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["supports_classification"] is True
        assert body["num_classes"] == 3
        assert body["class_labels"] == ["credential_stuffing", "brute_force", "reconnaissance"]

    def test_error_when_model_not_loaded(
        self, settings_no_auth, fake_provider: FakeProvider
    ) -> None:
        fake_provider.ready = False
        from inference_service.server import create_app

        app = create_app(cfg=settings_no_auth, provider=fake_provider)
        with TestClient(app) as c:
            r = c.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "error"
            assert r.json()["model_loaded"] is False


class TestMetrics:
    def test_exposition(self, client: TestClient) -> None:
        client.post("/embeddings", json={"texts": ["a", "b"]})
        client.post("/classify", json={"texts": ["a"], "top_k": 2})
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        text = r.text
        assert "inference_requests_total" in text
        assert "inference_latency_seconds" in text
        assert "inference_batch_size" in text


class TestErrorConversion:
    def test_oom_on_embeddings_returns_503(
        self, settings_no_auth, fake_provider: FakeProvider
    ) -> None:
        fake_provider.oom_on_embed = True
        from inference_service.server import create_app

        app = create_app(cfg=settings_no_auth, provider=fake_provider)
        with TestClient(app) as c:
            r = c.post("/embeddings", json={"texts": ["x"]})
            assert r.status_code == 503
            assert "OOM" in r.json()["detail"] or "memory" in r.json()["detail"].lower()

    def test_model_unavailable_returns_503(
        self, settings_no_auth, fake_provider: FakeProvider
    ) -> None:
        fake_provider.fail_model_unavailable = True
        from inference_service.server import create_app

        app = create_app(cfg=settings_no_auth, provider=fake_provider)
        with TestClient(app) as c:
            r = c.post("/embeddings", json={"texts": ["x"]})
            assert r.status_code == 503
