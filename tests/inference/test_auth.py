from __future__ import annotations

from fastapi.testclient import TestClient


class TestAuth:
    def test_rejects_missing_token(self, client_auth: TestClient) -> None:
        r = client_auth.post("/embeddings", json={"texts": ["x"]})
        assert r.status_code == 401
        assert r.headers.get("www-authenticate") == "Bearer"

    def test_rejects_wrong_scheme(self, client_auth: TestClient) -> None:
        r = client_auth.post(
            "/embeddings",
            json={"texts": ["x"]},
            headers={"Authorization": "Basic abcdef"},
        )
        assert r.status_code == 401

    def test_rejects_wrong_token(self, client_auth: TestClient) -> None:
        r = client_auth.post(
            "/embeddings",
            json={"texts": ["x"]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert r.status_code == 401

    def test_accepts_valid_token(self, client_auth: TestClient) -> None:
        r = client_auth.post(
            "/embeddings",
            json={"texts": ["x"]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert r.status_code == 200

    def test_health_does_not_require_auth(self, client_auth: TestClient) -> None:
        r = client_auth.get("/health")
        assert r.status_code == 200

    def test_metrics_does_not_require_auth(self, client_auth: TestClient) -> None:
        r = client_auth.get("/metrics")
        assert r.status_code == 200

    def test_no_auth_configured_allows_all(self, client: TestClient) -> None:
        r = client.post("/embeddings", json={"texts": ["x"]})
        assert r.status_code == 200

    def test_bearer_scheme_is_case_insensitive(self, client_auth: TestClient) -> None:
        r = client_auth.post(
            "/embeddings",
            json={"texts": ["x"]},
            headers={"Authorization": "bearer secret-token"},
        )
        assert r.status_code == 200
