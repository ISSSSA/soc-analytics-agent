"""Unit tests for soc_agent.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from soc_agent.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_env_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each test starts with no relevant env vars and no cached Settings."""
    for var in [
        "SECBERT_MODEL_PATH",
        "INFERENCE_SERVICE_URL",
        "INFERENCE_SERVICE_API_KEY",
        "LLM_MODEL",
        "LLM_MAX_CONCURRENT",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_TIMEOUT",
        "CHROMA_PERSIST_DIR",
        "PLAYBOOKS_DIR",
        "EMBEDDING_MODEL",
        "CACHE_DIR",
        "BATCH_SIZE",
        "HDBSCAN_MIN_CLUSTER_SIZE",
        "HDBSCAN_MIN_SAMPLES",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(var, raising=False)
    # Avoid picking up a real .env from repo root during tests.
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


class TestDefaults:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.llm_model == "anthropic/claude-haiku-4-5"
        assert s.llm_max_concurrent == 5
        assert s.llm_temperature == 0.3
        assert s.llm_max_tokens == 2000
        assert s.batch_size == 64
        assert s.hdbscan_min_cluster_size == 5
        assert s.hdbscan_min_samples == 3
        assert s.log_level == "INFO"
        assert s.embedding_model.startswith("sentence-transformers/")

    def test_paths_are_path_objects(self) -> None:
        s = Settings()
        assert isinstance(s.chroma_persist_dir, Path)
        assert isinstance(s.cache_dir, Path)
        assert isinstance(s.playbooks_dir, Path)

    def test_secret_is_none_by_default(self) -> None:
        s = Settings()
        assert s.inference_service_api_key is None


class TestEnvLoading:
    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
        monkeypatch.setenv("LLM_MAX_CONCURRENT", "10")
        monkeypatch.setenv("INFERENCE_SERVICE_URL", "https://gpu.example.com:8001")
        monkeypatch.setenv("INFERENCE_SERVICE_API_KEY", "secret-token")
        monkeypatch.setenv("BATCH_SIZE", "128")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        s = Settings()
        assert s.llm_model == "openai/gpt-4o-mini"
        assert s.llm_max_concurrent == 10
        assert str(s.inference_service_url).startswith("https://gpu.example.com")
        assert isinstance(s.inference_service_api_key, SecretStr)
        assert s.inference_service_api_key.get_secret_value() == "secret-token"
        assert s.batch_size == 128
        assert s.log_level == "DEBUG"

    def test_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("llm_model", "deepseek/deepseek-chat")
        s = Settings()
        assert s.llm_model == "deepseek/deepseek-chat"

    def test_ignores_unknown_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPLETELY_UNRELATED_VAR", "foo")
        s = Settings()  # must not raise
        assert s.llm_model == "anthropic/claude-haiku-4-5"


class TestValidation:
    def test_llm_max_concurrent_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MAX_CONCURRENT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_llm_temperature_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_TEMPERATURE", "3.0")
        with pytest.raises(ValidationError):
            Settings()

    def test_batch_size_bounds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BATCH_SIZE", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "SILLY")
        with pytest.raises(ValidationError):
            Settings()

    def test_hdbscan_min_cluster_size_min(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HDBSCAN_MIN_CLUSTER_SIZE", "1")
        with pytest.raises(ValidationError):
            Settings()


class TestHelpers:
    def test_ensure_dirs_creates_missing(self, tmp_path: Path) -> None:
        chroma = tmp_path / "chroma"
        cache = tmp_path / "cache"
        s = Settings(chroma_persist_dir=chroma, cache_dir=cache)
        assert not chroma.exists()
        s.ensure_dirs()
        assert chroma.is_dir()
        assert cache.is_dir()

    def test_ensure_dirs_idempotent(self, tmp_path: Path) -> None:
        s = Settings(chroma_persist_dir=tmp_path / "c", cache_dir=tmp_path / "k")
        s.ensure_dirs()
        s.ensure_dirs()  # must not raise

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()
        get_settings.cache_clear()
        assert get_settings() is get_settings()
