"""Runtime configuration loaded from environment / .env.

LLM provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY) are
intentionally NOT part of Settings — LiteLLM reads them from process env directly.
Keeping them out of the Pydantic model reduces the risk of leaking them via
logs or `settings.model_dump()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Agent-side runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Inference service --------------------------------------------------
    secbert_model_path: str = Field(
        default="issssssaaaa/secbert-siem",
        description="HF repo id or local path to the fine-tuned SecBERT checkpoint.",
    )
    inference_service_url: HttpUrl = Field(default=HttpUrl("http://localhost:8001"))
    inference_service_api_key: SecretStr | None = None

    # ---- LLM ---------------------------------------------------------------
    llm_model: str = Field(default="anthropic/claude-haiku-4-5")
    llm_max_concurrent: int = Field(default=5, ge=1, le=100)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=1, le=32000)
    llm_timeout: int = Field(default=60, ge=1, le=600)

    # ---- RAG ---------------------------------------------------------------
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    playbooks_dir: Path = Field(default=Path("./playbooks"))
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # ---- Pipeline ----------------------------------------------------------
    cache_dir: Path = Field(default=Path("./data/cache"))
    batch_size: int = Field(default=64, ge=1, le=2048)
    hdbscan_min_cluster_size: int = Field(default=5, ge=2)
    hdbscan_min_samples: int = Field(default=3, ge=1)

    # ---- Logging -----------------------------------------------------------
    log_level: LogLevel = "INFO"

    def ensure_dirs(self) -> None:
        """Create persistence directories if missing. Safe to call many times."""
        for d in (self.chroma_persist_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached Settings accessor.

    The cache avoids re-reading `.env` on every call. In tests, clear via
    `get_settings.cache_clear()` after mutating env vars.
    """
    return Settings()
