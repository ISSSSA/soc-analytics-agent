from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    secbert_model_path: str = Field(
        default="issssssaaaa/secbert-siem",
        description="HF repo id or local path to the fine-tuned SecBERT checkpoint.",
    )
    inference_service_url: HttpUrl = Field(default=HttpUrl("http://localhost:8001"))
    inference_service_api_key: SecretStr | None = None

    llm_model: str = Field(default="anthropic/claude-haiku-4-5")
    llm_max_concurrent: int = Field(default=5, ge=1, le=100)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2000, ge=1, le=32000)
    llm_timeout: int = Field(default=60, ge=1, le=600)

    openrouter_app_name: str | None = Field(
        default=None,
        description="X-Title header for OpenRouter dashboard identification.",
    )
    openrouter_site_url: str | None = Field(
        default=None,
        description="HTTP-Referer header for OpenRouter app rankings.",
    )

    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))
    playbooks_dir: Path = Field(default=Path("./playbooks"))
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    cache_dir: Path = Field(default=Path("./data/cache"))
    batch_size: int = Field(default=64, ge=1, le=2048)
    hdbscan_min_cluster_size: int = Field(default=5, ge=2)
    hdbscan_min_samples: int = Field(default=3, ge=1)

    log_level: LogLevel = "INFO"

    def ensure_dirs(self) -> None:
        for d in (self.chroma_persist_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
