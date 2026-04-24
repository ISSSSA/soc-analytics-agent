from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Device = Literal["auto", "cpu", "cuda", "mps"]


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    secbert_model_path: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HF repo id or local path. Defaults to MINI model for dev/tests.",
    )
    inference_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    max_batch_size: int = Field(default=256, ge=1, le=2048)
    max_texts_per_request: int = Field(default=512, ge=1, le=4096)
    max_text_length: int = Field(default=2048, ge=16, le=32768)

    rate_limit: str = Field(default="100/minute", description="slowapi format")

    device: Device = "auto"
    use_fp16: bool = True

    warmup_on_startup: bool = True

    log_level: LogLevel = "INFO"


@lru_cache(maxsize=1)
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
