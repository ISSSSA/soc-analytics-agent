"""Shared Pydantic models for soc_agent.

Layered by role in the pipeline:

- **Input** — `LogEntry` (SIEM dataset parsing, permissive)
- **Inference IO** — `EmbeddingsRequest/Response`, `ClassifyRequest/Response`
  (wire format with GPU service)
- **Aggregation** — `Cluster`, `Classification`, `DominantClass` (intermediate pipeline state)
- **RAG** — `PlaybookChunk`
- **Output** — `Recommendation` (strict, LLM-facing), `IncidentReport`, `PipelineResult`

`Recommendation` has `extra="forbid"` + no defaults so its JSON schema is valid for
OpenAI-style strict structured output (`response_format.json_schema.strict=True`).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    IPvAnyAddress,
    field_serializer,
    field_validator,
)

# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #


def _normalize_enum_str(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower().replace(" ", "_").replace("-", "_")
    return v


class SeverityLevel(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @classmethod
    def _missing_(cls, value: object) -> SeverityLevel | None:
        if isinstance(value, str):
            normalized = _normalize_enum_str(value)
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class EventType(StrEnum):
    AUTH = "auth"
    FIREWALL = "firewall"
    ENDPOINT = "endpoint"
    NETWORK = "network"
    CLOUD = "cloud"
    AI = "ai"
    IOT = "iot"
    IDS_ALERT = "ids_alert"

    @classmethod
    def _missing_(cls, value: object) -> EventType | None:
        if isinstance(value, str):
            normalized = _normalize_enum_str(value)
            for member in cls:
                if member.value == normalized:
                    return member
        return None


class Priority(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

    @classmethod
    def _missing_(cls, value: object) -> Priority | None:
        if isinstance(value, str):
            normalized = _normalize_enum_str(value)
            for member in cls:
                if member.value == normalized:
                    return member
        return None


# --------------------------------------------------------------------------- #
# Custom annotated types                                                       #
# --------------------------------------------------------------------------- #


_EMPTY_IP_MARKERS = {"", "n/a", "null", "none", "-", "unknown"}


def _empty_ip_to_none(v: Any) -> Any:
    """Coerce empty-ish strings to None before IPvAnyAddress validates."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in _EMPTY_IP_MARKERS:
        return None
    return v


def _validate_port(v: Any) -> int | None:
    """Coerce empty/'N/A'/None → None; validate 0 ≤ port ≤ 65535 otherwise."""
    if v is None or v == "":
        return None
    if isinstance(v, str) and v.strip().lower() in _EMPTY_IP_MARKERS:
        return None
    try:
        port = int(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid port: {v!r}") from e
    if not 0 <= port <= 65535:
        raise ValueError(f"port out of range [0, 65535]: {port}")
    return port


OptionalIP = Annotated[IPvAnyAddress | None, BeforeValidator(_empty_ip_to_none)]
OptionalPort = Annotated[int | None, BeforeValidator(_validate_port)]


# --------------------------------------------------------------------------- #
# Input — LogEntry (SIEM dataset row)                                          #
# --------------------------------------------------------------------------- #


class AdvancedMetadata(BaseModel):
    """Advanced metadata attached to a SIEM event. Permissive — dataset may vary."""

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    risk_score: float | None = None
    confidence: float | None = None
    geo_location: str | None = None


class BehavioralAnalytics(BaseModel):
    """Behavioral signals attached to a SIEM event. Permissive."""

    model_config = ConfigDict(extra="allow")

    baseline_deviation: float | None = None
    entropy: float | None = None
    frequency_anomaly: float | None = None
    sequence_anomaly: float | None = None


class LogEntry(BaseModel):
    """Single SIEM log entry.

    `description` is the text used for embeddings. `raw_log` (CEF) is stored for
    audit but MUST NOT be used for ML — it's noisy with SIEM prefixes.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: SeverityLevel
    description: str = Field(..., min_length=1)
    raw_log: str | None = None

    advanced_metadata: AdvancedMetadata = Field(default_factory=AdvancedMetadata)
    behavioral_analytics: BehavioralAnalytics = Field(default_factory=BehavioralAnalytics)

    user: str | None = None
    action: str | None = None  # 55 SecBERT classes — kept as free str, labels come from /health
    object: str | None = None

    src_ip: OptionalIP = None
    dst_ip: OptionalIP = None
    src_port: OptionalPort = None
    dst_port: OptionalPort = None
    protocol: str | None = None
    mac_address: str | None = None

    alert_type: str | None = None
    signature_id: str | None = None
    category: str | None = None

    cloud_service: str | None = None
    model_id: str | None = None

    @field_serializer("src_ip", "dst_ip")
    def _serialize_ip(self, ip: IPvAnyAddress | None) -> str | None:
        return str(ip) if ip is not None else None


# --------------------------------------------------------------------------- #
# Inference IO — /embeddings and /classify                                     #
# --------------------------------------------------------------------------- #


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(..., min_length=1, max_length=512)
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if len(t) > 2048:
                raise ValueError(f"texts[{i}]: length {len(t)} exceeds 2048 chars")
        return v


class EmbeddingsResponse(BaseModel):
    embeddings: list[list[float]]
    model_id: str
    pooling: Literal["mean", "cls"]
    normalized: bool
    processing_time_ms: float = Field(..., ge=0.0)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(..., min_length=1, max_length=512)
    top_k: int = Field(5, ge=1, le=55)
    return_all_scores: bool = False

    @field_validator("texts")
    @classmethod
    def _validate_texts(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if len(t) > 2048:
                raise ValueError(f"texts[{i}]: length {len(t)} exceeds 2048 chars")
        return v


class ClassPrediction(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    top_k: list[tuple[str, float]] = Field(default_factory=list)


class ClassifyResponse(BaseModel):
    predictions: list[ClassPrediction]
    model_id: str
    processing_time_ms: float = Field(..., ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    model_loaded: bool
    gpu_available: bool
    gpu_name: str | None = None
    gpu_memory_total_mb: int | None = None
    gpu_memory_used_mb: int | None = None
    supports_classification: bool
    num_classes: int | None = None
    class_labels: list[str] | None = None


# --------------------------------------------------------------------------- #
# Aggregation — clustering + classification                                    #
# --------------------------------------------------------------------------- #


class ClusterStats(BaseModel):
    """Raw HDBSCAN output for one cluster. cluster_id == -1 means noise."""

    cluster_id: int
    size: int = Field(..., ge=1)
    log_indices: list[int]


class Classification(BaseModel):
    """One classification result tied back to a log inside a cluster."""

    log_index: int = Field(..., ge=0)  # index within cluster.log_indices (or global)
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    top_k: list[tuple[str, float]] = Field(default_factory=list)


class DominantClass(BaseModel):
    """Aggregated statistic over a cluster's classifications.

    Serializes `class_name` as `"class"` in JSON to match the external schema.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    class_name: str = Field(..., alias="class")
    count: int = Field(..., ge=1)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)


class Cluster(BaseModel):
    """Enriched cluster — joins ClusterStats with aggregated LogEntry/Classification data."""

    cluster_id: int
    cluster_size: int = Field(..., ge=1)
    time_range: tuple[datetime, datetime]
    dominant_event_type: EventType
    dominant_classes: list[DominantClass]  # top-3 by count
    severity_breakdown: dict[SeverityLevel, int]
    unique_src_ips: int = Field(..., ge=0)
    unique_users_targeted: int = Field(..., ge=0)
    representative_logs: list[LogEntry]  # 3 logs closest to centroid
    all_log_indices: list[int]


# --------------------------------------------------------------------------- #
# RAG                                                                          #
# --------------------------------------------------------------------------- #


class PlaybookChunk(BaseModel):
    content: str
    source_file: str
    playbook_name: str
    section: str | None = None
    mitre_techniques: list[str] = Field(default_factory=list)
    score: float  # cosine similarity after MITRE boost; may exceed 1.0
    file_hash: str


# --------------------------------------------------------------------------- #
# Output — Recommendation (strict LLM output) and IncidentReport              #
# --------------------------------------------------------------------------- #


class Recommendation(BaseModel):
    """SOC analyst recommendation.

    Shape matches the JSON schema sent to the LLM in structured-output mode.
    `extra="forbid"` + no default values → strict-mode compatible.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(..., min_length=10, max_length=2000)
    threat_assessment: str = Field(..., min_length=10, max_length=2000)
    relevant_playbooks: list[str]
    immediate_actions: list[str] = Field(..., min_length=1, max_length=15)
    investigation_steps: list[str] = Field(..., min_length=1, max_length=15)
    mitre_techniques: list[str]
    priority: Priority


class IncidentReport(BaseModel):
    """Per-cluster SOC incident. Top-level output unit."""

    cluster_id: int
    cluster_size: int = Field(..., ge=1)
    time_range: tuple[datetime, datetime]
    dominant_event_type: EventType
    dominant_classes: list[DominantClass]
    severity_breakdown: dict[SeverityLevel, int]
    unique_src_ips: int = Field(..., ge=0)
    unique_users_targeted: int = Field(..., ge=0)
    representative_logs: list[LogEntry]
    recommendation: Recommendation

    # Fallback metadata — set when LLM failed and `recommendation` is a stub.
    recommendation_unavailable: bool = False
    fallback_reason: str | None = None


class LLMUsage(BaseModel):
    model: str
    total_cost_usd: float = Field(..., ge=0.0)
    total_tokens_in: int = Field(..., ge=0)
    total_tokens_out: int = Field(..., ge=0)


class PipelineStats(BaseModel):
    total_logs: int = Field(..., ge=0)
    n_clusters: int = Field(..., ge=0)
    n_noise: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0)
    llm_usage: LLMUsage


class PipelineResult(BaseModel):
    correlation_id: str
    incidents: list[IncidentReport]
    stats: PipelineStats
