from __future__ import annotations

import json
from datetime import UTC, datetime
from ipaddress import IPv4Address
from typing import Any

import pytest
from pydantic import ValidationError

from soc_agent.schemas import (
    Classification,
    ClassifyRequest,
    Cluster,
    ClusterStats,
    DominantClass,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EventType,
    HealthResponse,
    IncidentReport,
    LLMUsage,
    LogEntry,
    PipelineResult,
    PipelineStats,
    PlaybookChunk,
    Priority,
    Recommendation,
    SeverityLevel,
)


class TestEnums:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("high", SeverityLevel.HIGH),
            ("HIGH", SeverityLevel.HIGH),
            ("  High ", SeverityLevel.HIGH),
            ("critical", SeverityLevel.CRITICAL),
        ],
    )
    def test_severity_case_insensitive(self, raw: str, expected: SeverityLevel) -> None:
        assert SeverityLevel(raw) is expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("ids_alert", EventType.IDS_ALERT),
            ("IDS-ALERT", EventType.IDS_ALERT),
            ("IDS ALERT", EventType.IDS_ALERT),
            ("cloud", EventType.CLOUD),
        ],
    )
    def test_event_type_normalization(self, raw: str, expected: EventType) -> None:
        assert EventType(raw) is expected

    def test_event_type_unknown_rejected(self) -> None:
        with pytest.raises(ValueError):
            EventType("quantum_teleport")

    def test_priority_normalized(self) -> None:
        assert Priority("HIGH") is Priority.HIGH
        assert Priority("informational") is Priority.INFORMATIONAL


class TestLogEntry:
    def test_full_from_dict(self, sample_log_dict: dict[str, Any]) -> None:
        entry = LogEntry.model_validate(sample_log_dict)
        assert entry.event_id == "evt-001"
        assert entry.event_type is EventType.IDS_ALERT
        assert entry.severity is SeverityLevel.HIGH
        assert entry.src_ip == IPv4Address("203.0.113.42")
        assert entry.advanced_metadata.risk_score == 0.87
        assert entry.behavioral_analytics.entropy == 0.65

    def test_minimal(self) -> None:
        entry = LogEntry.model_validate(
            {
                "event_id": "x",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "auth",
                "severity": "low",
                "description": "ok",
                "raw_log": "CEF:0|Test|SIEM|1.0|100|auth event|3|src=10.0.0.1",
            }
        )
        assert entry.src_ip is None
        assert entry.user is None
        assert entry.advanced_metadata.risk_score is None

    @pytest.mark.parametrize("empty", ["", "N/A", "null", "None", "unknown", "-"])
    def test_empty_ip_becomes_none(self, empty: str) -> None:
        entry = LogEntry.model_validate(
            {
                "event_id": "x",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "auth",
                "severity": "low",
                "description": "ok",
                "raw_log": "CEF:0|Test|SIEM|1.0|100|auth event|3|src=10.0.0.1",
                "src_ip": empty,
                "dst_ip": empty,
            }
        )
        assert entry.src_ip is None
        assert entry.dst_ip is None

    def test_invalid_ip_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry.model_validate(
                {
                    "event_id": "x",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "event_type": "auth",
                    "severity": "low",
                    "description": "ok",
                    "src_ip": "not_an_ip",
                }
            )

    def test_invalid_port_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry.model_validate(
                {
                    "event_id": "x",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "event_type": "auth",
                    "severity": "low",
                    "description": "ok",
                    "src_port": 99999,
                }
            )

    def test_empty_port_becomes_none(self) -> None:
        entry = LogEntry.model_validate(
            {
                "event_id": "x",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "auth",
                "severity": "low",
                "description": "ok",
                "raw_log": "CEF:0|Test|SIEM|1.0|100|auth event|3|src=10.0.0.1",
                "src_port": "",
                "dst_port": "N/A",
            }
        )
        assert entry.src_port is None
        assert entry.dst_port is None

    def test_requires_description(self) -> None:
        with pytest.raises(ValidationError) as ei:
            LogEntry.model_validate(
                {
                    "event_id": "x",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "event_type": "auth",
                    "severity": "low",
                    "description": "",
                    "raw_log": "CEF:0|Test|SIEM|1.0|100|auth|3|src=10.0.0.1",
                }
            )
        assert "description" in str(ei.value)

    def test_extra_fields_allowed(self) -> None:
        entry = LogEntry.model_validate(
            {
                "event_id": "x",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "auth",
                "severity": "low",
                "description": "ok",
                "raw_log": "CEF:0|Test|SIEM|1.0|100|auth event|3|src=10.0.0.1",
                "custom_siem_field": "whatever",
            }
        )
        dumped = entry.model_dump()
        assert dumped["custom_siem_field"] == "whatever"

    def test_advanced_metadata_extra_allowed(self) -> None:
        entry = LogEntry.model_validate(
            {
                "event_id": "x",
                "timestamp": "2025-01-01T00:00:00Z",
                "event_type": "auth",
                "severity": "low",
                "description": "ok",
                "raw_log": "CEF:0|Test|SIEM|1.0|100|auth event|3|src=10.0.0.1",
                "advanced_metadata": {"risk_score": 0.5, "future_field": 123},
            }
        )
        assert entry.advanced_metadata.risk_score == 0.5
        dumped = entry.advanced_metadata.model_dump()
        assert dumped["future_field"] == 123

    def test_ip_serializes_as_string(self, sample_log_dict: dict[str, Any]) -> None:
        entry = LogEntry.model_validate(sample_log_dict)
        s = json.loads(entry.model_dump_json())
        assert s["src_ip"] == "203.0.113.42"
        assert s["dst_ip"] == "10.0.0.5"


class TestInferenceIO:
    def test_embeddings_request_defaults(self) -> None:
        req = EmbeddingsRequest(texts=["hello"])
        assert req.pooling == "mean"
        assert req.normalize is True

    def test_embeddings_request_rejects_empty_list(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest(texts=[])

    def test_embeddings_request_max_texts(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest(texts=["x"] * 513)

    def test_embeddings_request_max_text_length(self) -> None:
        with pytest.raises(ValidationError) as ei:
            EmbeddingsRequest(texts=["x" * 2049])
        assert "2048" in str(ei.value)

    def test_embeddings_request_forbids_extra(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingsRequest.model_validate({"texts": ["a"], "bogus": 1})

    def test_embeddings_response(self) -> None:
        r = EmbeddingsResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            model_id="secbert-v1",
            pooling="mean",
            normalized=True,
            processing_time_ms=42.5,
        )
        assert len(r.embeddings[0]) == 3

    def test_classify_request_top_k_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyRequest(texts=["x"], top_k=0)
        with pytest.raises(ValidationError):
            ClassifyRequest(texts=["x"], top_k=100)

    def test_health_response_minimal(self) -> None:
        h = HealthResponse(
            status="ok",
            model_loaded=True,
            gpu_available=True,
            supports_classification=True,
        )
        assert h.num_classes is None


class TestAggregation:
    def test_dominant_class_alias_in_out(self) -> None:
        dc = DominantClass.model_validate(
            {"class": "credential_stuffing", "count": 12, "avg_confidence": 0.87}
        )
        assert dc.class_name == "credential_stuffing"
        dumped = dc.model_dump(by_alias=True)
        assert dumped["class"] == "credential_stuffing"
        assert dc.model_dump()["class_name"] == "credential_stuffing"

    def test_dominant_class_forbids_extra(self) -> None:
        with pytest.raises(ValidationError):
            DominantClass.model_validate(
                {"class": "x", "count": 1, "avg_confidence": 0.5, "bogus": 1}
            )

    def test_classification_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Classification(log_index=0, label="x", score=1.5)

    def test_cluster_roundtrip(self, sample_cluster_kwargs: dict[str, Any]) -> None:
        c = Cluster.model_validate(sample_cluster_kwargs)
        assert c.cluster_id == 42
        assert c.dominant_event_type is EventType.IDS_ALERT
        assert c.dominant_classes[0].class_name == "credential_stuffing"
        assert c.severity_breakdown[SeverityLevel.CRITICAL] == 8

    def test_cluster_serializes_class_alias(self, sample_cluster_kwargs: dict[str, Any]) -> None:
        c = Cluster.model_validate(sample_cluster_kwargs)
        dumped = json.loads(c.model_dump_json(by_alias=True))
        assert dumped["dominant_classes"][0]["class"] == "credential_stuffing"

    def test_cluster_stats(self) -> None:
        cs = ClusterStats(cluster_id=-1, size=1, log_indices=[5])
        assert cs.cluster_id == -1


class TestRecommendation:
    def test_valid(self, sample_recommendation_kwargs: dict[str, Any]) -> None:
        r = Recommendation.model_validate(sample_recommendation_kwargs)
        assert r.priority is Priority.HIGH
        assert len(r.immediate_actions) == 2

    def test_requires_at_least_one_action(
        self, sample_recommendation_kwargs: dict[str, Any]
    ) -> None:
        sample_recommendation_kwargs["immediate_actions"] = []
        with pytest.raises(ValidationError):
            Recommendation.model_validate(sample_recommendation_kwargs)

    def test_requires_at_least_one_investigation_step(
        self, sample_recommendation_kwargs: dict[str, Any]
    ) -> None:
        sample_recommendation_kwargs["investigation_steps"] = []
        with pytest.raises(ValidationError):
            Recommendation.model_validate(sample_recommendation_kwargs)

    def test_summary_min_length(self, sample_recommendation_kwargs: dict[str, Any]) -> None:
        sample_recommendation_kwargs["summary"] = "short"
        with pytest.raises(ValidationError):
            Recommendation.model_validate(sample_recommendation_kwargs)

    def test_forbids_extra(self, sample_recommendation_kwargs: dict[str, Any]) -> None:
        sample_recommendation_kwargs["hallucinated_field"] = "nope"
        with pytest.raises(ValidationError):
            Recommendation.model_validate(sample_recommendation_kwargs)

    def test_json_schema_strict_compatible(self) -> None:
        schema = Recommendation.model_json_schema()
        assert schema.get("additionalProperties") is False
        required = set(schema["required"])
        expected = {
            "summary",
            "threat_assessment",
            "relevant_playbooks",
            "immediate_actions",
            "investigation_steps",
            "mitre_techniques",
            "priority",
        }
        assert required == expected

    def test_priority_normalization(self, sample_recommendation_kwargs: dict[str, Any]) -> None:
        sample_recommendation_kwargs["priority"] = "HIGH"
        r = Recommendation.model_validate(sample_recommendation_kwargs)
        assert r.priority is Priority.HIGH


class TestPlaybookChunk:
    def test_construct(self) -> None:
        pc = PlaybookChunk(
            content="# Immediate Actions\nBlock IPs...",
            source_file="credential_stuffing_response.md",
            playbook_name="Credential Stuffing Response",
            section="Immediate Actions",
            mitre_techniques=["T1110.004"],
            score=0.92,
            file_hash="abc123",
        )
        assert pc.mitre_techniques == ["T1110.004"]


class TestIncidentReport:
    def _build(
        self,
        cluster_kwargs: dict[str, Any],
        rec_kwargs: dict[str, Any],
    ) -> IncidentReport:
        return IncidentReport.model_validate(
            {
                **{k: v for k, v in cluster_kwargs.items() if k != "all_log_indices"},
                "recommendation": rec_kwargs,
            }
        )

    def test_construct_and_json(
        self,
        sample_cluster_kwargs: dict[str, Any],
        sample_recommendation_kwargs: dict[str, Any],
    ) -> None:
        ir = self._build(sample_cluster_kwargs, sample_recommendation_kwargs)
        assert ir.cluster_id == 42
        assert ir.recommendation.priority is Priority.HIGH
        assert ir.recommendation_unavailable is False

    def test_fallback_flags(
        self,
        sample_cluster_kwargs: dict[str, Any],
        sample_recommendation_kwargs: dict[str, Any],
    ) -> None:
        ir = IncidentReport.model_validate(
            {
                **{k: v for k, v in sample_cluster_kwargs.items() if k != "all_log_indices"},
                "recommendation": sample_recommendation_kwargs,
                "recommendation_unavailable": True,
                "fallback_reason": "LLM timeout after 3 retries",
            }
        )
        assert ir.recommendation_unavailable is True
        assert ir.fallback_reason is not None

    def test_roundtrip_json_by_alias(
        self,
        sample_cluster_kwargs: dict[str, Any],
        sample_recommendation_kwargs: dict[str, Any],
    ) -> None:
        ir = self._build(sample_cluster_kwargs, sample_recommendation_kwargs)
        dumped = json.loads(ir.model_dump_json(by_alias=True))
        assert dumped["dominant_classes"][0]["class"] == "credential_stuffing"
        again = IncidentReport.model_validate(dumped)
        assert again.cluster_id == ir.cluster_id


class TestPipelineResult:
    def test_construct(
        self,
        sample_cluster_kwargs: dict[str, Any],
        sample_recommendation_kwargs: dict[str, Any],
    ) -> None:
        ir = IncidentReport.model_validate(
            {
                **{k: v for k, v in sample_cluster_kwargs.items() if k != "all_log_indices"},
                "recommendation": sample_recommendation_kwargs,
            }
        )
        pr = PipelineResult(
            correlation_id="run-xyz",
            incidents=[ir],
            stats=PipelineStats(
                total_logs=1000,
                n_clusters=1,
                n_noise=0,
                processing_time_seconds=42.0,
                cache_hit_rate=0.8,
                llm_usage=LLMUsage(
                    model="anthropic/claude-haiku-4-5",
                    total_cost_usd=0.0023,
                    total_tokens_in=1840,
                    total_tokens_out=520,
                ),
            ),
        )
        assert pr.stats.cache_hit_rate == 0.8
        assert pr.incidents[0].cluster_id == 42

    def test_cache_hit_rate_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PipelineStats(
                total_logs=1,
                n_clusters=0,
                n_noise=0,
                processing_time_seconds=1.0,
                cache_hit_rate=1.5,
                llm_usage=LLMUsage(
                    model="x", total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0
                ),
            )


def test_timestamp_parses_iso_with_z(sample_log_dict: dict[str, Any]) -> None:
    entry = LogEntry.model_validate(sample_log_dict)
    assert entry.timestamp == datetime(2025, 3, 15, 10, 23, 0, tzinfo=UTC)
