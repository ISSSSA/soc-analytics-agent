from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from soc_agent.recommendations.generator import (
    GenerationResult,
    RecommendationGenerator,
    _fallback_recommendation,
    _unique_source_files,
)
from soc_agent.schemas import (
    Cluster,
    LogEntry,
    PlaybookChunk,
    Priority,
    Recommendation,
    SeverityLevel,
)


def _cluster(**overrides: Any) -> Cluster:
    t0 = datetime(2025, 3, 15, 10, 23, tzinfo=UTC)
    t1 = datetime(2025, 3, 15, 10, 47, tzinfo=UTC)
    log = LogEntry.model_validate(
        {
            "event_id": "evt-1",
            "timestamp": "2025-03-15T10:23:00Z",
            "event_type": "ids_alert",
            "severity": "high",
            "description": "Failed SSH logins from 15 IPs targeting admin (T1110.004)",
            "src_ip": "203.0.113.42",
            "user": "admin",
            "action": "credential_stuffing",
        }
    )
    data: dict[str, Any] = {
        "cluster_id": 42,
        "cluster_size": 15,
        "time_range": (t0, t1),
        "dominant_event_type": "ids_alert",
        "dominant_classes": [
            {"class": "credential_stuffing", "count": 12, "avg_confidence": 0.87},
        ],
        "severity_breakdown": {"critical": 8, "high": 7},
        "unique_src_ips": 15,
        "unique_users_targeted": 8,
        "representative_logs": [log],
        "all_log_indices": list(range(15)),
    }
    data.update(overrides)
    return Cluster.model_validate(data)


def _chunk(source_file: str, score: float = 0.9) -> PlaybookChunk:
    return PlaybookChunk(
        content=f"Content from {source_file}",
        source_file=source_file,
        playbook_name=source_file.replace(".md", ""),
        section="Immediate Actions",
        mitre_techniques=["T1110.004"],
        score=score,
        file_hash="h",
    )


def _valid_recommendation() -> Recommendation:
    return Recommendation.model_validate(
        {
            "summary": "Coordinated credential stuffing from 15 unique source IPs.",
            "threat_assessment": "High — automated campaign targeting admin SSH.",
            "relevant_playbooks": ["llm_cited.md"],
            "immediate_actions": [
                "Block source IPs at perimeter firewall",
                "Force password reset for affected accounts",
            ],
            "investigation_steps": [
                "Correlate timestamps with SSH auth log",
                "Check for successful logins from same IP pool",
            ],
            "mitre_techniques": ["T1110.004"],
            "priority": "high",
        }
    )


class TestUniqueSourceFiles:
    def test_preserves_first_seen_order(self) -> None:
        chunks = [_chunk("a.md"), _chunk("b.md"), _chunk("a.md"), _chunk("c.md")]
        assert _unique_source_files(chunks) == ["a.md", "b.md", "c.md"]

    def test_empty(self) -> None:
        assert _unique_source_files([]) == []

    def test_skips_blank_source(self) -> None:
        c = _chunk("")
        assert _unique_source_files([c]) == []


class TestFallback:
    def test_priority_from_max_severity(self) -> None:
        cluster = _cluster(severity_breakdown={"critical": 3, "high": 2})
        fb = _fallback_recommendation(cluster, RuntimeError("boom"))
        assert fb.priority is Priority.CRITICAL

    def test_priority_maps_low_to_low(self) -> None:
        cluster = _cluster(severity_breakdown={SeverityLevel.LOW.value: 1})
        fb = _fallback_recommendation(cluster, RuntimeError("boom"))
        assert fb.priority is Priority.LOW

    def test_content_mentions_failure(self) -> None:
        cluster = _cluster()
        fb = _fallback_recommendation(cluster, TimeoutError("slow"))
        assert "TimeoutError" in fb.summary
        assert "Manual SOC review" in fb.immediate_actions[0]
        assert len(fb.immediate_actions) >= 1
        assert len(fb.investigation_steps) >= 1
        assert fb.relevant_playbooks == []
        assert fb.mitre_techniques == []


class TestGenerator:
    @pytest.mark.asyncio
    async def test_happy_path_returns_recommendation(self) -> None:
        llm = MagicMock()
        llm.generate_structured = AsyncMock(return_value=_valid_recommendation())
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(
            return_value=[
                _chunk("credential_stuffing_response.md", 0.95),
                _chunk("insider_threat_detection.md", 0.55),
            ]
        )
        gen = RecommendationGenerator(llm, retriever)
        result = await gen.generate(_cluster())

        assert isinstance(result, GenerationResult)
        assert result.used_fallback is False
        assert result.fallback_reason is None
        assert result.recommendation.relevant_playbooks == [
            "credential_stuffing_response.md",
            "insider_threat_detection.md",
        ]
        assert len(result.playbook_chunks) == 2

    @pytest.mark.asyncio
    async def test_retriever_called_with_cluster_data(self) -> None:
        llm = MagicMock()
        llm.generate_structured = AsyncMock(return_value=_valid_recommendation())
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(return_value=[])
        gen = RecommendationGenerator(llm, retriever, top_k_playbooks=5, mitre_boost=0.3)
        await gen.generate(_cluster())

        retriever.retrieve_for_cluster_text.assert_called_once()
        args = retriever.retrieve_for_cluster_text.call_args
        assert args.args[0] == "ids_alert"
        assert args.args[1] == ["credential_stuffing"]
        assert "Failed SSH logins" in args.args[2]
        assert args.kwargs["top_k"] == 5
        assert args.kwargs["mitre_boost"] == 0.3

    @pytest.mark.asyncio
    async def test_llm_prompt_built_from_cluster(self) -> None:
        captured: dict[str, Any] = {}

        async def _cap(*, system: str, user: str, schema: Any) -> Recommendation:
            captured["system"] = system
            captured["user"] = user
            return _valid_recommendation()

        llm = MagicMock()
        llm.generate_structured = AsyncMock(side_effect=_cap)
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(
            return_value=[_chunk("credential_stuffing_response.md")]
        )
        gen = RecommendationGenerator(llm, retriever)
        await gen.generate(_cluster())

        assert "SOC analyst" in captured["system"]
        assert "Cluster ID:** 42" in captured["user"]
        assert "credential_stuffing_response.md" in captured["user"]

    @pytest.mark.asyncio
    async def test_no_playbooks_retrieved_does_not_enrich(self) -> None:
        llm = MagicMock()
        rec_input = _valid_recommendation()
        llm.generate_structured = AsyncMock(return_value=rec_input)
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(return_value=[])
        gen = RecommendationGenerator(llm, retriever)
        result = await gen.generate(_cluster())

        assert result.recommendation.relevant_playbooks == ["llm_cited.md"]


class TestGeneratorFallback:
    @pytest.mark.asyncio
    async def test_llm_exception_triggers_fallback(self) -> None:
        llm = MagicMock()
        llm.generate_structured = AsyncMock(side_effect=TimeoutError("llm died"))
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(
            return_value=[_chunk("credential_stuffing_response.md")]
        )
        gen = RecommendationGenerator(llm, retriever)
        result = await gen.generate(_cluster())

        assert result.used_fallback is True
        assert "TimeoutError" in (result.fallback_reason or "")
        assert len(result.playbook_chunks) == 1
        assert isinstance(result.recommendation, Recommendation)
        assert "Manual SOC review" in result.recommendation.immediate_actions[0]

    @pytest.mark.asyncio
    async def test_retriever_exception_propagates(self) -> None:
        llm = MagicMock()
        llm.generate_structured = AsyncMock(return_value=_valid_recommendation())
        retriever = MagicMock()
        retriever.retrieve_for_cluster_text = MagicMock(
            side_effect=RuntimeError("chroma dead")
        )
        gen = RecommendationGenerator(llm, retriever)
        with pytest.raises(RuntimeError, match="chroma dead"):
            await gen.generate(_cluster())
