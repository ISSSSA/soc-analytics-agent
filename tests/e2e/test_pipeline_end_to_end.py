"""End-to-end pipeline test on 100 synthetic SIEM logs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from soc_agent.clustering.clusterer import HDBSCANClusterer
from soc_agent.io import load_logs
from soc_agent.pipeline import SOCPipeline
from soc_agent.recommendations.generator import RecommendationGenerator
from soc_agent.reports import format_report
from soc_agent.schemas import (
    ClassPrediction,
    LLMUsage,
    PlaybookChunk,
    Priority,
    Recommendation,
)

_SCENARIOS = [
    {
        "prefix": "c",
        "count": 40,
        "description_template": (
            "Failed SSH login from IP 203.0.113.{i} targeting admin "
            "(credential stuffing attempt)"
        ),
        "event_type": "ids_alert",
        "severity": "high",
        "action": "credential_stuffing",
        "src_ip_template": "203.0.113.{i}",
        "user": "admin",
        "centroid": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    },
    {
        "prefix": "d",
        "count": 30,
        "description_template": (
            "Anomalous DNS query {i} with high-entropy subdomain — possible "
            "tunneling C2"
        ),
        "event_type": "network",
        "severity": "medium",
        "action": "dns_tunneling",
        "src_ip_template": "10.0.0.{i}",
        "user": "alice",
        "centroid": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    },
    {
        "prefix": "e",
        "count": 20,
        "description_template": (
            "Container {i} attempted privileged syscall (pivot_root) — "
            "container escape signal"
        ),
        "event_type": "endpoint",
        "severity": "critical",
        "action": "container_escape",
        "src_ip_template": "10.0.1.{i}",
        "user": "svc-runner",
        "centroid": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    },
]


def _make_logs_and_vectors() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    logs: list[dict[str, Any]] = []
    text_to_vec: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(1234)

    base = datetime(2025, 3, 15, 10, 0, tzinfo=UTC)

    idx = 0
    for scenario in _SCENARIOS:
        for i in range(scenario["count"]):
            ts = base + timedelta(seconds=idx * 10)
            desc = scenario["description_template"].format(i=i)
            log = {
                "event_id": f"{scenario['prefix']}{i}",
                "timestamp": ts.isoformat(),
                "event_type": scenario["event_type"],
                "severity": scenario["severity"],
                "description": desc,
                "raw_log": f"CEF:0|Vendor|SIEM|1.0|{scenario['prefix']}{i}|{scenario['action']}|5|src={scenario['src_ip_template'].format(i=(i % 250) + 1)}",
                "src_ip": scenario["src_ip_template"].format(i=(i % 250) + 1),
                "user": scenario["user"],
                "action": scenario["action"],
            }
            logs.append(log)
            noise = rng.normal(scale=0.01, size=4).astype(np.float32)
            vec = scenario["centroid"] + noise
            vec /= np.linalg.norm(vec) or 1.0
            text_to_vec[desc] = vec.astype(np.float32)
            idx += 1

    for i in range(10):
        ts = base + timedelta(seconds=idx * 10)
        desc = f"Miscellaneous event {i} with no clear pattern"
        log = {
            "event_id": f"n{i}",
            "timestamp": ts.isoformat(),
            "event_type": "auth",
            "severity": "low",
            "description": desc,
            "raw_log": f"CEF:0|Vendor|SIEM|1.0|n{i}|misc|1|user=user{i}",
            "user": f"user{i}",
        }
        logs.append(log)
        vec = rng.normal(size=4).astype(np.float32)
        vec /= np.linalg.norm(vec) or 1.0
        text_to_vec[desc] = vec
        idx += 1

    return logs, text_to_vec


def _mock_embedder(text_to_vec: dict[str, np.ndarray]) -> MagicMock:
    async def _embed(texts: list[str], *, pooling: str = "mean", normalize: bool = True):
        vectors = np.asarray([text_to_vec[t] for t in texts], dtype=np.float32)
        return vectors

    emb = MagicMock()
    emb.embed_texts = AsyncMock(side_effect=_embed)
    stats = MagicMock()
    stats.hit_rate = 0.0
    emb.stats = stats
    return emb


def _mock_classifier() -> MagicMock:
    async def _classify(texts: list[str], *, top_k: int = 5, return_all_scores: bool = False):
        preds: list[ClassPrediction] = []
        for t in texts:
            tl = t.lower()
            if "credential" in tl or "ssh" in tl:
                label, score = "credential_stuffing", 0.92
                other = [("brute_force", 0.05), ("reconnaissance", 0.03)]
            elif "dns" in tl:
                label, score = "dns_tunneling", 0.88
                other = [("data_exfiltration", 0.08), ("beaconing", 0.04)]
            elif "container" in tl or "pivot_root" in tl:
                label, score = "container_escape", 0.91
                other = [("privilege_escalation", 0.06), ("suspicious_process", 0.03)]
            else:
                label, score = "benign", 0.60
                other = [("unclassified", 0.40)]
            preds.append(
                ClassPrediction(label=label, score=score, top_k=[(label, score), *other])
            )
        return preds

    clf = MagicMock()
    clf.classify_texts = AsyncMock(side_effect=_classify)
    return clf


def _mock_retriever() -> MagicMock:
    def _retrieve_for_cluster_text(
        event_type: str,
        top_classes: list[str],
        representative_text: str,
        *,
        top_k: int = 5,
        mitre_boost: float = 0.3,
    ) -> list[PlaybookChunk]:
        label = (top_classes[0] if top_classes else event_type).lower()
        if "credential" in label or "brute" in label:
            src, mitre = "credential_stuffing_response.md", ["T1110.004"]
        elif "dns" in label or "tunnel" in label:
            src, mitre = "dns_tunneling_response.md", ["T1071.004"]
        elif "container" in label or "escape" in label:
            src, mitre = "container_escape_response.md", ["T1611"]
        else:
            src, mitre = "insider_threat_detection.md", ["T1078"]
        return [
            PlaybookChunk(
                content=f"Guidance for {src}: block, investigate, recover.",
                source_file=src,
                playbook_name=src.replace(".md", "").replace("_", " "),
                section="Immediate Actions",
                mitre_techniques=mitre,
                score=0.92,
                file_hash="hash",
            )
        ]

    ret = MagicMock()
    ret.retrieve_for_cluster_text = MagicMock(side_effect=_retrieve_for_cluster_text)
    return ret


def _mock_llm() -> MagicMock:
    async def _generate_structured(*, system: str, user: str, schema: Any) -> Recommendation:
        u = user.lower()
        if "credential" in u:
            priority = Priority.HIGH
            title = "Credential stuffing campaign — block & rotate."
            mitre = ["T1110.004"]
        elif "dns" in u:
            priority = Priority.MEDIUM
            title = "Suspected DNS tunneling — sinkhole & investigate endpoint."
            mitre = ["T1071.004"]
        elif "container" in u:
            priority = Priority.CRITICAL
            title = "Container escape attempt — cordon node and forensic snapshot."
            mitre = ["T1611"]
        else:
            priority = Priority.LOW
            title = "Mixed cluster — manual review."
            mitre = []
        return Recommendation(
            summary=title + " Detailed summary exceeds ten characters.",
            threat_assessment=title + " Assessment follows the pattern.",
            relevant_playbooks=[],
            immediate_actions=[
                "Block source IPs at perimeter firewall",
                "Notify on-call SOC analyst",
            ],
            investigation_steps=[
                "Correlate with recent auth / system logs",
                "Confirm scope across unique IPs and users",
            ],
            mitre_techniques=mitre,
            priority=priority,
        )

    llm = MagicMock()
    llm.generate_structured = AsyncMock(side_effect=_generate_structured)
    llm.get_usage_stats = MagicMock(
        return_value=LLMUsage(
            model="mock-llm-v1",
            total_cost_usd=0.0035,
            total_tokens_in=1200,
            total_tokens_out=480,
        )
    )
    return llm


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_on_100_logs(self, tmp_path: Path) -> None:  # noqa: PLR0915
        logs_data, text_to_vec = _make_logs_and_vectors()
        assert len(logs_data) == 100

        jsonl = tmp_path / "logs.jsonl"
        jsonl.write_text(
            "\n".join(json.dumps(d) for d in logs_data) + "\n", encoding="utf-8"
        )
        load = load_logs(jsonl)
        assert load.parsed == 100
        assert load.skipped == 0
        assert load.source_format == "jsonl"

        embedder = _mock_embedder(text_to_vec)
        classifier = _mock_classifier()
        retriever = _mock_retriever()
        llm = _mock_llm()
        generator = RecommendationGenerator(llm, retriever)
        pipeline = SOCPipeline(
            embedder=embedder,
            clusterer=HDBSCANClusterer(min_cluster_size=5, min_samples=3),
            classifier=classifier,
            generator=generator,
        )

        progress_events: list[tuple[str, int, int]] = []
        result = await pipeline.process(
            load.logs,
            correlation_id="e2e-run",
            progress_callback=lambda s, c, t: progress_events.append((s, c, t)),
        )

        assert result.correlation_id == "e2e-run"
        assert result.stats.total_logs == 100
        assert result.stats.n_clusters == 3
        assert result.stats.n_noise >= 0
        assert len(result.incidents) == 3
        assert result.stats.llm_usage.total_cost_usd > 0

        stages = {s for s, _, _ in progress_events}
        assert stages >= {"embedding", "clustering", "classifying", "recommending"}

        incidents_by_type = {inc.dominant_event_type.value: inc for inc in result.incidents}
        assert set(incidents_by_type) == {"ids_alert", "network", "endpoint"}

        cred = incidents_by_type["ids_alert"]
        assert cred.cluster_size >= 40
        assert cred.dominant_classes[0].class_name == "credential_stuffing"
        assert cred.dominant_classes[0].count == 40
        assert cred.unique_src_ips > 1
        assert cred.recommendation.priority is Priority.HIGH
        assert cred.recommendation.relevant_playbooks == ["credential_stuffing_response.md"]
        assert cred.recommendation_unavailable is False

        dns = incidents_by_type["network"]
        assert dns.cluster_size >= 30
        assert dns.dominant_classes[0].count == 30
        assert dns.recommendation.priority is Priority.MEDIUM
        assert "dns_tunneling_response.md" in dns.recommendation.relevant_playbooks

        esc = incidents_by_type["endpoint"]
        assert esc.cluster_size >= 20
        assert esc.dominant_classes[0].count == 20
        assert esc.recommendation.priority is Priority.CRITICAL

        for inc in result.incidents:
            assert len(inc.representative_logs) >= 1

        json_out = tmp_path / "report.json"
        json_out.write_text(
            result.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
        )
        parsed = json.loads(json_out.read_text())
        assert parsed["correlation_id"] == "e2e-run"
        assert parsed["stats"]["n_clusters"] == 3
        assert parsed["incidents"][0]["dominant_classes"][0]["class"]

        md = format_report(result, input_name="logs.jsonl")
        md_out = tmp_path / "report.md"
        md_out.write_text(md, encoding="utf-8")
        assert "# SOC Incident Report" in md
        assert "🔴 Critical Incidents" in md
        assert "🟠 High Incidents" in md
        assert "🟡 Medium Incidents" in md
        assert "credential_stuffing_response.md" in md
        assert "container_escape_response.md" in md
        assert "<details>" in md
        assert "Cluster size:**" in md

    @pytest.mark.asyncio
    async def test_llm_failure_triggers_fallback(self, tmp_path: Path) -> None:
        logs_data, text_to_vec = _make_logs_and_vectors()

        embedder = _mock_embedder(text_to_vec)
        classifier = _mock_classifier()
        retriever = _mock_retriever()

        llm = MagicMock()
        llm.generate_structured = AsyncMock(side_effect=TimeoutError("provider down"))
        llm.get_usage_stats = MagicMock(
            return_value=LLMUsage(
                model="mock-llm-v1", total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0
            )
        )
        generator = RecommendationGenerator(llm, retriever)

        pipeline = SOCPipeline(
            embedder=embedder,
            clusterer=HDBSCANClusterer(min_cluster_size=5, min_samples=3),
            classifier=classifier,
            generator=generator,
        )

        from soc_agent.schemas import LogEntry

        logs = [LogEntry.model_validate(d) for d in logs_data]
        result = await pipeline.process(logs)

        assert len(result.incidents) == 3
        for inc in result.incidents:
            assert inc.recommendation_unavailable is True
            assert inc.fallback_reason is not None
            assert "TimeoutError" in inc.fallback_reason
            assert "Manual SOC review" in inc.recommendation.immediate_actions[0]

        md = format_report(result)
        assert "Recommendation unavailable" in md
        assert "placeholder recommendations" in md

    @pytest.mark.asyncio
    async def test_skip_recommendations_skips_llm(self) -> None:
        logs_data, text_to_vec = _make_logs_and_vectors()

        embedder = _mock_embedder(text_to_vec)
        classifier = _mock_classifier()
        llm = MagicMock()
        llm.generate_structured = AsyncMock()
        llm.get_usage_stats = MagicMock(
            return_value=LLMUsage(
                model="x", total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0
            )
        )
        retriever = _mock_retriever()
        generator = RecommendationGenerator(llm, retriever)
        pipeline = SOCPipeline(
            embedder=embedder,
            clusterer=HDBSCANClusterer(min_cluster_size=5, min_samples=3),
            classifier=classifier,
            generator=generator,
        )

        from soc_agent.schemas import LogEntry

        logs = [LogEntry.model_validate(d) for d in logs_data]
        result = await pipeline.process(logs, skip_recommendations=True)

        assert len(result.incidents) == 3
        for inc in result.incidents:
            assert inc.recommendation_unavailable is True
            assert inc.fallback_reason == "skipped_by_user"
        llm.generate_structured.assert_not_awaited()
        retriever.retrieve_for_cluster_text.assert_not_called()
