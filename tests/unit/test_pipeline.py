from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from soc_agent.classification.classifier import ClassificationUnsupportedError
from soc_agent.clustering.clusterer import ClusterResult
from soc_agent.pipeline import (
    SOCPipeline,
    _aggregate_dominant_classes,
    _incident_skipped,
    _select_representative_logs,
)
from soc_agent.recommendations.generator import GenerationResult
from soc_agent.schemas import (
    ClassPrediction,
    ClusterStats,
    LLMUsage,
    LogEntry,
    Recommendation,
)


def _make_log(
    idx: int,
    *,
    event_type: str = "ids_alert",
    severity: str = "high",
    user: str = "admin",
    src_ip: str = "203.0.113.42",
    action: str = "credential_stuffing",
) -> LogEntry:
    t = datetime(2025, 3, 15, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=idx * 10)
    return LogEntry.model_validate(
        {
            "event_id": f"evt-{idx}",
            "timestamp": t.isoformat(),
            "event_type": event_type,
            "severity": severity,
            "description": f"description {idx}",
            "src_ip": src_ip,
            "user": user,
            "action": action,
        }
    )


def _fake_embeddings(n: int, dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _make_embedder(vectors: np.ndarray, hit_rate: float = 0.0) -> MagicMock:
    emb = MagicMock()
    emb.embed_texts = AsyncMock(return_value=vectors)
    stats = MagicMock()
    stats.hit_rate = hit_rate
    emb.stats = stats
    return emb


def _make_clusterer(result: ClusterResult) -> MagicMock:
    c = MagicMock()
    c.cluster = MagicMock(return_value=result)
    return c


def _valid_recommendation() -> Recommendation:
    return Recommendation.model_validate(
        {
            "summary": "Cluster summary placeholder, 10+ chars.",
            "threat_assessment": "Assessment placeholder, 10+ chars.",
            "relevant_playbooks": ["credential_stuffing_response.md"],
            "immediate_actions": ["Block IPs at perimeter firewall"],
            "investigation_steps": ["Correlate logins with SSH auth log"],
            "mitre_techniques": ["T1110.004"],
            "priority": "high",
        }
    )


class TestAggregateDominantClasses:
    def test_empty(self) -> None:
        assert _aggregate_dominant_classes([]) == []

    def test_top_k_by_count(self) -> None:
        preds = [
            ClassPrediction(label="a", score=0.9, top_k=[]),
            ClassPrediction(label="a", score=0.8, top_k=[]),
            ClassPrediction(label="b", score=0.7, top_k=[]),
            ClassPrediction(label="c", score=0.6, top_k=[]),
            ClassPrediction(label="c", score=0.5, top_k=[]),
            ClassPrediction(label="c", score=0.4, top_k=[]),
        ]
        dcs = _aggregate_dominant_classes(preds, k=2)
        assert [dc.class_name for dc in dcs] == ["c", "a"]
        assert dcs[0].count == 3
        assert dcs[0].avg_confidence == pytest.approx(0.5)
        assert dcs[1].count == 2
        assert dcs[1].avg_confidence == pytest.approx(0.85)


class TestSelectRepresentativeLogs:
    def test_picks_closest_to_centroid(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        logs = [_make_log(i) for i in range(4)]
        reps = _select_representative_logs(X, [0, 1, 2, 3], logs, count=3)
        assert len(reps) == 3
        rep_ids = [log.event_id for log in reps]
        assert "evt-3" not in rep_ids

    def test_empty_indices(self) -> None:
        X = _fake_embeddings(0)
        assert _select_representative_logs(X, [], [], count=3) == []


def _cluster_result(labels: list[int]) -> ClusterResult:
    from collections import defaultdict

    buckets: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[lab].append(i)
    clusters = [
        ClusterStats(cluster_id=cid, size=len(idx), log_indices=idx)
        for cid, idx in sorted(buckets.items())
    ]
    n_clusters = sum(1 for c in clusters if c.cluster_id >= 0)
    n_noise = sum(c.size for c in clusters if c.cluster_id == -1)
    return ClusterResult(
        labels=np.asarray(labels, dtype=np.int64),
        probabilities=np.ones(len(labels)),
        clusters=clusters,
        n_clusters=n_clusters,
        n_noise=n_noise,
    )


class TestPipelineHappyPath:
    @pytest.mark.asyncio
    async def test_full_flow(self) -> None:
        logs = [_make_log(i) for i in range(10)]
        embeddings = _fake_embeddings(10)
        cr = _cluster_result([0] * 5 + [1] * 5)

        embedder = _make_embedder(embeddings, hit_rate=0.5)
        clusterer = _make_clusterer(cr)
        classifier = MagicMock()
        classifier.classify_texts = AsyncMock(
            return_value=[ClassPrediction(label="credential_stuffing", score=0.9, top_k=[])] * 5
        )
        generator = MagicMock()
        generator.generate = AsyncMock(
            return_value=GenerationResult(
                recommendation=_valid_recommendation(),
                playbook_chunks=[],
            )
        )
        generator.get_llm_usage_stats = MagicMock(
            return_value=LLMUsage(
                model="anthropic/claude-haiku-4-5",
                total_cost_usd=0.01,
                total_tokens_in=1000,
                total_tokens_out=200,
            )
        )

        pipeline = SOCPipeline(
            embedder=embedder,
            clusterer=clusterer,
            classifier=classifier,
            generator=generator,
        )
        result = await pipeline.process(logs)

        assert len(result.incidents) == 2
        assert result.stats.total_logs == 10
        assert result.stats.n_clusters == 2
        assert result.stats.n_noise == 0
        assert result.stats.cache_hit_rate == 0.5
        assert result.stats.llm_usage.total_cost_usd == 0.01
        assert result.correlation_id

        for inc in result.incidents:
            assert inc.recommendation_unavailable is False
            assert inc.recommendation.priority.value == "high"
            assert inc.cluster_size == 5
            assert len(inc.representative_logs) > 0

    @pytest.mark.asyncio
    async def test_empty_logs(self) -> None:
        embedder = _make_embedder(np.empty((0, 4)))
        clusterer = _make_clusterer(
            ClusterResult(
                labels=np.empty(0, dtype=np.int64),
                probabilities=np.empty(0),
                clusters=[],
                n_clusters=0,
                n_noise=0,
            )
        )
        pipeline = SOCPipeline(embedder=embedder, clusterer=clusterer)
        result = await pipeline.process([])
        assert result.incidents == []
        assert result.stats.total_logs == 0

    @pytest.mark.asyncio
    async def test_dominant_classes_aggregated(self) -> None:
        logs = [_make_log(i) for i in range(6)]
        embeddings = _fake_embeddings(6)
        cr = _cluster_result([0] * 6)

        classifier = MagicMock()
        classifier.classify_texts = AsyncMock(
            return_value=[
                ClassPrediction(label="credential_stuffing", score=0.9, top_k=[]),
                ClassPrediction(label="credential_stuffing", score=0.85, top_k=[]),
                ClassPrediction(label="credential_stuffing", score=0.8, top_k=[]),
                ClassPrediction(label="brute_force", score=0.7, top_k=[]),
                ClassPrediction(label="brute_force", score=0.6, top_k=[]),
                ClassPrediction(label="reconnaissance", score=0.5, top_k=[]),
            ]
        )
        embedder = _make_embedder(embeddings)
        clusterer = _make_clusterer(cr)
        pipeline = SOCPipeline(
            embedder=embedder,
            clusterer=clusterer,
            classifier=classifier,
        )
        result = await pipeline.process(logs, skip_recommendations=True)

        inc = result.incidents[0]
        assert inc.dominant_classes[0].class_name == "credential_stuffing"
        assert inc.dominant_classes[0].count == 3
        assert inc.dominant_classes[0].avg_confidence == pytest.approx(0.85)
        assert inc.dominant_classes[1].class_name == "brute_force"
        assert inc.dominant_classes[1].count == 2


class TestPipelineDegradation:
    @pytest.mark.asyncio
    async def test_skip_recommendations(self) -> None:
        logs = [_make_log(i) for i in range(5)]
        embeddings = _fake_embeddings(5)
        cr = _cluster_result([0] * 5)

        pipeline = SOCPipeline(
            embedder=_make_embedder(embeddings),
            clusterer=_make_clusterer(cr),
        )
        result = await pipeline.process(logs, skip_recommendations=True)

        assert len(result.incidents) == 1
        inc = result.incidents[0]
        assert inc.recommendation_unavailable is True
        assert inc.fallback_reason == "skipped_by_user"
        assert "skipped" in inc.recommendation.summary.lower()

    @pytest.mark.asyncio
    async def test_no_classifier(self) -> None:
        logs = [_make_log(i) for i in range(5)]
        cr = _cluster_result([0] * 5)
        pipeline = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
            classifier=None,
        )
        result = await pipeline.process(logs, skip_recommendations=True)
        assert result.incidents[0].dominant_classes == []

    @pytest.mark.asyncio
    async def test_classifier_unsupported_disables_for_run(self) -> None:
        logs = [_make_log(i) for i in range(6)]
        cr = _cluster_result([0] * 3 + [1] * 3)

        classifier = MagicMock()
        classifier.classify_texts = AsyncMock(
            side_effect=ClassificationUnsupportedError("no head")
        )

        pipeline = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(6)),
            clusterer=_make_clusterer(cr),
            classifier=classifier,
        )
        result = await pipeline.process(logs, skip_recommendations=True)

        for inc in result.incidents:
            assert inc.dominant_classes == []
        assert classifier.classify_texts.await_count == 1

    @pytest.mark.asyncio
    async def test_generator_fallback_sets_unavailable_flag(self) -> None:
        logs = [_make_log(i) for i in range(5)]
        cr = _cluster_result([0] * 5)

        generator = MagicMock()
        generator.generate = AsyncMock(
            return_value=GenerationResult(
                recommendation=_valid_recommendation(),
                playbook_chunks=[],
                fallback_reason="TimeoutError: slow",
            )
        )
        generator.get_llm_usage_stats = MagicMock(
            return_value=LLMUsage(
                model="x", total_cost_usd=0.0, total_tokens_in=0, total_tokens_out=0
            )
        )
        pipeline = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
            generator=generator,
        )
        result = await pipeline.process(logs)
        inc = result.incidents[0]
        assert inc.recommendation_unavailable is True
        assert inc.fallback_reason == "TimeoutError: slow"

    @pytest.mark.asyncio
    async def test_include_noise_flag(self) -> None:
        logs = [_make_log(i) for i in range(5)]
        cr = _cluster_result([0, 0, 0, -1, -1])

        p_no_noise = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
            include_noise=False,
        )
        r1 = await p_no_noise.process(logs, skip_recommendations=True)
        assert len(r1.incidents) == 1
        assert r1.stats.n_noise == 2

        p_noise = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
            include_noise=True,
        )
        r2 = await p_noise.process(logs, skip_recommendations=True)
        assert len(r2.incidents) == 2
        noise_inc = next(i for i in r2.incidents if i.cluster_id == -1)
        assert noise_inc.cluster_size == 2


class TestPipelineProgress:
    @pytest.mark.asyncio
    async def test_progress_callback_receives_stages(self) -> None:
        logs = [_make_log(i) for i in range(5)]
        cr = _cluster_result([0] * 5)
        pipeline = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
        )
        events: list[tuple[str, int, int]] = []
        await pipeline.process(
            logs,
            skip_recommendations=True,
            progress_callback=lambda s, c, t: events.append((s, c, t)),
        )
        stages = {stage for stage, _, _ in events}
        assert "embedding" in stages
        assert "clustering" in stages

    @pytest.mark.asyncio
    async def test_progress_callback_exception_does_not_break(self) -> None:
        def _bad_cb(_s: str, _c: int, _t: int) -> None:
            raise RuntimeError("boom")

        logs = [_make_log(i) for i in range(5)]
        cr = _cluster_result([0] * 5)
        pipeline = SOCPipeline(
            embedder=_make_embedder(_fake_embeddings(5)),
            clusterer=_make_clusterer(cr),
        )
        result = await pipeline.process(
            logs, skip_recommendations=True, progress_callback=_bad_cb
        )
        assert len(result.incidents) == 1


class TestIncidentSkipped:
    def test_priority_from_severity(self) -> None:
        from soc_agent.schemas import Cluster

        cluster = Cluster.model_validate(
            {
                "cluster_id": 1,
                "cluster_size": 3,
                "time_range": (
                    datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2025, 1, 1, 1, tzinfo=UTC),
                ),
                "dominant_event_type": "ids_alert",
                "dominant_classes": [],
                "severity_breakdown": {"critical": 3},
                "unique_src_ips": 3,
                "unique_users_targeted": 1,
                "representative_logs": [],
                "all_log_indices": [0, 1, 2],
            }
        )
        incident = _incident_skipped(cluster)
        assert incident.recommendation.priority.value == "critical"
        assert incident.recommendation_unavailable is True
