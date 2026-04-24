from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from soc_agent.classification.classifier import (
    ClassificationUnsupportedError,
    ClassifierClient,
)
from soc_agent.clustering.clusterer import HDBSCANClusterer
from soc_agent.clustering.embedder import EmbedderClient
from soc_agent.recommendations.generator import (
    GenerationResult,
    RecommendationGenerator,
)
from soc_agent.recommendations.prompts import max_severity, severity_to_priority
from soc_agent.schemas import (
    ClassPrediction,
    Cluster,
    ClusterStats,
    DominantClass,
    EventType,
    IncidentReport,
    LLMUsage,
    LogEntry,
    PipelineResult,
    PipelineStats,
    Recommendation,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class _Dependencies:
    embedder: EmbedderClient
    clusterer: HDBSCANClusterer
    classifier: ClassifierClient | None
    generator: RecommendationGenerator | None


class SOCPipeline:
    """End-to-end SOC pipeline orchestrator: embed → cluster → classify → recommend."""

    def __init__(
        self,
        *,
        embedder: EmbedderClient,
        clusterer: HDBSCANClusterer,
        classifier: ClassifierClient | None = None,
        generator: RecommendationGenerator | None = None,
        include_noise: bool = False,
        representative_count: int = 3,
    ) -> None:
        self._deps = _Dependencies(
            embedder=embedder,
            clusterer=clusterer,
            classifier=classifier,
            generator=generator,
        )
        self._include_noise = include_noise
        self._representative_count = representative_count

    async def process(
        self,
        logs: list[LogEntry],
        *,
        correlation_id: str | None = None,
        skip_recommendations: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Run the full pipeline (embed → cluster → classify → recommend) on `logs`."""
        cid = correlation_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()

        if not logs:
            logger.warning("process() called with zero logs; returning empty result")
            return self._empty_result(cid, started)

        self._report(progress_callback, "embedding", 0, len(logs))
        embeddings = await self._deps.embedder.embed_texts(
            [log.description for log in logs]
        )
        self._report(progress_callback, "embedding", len(logs), len(logs))

        self._report(progress_callback, "clustering", 0, 1)
        cluster_result = self._deps.clusterer.cluster(embeddings)
        self._report(progress_callback, "clustering", 1, 1)

        kept = [
            cs for cs in cluster_result.clusters
            if cs.cluster_id >= 0 or self._include_noise
        ]

        classifications = await self._classify_all(kept, logs, progress_callback)

        clusters = [
            self._build_cluster(logs, embeddings, cs, classifications.get(cs.cluster_id, []))
            for cs in kept
        ]

        incidents = await self._generate_all(clusters, skip_recommendations, progress_callback)

        elapsed = time.perf_counter() - started
        stats = PipelineStats(
            total_logs=len(logs),
            n_clusters=cluster_result.n_clusters,
            n_noise=cluster_result.n_noise,
            processing_time_seconds=elapsed,
            cache_hit_rate=self._deps.embedder.stats.hit_rate,
            llm_usage=self._llm_usage(),
        )
        return PipelineResult(correlation_id=cid, incidents=incidents, stats=stats)

    async def _classify_all(
        self,
        clusters: list[ClusterStats],
        logs: list[LogEntry],
        progress_callback: ProgressCallback | None,
    ) -> dict[int, list[ClassPrediction]]:
        """Classify per cluster sequentially; first 501 disables classification for the rest."""
        results: dict[int, list[ClassPrediction]] = {}
        if self._deps.classifier is None or not clusters:
            return results

        classifier_disabled = False
        for i, cs in enumerate(clusters):
            if classifier_disabled:
                results[cs.cluster_id] = []
                continue
            texts = [logs[idx].description for idx in cs.log_indices]
            try:
                preds = await self._deps.classifier.classify_texts(texts)
                results[cs.cluster_id] = preds
            except ClassificationUnsupportedError:
                logger.warning(
                    "Inference service reports no classifier head — "
                    "disabling classification for this run."
                )
                classifier_disabled = True
                results[cs.cluster_id] = []
            self._report(
                progress_callback, "classifying", i + 1, len(clusters)
            )
        return results

    async def _generate_all(
        self,
        clusters: list[Cluster],
        skip: bool,
        progress_callback: ProgressCallback | None,
    ) -> list[IncidentReport]:
        if not clusters:
            return []

        if skip or self._deps.generator is None:
            return [_incident_skipped(c) for c in clusters]

        generator = self._deps.generator
        total = len(clusters)
        done = 0

        async def _one(cluster: Cluster) -> IncidentReport:
            nonlocal done
            result: GenerationResult = await generator.generate(cluster)
            done += 1
            self._report(progress_callback, "recommending", done, total)
            return _incident_from_result(cluster, result)

        return list(await asyncio.gather(*(_one(c) for c in clusters)))

    def _build_cluster(
        self,
        all_logs: list[LogEntry],
        embeddings: np.ndarray,
        stats: ClusterStats,
        classifications: list[ClassPrediction],
    ) -> Cluster:
        """Aggregate per-log data into a Cluster: time range, dominants, severity, IPs, users."""
        cluster_logs = [all_logs[i] for i in stats.log_indices]

        timestamps = [log.timestamp for log in cluster_logs]
        t_min, t_max = min(timestamps), max(timestamps)

        event_counts: Counter[EventType] = Counter(log.event_type for log in cluster_logs)
        dominant_event_type = event_counts.most_common(1)[0][0]

        severity_counts: dict[SeverityLevel, int] = {}
        for log in cluster_logs:
            severity_counts[log.severity] = severity_counts.get(log.severity, 0) + 1

        unique_src_ips = len({log.src_ip for log in cluster_logs if log.src_ip is not None})
        unique_users = len({log.user for log in cluster_logs if log.user is not None})

        representative_logs = _select_representative_logs(
            embeddings,
            stats.log_indices,
            all_logs,
            count=self._representative_count,
        )
        dominant_classes = _aggregate_dominant_classes(classifications, k=3)

        return Cluster(
            cluster_id=stats.cluster_id,
            cluster_size=stats.size,
            time_range=(t_min, t_max),
            dominant_event_type=dominant_event_type,
            dominant_classes=dominant_classes,
            severity_breakdown=severity_counts,
            unique_src_ips=unique_src_ips,
            unique_users_targeted=unique_users,
            representative_logs=representative_logs,
            all_log_indices=list(stats.log_indices),
        )

    @staticmethod
    def _report(
        cb: ProgressCallback | None,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        if cb is None:
            return
        try:
            cb(stage, current, total)
        except Exception:
            logger.debug("progress_callback raised; ignoring", exc_info=True)

    def _llm_usage(self) -> LLMUsage:
        if self._deps.generator is None:
            return LLMUsage(
                model="(skipped)",
                total_cost_usd=0.0,
                total_tokens_in=0,
                total_tokens_out=0,
            )
        return self._deps.generator.get_llm_usage_stats()

    def _empty_result(self, cid: str, started: float) -> PipelineResult:
        elapsed = time.perf_counter() - started
        return PipelineResult(
            correlation_id=cid,
            incidents=[],
            stats=PipelineStats(
                total_logs=0,
                n_clusters=0,
                n_noise=0,
                processing_time_seconds=elapsed,
                cache_hit_rate=self._deps.embedder.stats.hit_rate,
                llm_usage=self._llm_usage(),
            ),
        )


def _select_representative_logs(
    embeddings: np.ndarray,
    log_indices: list[int],
    all_logs: list[LogEntry],
    *,
    count: int,
) -> list[LogEntry]:
    """Pick up to `count` logs closest to the cluster centroid by cosine similarity."""
    if not log_indices:
        return []
    idx_arr = np.asarray(log_indices, dtype=np.int64)
    cluster_emb = embeddings[idx_arr]
    centroid = cluster_emb.mean(axis=0)
    similarities = cluster_emb @ centroid
    top = np.argsort(-similarities)[:count]
    return [all_logs[int(idx_arr[i])] for i in top]


def _aggregate_dominant_classes(
    classifications: list[ClassPrediction],
    *,
    k: int = 3,
) -> list[DominantClass]:
    """Group predictions by label and return the top-`k` by count, with mean score."""
    if not classifications:
        return []
    by_label: dict[str, list[float]] = {}
    for pred in classifications:
        by_label.setdefault(pred.label, []).append(pred.score)

    items = [
        DominantClass(
            class_name=label,
            count=len(scores),
            avg_confidence=sum(scores) / len(scores),
        )
        for label, scores in by_label.items()
    ]
    items.sort(key=lambda dc: dc.count, reverse=True)
    return items[:k]


def _incident_from_result(cluster: Cluster, result: GenerationResult) -> IncidentReport:
    return IncidentReport(
        cluster_id=cluster.cluster_id,
        cluster_size=cluster.cluster_size,
        time_range=cluster.time_range,
        dominant_event_type=cluster.dominant_event_type,
        dominant_classes=cluster.dominant_classes,
        severity_breakdown=cluster.severity_breakdown,
        unique_src_ips=cluster.unique_src_ips,
        unique_users_targeted=cluster.unique_users_targeted,
        representative_logs=cluster.representative_logs,
        recommendation=result.recommendation,
        recommendation_unavailable=result.used_fallback,
        fallback_reason=result.fallback_reason,
    )


def _incident_skipped(cluster: Cluster) -> IncidentReport:
    sev = max_severity(cluster.severity_breakdown)
    priority = severity_to_priority(sev)
    placeholder = Recommendation(
        summary=(
            f"Recommendation generation was skipped for this cluster of "
            f"{cluster.cluster_size} {cluster.dominant_event_type.value} events."
        ),
        threat_assessment=(
            "Not assessed — run without --skip-recommendations to generate "
            "automated analyst guidance."
        ),
        relevant_playbooks=[],
        immediate_actions=["Re-run pipeline without --skip-recommendations"],
        investigation_steps=["Review representative logs and cluster metadata"],
        mitre_techniques=[],
        priority=priority,
    )
    return IncidentReport(
        cluster_id=cluster.cluster_id,
        cluster_size=cluster.cluster_size,
        time_range=cluster.time_range,
        dominant_event_type=cluster.dominant_event_type,
        dominant_classes=cluster.dominant_classes,
        severity_breakdown=cluster.severity_breakdown,
        unique_src_ips=cluster.unique_src_ips,
        unique_users_targeted=cluster.unique_users_targeted,
        representative_logs=cluster.representative_logs,
        recommendation=placeholder,
        recommendation_unavailable=True,
        fallback_reason="skipped_by_user",
    )
