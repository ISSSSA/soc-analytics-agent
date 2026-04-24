from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from soc_agent.rag.retriever import PlaybookRetriever
from soc_agent.recommendations.llm_client import LLMClient
from soc_agent.recommendations.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    max_severity,
    severity_to_priority,
)
from soc_agent.schemas import Cluster, LLMUsage, PlaybookChunk, Recommendation

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    recommendation: Recommendation
    playbook_chunks: list[PlaybookChunk]
    fallback_reason: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.fallback_reason is not None


class RecommendationGenerator:
    """Generate a structured Recommendation per cluster, with safe fallback on LLM failure."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: PlaybookRetriever,
        *,
        top_k_playbooks: int = 5,
        mitre_boost: float = 0.3,
    ) -> None:
        self._llm = llm_client
        self._retriever = retriever
        self._top_k = top_k_playbooks
        self._mitre_boost = mitre_boost

    def get_llm_usage_stats(self) -> LLMUsage:
        return self._llm.get_usage_stats()

    async def generate(self, cluster: Cluster) -> GenerationResult:
        """Retrieve playbooks, call LLM, return GenerationResult; never raises on LLM error."""
        chunks = await self._retrieve(cluster)

        try:
            rec = await self._llm.generate_structured(
                system=SYSTEM_PROMPT,
                user=build_user_prompt(cluster, chunks),
                schema=Recommendation,
            )
        except Exception as e:
            logger.warning(
                "LLM generation failed for cluster %d: %s: %s",
                cluster.cluster_id,
                e.__class__.__name__,
                e,
            )
            fallback = _fallback_recommendation(cluster, e)
            return GenerationResult(
                recommendation=fallback,
                playbook_chunks=chunks,
                fallback_reason=f"{e.__class__.__name__}: {e}",
            )

        retrieved_sources = _unique_source_files(chunks)
        if retrieved_sources:
            rec = rec.model_copy(update={"relevant_playbooks": retrieved_sources})

        return GenerationResult(recommendation=rec, playbook_chunks=chunks)

    async def _retrieve(self, cluster: Cluster) -> list[PlaybookChunk]:
        top_classes = [dc.class_name for dc in cluster.dominant_classes[:3]]
        rep_text = (
            cluster.representative_logs[0].description
            if cluster.representative_logs
            else ""
        )
        return await asyncio.to_thread(
            self._retriever.retrieve_for_cluster_text,
            cluster.dominant_event_type.value,
            top_classes,
            rep_text,
            top_k=self._top_k,
            mitre_boost=self._mitre_boost,
        )


def _unique_source_files(chunks: list[PlaybookChunk]) -> list[str]:
    seen: dict[str, None] = {}
    for c in chunks:
        if c.source_file:
            seen.setdefault(c.source_file, None)
    return list(seen.keys())


def _fallback_recommendation(cluster: Cluster, error: BaseException) -> Recommendation:
    sev = max_severity(cluster.severity_breakdown)
    priority = severity_to_priority(sev)
    err_name = error.__class__.__name__

    return Recommendation(
        summary=(
            f"Automated recommendation is unavailable — LLM generation failed "
            f"({err_name}). Cluster of {cluster.cluster_size} "
            f"{cluster.dominant_event_type.value} events (max severity: "
            f"{sev.value}) requires manual SOC review."
        ),
        threat_assessment=(
            f"Threat assessment could not be generated automatically. "
            f"The cluster contains {cluster.cluster_size} events spanning "
            f"{cluster.unique_src_ips} unique source IP(s) and affecting "
            f"{cluster.unique_users_targeted} user(s). Severity reaches "
            f"{sev.value}."
        ),
        relevant_playbooks=[],
        immediate_actions=[
            "Manual SOC review required — automated LLM generation failed",
            "Preserve representative logs and cluster metadata for analyst triage",
        ],
        investigation_steps=[
            "Inspect the representative events in this cluster for common indicators",
            "Determine whether the activity is ongoing or already contained",
            "Escalate based on observed business impact and data sensitivity",
        ],
        mitre_techniques=[],
        priority=priority,
    )
