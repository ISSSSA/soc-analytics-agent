from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from soc_agent.schemas import (
    DominantClass,
    LogEntry,
    PlaybookChunk,
    Priority,
    SeverityLevel,
)

if TYPE_CHECKING:
    from soc_agent.schemas import Cluster


SYSTEM_PROMPT = """\
You are a senior SOC analyst with 10+ years of experience in threat hunting
and incident response.

Your task is to analyse a cluster of related security events and generate
structured, actionable recommendations based on the provided SOC playbooks.

CORE RULES:
1. Ground all recommendations in the provided logs and playbooks. Do NOT
   invent IoCs, threat-actor attributions, or MITRE techniques that are not
   present in the data.
2. Be specific. "Block malicious IPs" is vague; "Block source IPs
   [a.b.c.d, e.f.g.h] at the perimeter firewall via ACL rule" is actionable.
3. Immediate actions must be executable within 15 minutes by a Tier 1
   analyst — no vague "investigate further".
4. Investigation steps must be queries or checks that produce answers, not
   open-ended directives.
5. Cite relevant playbook sections by name when applicable.
6. Assign priority from the severity of events, the scope (unique IPs,
   unique users), and the presence of post-exploitation indicators.

OUTPUT: Valid JSON matching the Recommendation schema. No prose outside JSON.
"""


USER_PROMPT_TEMPLATE = """\
# Incident Cluster Analysis

## Cluster Overview
- **Cluster ID:** {cluster_id}
- **Size:** {cluster_size} events
- **Time range:** {time_start} → {time_end} (span: {time_span})
- **Dominant event type:** {event_type}
- **Unique source IPs:** {unique_ips}
- **Unique users affected:** {unique_users}

## Classification Results

{classifications_table}

## Severity Distribution

{severity_breakdown}

## Representative Events (most typical in cluster)

{sample_logs_block}

## Relevant SOC Playbooks (retrieved via semantic search)

{playbooks_block}

---

Generate a structured incident recommendation focused on the SPECIFIC activity
observed in these events. Prioritise actions that address the actual threat
pattern, not generic advice.
"""


SEVERITY_RANK: dict[SeverityLevel, int] = {
    SeverityLevel.INFO: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
    SeverityLevel.EMERGENCY: 5,
}


def max_severity(breakdown: dict[SeverityLevel, int]) -> SeverityLevel:
    present = [s for s, count in breakdown.items() if count > 0]
    if not present:
        return SeverityLevel.LOW
    return max(present, key=lambda s: SEVERITY_RANK.get(s, 0))


def severity_to_priority(sev: SeverityLevel) -> Priority:
    if sev in (SeverityLevel.EMERGENCY, SeverityLevel.CRITICAL):
        return Priority.CRITICAL
    if sev == SeverityLevel.HIGH:
        return Priority.HIGH
    if sev == SeverityLevel.MEDIUM:
        return Priority.MEDIUM
    if sev == SeverityLevel.LOW:
        return Priority.LOW
    return Priority.INFORMATIONAL


def format_time_span(t0: datetime, t1: datetime) -> str:
    delta = t1 - t0
    secs = int(max(delta.total_seconds(), 0))
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    if secs < 86400:
        return f"{secs // 3600}h {(secs % 3600) // 60}m"
    return f"{secs // 86400}d {(secs % 86400) // 3600}h"


def format_classifications_table(dominant_classes: list[DominantClass]) -> str:
    header = "| Class | Count | Avg Confidence |\n|-------|-------|----------------|"
    if not dominant_classes:
        return header + "\n| (no classifications available) | — | — |"
    rows = [
        f"| {dc.class_name} | {dc.count} | {dc.avg_confidence:.2f} |"
        for dc in dominant_classes
    ]
    return "\n".join([header, *rows])


def format_severity_breakdown(breakdown: dict[SeverityLevel, int]) -> str:
    if not breakdown:
        return "- (no severity data)"
    items = sorted(
        (item for item in breakdown.items() if item[1] > 0),
        key=lambda x: SEVERITY_RANK.get(x[0], 0),
        reverse=True,
    )
    return "\n".join(f"- **{sev.value}**: {count}" for sev, count in items)


def format_sample_logs(logs: list[LogEntry], *, max_count: int = 3) -> str:
    if not logs:
        return "_(no representative logs available)_"
    parts: list[str] = []
    for i, log in enumerate(logs[:max_count], start=1):
        parts.append(
            f"### Event {i}\n"
            f"- Time: {log.timestamp.isoformat()}\n"
            f"- Event type: {log.event_type.value}\n"
            f"- Severity: {log.severity.value}\n"
            f"- Description: {log.description}\n"
            f"- Source IP: {log.src_ip or 'n/a'}\n"
            f"- Destination IP: {log.dst_ip or 'n/a'}\n"
            f"- User: {log.user or 'n/a'}\n"
            f"- Action: {log.action or 'n/a'}"
        )
    return "\n\n".join(parts)


def format_playbooks_block(
    chunks: list[PlaybookChunk],
    *,
    max_chars_per_chunk: int = 500,
) -> str:
    if not chunks:
        return "_(no playbooks retrieved)_"
    parts: list[str] = []
    for chunk in chunks:
        section = chunk.section or "—"
        content = chunk.content.strip()
        if len(content) > max_chars_per_chunk:
            content = content[:max_chars_per_chunk].rstrip() + "…"
        parts.append(
            f"### Playbook: {chunk.source_file} — "
            f"Section: {section} (relevance: {chunk.score:.2f})\n"
            f"{content}\n\n---"
        )
    return "\n".join(parts)


def build_user_prompt(cluster: Cluster, playbook_chunks: list[PlaybookChunk]) -> str:
    """Render USER_PROMPT_TEMPLATE with the cluster + retrieved playbooks."""
    t0, t1 = cluster.time_range
    return USER_PROMPT_TEMPLATE.format(
        cluster_id=cluster.cluster_id,
        cluster_size=cluster.cluster_size,
        time_start=t0.isoformat(),
        time_end=t1.isoformat(),
        time_span=format_time_span(t0, t1),
        event_type=cluster.dominant_event_type.value,
        unique_ips=cluster.unique_src_ips,
        unique_users=cluster.unique_users_targeted,
        classifications_table=format_classifications_table(cluster.dominant_classes),
        severity_breakdown=format_severity_breakdown(cluster.severity_breakdown),
        sample_logs_block=format_sample_logs(cluster.representative_logs),
        playbooks_block=format_playbooks_block(playbook_chunks),
    )
