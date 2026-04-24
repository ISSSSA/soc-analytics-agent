from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from soc_agent.reports.markdown_formatter import format_report
from soc_agent.schemas import (
    DominantClass,
    EventType,
    IncidentReport,
    LLMUsage,
    LogEntry,
    PipelineResult,
    PipelineStats,
    Priority,
    Recommendation,
    SeverityLevel,
)


def _log(event_id: str = "evt-1") -> LogEntry:
    return LogEntry.model_validate(
        {
            "event_id": event_id,
            "timestamp": "2025-03-15T10:23:00Z",
            "event_type": "ids_alert",
            "severity": "high",
            "description": f"Failed SSH login — {event_id}",
            "src_ip": "203.0.113.42",
            "dst_ip": "10.0.0.5",
            "user": "admin",
            "action": "credential_stuffing",
        }
    )


def _rec(priority: Priority = Priority.HIGH, *, fallback: bool = False) -> Recommendation:
    return Recommendation(
        summary="Cluster summary that is plenty long to pass validation.",
        threat_assessment="Assessment content long enough.",
        relevant_playbooks=["credential_stuffing_response.md"],
        immediate_actions=["Block source IPs via ACL", "Force password reset"],
        investigation_steps=["Check SSH auth log", "Find successful logins"],
        mitre_techniques=["T1110.004"],
        priority=priority,
    )


def _incident(
    cluster_id: int,
    *,
    priority: Priority = Priority.HIGH,
    size: int = 10,
    fallback: bool = False,
    fallback_reason: str | None = None,
    event_type: str = "ids_alert",
    dominant_class: str | None = "credential_stuffing",
    representative: int = 2,
    **overrides: Any,
) -> IncidentReport:
    t0 = datetime(2025, 3, 15, 10, 23, tzinfo=UTC)
    t1 = datetime(2025, 3, 15, 10, 47, tzinfo=UTC)
    dc: list[DominantClass] = []
    if dominant_class:
        dc = [
            DominantClass(class_name=dominant_class, count=size, avg_confidence=0.87),
        ]
    data: dict[str, Any] = {
        "cluster_id": cluster_id,
        "cluster_size": size,
        "time_range": (t0, t1),
        "dominant_event_type": EventType(event_type),
        "dominant_classes": dc,
        "severity_breakdown": {SeverityLevel.CRITICAL: 4, SeverityLevel.HIGH: 6},
        "unique_src_ips": 12,
        "unique_users_targeted": 5,
        "representative_logs": [_log(f"evt-{cluster_id}-{i}") for i in range(representative)],
        "recommendation": _rec(priority),
        "recommendation_unavailable": fallback,
        "fallback_reason": fallback_reason,
    }
    data.update(overrides)
    return IncidentReport(**data)


def _result(
    incidents: list[IncidentReport],
    *,
    total_logs: int = 100,
    n_noise: int = 0,
    cache_hit: float = 0.25,
    cost: float = 0.12,
) -> PipelineResult:
    return PipelineResult(
        correlation_id="abc123",
        incidents=incidents,
        stats=PipelineStats(
            total_logs=total_logs,
            n_clusters=len(incidents),
            n_noise=n_noise,
            processing_time_seconds=90.0,
            cache_hit_rate=cache_hit,
            llm_usage=LLMUsage(
                model="anthropic/claude-haiku-4-5",
                total_cost_usd=cost,
                total_tokens_in=5000,
                total_tokens_out=2000,
            ),
        ),
    )


_FIXED_NOW = datetime(2026, 4, 24, 12, 30, 0)


class TestHeader:
    def test_includes_run_metadata(self) -> None:
        report = format_report(
            _result([_incident(0)], total_logs=250),
            input_name="batch.jsonl",
            generated_at=_FIXED_NOW,
        )
        assert "# SOC Incident Report" in report
        assert "2026-04-24 12:30:00" in report
        assert "abc123" in report
        assert "batch.jsonl" in report
        assert "250" in report
        assert "anthropic/claude-haiku-4-5" in report
        assert "25.0%" in report
        assert "$0.1200" in report

    def test_no_input_name_fallback(self) -> None:
        report = format_report(_result([_incident(0)]), generated_at=_FIXED_NOW)
        assert "Events analysed" in report
        assert "Input:" not in report


class TestSummaryTable:
    def test_priority_counts(self) -> None:
        incs = [
            _incident(1, priority=Priority.CRITICAL),
            _incident(2, priority=Priority.CRITICAL),
            _incident(3, priority=Priority.HIGH),
            _incident(4, priority=Priority.LOW),
        ]
        report = format_report(_result(incs), generated_at=_FIXED_NOW)
        assert "🔴 Critical | 2" in report
        assert "🟠 High | 1" in report
        assert "🔵 Low | 1" in report
        assert "🟡 Medium" not in report

    def test_noise_shown_when_present(self) -> None:
        report = format_report(
            _result([_incident(0)], n_noise=42), generated_at=_FIXED_NOW
        )
        assert "Noise (outlier points) | 42" in report

    def test_noise_hidden_when_zero(self) -> None:
        report = format_report(_result([_incident(0)]), generated_at=_FIXED_NOW)
        assert "Noise" not in report

    def test_fallback_callout(self) -> None:
        incs = [
            _incident(1, priority=Priority.HIGH, fallback=True, fallback_reason="timeout"),
        ]
        report = format_report(_result(incs), generated_at=_FIXED_NOW)
        assert "placeholder recommendations" in report

    def test_no_fallback_callout(self) -> None:
        report = format_report(_result([_incident(0)]), generated_at=_FIXED_NOW)
        assert "placeholder recommendations" not in report


class TestPrioritySections:
    def test_priority_ordering(self) -> None:
        incs = [
            _incident(1, priority=Priority.LOW),
            _incident(2, priority=Priority.CRITICAL),
            _incident(3, priority=Priority.MEDIUM),
            _incident(4, priority=Priority.HIGH),
        ]
        report = format_report(_result(incs), generated_at=_FIXED_NOW)
        critical_pos = report.index("🔴 Critical Incidents")
        high_pos = report.index("🟠 High Incidents")
        medium_pos = report.index("🟡 Medium Incidents")
        low_pos = report.index("🔵 Low Incidents")
        assert critical_pos < high_pos < medium_pos < low_pos

    def test_within_priority_sorted_by_size(self) -> None:
        incs = [
            _incident(1, priority=Priority.HIGH, size=5),
            _incident(2, priority=Priority.HIGH, size=50),
            _incident(3, priority=Priority.HIGH, size=20),
        ]
        report = format_report(_result(incs), generated_at=_FIXED_NOW)
        i1 = report.index("Incident #1 ")
        i2 = report.index("Incident #2 ")
        i3 = report.index("Incident #3 ")
        assert i2 < i3 < i1

    def test_empty_priority_omitted(self) -> None:
        incs = [_incident(0, priority=Priority.CRITICAL)]
        report = format_report(_result(incs), generated_at=_FIXED_NOW)
        assert "🟠 High Incidents" not in report
        assert "🔵 Low Incidents" not in report


class TestIncidentRender:
    def test_content(self) -> None:
        inc = _incident(
            42, priority=Priority.CRITICAL, size=15, dominant_class="credential_stuffing"
        )
        report = format_report(_result([inc]), generated_at=_FIXED_NOW)
        assert "Incident #42 — Credential Stuffing (15 events)" in report
        assert "Cluster size:** 15 events" in report
        assert "2025-03-15 10:23 → 10:47" in report
        assert "ids_alert / credential_stuffing (87% confidence)" in report
        assert "4 critical" in report
        assert "6 high" in report
        assert "T1110.004" in report
        assert "`credential_stuffing_response.md`" in report
        assert "**Summary:**" in report
        assert "**Threat assessment:**" in report
        assert "**Immediate actions:**" in report
        assert "1. Block source IPs via ACL" in report

    def test_representative_logs_collapsed(self) -> None:
        report = format_report(
            _result([_incident(0, size=12, representative=3)]),
            generated_at=_FIXED_NOW,
        )
        assert "<details>" in report
        assert "Representative logs (3 of 12)" in report
        assert "</details>" in report
        assert "evt-0-0" in report
        assert "evt-0-1" in report

    def test_fallback_warning(self) -> None:
        inc = _incident(
            7,
            priority=Priority.HIGH,
            fallback=True,
            fallback_reason="TimeoutError: llm slow",
        )
        report = format_report(_result([inc]), generated_at=_FIXED_NOW)
        assert "**Recommendation unavailable**" in report
        assert "TimeoutError" in report

    def test_no_dominant_class_uses_event_type(self) -> None:
        inc = _incident(
            9, priority=Priority.MEDIUM, dominant_class=None, event_type="network"
        )
        report = format_report(_result([inc]), generated_at=_FIXED_NOW)
        assert "Incident #9 — network (" in report


class TestEdgeCases:
    def test_empty_incidents(self) -> None:
        report = format_report(_result([]), generated_at=_FIXED_NOW)
        assert "# SOC Incident Report" in report
        assert "## Summary" in report
        assert "Total incidents** | **0**" in report

    def test_cross_day_time_range(self) -> None:
        inc = _incident(
            0,
            time_range=(
                datetime(2025, 3, 15, 23, 59, tzinfo=UTC),
                datetime(2025, 3, 16, 0, 30, tzinfo=UTC),
            ),
        )
        report = format_report(_result([inc]), generated_at=_FIXED_NOW)
        assert "2025-03-15 23:59 → 2025-03-16 00:30" in report
