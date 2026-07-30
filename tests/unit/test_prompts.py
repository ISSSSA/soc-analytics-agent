from __future__ import annotations

from datetime import UTC, datetime

from soc_agent.recommendations.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_classifications_table,
    format_playbooks_block,
    format_sample_logs,
    format_severity_breakdown,
    format_time_span,
    max_severity,
    severity_to_priority,
)
from soc_agent.schemas import (
    Cluster,
    DominantClass,
    LogEntry,
    PlaybookChunk,
    Priority,
    SeverityLevel,
)


class TestStaticPrompts:
    def test_system_prompt_contains_core_rules(self) -> None:
        assert "SOC analyst" in SYSTEM_PROMPT
        assert "CORE RULES" in SYSTEM_PROMPT
        assert "Ground all recommendations" in SYSTEM_PROMPT
        assert "15 minutes" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT


class TestSeverityHelpers:
    def test_max_severity_picks_highest(self) -> None:
        bd = {
            SeverityLevel.LOW: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 1,
        }
        assert max_severity(bd) is SeverityLevel.CRITICAL

    def test_max_severity_ignores_zero_counts(self) -> None:
        bd = {SeverityLevel.CRITICAL: 0, SeverityLevel.MEDIUM: 5}
        assert max_severity(bd) is SeverityLevel.MEDIUM

    def test_max_severity_empty_defaults_low(self) -> None:
        assert max_severity({}) is SeverityLevel.LOW

    def test_severity_to_priority_mapping(self) -> None:
        assert severity_to_priority(SeverityLevel.EMERGENCY) is Priority.CRITICAL
        assert severity_to_priority(SeverityLevel.CRITICAL) is Priority.CRITICAL
        assert severity_to_priority(SeverityLevel.HIGH) is Priority.HIGH
        assert severity_to_priority(SeverityLevel.MEDIUM) is Priority.MEDIUM
        assert severity_to_priority(SeverityLevel.LOW) is Priority.LOW
        assert severity_to_priority(SeverityLevel.INFO) is Priority.INFORMATIONAL


class TestTimeSpan:
    def test_seconds(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        t1 = datetime(2025, 1, 1, 0, 0, 42, tzinfo=UTC)
        assert format_time_span(t0, t1) == "42s"

    def test_minutes(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        t1 = datetime(2025, 1, 1, 0, 5, 30, tzinfo=UTC)
        assert format_time_span(t0, t1) == "5m 30s"

    def test_hours(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        t1 = datetime(2025, 1, 1, 2, 15, 0, tzinfo=UTC)
        assert format_time_span(t0, t1) == "2h 15m"

    def test_days(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        t1 = datetime(2025, 1, 3, 6, 0, 0, tzinfo=UTC)
        assert format_time_span(t0, t1) == "2d 6h"

    def test_zero_or_negative(self) -> None:
        t0 = datetime(2025, 1, 1, tzinfo=UTC)
        assert format_time_span(t0, t0) == "0s"


class TestClassificationsTable:
    def test_renders_rows(self) -> None:
        dcs = [
            DominantClass.model_validate(
                {"class": "credential_stuffing", "count": 12, "avg_confidence": 0.87}
            ),
            DominantClass.model_validate(
                {"class": "brute_force", "count": 3, "avg_confidence": 0.72}
            ),
        ]
        out = format_classifications_table(dcs)
        assert "| Class | Count | Avg Confidence |" in out
        assert "credential_stuffing" in out
        assert "0.87" in out

    def test_empty_placeholder(self) -> None:
        out = format_classifications_table([])
        assert "(no classifications available)" in out


class TestSeverityBreakdown:
    def test_sorted_highest_first(self) -> None:
        bd = {
            SeverityLevel.LOW: 2,
            SeverityLevel.CRITICAL: 8,
            SeverityLevel.HIGH: 7,
        }
        out = format_severity_breakdown(bd)
        lines = out.splitlines()
        assert lines[0].endswith("8")
        assert "critical" in lines[0]
        assert "high" in lines[1]
        assert "low" in lines[2]

    def test_skips_zero_counts(self) -> None:
        bd = {SeverityLevel.HIGH: 0, SeverityLevel.MEDIUM: 3}
        out = format_severity_breakdown(bd)
        assert "high" not in out
        assert "medium" in out

    def test_empty_placeholder(self) -> None:
        assert "no severity data" in format_severity_breakdown({})


class TestSampleLogs:
    def _log(self, event_id: str, desc: str = "event desc") -> LogEntry:
        return LogEntry.model_validate(
            {
                "event_id": event_id,
                "timestamp": "2025-03-15T10:23:00Z",
                "event_type": "ids_alert",
                "severity": "high",
                "description": desc,
                "raw_log": f"CEF:0|Vendor|IDS|1.0|{event_id}|event|8|src=203.0.113.42",
                "src_ip": "203.0.113.42",
                "user": "admin",
                "action": "credential_stuffing",
            }
        )

    def test_renders_up_to_three(self) -> None:
        logs = [self._log(f"e{i}") for i in range(5)]
        out = format_sample_logs(logs)
        assert out.count("### Event") == 3

    def test_empty(self) -> None:
        assert "no representative logs" in format_sample_logs([])

    def test_includes_key_fields(self) -> None:
        log = self._log("e1", desc="SSH brute force")
        out = format_sample_logs([log])
        assert "SSH brute force" in out
        assert "203.0.113.42" in out
        assert "admin" in out
        assert "credential_stuffing" in out
        assert "ids_alert" in out


class TestPlaybooksBlock:
    def _chunk(self, src: str, score: float = 0.9, content: str = "body") -> PlaybookChunk:
        return PlaybookChunk(
            content=content,
            source_file=src,
            playbook_name=src.replace(".md", "").replace("_", " "),
            section="Immediate Actions",
            mitre_techniques=["T1110.004"],
            score=score,
            file_hash="abc123",
        )

    def test_renders_chunks(self) -> None:
        chunks = [self._chunk("a.md", 0.91), self._chunk("b.md", 0.75)]
        out = format_playbooks_block(chunks)
        assert "### Playbook: a.md" in out
        assert "### Playbook: b.md" in out
        assert "0.91" in out
        assert "Immediate Actions" in out

    def test_truncates_long_content(self) -> None:
        big = self._chunk("c.md", 0.5, content="x" * 2000)
        out = format_playbooks_block([big], max_chars_per_chunk=100)
        assert "…" in out
        assert "x" * 2000 not in out

    def test_empty_placeholder(self) -> None:
        assert "no playbooks retrieved" in format_playbooks_block([])


def _build_cluster() -> Cluster:
    t0 = datetime(2025, 3, 15, 10, 23, tzinfo=UTC)
    t1 = datetime(2025, 3, 15, 10, 47, tzinfo=UTC)
    log = LogEntry.model_validate(
        {
            "event_id": "evt-1",
            "timestamp": "2025-03-15T10:23:00Z",
            "event_type": "ids_alert",
            "severity": "high",
            "description": "Failed SSH login attempts",
            "raw_log": "CEF:0|Vendor|IDS|1.0|1001|Failed SSH|8|src=203.0.113.42",
            "src_ip": "203.0.113.42",
            "user": "admin",
            "action": "credential_stuffing",
        }
    )
    return Cluster.model_validate(
        {
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
    )


class TestBuildUserPrompt:
    def test_contains_all_sections(self) -> None:
        cluster = _build_cluster()
        chunks = [
            PlaybookChunk(
                content="Block source IPs",
                source_file="credential_stuffing_response.md",
                playbook_name="Credential Stuffing Response",
                section="Immediate Actions",
                mitre_techniques=["T1110.004"],
                score=0.91,
                file_hash="h",
            )
        ]
        out = build_user_prompt(cluster, chunks)
        assert "Cluster ID:** 42" in out
        assert "Size:** 15 events" in out
        assert "ids_alert" in out
        assert "credential_stuffing" in out
        assert "Unique source IPs:** 15" in out
        assert "Unique users affected:** 8" in out
        assert "Block source IPs" in out
        assert "credential_stuffing_response.md" in out
        assert "24m" in out

    def test_no_playbooks_uses_placeholder(self) -> None:
        out = build_user_prompt(_build_cluster(), [])
        assert "no playbooks retrieved" in out

    def test_event_type_as_string(self) -> None:
        out = build_user_prompt(_build_cluster(), [])
        assert "EventType." not in out
        assert "ids_alert" in out
