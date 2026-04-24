from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest


@pytest.fixture
def sample_log_dict() -> dict[str, Any]:
    return {
        "event_id": "evt-001",
        "timestamp": "2025-03-15T10:23:00Z",
        "event_type": "ids_alert",
        "severity": "high",
        "description": "Multiple failed SSH login attempts from 203.0.113.42 targeting admin",
        "raw_log": "CEF:0|Vendor|Product|1.0|1001|Failed SSH|3|src=203.0.113.42",
        "advanced_metadata": {
            "session_id": "sess-abc",
            "risk_score": 0.87,
            "confidence": 0.91,
            "geo_location": "US",
        },
        "behavioral_analytics": {
            "baseline_deviation": 3.4,
            "entropy": 0.65,
            "frequency_anomaly": 0.88,
            "sequence_anomaly": 0.42,
        },
        "user": "admin",
        "action": "credential_stuffing",
        "object": "ssh_service",
        "src_ip": "203.0.113.42",
        "dst_ip": "10.0.0.5",
        "src_port": 44532,
        "dst_port": 22,
        "protocol": "TCP",
        "mac_address": "00:1A:2B:3C:4D:5E",
        "alert_type": "authentication_failure",
        "signature_id": "SIG-1001",
        "category": "credential_access",
    }


@pytest.fixture
def sample_cluster_kwargs() -> dict[str, Any]:
    t0 = datetime(2025, 3, 15, 10, 23, 0, tzinfo=UTC)
    t1 = datetime(2025, 3, 15, 10, 47, 12, tzinfo=UTC)
    return {
        "cluster_id": 42,
        "cluster_size": 15,
        "time_range": (t0, t1),
        "dominant_event_type": "ids_alert",
        "dominant_classes": [
            {"class": "credential_stuffing", "count": 12, "avg_confidence": 0.87},
            {"class": "brute_force", "count": 3, "avg_confidence": 0.72},
        ],
        "severity_breakdown": {"critical": 8, "high": 7},
        "unique_src_ips": 15,
        "unique_users_targeted": 8,
        "representative_logs": [],
        "all_log_indices": list(range(15)),
    }


@pytest.fixture
def sample_recommendation_kwargs() -> dict[str, Any]:
    return {
        "summary": "Coordinated credential stuffing attack from 15 unique source IPs.",
        "threat_assessment": "High — automated campaign targeting admin SSH.",
        "relevant_playbooks": ["credential_stuffing_response.md"],
        "immediate_actions": [
            "Block source IPs at perimeter firewall",
            "Force password reset for affected accounts",
        ],
        "investigation_steps": [
            "Correlate timestamps with SSH auth log",
            "Check for successful logins from the same IP pool",
        ],
        "mitre_techniques": ["T1110.004"],
        "priority": "high",
    }
