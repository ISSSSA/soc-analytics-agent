from __future__ import annotations

from pathlib import Path

import pytest

from soc_agent.rag.indexer import PlaybookIndexer
from soc_agent.rag.retriever import PlaybookRetriever, _mitre_overlap

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


SAMPLE_PB_CRED = """# Playbook: Credential Stuffing Response

## Overview
Automated replay of breach credentials against auth endpoints.
MITRE: T1110.004, T1078.

## Immediate Actions
Block source IPs at perimeter firewall via ACL rule.
"""

SAMPLE_PB_DNS = """# Playbook: DNS Tunneling Response

## Overview
Covert command-and-control via DNS query payloads. MITRE T1071.004, T1572.

## Immediate Actions
Sinkhole the suspicious domain at the internal resolver.
"""


@pytest.fixture
def indexed_store(tmp_path: Path):  # type: ignore[no-untyped-def]
    pb_dir = tmp_path / "playbooks"
    pb_dir.mkdir()
    (pb_dir / "credential_stuffing_response.md").write_text(
        SAMPLE_PB_CRED, encoding="utf-8"
    )
    (pb_dir / "dns_tunneling_response.md").write_text(SAMPLE_PB_DNS, encoding="utf-8")

    chroma_dir = tmp_path / "chroma"
    indexer = PlaybookIndexer(playbooks_dir=pb_dir, chroma_persist_dir=chroma_dir)
    stats = indexer.index()
    assert stats.indexed_files == 2
    return chroma_dir


class TestMitreOverlap:
    def test_exact_match(self) -> None:
        assert _mitre_overlap(["T1110.004"], {"T1110.004"}, {"T1110"}) is True

    def test_parent_match(self) -> None:
        assert _mitre_overlap(["T1110.004"], {"T1110"}, {"T1110"}) is True

    def test_no_overlap(self) -> None:
        assert _mitre_overlap(["T1078"], {"T1087"}, {"T1087"}) is False

    def test_empty_chunk(self) -> None:
        assert _mitre_overlap([], {"T1078"}, {"T1078"}) is False


class TestRetriever:
    def test_retrieves_relevant_for_credential_stuffing(
        self, indexed_store: Path
    ) -> None:
        r = PlaybookRetriever(indexed_store)
        chunks = r.retrieve("Credential stuffing brute force attack", top_k=2)
        assert len(chunks) > 0
        top = chunks[0]
        assert top.source_file == "credential_stuffing_response.md"

    def test_retrieves_relevant_for_dns(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        chunks = r.retrieve(
            "DNS tunneling covert channel data exfiltration", top_k=2
        )
        assert len(chunks) > 0
        assert chunks[0].source_file == "dns_tunneling_response.md"

    def test_mitre_boost_prefers_tagged_chunk(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        chunks_no_hint = r.retrieve("security incident", top_k=1)
        chunks_with_hint = r.retrieve(
            "security incident", top_k=1, mitre_hints=["T1110.004"]
        )
        assert chunks_with_hint[0].source_file == "credential_stuffing_response.md"
        assert chunks_with_hint[0].score > chunks_no_hint[0].score - 0.01

    def test_returns_empty_for_empty_collection(self, tmp_path: Path) -> None:
        r = PlaybookRetriever(tmp_path / "chroma")
        assert r.retrieve("anything") == []

    def test_top_k_zero(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        assert r.retrieve("query", top_k=0) == []

    def test_chunk_mitre_parsed_from_metadata(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        chunks = r.retrieve("credential stuffing", top_k=5)
        with_mitre = [c for c in chunks if c.mitre_techniques]
        assert with_mitre, "expected at least one chunk with MITRE tags"

    def test_retrieve_for_cluster_text(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        chunks = r.retrieve_for_cluster_text(
            event_type="ids_alert",
            top_classes=["credential_stuffing", "brute_force"],
            representative_text="Multiple failed SSH from 15 IPs (T1110.004)",
            top_k=1,
        )
        assert chunks[0].source_file == "credential_stuffing_response.md"

    def test_count(self, indexed_store: Path) -> None:
        r = PlaybookRetriever(indexed_store)
        assert r.count() > 0
