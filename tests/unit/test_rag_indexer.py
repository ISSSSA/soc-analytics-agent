from __future__ import annotations

from pathlib import Path

import pytest

from soc_agent.rag.indexer import (
    PlaybookIndexer,
    _extract_mitre,
    _extract_playbook_name,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


SAMPLE_PLAYBOOK = """# Playbook: Sample Attack Response

## Overview
A short overview of the scenario. It references MITRE T1110.004 and T1078.

## Detection Indicators
- Signal A
- Signal B with T1556

## Immediate Actions
1. Step one.
2. Step two.
"""


class TestExtractMitre:
    def test_finds_simple(self) -> None:
        assert _extract_mitre("See T1110 for details") == ["T1110"]

    def test_finds_sub_technique(self) -> None:
        assert _extract_mitre("T1110.004 credential stuffing") == ["T1110.004"]

    def test_deduplicates_and_sorts(self) -> None:
        s = "T1078 and later T1078 and T1110.004 and T1110"
        assert _extract_mitre(s) == ["T1078", "T1110", "T1110.004"]

    def test_none_returns_empty(self) -> None:
        assert _extract_mitre("no techniques here") == []


class TestExtractPlaybookName:
    def test_strips_prefix(self) -> None:
        assert (
            _extract_playbook_name("# Playbook: Credential Stuffing\n", fallback="x.md")
            == "Credential Stuffing"
        )

    def test_plain_h1(self) -> None:
        assert _extract_playbook_name("# My Topic\n", fallback="x.md") == "My Topic"

    def test_fallback_when_no_h1(self) -> None:
        assert _extract_playbook_name("no header\n", fallback="foo.md") == "foo"


@pytest.fixture
def playbooks_dir(tmp_path: Path) -> Path:
    d = tmp_path / "playbooks"
    d.mkdir()
    (d / "sample.md").write_text(SAMPLE_PLAYBOOK, encoding="utf-8")
    return d


@pytest.fixture
def indexer(playbooks_dir: Path, tmp_path: Path) -> PlaybookIndexer:
    return PlaybookIndexer(
        playbooks_dir=playbooks_dir,
        chroma_persist_dir=tmp_path / "chroma",
    )


class TestIndexer:
    def test_indexes_sample(self, indexer: PlaybookIndexer) -> None:
        stats = indexer.index()
        assert stats.indexed_files == 1
        assert stats.skipped_files == 0
        assert stats.total_chunks >= 1

    def test_idempotent_second_call(self, indexer: PlaybookIndexer) -> None:
        first = indexer.index()
        second = indexer.index()
        assert first.indexed_files == 1
        assert second.indexed_files == 0
        assert second.skipped_files == 1

    def test_rebuild_reindexes(self, indexer: PlaybookIndexer) -> None:
        indexer.index()
        rebuilt = indexer.index(rebuild=True)
        assert rebuilt.indexed_files == 1
        assert rebuilt.skipped_files == 0

    def test_change_detected_by_hash(
        self, indexer: PlaybookIndexer, playbooks_dir: Path
    ) -> None:
        indexer.index()
        (playbooks_dir / "sample.md").write_text(
            SAMPLE_PLAYBOOK + "\n## New Section\nNew content with T1087.",
            encoding="utf-8",
        )
        stats = indexer.index()
        assert stats.indexed_files == 1
        assert stats.skipped_files == 0

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        indexer = PlaybookIndexer(
            playbooks_dir=tmp_path / "nope",
            chroma_persist_dir=tmp_path / "chroma",
        )
        with pytest.raises(FileNotFoundError):
            indexer.index()

    def test_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "playbooks"
        empty.mkdir()
        indexer = PlaybookIndexer(
            playbooks_dir=empty,
            chroma_persist_dir=tmp_path / "chroma",
        )
        stats = indexer.index()
        assert stats.total_chunks == 0
        assert stats.indexed_files == 0

    def test_chunk_metadata_contains_mitre(
        self, indexer: PlaybookIndexer, tmp_path: Path
    ) -> None:
        indexer.index()
        import chromadb

        client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
        col = client.get_collection("soc_playbooks")
        items = col.get(include=["metadatas"])
        metas = items["metadatas"] or []
        all_mitre: set[str] = set()
        for m in metas:
            if not m:
                continue
            all_mitre.update(
                t for t in str(m.get("mitre_techniques") or "").split(",") if t
            )
        assert "T1110.004" in all_mitre
        assert "T1078" in all_mitre
