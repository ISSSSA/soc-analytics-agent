from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

MITRE_REGEX = re.compile(r"T\d{4}(?:\.\d{3})?")
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
DEFAULT_COLLECTION = "soc_playbooks"


@dataclass
class IndexStats:
    indexed_files: int
    skipped_files: int
    total_chunks: int


class PlaybookIndexer:
    def __init__(
        self,
        playbooks_dir: Path,
        chroma_persist_dir: Path,
        *,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self._playbooks_dir = Path(playbooks_dir)
        self._chroma_dir = Path(chroma_persist_dir)
        self._embedding_model = embedding_model
        self._collection_name = collection_name

        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._ef: Any = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )

    def index(self, rebuild: bool = False) -> IndexStats:
        if not self._playbooks_dir.exists():
            raise FileNotFoundError(self._playbooks_dir)

        if rebuild:
            try:
                self._client.delete_collection(self._collection_name)
                logger.info("Dropped collection %s for rebuild", self._collection_name)
            except Exception:
                logger.debug("No existing collection to drop")

        collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

        files = sorted(self._playbooks_dir.glob("*.md"))
        if not files:
            logger.warning("No .md files found in %s", self._playbooks_dir)
            return IndexStats(indexed_files=0, skipped_files=0, total_chunks=0)

        existing: dict[str, str] = {}
        try:
            all_items = collection.get(include=["metadatas"])
            metadatas = all_items.get("metadatas") or []
            for md in metadatas:
                if not md:
                    continue
                src = md.get("source_file")
                fh = md.get("file_hash")
                if isinstance(src, str) and isinstance(fh, str):
                    existing.setdefault(src, fh)
        except Exception:
            logger.debug("Could not read existing metadata; assuming empty")

        indexed = 0
        skipped = 0
        total_chunks = 0
        for path in files:
            content = path.read_text(encoding="utf-8")
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            src_name = path.name
            if existing.get(src_name) == file_hash and not rebuild:
                skipped += 1
                logger.info("Skipping %s (unchanged)", src_name)
                continue

            with contextlib.suppress(Exception):
                collection.delete(where={"source_file": src_name})

            chunks = self._chunkify(content, src_name, file_hash)
            if not chunks:
                logger.warning("%s produced no chunks (empty file?)", src_name)
                continue

            ids = [c["id"] for c in chunks]
            documents = [c["document"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            indexed += 1
            total_chunks += len(chunks)
            logger.info("Indexed %s: %d chunks", src_name, len(chunks))

        return IndexStats(
            indexed_files=indexed,
            skipped_files=skipped,
            total_chunks=total_chunks,
        )

    def _chunkify(
        self, content: str, source_file: str, file_hash: str
    ) -> list[dict[str, Any]]:
        playbook_name = _extract_playbook_name(content, fallback=source_file)
        raw = self._splitter.split_text(content)
        chunks: list[dict[str, Any]] = []
        for i, doc in enumerate(raw):
            text = doc.page_content.strip()
            if not text:
                continue
            section = (
                doc.metadata.get("h3")
                or doc.metadata.get("h2")
                or doc.metadata.get("h1")
                or None
            )
            mitre_list = _extract_mitre(text)
            chunks.append(
                {
                    "id": f"{source_file}::{i}",
                    "document": text,
                    "metadata": {
                        "source_file": source_file,
                        "playbook_name": playbook_name,
                        "section": section or "",
                        "mitre_techniques": ",".join(mitre_list),
                        "file_hash": file_hash,
                    },
                }
            )
        return chunks


def _extract_mitre(text: str) -> list[str]:
    return sorted(set(MITRE_REGEX.findall(text)))


def _extract_playbook_name(content: str, *, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip().removeprefix("Playbook:").strip()
    return Path(fallback).stem
