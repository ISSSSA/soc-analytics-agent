from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from soc_agent.rag.indexer import DEFAULT_COLLECTION, _extract_mitre
from soc_agent.schemas import PlaybookChunk

logger = logging.getLogger(__name__)


class PlaybookRetriever:
    """Извлечение релевантных фрагментов плейбуков из векторной базы Chroma."""
    def __init__(
        self,
        chroma_persist_dir: Path,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._chroma_dir = Path(chroma_persist_dir)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._ef: Any = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

    def _collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )

    def count(self) -> int:
        try:
            return self._collection().count()
        except Exception:
            return 0

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        mitre_hints: list[str] | None = None,
        mitre_boost: float = 0.3,
        over_fetch: int = 4,
    ) -> list[PlaybookChunk]:
        """Retrieve top-k playbook chunks; matching MITRE tags get an additive score boost."""
        if top_k <= 0:
            return []

        collection = self._collection()
        if collection.count() == 0:
            logger.warning("Collection %s is empty", self._collection_name)
            return []

        n_fetch = min(max(top_k * over_fetch, top_k), collection.count())
        results = collection.query(
            query_texts=[query],
            n_results=n_fetch,
            include=["documents", "metadatas", "distances"],
        )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        hints_set: set[str] = set(mitre_hints or [])
        parent_hints: set[str] = {h.split(".", 1)[0] for h in hints_set}

        candidates: list[PlaybookChunk] = []
        for doc, meta, dist in zip(docs, metas, dists, strict=True):
            if not meta:
                continue
            base_score = 1.0 - float(dist)
            raw_mitre = meta.get("mitre_techniques") or ""
            chunk_mitre = [m for m in str(raw_mitre).split(",") if m]
            overlap = _mitre_overlap(chunk_mitre, hints_set, parent_hints)
            score = base_score + (mitre_boost if overlap else 0.0)
            candidates.append(
                PlaybookChunk(
                    content=str(doc),
                    source_file=str(meta.get("source_file", "")),
                    playbook_name=str(meta.get("playbook_name", "")),
                    section=str(meta.get("section") or "") or None,
                    mitre_techniques=chunk_mitre,
                    score=score,
                    file_hash=str(meta.get("file_hash", "")),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def retrieve_for_cluster_text(
        self,
        event_type: str,
        top_classes: list[str],
        representative_text: str,
        *,
        top_k: int = 5,
        mitre_boost: float = 0.3,
    ) -> list[PlaybookChunk]:
        query = (
            f"Security event: {event_type}. "
            f"Main activities: {', '.join(top_classes)}. "
            f"Sample: {representative_text}"
        )
        hints = _extract_mitre(representative_text)
        return self.retrieve(
            query,
            top_k=top_k,
            mitre_hints=hints,
            mitre_boost=mitre_boost,
        )


def _mitre_overlap(
    chunk_mitre: list[str],
    hints: set[str],
    parent_hints: set[str],
) -> bool:
    if not chunk_mitre:
        return False
    for m in chunk_mitre:
        if m in hints:
            return True
        if m.split(".", 1)[0] in parent_hints:
            return True
    return False
