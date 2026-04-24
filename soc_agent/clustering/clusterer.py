from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import hdbscan
import numpy as np

from soc_agent.schemas import ClusterStats

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    clusters: list[ClusterStats]
    n_clusters: int
    n_noise: int


class HDBSCANClusterer:
    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        normalize: bool = True,
    ) -> None:
        if min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {min_cluster_size}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples
        self._metric = metric
        self._cluster_selection_method = cluster_selection_method
        self._normalize = normalize

    def cluster(self, embeddings: np.ndarray) -> ClusterResult:
        """Cluster an (N, D) float array; L2-normalizes when configured so euclidean ≈ cosine."""
        if embeddings.size == 0:
            return ClusterResult(
                labels=np.empty(0, dtype=np.int64),
                probabilities=np.empty(0, dtype=np.float64),
                clusters=[],
                n_clusters=0,
                n_noise=0,
            )
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D (N, D); got shape {embeddings.shape}"
            )

        X = embeddings.astype(np.float64, copy=False)
        if self._normalize:
            X = _l2_normalize(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            min_samples=self._min_samples,
            metric=self._metric,
            cluster_selection_method=self._cluster_selection_method,
        )
        labels: np.ndarray = clusterer.fit_predict(X)
        probabilities: np.ndarray = clusterer.probabilities_

        clusters = _aggregate(labels)
        n_noise = int(np.sum(labels == -1))
        n_clusters = sum(1 for c in clusters if c.cluster_id >= 0)

        logger.info(
            "HDBSCAN: n=%d → clusters=%d noise=%d",
            len(labels),
            n_clusters,
            n_noise,
        )
        return ClusterResult(
            labels=labels.astype(np.int64, copy=False),
            probabilities=probabilities.astype(np.float64, copy=False),
            clusters=clusters,
            n_clusters=n_clusters,
            n_noise=n_noise,
        )


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return X / safe


def _aggregate(labels: np.ndarray) -> list[ClusterStats]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels.tolist()):
        buckets[int(lab)].append(idx)
    return [
        ClusterStats(cluster_id=cid, size=len(indices), log_indices=indices)
        for cid, indices in sorted(buckets.items())
    ]
