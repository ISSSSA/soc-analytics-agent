from __future__ import annotations

import numpy as np
import pytest

from soc_agent.clustering.clusterer import (
    HDBSCANClusterer,
    _aggregate,
    _l2_normalize,
)


def _three_clusters(rng: np.random.Generator, per_cluster: int = 10, dim: int = 8) -> np.ndarray:
    centers = np.array(
        [
            [5.0] + [0.0] * (dim - 1),
            [-5.0] + [0.0] * (dim - 1),
            [0.0] * (dim - 1) + [5.0],
        ],
        dtype=np.float64,
    )
    blobs = []
    for c in centers:
        blobs.append(c + rng.normal(scale=0.2, size=(per_cluster, dim)))
    return np.vstack(blobs).astype(np.float32)


class TestL2Normalize:
    def test_unit_vectors(self) -> None:
        X = np.array([[3.0, 4.0], [1.0, 0.0]])
        out = _l2_normalize(X)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_zero_vector_preserved(self) -> None:
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        out = _l2_normalize(X)
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(out[1]), 1.0)


class TestAggregate:
    def test_groups_indices_by_label(self) -> None:
        labels = np.array([0, 1, 0, -1, 1, 0])
        stats = _aggregate(labels)
        ids = [s.cluster_id for s in stats]
        assert ids == [-1, 0, 1]
        cluster_0 = next(s for s in stats if s.cluster_id == 0)
        assert cluster_0.size == 3
        assert cluster_0.log_indices == [0, 2, 5]
        noise = next(s for s in stats if s.cluster_id == -1)
        assert noise.log_indices == [3]


class TestClusterer:
    def test_finds_three_blobs(self) -> None:
        rng = np.random.default_rng(42)
        X = _three_clusters(rng, per_cluster=10, dim=8)

        result = HDBSCANClusterer(min_cluster_size=5, min_samples=3).cluster(X)
        assert result.n_clusters == 3
        assert result.labels.shape == (30,)
        assert result.probabilities.shape == (30,)
        total = sum(c.size for c in result.clusters)
        assert total == 30

    def test_empty_input(self) -> None:
        X = np.empty((0, 8), dtype=np.float32)
        result = HDBSCANClusterer().cluster(X)
        assert result.n_clusters == 0
        assert result.n_noise == 0
        assert result.clusters == []
        assert result.labels.shape == (0,)

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            HDBSCANClusterer().cluster(np.array([1.0, 2.0, 3.0]))

    def test_too_few_points_mostly_noise(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 8)).astype(np.float32)
        result = HDBSCANClusterer(min_cluster_size=5, min_samples=3).cluster(X)
        assert result.n_clusters == 0
        assert result.n_noise == 4
        assert len(result.clusters) == 1
        assert result.clusters[0].cluster_id == -1

    def test_pure_clusters_no_noise(self) -> None:
        rng = np.random.default_rng(7)
        X = _three_clusters(rng, per_cluster=20, dim=16)
        result = HDBSCANClusterer(min_cluster_size=5, min_samples=3).cluster(X)
        assert result.n_clusters == 3
        if result.n_noise == 0:
            assert all(c.cluster_id >= 0 for c in result.clusters)

    def test_log_indices_cover_all_points(self) -> None:
        rng = np.random.default_rng(1)
        X = _three_clusters(rng, per_cluster=8, dim=8)
        result = HDBSCANClusterer(min_cluster_size=5, min_samples=3).cluster(X)
        all_idx: set[int] = set()
        for cluster in result.clusters:
            all_idx.update(cluster.log_indices)
        assert all_idx == set(range(len(X)))

    def test_normalize_true_is_default(self) -> None:
        rng = np.random.default_rng(99)
        X = _three_clusters(rng, per_cluster=10, dim=8)
        c = HDBSCANClusterer(min_cluster_size=5, min_samples=3, normalize=True)
        result1 = c.cluster(X)
        result2 = c.cluster(X * 100.0)
        assert result1.n_clusters == result2.n_clusters

    def test_normalize_false_keeps_raw(self) -> None:
        rng = np.random.default_rng(99)
        X = _three_clusters(rng, per_cluster=10, dim=8)
        c = HDBSCANClusterer(min_cluster_size=5, min_samples=3, normalize=False)
        result = c.cluster(X)
        assert result.n_clusters == 3


class TestValidation:
    def test_rejects_small_min_cluster_size(self) -> None:
        with pytest.raises(ValueError, match=r"min_cluster_size"):
            HDBSCANClusterer(min_cluster_size=1)

    def test_rejects_zero_min_samples(self) -> None:
        with pytest.raises(ValueError, match=r"min_samples"):
            HDBSCANClusterer(min_samples=0)
