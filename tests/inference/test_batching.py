from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from inference_service.batching import split_batches
from inference_service.model_loader import (
    OOMError,
    SecBERTProvider,
    _is_oom,
)


class TestSplitBatches:
    def test_exact_multiple(self) -> None:
        batches = list(split_batches([1, 2, 3, 4], 2))
        assert batches == [[1, 2], [3, 4]]

    def test_remainder(self) -> None:
        batches = list(split_batches([1, 2, 3, 4, 5], 2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_single_batch(self) -> None:
        assert list(split_batches([1, 2, 3], 100)) == [[1, 2, 3]]

    def test_empty(self) -> None:
        assert list(split_batches([], 5)) == []

    def test_negative_batch_size(self) -> None:
        with pytest.raises(ValueError):
            list(split_batches([1, 2], 0))


class TestIsOOM:
    def test_torch_oom_class(self) -> None:
        import torch

        err = torch.cuda.OutOfMemoryError("CUDA out of memory")
        assert _is_oom(err) is True

    def test_runtime_error_with_message(self) -> None:
        assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate...")) is True

    def test_generic_runtime_error_not_oom(self) -> None:
        assert _is_oom(RuntimeError("shape mismatch")) is False

    def test_value_error_not_oom(self) -> None:
        assert _is_oom(ValueError("bad input")) is False


class TestOOMRetryEmbed:
    """Exercise the OOM-retry fallback by stubbing _forward_embed on a real SecBERTProvider."""

    def _make_provider(self) -> SecBERTProvider:
        p = SecBERTProvider.__new__(SecBERTProvider)
        p._model_path = "fake"
        p._max_batch_size = 256
        p._ready = True
        p._is_classifier = False
        p._class_labels = None
        import threading

        p._lock = threading.Lock()
        return p

    def test_recovers_after_batch_1_retry(self) -> None:
        p = self._make_provider()
        big_err = RuntimeError("CUDA out of memory. Big batch.")
        calls: list[int] = []

        def fake_forward(
            texts: list[str], pooling: Any, normalize: bool, batch_size: int
        ) -> list[list[float]]:
            calls.append(batch_size)
            if batch_size == 256:
                raise big_err
            return [[1.0, 2.0]] * len(texts)

        with patch.object(p, "_forward_embed", side_effect=fake_forward), patch(
            "inference_service.model_loader._clear_cuda_cache"
        ):
            result = p.embed(["a", "b"], pooling="mean", normalize=True)

        assert calls == [256, 1]
        assert result == [[1.0, 2.0], [1.0, 2.0]]

    def test_raises_oom_error_when_batch_1_still_oom(self) -> None:
        p = self._make_provider()
        err1 = RuntimeError("CUDA out of memory.")
        err2 = RuntimeError("CUDA out of memory at bs=1.")

        def fake_forward(
            texts: list[str], pooling: Any, normalize: bool, batch_size: int
        ) -> list[list[float]]:
            raise err1 if batch_size == 256 else err2

        with patch.object(p, "_forward_embed", side_effect=fake_forward), patch(
            "inference_service.model_loader._clear_cuda_cache"
        ), pytest.raises(OOMError):
            p.embed(["a"], pooling="mean", normalize=True)

    def test_non_oom_error_not_retried(self) -> None:
        p = self._make_provider()
        mock_forward = MagicMock(side_effect=RuntimeError("shape mismatch"))
        with (
            patch.object(p, "_forward_embed", mock_forward),
            pytest.raises(RuntimeError, match="shape mismatch"),
        ):
            p.embed(["a"], pooling="mean", normalize=True)
        assert mock_forward.call_count == 1
