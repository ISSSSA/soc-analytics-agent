from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from inference_service.batching import split_batches

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OOMError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class ClassificationUnsupportedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassifyPred:
    label: str
    score: float
    top_k: list[tuple[str, float]]


Pooling = Literal["mean", "cls"]


@runtime_checkable
class ModelProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def is_ready(self) -> bool: ...

    @property
    def supports_classification(self) -> bool: ...

    @property
    def num_classes(self) -> int | None: ...

    @property
    def class_labels(self) -> list[str] | None: ...

    def device_info(self) -> dict[str, Any]: ...

    def embed(
        self,
        texts: list[str],
        pooling: Pooling,
        normalize: bool,
    ) -> list[list[float]]: ...

    def classify(self, texts: list[str], top_k: int) -> list[ClassifyPred]: ...

    def warmup(self) -> None: ...


class SecBERTProvider:
    """HuggingFace-backed provider; auto-detects classifier head and retries batch=1 on OOM."""

    def __init__(
        self,
        model_path: str,
        *,
        device: Literal["auto", "cpu", "cuda", "mps"] = "auto",
        use_fp16: bool = True,
        max_batch_size: int = 256,
        hf_token: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._requested_device = device
        self._use_fp16 = use_fp16
        self._max_batch_size = max_batch_size
        self._hf_token = hf_token

        self._tokenizer: Any = None
        self._model: Any = None
        self._config: Any = None
        self._device: Any = None
        self._is_classifier: bool = False
        self._class_labels: list[str] | None = None
        self._ready: bool = False
        self._lock = threading.Lock()

        self._load()

    @property
    def model_id(self) -> str:
        return self._model_path

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def supports_classification(self) -> bool:
        return self._is_classifier

    @property
    def num_classes(self) -> int | None:
        return len(self._class_labels) if self._class_labels is not None else None

    @property
    def class_labels(self) -> list[str] | None:
        return list(self._class_labels) if self._class_labels is not None else None

    def device_info(self) -> dict[str, Any]:
        import torch

        info: dict[str, Any] = {
            "device": str(self._device) if self._device is not None else None,
            "gpu_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            idx = 0
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            info["gpu_memory_total_mb"] = int(props.total_memory / (1024 * 1024))
            info["gpu_memory_used_mb"] = int(torch.cuda.memory_allocated(idx) / (1024 * 1024))
        return info

    def _resolve_device(self) -> Any:
        import torch

        if self._requested_device == "cpu":
            return torch.device("cpu")
        if self._requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but CUDA is not available")
            return torch.device("cuda")
        if self._requested_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("device=mps requested but MPS is not available")
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self) -> None:
        """Load tokenizer/config/model; classifier head only when checkpoint declares it."""
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        logger.info("Loading tokenizer from %s", self._model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, token=self._hf_token)

        logger.info("Loading config from %s", self._model_path)
        config = AutoConfig.from_pretrained(self._model_path, token=self._hf_token)
        self._config = config

        archs: list[str] = list(getattr(config, "architectures", None) or [])
        has_classifier = any("ForSequenceClassification" in a for a in archs)
        self._device = self._resolve_device()
        use_fp16 = self._use_fp16 and self._device.type == "cuda"

        if has_classifier:
            logger.info(
                "Loading AutoModelForSequenceClassification (num_labels=%d)",
                config.num_labels,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path, token=self._hf_token
            )
            self._is_classifier = True
            id2label = getattr(config, "id2label", None) or {}
            if id2label:
                self._class_labels = [id2label[i] for i in sorted(id2label.keys())]
            else:
                self._class_labels = [f"LABEL_{i}" for i in range(config.num_labels)]
        else:
            logger.info("Loading AutoModel (no classifier head detected)")
            model = AutoModel.from_pretrained(self._model_path, token=self._hf_token)
            self._is_classifier = False
            self._class_labels = None

        model.eval()
        model.to(self._device)
        if use_fp16:
            model.half()

        self._model = model
        self._ready = True
        logger.info(
            "Model loaded: device=%s fp16=%s classifier=%s labels=%d",
            self._device,
            use_fp16,
            self._is_classifier,
            (self.num_classes or 0),
        )

    def warmup(self) -> None:
        if not self._ready:
            raise ModelUnavailableError("model not loaded")
        logger.info("Warming up with 8 dummy texts")
        dummy = ["warmup text"] * 8
        self.embed(dummy, pooling="mean", normalize=True)
        if self._is_classifier:
            self.classify(dummy, top_k=3)

    def _encode(self, texts: list[str], max_length: int = 512) -> dict[str, Any]:
        enc: Any = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return dict(enc)

    def _forward_embed(
        self,
        texts: list[str],
        pooling: Pooling,
        normalize: bool,
        batch_size: int,
    ) -> list[list[float]]:
        """Tokenize, encode, mean/CLS-pool, optionally L2-normalize, in chunks of `batch_size`."""
        import torch

        out: list[list[float]] = []
        base = self._model.base_model if self._is_classifier else self._model
        for chunk in split_batches(texts, batch_size):
            enc = self._encode(chunk)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = base(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            if pooling == "mean":
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
            else:
                pooled = hidden[:, 0, :]
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.extend(pooled.float().cpu().tolist())
        return out

    def embed(
        self,
        texts: list[str],
        pooling: Pooling,
        normalize: bool,
    ) -> list[list[float]]:
        if not self._ready:
            raise ModelUnavailableError("model not loaded")
        with self._lock:
            try:
                return self._forward_embed(texts, pooling, normalize, self._max_batch_size)
            except Exception as e:
                if not _is_oom(e):
                    raise
                _clear_cuda_cache()
                logger.warning("CUDA OOM at batch=%d; retrying with batch=1", self._max_batch_size)
                try:
                    return self._forward_embed(texts, pooling, normalize, 1)
                except Exception as e2:
                    if _is_oom(e2):
                        raise OOMError("CUDA OOM even at batch_size=1") from e2
                    raise

    def _forward_classify(
        self,
        texts: list[str],
        top_k: int,
        batch_size: int,
    ) -> list[ClassifyPred]:
        import torch

        if not self._is_classifier or self._class_labels is None:
            raise ClassificationUnsupportedError("model has no classifier head")

        out: list[ClassifyPred] = []
        k = min(top_k, len(self._class_labels))
        for chunk in split_batches(texts, batch_size):
            enc = self._encode(chunk)
            enc = {key: v.to(self._device) for key, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            topk_vals, topk_idx = torch.topk(probs, k=k, dim=-1)
            vals_list = topk_vals.cpu().tolist()
            idx_list = topk_idx.cpu().tolist()
            for row_vals, row_idx in zip(vals_list, idx_list, strict=True):
                top_items = [
                    (self._class_labels[i], float(v))
                    for v, i in zip(row_vals, row_idx, strict=True)
                ]
                out.append(
                    ClassifyPred(
                        label=top_items[0][0],
                        score=top_items[0][1],
                        top_k=top_items,
                    )
                )
        return out

    def classify(self, texts: list[str], top_k: int) -> list[ClassifyPred]:
        if not self._ready:
            raise ModelUnavailableError("model not loaded")
        if not self._is_classifier:
            raise ClassificationUnsupportedError("model has no classifier head")
        with self._lock:
            try:
                return self._forward_classify(texts, top_k, self._max_batch_size)
            except Exception as e:
                if not _is_oom(e):
                    raise
                _clear_cuda_cache()
                logger.warning("CUDA OOM at batch=%d; retrying with batch=1", self._max_batch_size)
                try:
                    return self._forward_classify(texts, top_k, 1)
                except Exception as e2:
                    if _is_oom(e2):
                        raise OOMError("CUDA OOM even at batch_size=1") from e2
                    raise


def _is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(e)
    return "CUDA out of memory" in msg or "out of memory" in msg.lower()


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
