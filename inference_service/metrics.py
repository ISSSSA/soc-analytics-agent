from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client import generate_latest as _generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests by endpoint and status.",
    labelnames=("endpoint", "status"),
    registry=registry,
)

LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference endpoint latency.",
    labelnames=("endpoint",),
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=registry,
)

BATCH_SIZE = Histogram(
    "inference_batch_size",
    "Number of texts per request.",
    labelnames=("endpoint",),
    buckets=(1, 4, 16, 32, 64, 128, 256, 512),
    registry=registry,
)

GPU_MEMORY_USED_MB = Gauge(
    "inference_gpu_memory_used_mb",
    "Current GPU memory used, in MiB.",
    registry=registry,
)


def render() -> bytes:
    return _generate_latest(registry)
