from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def split_batches(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
