from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from soc_agent.schemas import LogEntry

FileFormat = Literal["csv", "json", "jsonl"]
FormatOpt = Literal["auto", "csv", "json", "jsonl"]

NESTED_JSON_FIELDS: frozenset[str] = frozenset({"advanced_metadata", "behavioral_analytics"})


class UnsupportedFormatError(ValueError):
    pass


class ParseError(BaseModel):
    row: int
    error: str
    raw_preview: str | None = None


class LoadResult(BaseModel):
    logs: list[LogEntry]
    total_rows: int = Field(..., ge=0)
    parsed: int = Field(..., ge=0)
    skipped: int = Field(..., ge=0)
    errors: list[ParseError] = Field(default_factory=list)
    source_format: FileFormat

    @property
    def skip_rate(self) -> float:
        return self.skipped / self.total_rows if self.total_rows else 0.0


_EXT_TO_FORMAT: dict[str, FileFormat] = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
}


def detect_format(path: Path) -> FileFormat:
    """Detect format by extension; fall back to sniffing the first non-empty line."""
    if not path.exists():
        raise FileNotFoundError(path)
    fmt = _EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt is not None:
        return fmt
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                return "json"
            if stripped.startswith("{"):
                return "jsonl"
            if "," in stripped:
                return "csv"
            break
    raise UnsupportedFormatError(f"Cannot detect format for {path}")


def _clean_csv_row(row: dict[str, str | None]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        if v is None or v == "":
            continue
        if k in NESTED_JSON_FIELDS and isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("{"):
                try:
                    cleaned[k] = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    pass
        cleaned[k] = v
    return cleaned


def _iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _clean_csv_row(row)


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise UnsupportedFormatError(
            f"JSON file must contain a top-level list; got {type(data).__name__}"
        )
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise UnsupportedFormatError(
                f"JSON list element at index {i} must be an object; got {type(row).__name__}"
            )
        yield row


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise UnsupportedFormatError(
                    f"Invalid JSON on line {line_no}: {e.msg}"
                ) from e
            if not isinstance(row, dict):
                raise UnsupportedFormatError(
                    f"JSONL line {line_no} must be an object; got {type(row).__name__}"
                )
            yield row


def _iter_raw(path: Path, fmt: FileFormat) -> Iterator[dict[str, Any]]:
    if fmt == "csv":
        return _iter_csv(path)
    if fmt == "json":
        return _iter_json(path)
    if fmt == "jsonl":
        return _iter_jsonl(path)
    raise UnsupportedFormatError(f"Unsupported format: {fmt}")  # pragma: no cover


def iter_logs(
    path: Path,
    *,
    format: FormatOpt = "auto",
    skip_invalid: bool = True,
) -> Iterator[LogEntry]:
    """Stream LogEntry records from `path`, skipping invalid rows by default."""
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(f"{path} is not a regular file")
    fmt: FileFormat = detect_format(path) if format == "auto" else format
    return _iter_logs_impl(path, fmt, skip_invalid)


def _iter_logs_impl(path: Path, fmt: FileFormat, skip_invalid: bool) -> Iterator[LogEntry]:
    for row in _iter_raw(path, fmt):
        try:
            yield LogEntry.model_validate(row)
        except ValidationError:
            if not skip_invalid:
                raise
            continue


def load_logs(
    path: Path,
    *,
    format: FormatOpt = "auto",
    skip_invalid: bool = True,
    max_errors_stored: int = 10,
) -> LoadResult:
    """Eagerly load and validate all log rows from `path`, returning parse statistics."""
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(f"{path} is not a regular file")
    fmt: FileFormat = detect_format(path) if format == "auto" else format

    logs: list[LogEntry] = []
    errors: list[ParseError] = []
    total = 0
    skipped = 0

    for row in _iter_raw(path, fmt):
        total += 1
        try:
            logs.append(LogEntry.model_validate(row))
        except ValidationError as e:
            if not skip_invalid:
                raise
            skipped += 1
            if len(errors) < max_errors_stored:
                preview: str | None
                try:
                    preview = json.dumps(row, default=str)[:200]
                except (TypeError, ValueError):
                    preview = str(row)[:200]
                errors.append(ParseError(row=total, error=str(e)[:500], raw_preview=preview))

    return LoadResult(
        logs=logs,
        total_rows=total,
        parsed=len(logs),
        skipped=skipped,
        errors=errors,
        source_format=fmt,
    )
