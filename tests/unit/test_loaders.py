from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from soc_agent.io.loaders import (
    LoadResult,
    UnsupportedFormatError,
    detect_format,
    iter_logs,
    load_logs,
)


def _minimal_log(event_id: str = "evt-1", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_id": event_id,
        "timestamp": "2025-03-15T10:23:00Z",
        "event_type": "ids_alert",
        "severity": "high",
        "description": f"event {event_id}",
    }
    base.update(overrides)
    return base


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


class TestDetectFormat:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("logs.csv", "csv"),
            ("logs.JSON", "json"),
            ("logs.jsonl", "jsonl"),
            ("logs.ndjson", "jsonl"),
        ],
    )
    def test_by_extension(self, tmp_path: Path, name: str, expected: str) -> None:
        p = tmp_path / name
        p.write_text("dummy", encoding="utf-8")
        assert detect_format(p) == expected

    def test_sniff_json_array(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.dat"
        p.write_text("  \n[{}]", encoding="utf-8")
        assert detect_format(p) == "json"

    def test_sniff_jsonl_object_per_line(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.dat"
        p.write_text('{"event_id": "x"}\n{"event_id": "y"}\n', encoding="utf-8")
        assert detect_format(p) == "jsonl"

    def test_sniff_csv_by_commas(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.dat"
        p.write_text("event_id,timestamp\nx,2025-01-01T00:00:00Z\n", encoding="utf-8")
        assert detect_format(p) == "csv"

    def test_unsupported(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.dat"
        p.write_text("single_token_no_commas\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            detect_format(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            detect_format(tmp_path / "nonexistent.json")


class TestLoadJson:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.json"
        _write_json(p, [_minimal_log("a"), _minimal_log("b")])
        result = load_logs(p)
        assert isinstance(result, LoadResult)
        assert result.source_format == "json"
        assert result.parsed == 2
        assert result.skipped == 0
        assert [log.event_id for log in result.logs] == ["a", "b"]

    def test_not_a_list(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.json"
        p.write_text('{"event_id": "x"}', encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            load_logs(p)

    def test_list_element_not_object(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.json"
        p.write_text('[{"event_id": "x"}, 42]', encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            load_logs(p)

    def test_utf8_bom_tolerated(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.json"
        p.write_bytes(b"\xef\xbb\xbf" + json.dumps([_minimal_log("a")]).encode())
        result = load_logs(p)
        assert result.parsed == 1


class TestLoadJsonl:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [_minimal_log("a"), _minimal_log("b"), _minimal_log("c")])
        result = load_logs(p)
        assert result.source_format == "jsonl"
        assert result.parsed == 3

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        p.write_text(
            json.dumps(_minimal_log("a")) + "\n\n\n" + json.dumps(_minimal_log("b")) + "\n",
            encoding="utf-8",
        )
        result = load_logs(p)
        assert result.parsed == 2

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        p.write_text(json.dumps(_minimal_log("a")) + "\n{not_json}\n", encoding="utf-8")
        with pytest.raises(UnsupportedFormatError):
            load_logs(p)


class TestLoadCsv:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        _write_csv(p, [_minimal_log("a"), _minimal_log("b")])
        result = load_logs(p)
        assert result.source_format == "csv"
        assert result.parsed == 2

    def test_empty_cells_become_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        row = _minimal_log("a", user="", src_ip="")
        _write_csv(p, [row])
        result = load_logs(p)
        assert result.parsed == 1
        assert result.logs[0].user is None
        assert result.logs[0].src_ip is None

    def test_nested_json_field(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        row = _minimal_log(
            "a",
            advanced_metadata=json.dumps(
                {"session_id": "s1", "risk_score": 0.9, "confidence": 0.8}
            ),
        )
        _write_csv(p, [row])
        result = load_logs(p)
        assert result.parsed == 1
        log = result.logs[0]
        assert log.advanced_metadata.session_id == "s1"
        assert log.advanced_metadata.risk_score == 0.9

    def test_nested_field_not_json_is_passed_through(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        row = _minimal_log("a", advanced_metadata="not_json")
        _write_csv(p, [row])
        result = load_logs(p)
        assert result.skipped == 1
        assert result.parsed == 0

    def test_numeric_strings_coerced(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        row = _minimal_log("a", src_port="44532", dst_port="22")
        _write_csv(p, [row])
        result = load_logs(p)
        log = result.logs[0]
        assert log.src_port == 44532
        assert log.dst_port == 22

    def test_extra_columns_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.csv"
        row = _minimal_log("a", my_custom_field="hello")
        _write_csv(p, [row])
        result = load_logs(p)
        dumped = result.logs[0].model_dump()
        assert dumped["my_custom_field"] == "hello"


class TestErrorHandling:
    def test_invalid_row_skipped_by_default(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        good = _minimal_log("a")
        bad = {"event_id": "b"}
        _write_jsonl(p, [good, bad, _minimal_log("c")])
        result = load_logs(p)
        assert result.parsed == 2
        assert result.skipped == 1
        assert result.total_rows == 3
        assert len(result.errors) == 1
        assert result.errors[0].row == 2

    def test_skip_false_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [_minimal_log("a"), {"event_id": "b"}])
        with pytest.raises(ValidationError):
            load_logs(p, skip_invalid=False)

    def test_error_storage_capped(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        rows = [{"event_id": str(i)} for i in range(20)]
        _write_jsonl(p, rows)
        result = load_logs(p, max_errors_stored=5)
        assert result.skipped == 20
        assert len(result.errors) == 5

    def test_error_preview_truncated(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        bad = {"event_id": "x", "description": "y" * 2000}
        _write_jsonl(p, [bad])
        result = load_logs(p)
        assert result.skipped == 1
        preview = result.errors[0].raw_preview
        assert preview is not None
        assert len(preview) <= 200

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        p.write_text("", encoding="utf-8")
        result = load_logs(p)
        assert result.total_rows == 0
        assert result.parsed == 0
        assert result.logs == []

    def test_skip_rate_property(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [_minimal_log("a"), {"event_id": "b"}, {"event_id": "c"}])
        result = load_logs(p)
        assert result.skip_rate == pytest.approx(2 / 3)

    def test_skip_rate_on_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        p.write_text("", encoding="utf-8")
        assert load_logs(p).skip_rate == 0.0


class TestIterLogs:
    def test_is_iterator(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [_minimal_log("a"), _minimal_log("b")])
        it = iter_logs(p)
        ids = [log.event_id for log in it]
        assert ids == ["a", "b"]

    def test_skips_invalid_silently(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [_minimal_log("a"), {"event_id": "bad"}, _minimal_log("c")])
        ids = [log.event_id for log in iter_logs(p)]
        assert ids == ["a", "c"]

    def test_skip_false_raises_on_iteration(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.jsonl"
        _write_jsonl(p, [{"event_id": "bad"}])
        it = iter_logs(p, skip_invalid=False)
        with pytest.raises(ValidationError):
            next(it)

    def test_missing_file_raises_eagerly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            iter_logs(tmp_path / "nope.json")


class TestFormatOverride:
    def test_force_jsonl_on_unknown_ext(self, tmp_path: Path) -> None:
        p = tmp_path / "logs.dat"
        _write_jsonl(p, [_minimal_log("a")])
        result = load_logs(p, format="jsonl")
        assert result.parsed == 1

    def test_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            load_logs(tmp_path)
