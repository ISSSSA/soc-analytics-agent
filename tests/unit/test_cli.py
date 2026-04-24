from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from soc_agent import cli
from soc_agent.schemas import (
    EventType,
    IncidentReport,
    LLMUsage,
    PipelineResult,
    PipelineStats,
    Priority,
    Recommendation,
    SeverityLevel,
)

runner = CliRunner()


def _sample_result() -> PipelineResult:
    rec = Recommendation(
        summary="Summary long enough.",
        threat_assessment="Assessment long enough.",
        relevant_playbooks=["cred.md"],
        immediate_actions=["Block IPs"],
        investigation_steps=["Check SSH logs"],
        mitre_techniques=["T1110.004"],
        priority=Priority.HIGH,
    )
    incident = IncidentReport(
        cluster_id=0,
        cluster_size=5,
        time_range=(
            datetime(2025, 3, 15, 10, 0, tzinfo=UTC),
            datetime(2025, 3, 15, 10, 5, tzinfo=UTC),
        ),
        dominant_event_type=EventType.IDS_ALERT,
        dominant_classes=[],
        severity_breakdown={SeverityLevel.HIGH: 5},
        unique_src_ips=5,
        unique_users_targeted=2,
        representative_logs=[],
        recommendation=rec,
    )
    return PipelineResult(
        correlation_id="run-test",
        incidents=[incident],
        stats=PipelineStats(
            total_logs=5,
            n_clusters=1,
            n_noise=0,
            processing_time_seconds=1.2,
            cache_hit_rate=0.3,
            llm_usage=LLMUsage(
                model="test-model",
                total_cost_usd=0.0042,
                total_tokens_in=500,
                total_tokens_out=100,
            ),
        ),
    )


def _fake_pipeline(result: PipelineResult) -> MagicMock:
    p = MagicMock()
    p.process = AsyncMock(return_value=result)
    return p


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from soc_agent.config import get_settings

    get_settings.cache_clear()


class TestVersion:
    def test_prints_version(self) -> None:
        result = runner.invoke(cli.app, ["version"])
        assert result.exit_code == 0
        assert "SOC Agent" in result.stdout
        from soc_agent import __version__

        assert __version__ in result.stdout


class TestIndexPlaybooks:
    def test_indexes_playbooks_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pb_dir = tmp_path / "playbooks"
        pb_dir.mkdir()
        (pb_dir / "sample.md").write_text(
            "# Playbook: Sample\n\n## Overview\nShort (T1110)\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("PLAYBOOKS_DIR", str(pb_dir))
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

        result = runner.invoke(cli.app, ["index-playbooks"])
        assert result.exit_code == 0, result.stdout
        assert "indexed=1" in result.stdout

    def test_override_dir_flag(self, tmp_path: Path) -> None:
        pb_dir = tmp_path / "custom"
        pb_dir.mkdir()
        (pb_dir / "a.md").write_text(
            "# Playbook: Custom\n\n## Overview\nContent (T1078)",
            encoding="utf-8",
        )
        result = runner.invoke(
            cli.app,
            [
                "index-playbooks",
                "--playbooks-dir",
                str(pb_dir),
                "--rebuild",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "indexed=1" in result.stdout


class TestHealth:
    def test_all_up(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))

        async def _fake_collect(_s: Any) -> list[tuple[str, bool, str]]:
            return [
                ("Inference (SecBERT)", True, "model_loaded=True, gpu=fake"),
                ("LLM (anthropic/claude-haiku-4-5)", True, "API key configured"),
                ("ChromaDB", True, "5 chunks indexed"),
            ]

        monkeypatch.setattr(cli, "_collect_health", _fake_collect)
        result = runner.invoke(cli.app, ["health"])
        assert result.exit_code == 0
        assert "UP" in result.stdout
        assert "Inference" in result.stdout
        assert "ChromaDB" in result.stdout

    def test_any_down_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_collect(_s: Any) -> list[tuple[str, bool, str]]:
            return [
                ("Inference (SecBERT)", False, "connection refused"),
                ("LLM (x)", True, "ok"),
                ("ChromaDB", True, "ok"),
            ]

        monkeypatch.setattr(cli, "_collect_health", _fake_collect)
        result = runner.invoke(cli.app, ["health"])
        assert result.exit_code == 1
        assert "DOWN" in result.stdout


class TestLLMConfigCheck:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        s = MagicMock()
        s.llm_model = "anthropic/claude-haiku-4-5"
        _, ok, details = cli._check_llm_config(s)
        assert ok is False
        assert "ANTHROPIC_API_KEY" in details

    def test_present_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        s = MagicMock()
        s.llm_model = "anthropic/claude-haiku-4-5"
        _, ok, _ = cli._check_llm_config(s)
        assert ok is True


def _write_jsonl(path: Path, n: int = 5) -> Path:
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "event_id": f"e{i}",
                    "timestamp": "2025-03-15T10:23:00Z",
                    "event_type": "ids_alert",
                    "severity": "high",
                    "description": f"log {i}",
                }
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestAnalyze:
    def test_happy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_file = _write_jsonl(tmp_path / "logs.jsonl", 5)
        output_file = tmp_path / "out.json"
        pipeline = _fake_pipeline(_sample_result())
        monkeypatch.setattr(cli, "_build_pipeline", lambda s, *, include_noise: pipeline)

        result = runner.invoke(
            cli.app,
            [
                "analyze",
                str(input_file),
                "--output",
                str(output_file),
                "--skip-recommendations",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert output_file.exists()
        payload = json.loads(output_file.read_text())
        assert payload["correlation_id"] == "run-test"
        assert payload["stats"]["total_logs"] == 5
        assert "Run summary" in result.stdout

    def test_writes_markdown_stub_if_formatter_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_file = _write_jsonl(tmp_path / "logs.jsonl", 3)
        out = tmp_path / "r.json"
        monkeypatch.setattr(
            cli, "_build_pipeline", lambda s, *, include_noise: _fake_pipeline(_sample_result())
        )
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *a: Any, **kw: Any) -> Any:
            if name == "soc_agent.reports.markdown_formatter":
                raise ImportError("not installed")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        result = runner.invoke(
            cli.app,
            ["analyze", str(input_file), "--output", str(out), "--markdown", "-s"],
        )
        result = runner.invoke(
            cli.app,
            ["analyze", str(input_file), "--output", str(out), "--markdown"],
        )
        assert result.exit_code == 0, result.stdout
        md = out.with_suffix(".md")
        assert md.exists()
        assert "Markdown formatter not yet installed" in md.read_text()

    def test_empty_input_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            cli, "_build_pipeline", lambda s, *, include_noise: _fake_pipeline(_sample_result())
        )
        result = runner.invoke(cli.app, ["analyze", str(empty)])
        assert result.exit_code == 1
        assert "No valid logs" in result.stdout

    def test_missing_input_file(self) -> None:
        result = runner.invoke(cli.app, ["analyze", "/nonexistent.jsonl"])
        assert result.exit_code != 0

    def test_passes_skip_recommendations_to_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        input_file = _write_jsonl(tmp_path / "logs.jsonl", 2)
        pipeline = _fake_pipeline(_sample_result())
        monkeypatch.setattr(cli, "_build_pipeline", lambda s, *, include_noise: pipeline)
        result = runner.invoke(
            cli.app,
            ["analyze", str(input_file), "--skip-recommendations"],
        )
        assert result.exit_code == 0
        call = pipeline.process.await_args
        assert call.kwargs["skip_recommendations"] is True
