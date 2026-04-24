from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


class TestInfraFilesExist:
    @pytest.mark.parametrize(
        "name",
        [
            "Dockerfile.inference",
            "Dockerfile.agent",
            "docker-compose.yml",
            "Makefile",
            ".dockerignore",
            "README.md",
            "DEPLOYMENT.md",
            ".env.example",
        ],
    )
    def test_present(self, name: str) -> None:
        p = ROOT / name
        assert p.exists(), f"{name} is missing"
        assert p.stat().st_size > 0


class TestDockerCompose:
    def test_parses_as_yaml(self) -> None:
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        assert "services" in data
        assert "inference_service" in data["services"]
        assert "agent" in data["services"]

    def test_inference_exposes_8001(self) -> None:
        yaml = pytest.importorskip("yaml")
        data = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
        ports = data["services"]["inference_service"]["ports"]
        assert any(":8001" in p for p in ports)


class TestDockerfiles:
    def test_inference_uses_cuda_base(self) -> None:
        text = (ROOT / "Dockerfile.inference").read_text()
        assert "nvidia/cuda:12.1" in text
        assert "inference_service.server:app" in text
        assert "--workers" in text and '"1"' in text

    def test_agent_uses_python_slim(self) -> None:
        text = (ROOT / "Dockerfile.agent").read_text()
        assert "python:3.11-slim" in text
        assert 'ENTRYPOINT ["soc-agent"]' in text


class TestMakefile:
    def test_has_core_targets(self) -> None:
        text = (ROOT / "Makefile").read_text()
        for target in ("up", "down", "logs", "health", "analyze", "index", "test", "clean"):
            assert f"\n{target}:" in text, f"Makefile target {target} missing"
