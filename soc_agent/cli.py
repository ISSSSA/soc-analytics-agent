from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from soc_agent import __version__
from soc_agent.classification.classifier import ClassifierClient
from soc_agent.clustering.clusterer import HDBSCANClusterer
from soc_agent.clustering.embedder import EmbedderClient
from soc_agent.config import Settings, get_settings
from soc_agent.io import load_logs
from soc_agent.logging_setup import configure_logging
from soc_agent.pipeline import SOCPipeline
from soc_agent.rag.indexer import PlaybookIndexer
from soc_agent.rag.retriever import PlaybookRetriever
from soc_agent.recommendations.generator import RecommendationGenerator
from soc_agent.recommendations.llm_client import LLMClient, build_openrouter_headers
from soc_agent.schemas import PipelineResult

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="soc-agent",
    help="SOC agent: cluster SIEM logs, classify, and generate LLM recommendations.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("version")
def version_cmd() -> None:
    s = get_settings()
    table = Table(title=f"SOC Agent {__version__}", show_header=False)
    table.add_row("Version", __version__)
    table.add_row("Python package", "soc_agent")
    table.add_row("Inference URL", str(s.inference_service_url))
    table.add_row("SecBERT path", s.secbert_model_path)
    table.add_row("LLM model", s.llm_model)
    table.add_row("Embedding model", s.embedding_model)
    table.add_row("Batch size", str(s.batch_size))
    table.add_row(
        "HDBSCAN (min_cluster / min_samples)",
        f"{s.hdbscan_min_cluster_size} / {s.hdbscan_min_samples}",
    )
    console.print(table)


@app.command("index-playbooks")
def index_playbooks_cmd(
    playbooks_dir: Annotated[
        Path | None,
        typer.Option("--playbooks-dir", help="Override the configured playbooks dir."),
    ] = None,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Drop the collection and reindex everything.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)
    pb_dir = playbooks_dir or settings.playbooks_dir
    console.print(f"[cyan]Indexing playbooks from[/cyan] {pb_dir}")
    indexer = PlaybookIndexer(
        playbooks_dir=pb_dir,
        chroma_persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )
    stats = indexer.index(rebuild=rebuild)
    console.print(
        f"[green]✓[/green] indexed={stats.indexed_files} "
        f"skipped={stats.skipped_files} total_chunks={stats.total_chunks}"
    )


@app.command("health")
def health_cmd() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    table = Table(title="SOC Agent — Upstream Health", show_header=True)
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Details", overflow="fold")

    rows = asyncio.run(_collect_health(settings))
    for name, ok, details in rows:
        mark = "[green]✓ UP[/green]" if ok else "[red]✗ DOWN[/red]"
        table.add_row(name, mark, details)
    console.print(table)
    if not all(ok for _, ok, _ in rows):
        raise typer.Exit(code=1)


async def _collect_health(settings: Settings) -> list[tuple[str, bool, str]]:
    import httpx

    rows: list[tuple[str, bool, str]] = []

    key = (
        settings.inference_service_api_key.get_secret_value()
        if settings.inference_service_api_key
        else None
    )
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        async with httpx.AsyncClient(
            base_url=str(settings.inference_service_url),
            timeout=5.0,
            headers=headers,
        ) as client:
            r = await client.get("/health")
            if r.status_code == 200:
                body = r.json()
                msg = (
                    f"model_loaded={body.get('model_loaded')}, "
                    f"gpu={body.get('gpu_name') or 'CPU'}"
                )
                rows.append(("Inference (SecBERT)", True, msg))
            else:
                rows.append(
                    ("Inference (SecBERT)", False, f"HTTP {r.status_code}")
                )
    except Exception as e:
        rows.append(("Inference (SecBERT)", False, f"{e.__class__.__name__}: {e}"))

    rows.append(_check_llm_config(settings))
    rows.append(_check_chroma(settings))
    return rows


def _check_llm_config(settings: Settings) -> tuple[str, bool, str]:
    import os

    provider = settings.llm_model.split("/", 1)[0] if "/" in settings.llm_model else ""
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    expected = env_map.get(provider)
    if expected and not os.environ.get(expected):
        return (
            f"LLM ({settings.llm_model})",
            False,
            f"{expected} not set in env",
        )
    return (
        f"LLM ({settings.llm_model})",
        True,
        "API key configured" if expected else "No key check for this provider",
    )


def _check_chroma(settings: Settings) -> tuple[str, bool, str]:
    try:
        retriever = PlaybookRetriever(
            chroma_persist_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
        )
        n = retriever.count()
    except Exception as e:
        return ("ChromaDB", False, f"{e.__class__.__name__}: {e}")
    if n == 0:
        return ("ChromaDB", False, "No chunks indexed (run `soc-agent index-playbooks`)")
    return ("ChromaDB", True, f"{n} chunks indexed")


@app.command("analyze")
def analyze_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="CSV / JSON / JSONL SIEM log file",
        ),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="JSON output path")
    ] = None,
    markdown: Annotated[
        bool, typer.Option("--markdown", "-m", help="Also write a Markdown report")
    ] = False,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", "-b", help="Embedding batch size"),
    ] = None,
    min_cluster_size: Annotated[
        int | None,
        typer.Option("--min-cluster-size", help="Override HDBSCAN min_cluster_size"),
    ] = None,
    skip_recommendations: Annotated[
        bool,
        typer.Option("--skip-recommendations", help="Do not call the LLM"),
    ] = False,
    include_noise: Annotated[
        bool,
        typer.Option("--include-noise", help="Emit a noise incident (cluster_id=-1)"),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)
    if batch_size is not None:
        settings = settings.model_copy(update={"batch_size": batch_size})
    if min_cluster_size is not None:
        settings = settings.model_copy(update={"hdbscan_min_cluster_size": min_cluster_size})
    settings.ensure_dirs()

    result = asyncio.run(
        _run_analyze(
            settings=settings,
            input_file=input_file,
            skip_recommendations=skip_recommendations,
            include_noise=include_noise,
        )
    )
    _write_outputs(result, output, markdown)
    _print_summary(result)


def _build_pipeline(settings: Settings, *, include_noise: bool) -> SOCPipeline:
    api_key = (
        settings.inference_service_api_key.get_secret_value()
        if settings.inference_service_api_key
        else None
    )
    base_url = str(settings.inference_service_url)

    embedder = EmbedderClient(
        base_url=base_url,
        api_key=api_key,
        cache_dir=settings.cache_dir,
        batch_size=settings.batch_size,
    )
    classifier = ClassifierClient(
        base_url=base_url,
        api_key=api_key,
        batch_size=settings.batch_size,
    )
    clusterer = HDBSCANClusterer(
        min_cluster_size=settings.hdbscan_min_cluster_size,
        min_samples=settings.hdbscan_min_samples,
    )
    retriever = PlaybookRetriever(
        chroma_persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )
    extra_headers = build_openrouter_headers(
        app_name=settings.openrouter_app_name,
        site_url=settings.openrouter_site_url,
    )
    llm = LLMClient(
        model=settings.llm_model,
        max_concurrent=settings.llm_max_concurrent,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        extra_headers=extra_headers or None,
    )
    generator = RecommendationGenerator(llm_client=llm, retriever=retriever)

    return SOCPipeline(
        embedder=embedder,
        clusterer=clusterer,
        classifier=classifier,
        generator=generator,
        include_noise=include_noise,
    )


async def _run_analyze(
    *,
    settings: Settings,
    input_file: Path,
    skip_recommendations: bool,
    include_noise: bool,
) -> PipelineResult:
    console.print(f"[cyan]Loading logs from[/cyan] {input_file}")
    load = load_logs(input_file)
    console.print(
        f"[cyan]→ parsed {load.parsed}/{load.total_rows} "
        f"(skipped {load.skipped}, format={load.source_format})[/cyan]"
    )
    if load.parsed == 0:
        console.print("[red]No valid logs to analyse — aborting.[/red]")
        raise typer.Exit(code=1)

    pipeline = _build_pipeline(settings, include_noise=include_noise)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_ids: dict[str, TaskID] = {}

        def cb(stage: str, current: int, total: int) -> None:
            label = {
                "embedding": "[blue]Embedding",
                "clustering": "[green]Clustering",
                "classifying": "[yellow]Classifying",
                "recommending": "[magenta]Generating recommendations",
            }.get(stage, stage)
            if stage not in task_ids:
                task_ids[stage] = progress.add_task(label, total=total)
            progress.update(task_ids[stage], completed=current, total=total)

        return await pipeline.process(
            load.logs,
            skip_recommendations=skip_recommendations,
            progress_callback=cb,
        )


def _default_output_path(input_file: Path, *, ext: str) -> Path:
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    return reports / f"report_{input_file.stem}_{stamp}.{ext}"


def _write_outputs(result: PipelineResult, output: Path | None, markdown: bool) -> None:
    json_path = output or _default_output_path(Path(result.correlation_id or "run"), ext="json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        result.model_dump_json(indent=2, by_alias=True),
        encoding="utf-8",
    )
    console.print(f"[green]✓[/green] JSON: {json_path}")

    if markdown:
        md_path = json_path.with_suffix(".md")
        try:
            from soc_agent.reports.markdown_formatter import format_report

            md_path.write_text(format_report(result), encoding="utf-8")
        except ImportError:
            md_path.write_text(
                f"# SOC Incident Report\n\n"
                f"Markdown formatter not yet installed. "
                f"See JSON: {json_path.name}\n",
                encoding="utf-8",
            )
        console.print(f"[green]✓[/green] Markdown: {md_path}")


def _print_summary(result: PipelineResult) -> None:
    from collections import Counter

    priorities: Counter[str] = Counter(
        inc.recommendation.priority.value for inc in result.incidents
    )
    fallback_count = sum(1 for inc in result.incidents if inc.recommendation_unavailable)

    table = Table(title="Run summary", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total logs", str(result.stats.total_logs))
    table.add_row("Incidents", str(len(result.incidents)))
    table.add_row("Noise points", str(result.stats.n_noise))
    table.add_row("Cache hit rate", f"{result.stats.cache_hit_rate:.1%}")
    table.add_row("Processing time", f"{result.stats.processing_time_seconds:.1f}s")
    tok_in = result.stats.llm_usage.total_tokens_in
    tok_out = result.stats.llm_usage.total_tokens_out
    cost = result.stats.llm_usage.total_cost_usd
    table.add_row("LLM cost", f"${cost:.4f} ({tok_in}→{tok_out} tokens)")
    table.add_row("Fallback recommendations", str(fallback_count))
    for pri in ("critical", "high", "medium", "low", "informational"):
        if priorities.get(pri):
            table.add_row(f"Priority {pri}", str(priorities[pri]))
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
