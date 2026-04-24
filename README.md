# SOC Agent

Batch analysis of SIEM log exports:

```
logs.jsonl ──▶ SecBERT embeddings (GPU) ──▶ HDBSCAN clustering
                                             ├─▶ SecBERT classifier (GPU)
                                             ├─▶ RAG retrieve (SOC playbooks)
                                             └─▶ Cloud LLM (LiteLLM) ──▶ IncidentReport[]
                                                                         + Markdown
```

- **GPU process** — `inference_service/` (FastAPI + fine-tuned SecBERT).
  Hosted isolated on a rented GPU (Vast.ai, RunPod) or locally.
- **CPU process** — `soc_agent/` CLI. Talks to the GPU service over HTTP,
  calls a cloud LLM through [LiteLLM](https://docs.litellm.ai/) for
  provider-agnostic structured output, and writes an analyst-facing report.

## Quickstart

### Local dev (no Docker, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[agent,inference,dev]"
# Run the CPU/GPU services together on this machine:
cp .env.example .env       # fill in ANTHROPIC_API_KEY, INFERENCE_SERVICE_API_KEY
uvicorn inference_service.server:app --port 8001 --workers 1 &
soc-agent index-playbooks
soc-agent health
soc-agent analyze input/sample.jsonl --markdown
```

On CPU the inference service defaults to
`sentence-transformers/all-MiniLM-L6-v2` (embeddings only; `/classify` will
return 501). Override `SECBERT_MODEL_PATH` for the real checkpoint.

### Docker Compose (GPU on the same host)

```bash
cp .env.example .env   # edit keys
make up                # inference_service on :8001
make health            # verify all upstreams up
make index             # build the playbook Chroma index
make analyze FILE=logs.jsonl
```

Outputs land in `reports/report_<run>_<timestamp>.json` and `.md`.

### Split deployment (agent local, GPU rented)

1. Deploy `Dockerfile.inference` on a GPU host — see [DEPLOYMENT.md](DEPLOYMENT.md).
2. On your laptop, keep only the agent side. Point `INFERENCE_SERVICE_URL` in
   `.env` to the remote host (tunneled via SSH or exposed via HTTPS).
3. `pip install -e ".[agent]"` and run `soc-agent analyze ...` directly.

## Configuration

All runtime settings live in `.env` (see `.env.example`). Key ones:

| Variable | Purpose | Default |
| --- | --- | --- |
| `SECBERT_MODEL_PATH` | HF repo id or local path to fine-tuned SecBERT | `issssssaaaa/secbert-siem` |
| `INFERENCE_SERVICE_URL` | Agent → GPU URL | `http://localhost:8001` |
| `INFERENCE_SERVICE_API_KEY` | Bearer token (same on both sides) | _(required in prod)_ |
| `LLM_MODEL` | LiteLLM model id | `anthropic/claude-haiku-4-5` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` | Provider key | _(one required)_ |
| `LLM_MAX_CONCURRENT` | Requests in flight to LLM provider | `5` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | Minimum cluster | `5` |
| `HDBSCAN_MIN_SAMPLES` | HDBSCAN min_samples | `3` |
| `BATCH_SIZE` | Embed batch to GPU | `64` |

LLM provider API keys are read by LiteLLM from `os.environ` directly — they
are intentionally **not** loaded into the `Settings` model to avoid leaking
them via `settings.model_dump()`.

## CLI

```text
soc-agent analyze <file>         # full pipeline
soc-agent index-playbooks        # (re)build Chroma index
soc-agent health                 # inference + LLM + Chroma up-check
soc-agent version                # show config
```

Common flags for `analyze`:

- `--output path.json` — override JSON output location.
- `--markdown` — also write a Markdown incident report.
- `--skip-recommendations` — run without LLM (embed + cluster only).
- `--include-noise` — emit a synthetic incident for HDBSCAN outliers.
- `--min-cluster-size N` — override the HDBSCAN parameter at runtime.
- `--verbose` — DEBUG logging.

## Architecture

| Component | Module | Notes |
| --- | --- | --- |
| Log I/O | `soc_agent.io.loaders` | CSV / JSON / JSONL, skips malformed rows |
| GPU service | `inference_service/` | FastAPI, Bearer auth, Prometheus metrics, OOM retry |
| Embedder client | `soc_agent.clustering.embedder` | diskcache, tenacity retries |
| Clusterer | `soc_agent.clustering.clusterer` | HDBSCAN on L2-normalized vectors |
| Classifier client | `soc_agent.classification.classifier` | Same retry/auth pattern |
| RAG | `soc_agent.rag` | Chroma + SentenceTransformer + MITRE boost |
| LLM client | `soc_agent.recommendations.llm_client` | LiteLLM, strict JSON schema + json_object fallback |
| Generator | `soc_agent.recommendations.generator` | Fallback on LLM failure |
| Pipeline | `soc_agent.pipeline` | Orchestrator |
| CLI | `soc_agent.cli` | Typer + Rich |
| Reports | `soc_agent.reports` | Markdown |

## Development

```bash
make test         # pytest
make lint         # ruff
make typecheck    # mypy strict
```

Tests run entirely on mocks / the tiny MINI model — no GPU required, no
external API calls. The integration tests that touch the real MINI weights
are auto-skipped when `transformers`/`torch` are missing.

## License

MIT — see [LICENSE](LICENSE) (pending).
