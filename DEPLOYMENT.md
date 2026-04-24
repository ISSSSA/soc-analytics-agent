# Deployment

Three supported topologies:

- **All-in-one GPU host** — `docker compose up` runs both the inference
  service and the agent CLI in the same Docker network. Easiest for small
  runs; wastes GPU when the agent is idle.
- **Split (recommended)** — the inference service lives on a rented GPU
  (Vast.ai / RunPod), the agent CLI runs locally or on a cheap CPU VM. Pay
  for the GPU only while processing.
- **On-demand (Modal)** — cold-start the GPU service per run via Modal
  Labs; only pay for seconds used.

## Vast.ai (recommended for ad-hoc runs)

1. Pick a machine.
   - **GPU**: RTX 4090 24 GB (or L40S / H100 if you need throughput).
   - **Filter**: CUDA ≥ 12.1, DLPerf ≥ 20, reliability ≥ 95 %.
   - **Template**: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`.
   - **Disk**: 40 GB is plenty for the model + HF cache.
2. Expose HTTP port `8001` in the instance configuration.
3. On-start script:
   ```bash
   cd /workspace
   apt-get update && apt-get install -y git
   git clone https://github.com/<you>/siem-agent.git && cd siem-agent
   # Supply the bearer token at launch:
   export INFERENCE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
   echo "INFERENCE_API_KEY=$INFERENCE_API_KEY" > .env
   docker compose up -d --build inference_service
   ```
4. Copy the token back to your local `.env` as `INFERENCE_SERVICE_API_KEY`.
5. Tunnel to the service while you run the agent locally:
   ```bash
   ssh -NL 8001:localhost:8001 root@<vast-host>    # keep open
   soc-agent health                                 # points at localhost:8001
   ```
6. Alternatively, expose via Caddy/Traefik on a real domain with TLS and
   tighten the `allowed_hosts` / IP allow-list — **never expose plain HTTP
   on a public IP, the bearer token alone is not enough**.

## RunPod (more stable than Vast, slightly pricier)

1. Template: **PyTorch 2.3 CUDA 12.1**.
2. GPU: RTX 4090 or A40.
3. Volume: persistent, 20 GB for HF cache (mounted at `/models`).
4. Expose HTTP service on port `8001` — RunPod issues a URL like
   `https://<pod-id>-8001.proxy.runpod.net`.
5. Set `INFERENCE_SERVICE_URL` on the agent side to that URL. TLS is handled
   by RunPod's proxy.

## Modal (per-second billing)

When processing is episodic (weekly batch, not 24/7) Modal is cheaper.

- Cold start ~30 s (model load).
- Processing 100 k logs: ~30 min on an A10 ≈ **$0.50**.
- Create a Modal app from `inference_service/` — mount the HF cache volume
  so the second run starts warm.
- The agent keeps the same HTTP contract — swap `INFERENCE_SERVICE_URL`.

## Cost reference

| Workload | GPU | LLM | Total |
| --- | --- | --- | --- |
| 100 k logs (~500 incidents) | RTX 4090 @ $0.35/hr × 30 min = **$0.18** | Claude Haiku 4.5 × 500 recs ≈ **$1.50** | **~$2** |
| 24/7 standby | ~$250 / month | — | Prefer on-demand |

## Security checklist (before production)

- [ ] `INFERENCE_SERVICE_API_KEY` generated with `secrets.token_urlsafe(32)` and **not** committed.
- [ ] TLS terminator (Caddy / Traefik / nginx) in front of the inference service — never plain HTTP on a public IP.
- [ ] IP allow-list on the GPU host: only the agent machine's public IP reaches `:8001`.
- [ ] Rate limiting enabled (default: `100/minute` per source IP via slowapi).
- [ ] Log redaction: scrub PII from descriptions before they reach the LLM if any are present (document-level filter or prompt-level instruction).
- [ ] `.env` files and Docker secrets managed via a secret manager (1Password, Vault, AWS SM), not plaintext on disk.
- [ ] LLM API keys rotated quarterly; monitor cost via `litellm.completion_cost()` telemetry in `PipelineStats.llm_usage`.
- [ ] Pinned container image digests in production (don't rely on `:latest`).

## Monitoring

- **Prometheus** — the inference service exposes `/metrics` in Prometheus
  text format: request counts, latency histogram, batch size, GPU memory.
  Scrape with any Prometheus-compatible agent.
- **Correlation ID** — every `PipelineResult` carries a `correlation_id`
  (short UUID). Surface it in structured logs to correlate the full run.
- **Cache hit rate** — exposed in `PipelineResult.stats.cache_hit_rate`;
  falling hit rate often means the SecBERT checkpoint changed and the cache
  needs pruning (`make clean`).

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `soc-agent health` → Inference DOWN | SSH tunnel down, pod rebooted | Re-open tunnel, check GPU host is healthy |
| `/classify` returns 501 | Loaded model has no classifier head (e.g., MINI) | Set `SECBERT_MODEL_PATH` to the fine-tuned checkpoint |
| LLM 401 / auth error | Provider key not set in env | `echo $ANTHROPIC_API_KEY` — confirm shell sees it |
| Pipeline slow first run | HF download + diskcache cold | Keep `./data/cache` between runs |
| OOM on GPU | Batch too big for this GPU | Lower `BATCH_SIZE` in `.env` |
| LLM cost surprisingly high | Many clusters + long prompts | Use `--skip-recommendations` to dry-run, then generate only for top-N |
