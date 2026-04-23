# SOC Agent (cloud LLM версия) — инструкции проекта

> **Архитектура:** SecBERT на арендованной RTX 4090 (FastAPI + transformers) + Cloud LLM через LiteLLM (Claude/OpenAI/DeepSeek) + ChromaDB RAG локально + Python CLI агент для батч-обработки CSV/JSON.

> **Работаем итеративно:** архитектура → модуль → тесты → следующий модуль. НЕ писать код пока пользователь не апрувнет структуру. После каждого модуля ждать "go" от пользователя.

---

## Преимущества архитектуры

1. **GPU только для SecBERT** — 4090 24GB избыточна, но даст огромный throughput батчей
2. **LLM через API** — лучшее качество рекомендаций без GPU оверхеда
3. **LiteLLM** — единый интерфейс ко всем LLM провайдерам, смена через env переменную
4. **Изоляция отказов** — падение LLM не трогает SecBERT, и наоборот
5. **Дешевле** — платите за GPU только когда реально обрабатываете логи, LLM по usage

---

## ЗАДАЧА

Батч-обработка файла (CSV/JSON) с SIEM логами:
1. Эмбеддинг через fine-tuned SecBERT (HTTP к GPU сервису на RTX 4090)
2. HDBSCAN кластеризация (min_cluster_size=5, min_samples=3, metric='euclidean', cluster_selection_method='eom', после L2-нормализации эмбеддингов)
3. Классификация каждого лога в кластере (тот же SecBERT, /classify endpoint)
4. Retrieve релевантные SOC playbook'и из ChromaDB (top-5 chunks)
5. Генерация структурированных SOC рекомендаций через Cloud LLM (LiteLLM → Claude/OpenAI/DeepSeek)
6. Выход: JSON с IncidentReport[] + markdown отчёт для аналитика

---

## СТЕК

**GPU сервис (арендованная RTX 4090 24GB, Vast.ai или RunPod):**
- Единственный сервис на GPU — inference_service
- FastAPI + transformers, порт 8001
- Endpoints: /embeddings, /classify, /health, /metrics
- FP16, batch до 256 (4090 это легко тянет для SecBERT)
- Docker контейнер с nvidia/cuda:12.1

**CPU агент (может быть на том же хосте или отдельно):**
- Python 3.11+
- httpx async для HTTP к inference service
- hdbscan, numpy, pandas для кластеризации
- chromadb (persistent mode, локально)
- sentence-transformers/all-MiniLM-L6-v2 для эмбеддингов playbook'ов (НЕ SecBERT — он переобучен на логи, general text эмбеддит хуже)
- **litellm** для вызова Cloud LLM — даёт единый OpenAI-совместимый интерфейс к Claude/GPT/DeepSeek/любому провайдеру
- pydantic v2, pydantic-settings, typer, rich, structlog, tenacity, diskcache
- pytest, pytest-asyncio, httpx_mock

**LLM провайдер:**
- По умолчанию через env: `LLM_MODEL=anthropic/claude-haiku-4-5` (оптимум цена/качество)
- Альтернативы через env: `openai/gpt-4o`, `openai/gpt-4o-mini`, `deepseek/deepseek-chat`
- LiteLLM автоматически подхватывает API ключи из env (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY)
- Все вызовы через `litellm.acompletion()` с `response_format` для structured output

---

## ФОРМАТ ДАННЫХ

**Вход:** Advanced_SIEM_Dataset (darkknight25). Ключевые поля:
- `event_id`, `timestamp`, `event_type` (8 классов: auth/firewall/endpoint/network/cloud/ai/iot/ids_alert)
- `severity` (6 уровней), `description` (чистый текст — используем для эмбеддингов)
- `raw_log` (CEF формат — НЕ использовать, он зашумлён префиксами SIEM)
- `advanced_metadata` dict: {session_id, risk_score, confidence, geo_location}
- `behavioral_analytics` dict: {baseline_deviation, entropy, frequency_anomaly, sequence_anomaly}
- `user`, `action` (55 классов — классификационная цель fine-tuned SecBERT), `object`
- `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `mac_address`
- `alert_type`, `signature_id`, `category` (для ids_alert)
- `cloud_service`, `model_id` (для cloud/ai)

**Выход — IncidentReport (JSON):**
```json
{
  "cluster_id": 42,
  "cluster_size": 15,
  "time_range": ["2025-03-15T10:23:00", "2025-03-15T10:47:12"],
  "dominant_event_type": "ids_alert",
  "dominant_classes": [
    {"class": "credential_stuffing", "count": 12, "avg_confidence": 0.87},
    {"class": "brute_force", "count": 3, "avg_confidence": 0.72}
  ],
  "severity_breakdown": {"critical": 8, "high": 7},
  "unique_src_ips": 15,
  "unique_users_targeted": 8,
  "representative_logs": ["...3 лога..."],
  "recommendation": {
    "summary": "Скоординированная атака credential stuffing...",
    "threat_assessment": "High — автоматизированная атака с 15 IP...",
    "relevant_playbooks": ["credential_stuffing_response.md"],
    "immediate_actions": ["Заблокировать IP в firewall", "..."],
    "investigation_steps": ["...", "..."],
    "mitre_techniques": ["T1110.004"],
    "priority": "high"
  }
}
```

---

## ТРЕБОВАНИЯ

1. Type hints везде, mypy clean, pydantic v2 модели для всех интерфейсов
2. Async HTTP, где возможно (httpx.AsyncClient)
3. Кеш эмбеддингов на диск через diskcache по хешу текста
4. Retry с exponential backoff (tenacity) на всех HTTP вызовах
5. Graceful degradation: если LLM упал → выдаём кластеры и классификации с пометкой `recommendation_unavailable` (важно для SOC — нельзя блокировать всю обработку)
6. Rate limiting для LLM вызовов: `max_concurrent_llm_requests=5` (не класть API провайдера)
7. Docker Compose для GPU сервиса + отдельный Dockerfile для CLI агента
8. `.env.example`, `README.md` с quickstart, `DEPLOYMENT.md` для Vast.ai/RunPod
9. Unit тесты для каждого модуля + один end-to-end на моках
10. structlog для логирования с `correlation_id` по каждому запуску

---

## КОНФИГУРАЦИЯ (.env)

```
# Inference service (GPU)
SECBERT_MODEL_PATH=issssssaaaa/secbert-siem  # HF repo id или локальный путь
INFERENCE_SERVICE_URL=http://<ip-арендованной-GPU>:8001
INFERENCE_SERVICE_API_KEY=<bearer-token-для-аутентификации>

# LLM (Cloud)
LLM_MODEL=anthropic/claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-...
# или
# LLM_MODEL=openai/gpt-4o-mini
# OPENAI_API_KEY=sk-...
# или
# LLM_MODEL=deepseek/deepseek-chat
# DEEPSEEK_API_KEY=sk-...

LLM_MAX_CONCURRENT=5
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2000

# RAG
CHROMA_PERSIST_DIR=./data/chroma
PLAYBOOKS_DIR=./playbooks
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Pipeline
CACHE_DIR=./data/cache
BATCH_SIZE=64
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPES=3

# Logging
LOG_LEVEL=INFO
```

---

## ПЛАН РАБОТЫ

**Шаг 1:** Предложить дерево файлов + ключевые Pydantic модели (schemas.py). Ждать апрув.

**Шаг 2 (после апрува):** реализация по одному модулю с тестами, в порядке:
  1. `schemas.py` + `config.py`
  2. `io/loaders.py` (CSV/JSON/JSONL reader с pydantic валидацией)
  3. `inference_service/` (FastAPI + SecBERT, Dockerfile.inference)
  4. `clustering/embedder.py` (HTTP клиент к /embeddings с кешем)
  5. `clustering/clusterer.py` (HDBSCAN wrapper)
  6. `classification/classifier.py` (HTTP клиент к /classify)
  7. `rag/` (indexer + retriever + 5 sample playbook.md)
  8. `recommendations/llm_client.py` (LiteLLM wrapper)
  9. `recommendations/prompts.py` + `generator.py`
  10. `pipeline.py` (оркестратор)
  11. `cli.py` (Typer)
  12. `reports/markdown_formatter.py`
  13. `docker-compose.yml` + `Dockerfile.inference` + `Dockerfile.agent`
  14. End-to-end тест на 100 моковых логах

**После каждого модуля ждать "go" от пользователя.**

---

## ДЕТАЛИ МОДУЛЕЙ

### Inference Service (GPU)

**Endpoints:**

`POST /embeddings` — auth: Bearer token, body: `{"texts": [...], "pooling": "mean"|"cls", "normalize": true}`, response: `{embeddings, model_id, pooling, normalized, processing_time_ms}`

`POST /classify` — auth: Bearer token, body: `{"texts": [...], "top_k": 5, "return_all_scores": false}`, response: `{predictions: [{label, score, top_k}], model_id, processing_time_ms}`

`GET /health` — `{status, model_loaded, gpu_available, gpu_name, gpu_memory_total_mb, gpu_memory_used_mb, supports_classification, num_classes, class_labels}`

`GET /metrics` — Prometheus формат

**Реализация:**
- Lifespan: загрузка из SECBERT_MODEL_PATH; автодетект архитектуры (classifier.weight → AutoModelForSequenceClassification, иначе AutoModel); model.eval() + model.half() для FP16; warmup dummy батчом
- Батчинг: MAX_BATCH_SIZE=256; резать большие запросы; mean pooling = `(last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1e-9)`; L2 нормализация через `torch.nn.functional.normalize(dim=-1)`
- Безопасность: Bearer token в dependency; slowapi 100 req/min per IP; CORS только локалхост; max_texts_per_request=512, max_text_length=2048
- OOM: empty_cache() + retry с batch_size=1 → 503
- Dockerfile: nvidia/cuda:12.1.0-runtime-ubuntu22.04, `--workers 1`
- Тесты: MINI модель (all-MiniLM-L6-v2) вместо SecBERT — тесты за секунды

### LiteLLM Client

```python
litellm.drop_params = True  # для unsupported params не падать
```

- `LLMClient.generate_structured(system, user, schema)` — retry(stop=3, wait=exp(2,4,30)), async semaphore (max_concurrent)
- response_format: json_schema (strict) с `schema.model_json_schema()`
- Cost tracking через `litellm.completion_cost()`
- Fallback: если json_schema не поддерживается → BadRequestError → retry с `{"type": "json_object"}` + схема в промпте + Pydantic валидация вручную
- Для Claude — LiteLLM маппит response_format в tool_use автоматически

### RAG модуль

Создать 5 playbook.md (~3000-5000 символов каждый, на английском):
1. `credential_stuffing_response.md` — T1110.004, T1078
2. `dns_tunneling_response.md` — T1071.004, T1572
3. `container_escape_response.md` — T1611, T1068
4. `ai_model_poisoning_response.md` — LLM01/LLM03 OWASP, T1565
5. `insider_threat_detection.md` — T1078, T1087

Структура playbook: Overview, Detection Indicators, MITRE ATT&CK Mapping, Immediate Actions (first 15 minutes), Investigation Steps, Containment & Eradication, Recovery, Post-Incident.

Indexer:
- MarkdownHeaderTextSplitter по H1/H2/H3
- SentenceTransformer для эмбеддингов
- MITRE regex: `r'T\d{4}(\.\d{3})?'`
- ChromaDB: `hnsw:space=cosine`, metadata: `{source_file, playbook_name, section, mitre_techniques (comma-separated — chromadb не любит lists), file_hash}`
- Идемпотентность: по file_hash, `--rebuild` пересоздаёт

Retriever:
- Query: `f"Security event: {dominant_event_type}. Main activities: {top-3 classes}. Sample: {representative_logs[0].description}"`
- MITRE boost: если MITRE техники упомянуты — boost scores chunks где mitre_techniques содержит их (boost=0.3)

### Recommendations Generator

Ключевые правила в SYSTEM_PROMPT:
1. Ground в provided logs/playbooks. НЕ выдумывать IoCs, attribution, MITRE не из данных.
2. Specific: "Block IPs [a.b.c.d, e.f.g.h] at perimeter firewall via ACL" вместо "Block malicious IPs"
3. Immediate actions — executable за 15 минут Tier 1 аналитиком
4. Investigation steps дают ответы, не просто "investigate further"
5. Cite playbook sections по имени
6. Priority на основе: severity, scope (unique IPs/users), post-exploitation indicators

Fallback: если LLM упал → минимальный Recommendation: `summary="LLM generation failed: {error}. Manual review required."`, `priority=max severity`, `immediate_actions=["Manual SOC review required"]`.

### Pipeline

```python
async def process(logs, progress_callback=None, skip_recommendations=False) -> PipelineResult:
    # 1. Extract description texts
    # 2. Embed batch-by-batch (cache по sha256(text))
    # 3. HDBSCAN clustering
    # 4. For each cluster (incl. noise as singletons):
    #    a. Classify all logs in cluster (batched)
    #    b. Build Cluster object
    #    c. If not skip_recommendations: generate Recommendation
```

Параллелизм:
- Embedding последовательно (GPU one-at-a-time)
- Classification одновременно с LLM generation для готовых кластеров
- LLM вызовы параллельно (до LLM_MAX_CONCURRENT) через asyncio.gather

### CLI (Typer)

```
soc-agent analyze <file> [--output/-o] [--markdown/-m] [--batch-size/-b] [--config/-c]
                         [--min-cluster-size] [--skip-recommendations] [--verbose/-v]
soc-agent index-playbooks [--playbooks-dir] [--rebuild]
soc-agent health
soc-agent version
```

Rich progress bars + Rich table для health. Цвета: cyan (loading), blue (embedding), green (clustering), yellow (classifying), magenta (recommendations).

### Markdown Report

Summary таблица с priority (🔴 Critical / 🟠 High / 🟡 Medium / 🔵 Low / ⚪ Noise). Секции по priority, инциденты внутри каждой — с representative logs в `<details>`.

---

## Docker / Deployment

**docker-compose.yml** два варианта:
- A: всё локально (inference_service с `runtime: nvidia`, healthcheck, + agent как CLI через `docker compose run --rm agent`)
- B: inference_service на Vast.ai, агент локально с `INFERENCE_SERVICE_URL=https://<vast-host>:8001`

**Vast.ai:** RTX 4090 24GB, CUDA ≥ 12.1, DLPerf ≥ 20, RELIABILITY ≥ 95%. Template: nvidia/cuda:12.1.0-runtime-ubuntu22.04. Expose 8001. SSH tunnel для доступа.

**RunPod:** "PyTorch 2.3 CUDA 12.1", RTX 4090 / A40, persistent volume 20GB. HTTPS endpoint вида `https://<pod-id>-8001.proxy.runpod.net`.

**Security checklist:** Bearer token не в git, HTTPS через Caddy/Traefik, IP whitelist, rate limiting, PII не в логи, API ключи в secret manager.

**Cost:** RTX 4090 Vast.ai ~$0.35/hr; 100k логов ≈ $0.35 GPU + $1.50 LLM (Claude Haiku, ~500 incidents) ≈ **$2 total**.

**Альтернатива: Modal** — per-second billing, cold start ~30s, 100k логов ~30 мин ≈ $0.50 (A10).

**Makefile:** up, down, logs, health, analyze (FILE=), index, test, clean.

---

## ДОПОЛНИТЕЛЬНЫЕ УКАЗАНИЯ

### Если предлагается другая структура

Это нормально. Если предложение логичнее — соглашаться. Главное чтобы было:
- Изоляция GPU сервиса от агента (разные процессы)
- LLM через LiteLLM (не прямой SDK к одному провайдеру)
- Async пайплайн для параллелизма HTTP вызовов
- Кеширование эмбеддингов

### Выбор провайдера LLM

**По умолчанию — Claude Haiku 4.5:** $0.80 / $4.00 за 1M токенов, отлично следует инструкциям, fast response (<2 sec), лучший структурированный вывод в классе.

**Если нужно дешевле — DeepSeek V3 API:** ~$0.27 / $1.10 за 1M токенов, качество чуть ниже, риск доступности (rate limits).

**Для production / критичных случаев — Claude Sonnet 4.6:** дороже, но качество существенно выше на сложных multi-step расследованиях.

### Fine-tuned SecBERT

Чекпоинт на Kaggle: `/kaggle/input/models/issssssaaaa/secbert-fine-tuned/pytorch/default/1/secbert_siem_final/`. Планируется перенос на HuggingFace Hub. `SECBERT_MODEL_PATH` должен работать и с HF repo id (`issssssaaaa/secbert-siem`), и с локальным путём.

### Порядок проверки работоспособности

1. Локально на CPU — unit тесты (модели mock'ированы)
2. Inference service на CPU (медленно, проверка кода)
3. Аренда GPU на час, деплой inference, `soc-agent health`
4. `soc-agent analyze` на 100 логах → проверка отчёта
5. Если ок — масштабирование
