.PHONY: help up down logs health analyze index test lint typecheck clean build

help:
	@echo "Targets:"
	@echo "  make up            — start the inference service (GPU)"
	@echo "  make down          — stop the stack"
	@echo "  make logs          — tail inference_service logs"
	@echo "  make health        — run `soc-agent health` in the agent container"
	@echo "  make index         — rebuild the Chroma playbook index"
	@echo "  make analyze FILE=logs.jsonl  — run the pipeline on input/\$$FILE"
	@echo "  make test          — run the full test suite locally"
	@echo "  make lint          — ruff check"
	@echo "  make typecheck     — mypy strict"
	@echo "  make build         — build both Docker images"
	@echo "  make clean         — wipe local data/ cache/ chroma/ reports/"

up:
	docker compose up -d inference_service

down:
	docker compose down

logs:
	docker compose logs -f inference_service

health:
	docker compose run --rm agent soc-agent health

index:
	docker compose run --rm agent soc-agent index-playbooks

analyze:
	@test -n "$(FILE)" || (echo "Usage: make analyze FILE=<name>"; exit 1)
	docker compose run --rm agent soc-agent analyze /app/input/$(FILE) --markdown --verbose

build:
	docker compose build

test:
	pytest tests/ -q

lint:
	ruff check soc_agent/ inference_service/ tests/

typecheck:
	mypy soc_agent/ inference_service/

clean:
	rm -rf data/cache data/chroma reports/* .pytest_cache .mypy_cache .ruff_cache
