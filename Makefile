.PHONY: install test lint fmt run docker-build docker-up demo

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

fmt:
	uv run ruff format src tests && uv run ruff check --fix src tests

run:
	uv run uvicorn market_agent.api.app:app --reload --port 8000

docker-build:
	docker build -t market-agent .

docker-up:
	docker compose up --build

demo:
	bash scripts/demo.sh
