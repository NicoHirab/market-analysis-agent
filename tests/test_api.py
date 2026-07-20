import json

import pytest
from fastapi.testclient import TestClient

from market_agent.api.app import create_app
from market_agent.core.config import Settings


@pytest.fixture()
def client():
    with TestClient(create_app(Settings(_env_file=None))) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"


def test_analysis_sync_happy_path(client):
    resp = client.post(
        "/api/v1/analyses?wait=true",
        json={"query": "iPhone 16", "language": "fr"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["result"]["report"]["product"] == "iPhone 16"
    assert body["result"]["report_markdown"].startswith("# Rapport")
    assert body["meta"]["llm_calls"] >= 3
    assert body["meta"]["provider"] == "mock"


def test_analysis_async_then_poll(client):
    resp = client.post("/api/v1/analyses", json={"query": "PS5"})
    assert resp.status_code == 202
    job_id = resp.json()["id"]
    # TestClient runs the loop between requests; poll until terminal.
    for _ in range(50):
        body = client.get(f"/api/v1/analyses/{job_id}").json()
        if body["status"] in ("done", "failed"):
            break
    assert body["status"] == "done"
    assert body["result"]["report"] is not None


def test_validation_and_404(client):
    assert client.post("/api/v1/analyses", json={"query": "x"}).status_code == 422
    assert client.post("/api/v1/analyses", json={}).status_code == 422
    assert client.get("/api/v1/analyses/unknown").status_code == 404
    assert client.get("/api/v1/analyses/unknown/report.md").status_code == 404


def test_unknown_platform_yields_failed_resource(client):
    body = client.post(
        "/api/v1/analyses?wait=true", json={"query": "PS5", "platforms": ["ebay"]}
    ).json()
    assert body["status"] == "failed"
    assert body["result"]["errors"]


def test_list_analyses(client):
    client.post("/api/v1/analyses?wait=true", json={"query": "Dyson V15"})
    items = client.get("/api/v1/analyses").json()
    assert items and items[0]["query"] == "Dyson V15"


def test_report_markdown_endpoint(client):
    job_id = client.post("/api/v1/analyses?wait=true", json={"query": "AirPods Pro"}).json()["id"]
    resp = client.get(f"/api/v1/analyses/{job_id}/report.md")
    assert resp.status_code == 200
    assert resp.text.startswith("# Rapport d'analyse")


def test_sse_stream_replays_until_terminal(client):
    job_id = client.post("/api/v1/analyses?wait=true", json={"query": "Kindle"}).json()["id"]
    events = []
    with client.stream("GET", f"/api/v1/analyses/{job_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line.removeprefix("data:").strip()))
    assert events[0]["type"] == "analysis_started"
    assert events[-1]["type"] == "analysis_completed"
    assert any(e["type"] == "node_completed" and e["node"] == "planner" for e in events)


def test_startup_fails_fast_on_missing_key():
    with (
        pytest.raises(RuntimeError, match="LLM_API_KEY"),
        TestClient(create_app(Settings(_env_file=None, llm_provider="groq"))),
    ):
        pass


async def test_sse_stream_tails_live_events_to_terminal():
    import asyncio
    import json

    from httpx import ASGITransport, AsyncClient

    from market_agent.api.registry import JobRegistry
    from market_agent.api.service import AnalysisService

    settings = Settings(_env_file=None)
    app = create_app(settings)
    # ASGITransport does not run lifespan, so wire app.state manually.
    registry = JobRegistry()
    app.state.registry = registry
    app.state.service = AnalysisService(registry, settings)

    job = registry.create({"query": "live tail"})
    registry.publish(job.id, {"type": "analysis_started"})  # snapshot event

    async def publisher():
        await asyncio.sleep(0.05)
        registry.publish(job.id, {"type": "node_completed", "node": "planner", "elapsed_ms": 1})
        await asyncio.sleep(0.05)
        registry.publish(job.id, {"type": "analysis_completed", "status": "done"})

    events: list[dict] = []

    async def consume():
        transport = ASGITransport(app=app)
        async with (
            AsyncClient(transport=transport, base_url="http://t") as ac,
            ac.stream("GET", f"/api/v1/analyses/{job.id}/events") as resp,
        ):
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[len("data:") :].strip()))
                    if events[-1]["type"] == "analysis_completed":
                        break

    await asyncio.gather(consume(), publisher())

    # "analysis_started" comes from the snapshot; the other two arrive via the
    # live queue — proving the await-queue.get() tail loop runs and terminates.
    assert [e["type"] for e in events] == [
        "analysis_started",
        "node_completed",
        "analysis_completed",
    ]
