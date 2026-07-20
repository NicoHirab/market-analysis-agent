import asyncio

import pytest

from market_agent.api.registry import JobRegistry, JobStatus
from market_agent.api.service import AnalysisService
from market_agent.core.config import Settings


def test_registry_create_get_list():
    reg = JobRegistry()
    a = reg.create({"query": "a"})
    b = reg.create({"query": "b"})
    assert reg.get(a.id).request["query"] == "a"
    assert [j.id for j in reg.list_jobs()] == [b.id, a.id]
    assert reg.get("nope") is None


async def test_registry_pubsub_replays_and_streams():
    reg = JobRegistry()
    job = reg.create({"query": "x"})
    reg.publish(job.id, {"type": "analysis_started"})
    snapshot, queue = reg.subscribe(job.id)
    assert snapshot == [{"type": "analysis_started"}]
    reg.publish(job.id, {"type": "node_completed", "node": "planner", "elapsed_ms": 3})
    live = queue.get_nowait()
    assert live["node"] == "planner"
    reg.unsubscribe(job.id, queue)


def _service() -> AnalysisService:
    return AnalysisService(JobRegistry(), Settings(_env_file=None))


async def test_service_runs_analysis_to_done():
    svc = _service()
    job = await svc.start("iPhone 16")
    job = await svc.wait(job.id)
    assert job.status == JobStatus.DONE
    assert job.result["report"]["product"] == "iPhone 16"
    assert "Rapport d'analyse" in job.result["report_markdown"]
    nodes_seen = {e.get("node") for e in job.events if e["type"] == "node_completed"}
    assert {"planner", "collect", "synthesize"} <= nodes_seen
    assert job.events[-1] == {"type": "analysis_completed", "status": "done"}
    assert job.meta["llm_calls"] >= 3
    assert job.meta["provider"] == "mock"
    assert job.meta["judge_score"] is not None


async def test_service_marks_failed_on_total_collection_failure():
    svc = _service()
    job = await svc.start("iPhone 16", platforms=["ebay"])  # unknown platform
    job = await svc.wait(job.id)
    assert job.status == JobStatus.FAILED
    assert job.result["report"] is None
    assert job.result["errors"]
    assert job.events[-1]["status"] == "failed"


async def test_wait_unknown_job_raises():
    with pytest.raises(KeyError):
        await _service().wait("nope")


async def test_service_timeout_records_typed_error_and_completes(monkeypatch):
    svc = AnalysisService(JobRegistry(), Settings(_env_file=None, analysis_timeout_s=0.02))

    async def _slow(job, initial):
        await asyncio.sleep(1.0)
        return {}

    monkeypatch.setattr(svc, "_stream_run", _slow)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert job.result["errors"] and job.result["errors"][0]["code"] == "TIMEOUT"
    assert job.meta["degraded"] is True
    assert job.events[-1] == {"type": "analysis_completed", "status": "failed"}


async def test_service_crash_records_error_and_completes(monkeypatch):
    svc = _service()

    async def _boom(job, initial):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(svc, "_stream_run", _boom)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert any("kaboom" in e["message"] for e in job.result["errors"])
    assert job.events[-1]["type"] == "analysis_completed"


async def test_service_finalization_failure_still_completes(monkeypatch):
    svc = _service()

    def _boom_meta(state, duration_ms):
        raise RuntimeError("meta boom")

    monkeypatch.setattr(svc, "_build_meta", _boom_meta)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert job.events[-1]["type"] == "analysis_completed"
