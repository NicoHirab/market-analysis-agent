import asyncio

import pytest

from market_agent.api.registry import JobRegistry, JobStatus
from market_agent.api.service import AnalysisService
from market_agent.core.config import Settings


def test_registry_create_get_list():
    reg = JobRegistry()
    a = reg.create({"product": "a"})
    b = reg.create({"product": "b"})
    assert reg.get(a.id).request["product"] == "a"
    assert [j.id for j in reg.list_jobs()] == [b.id, a.id]
    assert reg.get("nope") is None


def _service() -> AnalysisService:
    return AnalysisService(JobRegistry(), Settings(_env_file=None))


async def test_service_runs_analysis_to_done():
    svc = _service()
    job = await svc.start("iPhone 16")
    job = await svc.wait(job.id)
    assert job.status == JobStatus.DONE
    assert job.result["report"]["product"] == "iPhone 16"
    assert "Rapport d'analyse" in job.result["report_markdown"]
    assert job.meta["llm_calls"] >= 3
    assert job.meta["provider"] == "mock"
    assert job.meta["judge_passed"] is True
    assert job.meta["judge_criteria"] == {
        "grounding": True,
        "completeness": True,
        "actionability": True,
    }


async def test_service_marks_failed_on_total_collection_failure():
    svc = _service()
    job = await svc.start("iPhone 16", platforms=["ebay"])  # unknown platform
    job = await svc.wait(job.id)
    assert job.status == JobStatus.FAILED
    assert job.result["report"] is None
    assert job.result["errors"]


async def test_wait_unknown_job_raises():
    with pytest.raises(KeyError):
        await _service().wait("nope")


async def test_service_timeout_records_typed_error(monkeypatch):
    svc = AnalysisService(JobRegistry(), Settings(_env_file=None, analysis_timeout_s=0.02))

    async def _slow(initial):
        await asyncio.sleep(1.0)
        return {}

    monkeypatch.setattr(svc, "_run_graph", _slow)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert job.result["errors"] and job.result["errors"][0]["code"] == "TIMEOUT"
    assert job.meta["degraded"] is True


async def test_service_crash_records_error(monkeypatch):
    svc = _service()

    async def _boom(initial):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(svc, "_run_graph", _boom)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert any("kaboom" in e["message"] for e in job.result["errors"])


async def test_service_archives_run_to_disk(tmp_path):
    import json

    runs = tmp_path / "runs"
    svc = AnalysisService(JobRegistry(), Settings(_env_file=None, runs_dir=str(runs)))
    await svc.wait((await svc.start("iPhone 16")).id)
    run_dirs = list(runs.iterdir())
    assert len(run_dirs) == 1
    assert "iphone-16" in run_dirs[0].name
    data = json.loads((run_dirs[0] / "analysis.json").read_text())
    assert data["status"] == "done"
    assert data["result"]["report"]["product"] == "iPhone 16"
    assert (run_dirs[0] / "report.md").read_text().startswith("# Rapport")


async def test_service_archiving_failure_is_not_fatal(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")  # mkdir under a file will raise
    svc = AnalysisService(JobRegistry(), Settings(_env_file=None, runs_dir=str(blocker / "runs")))
    job = await svc.wait((await svc.start("PS5")).id)
    assert job.status == JobStatus.DONE  # analysis unaffected by archive failure


async def test_service_finalization_failure_still_completes(monkeypatch):
    svc = _service()

    def _boom_meta(state, duration_ms):
        raise RuntimeError("meta boom")

    monkeypatch.setattr(svc, "_build_meta", _boom_meta)
    job = await svc.wait((await svc.start("iPhone 16")).id)
    assert job.status == JobStatus.FAILED
    assert job.result["report"] is None
