import asyncio
import json
import re
import time
from pathlib import Path

from market_agent.agent.graph import build_graph, make_initial_state
from market_agent.agent.nodes import AgentNodes
from market_agent.api.registry import Job, JobRegistry, JobStatus
from market_agent.core.config import Settings
from market_agent.core.errors import AnalysisError, ErrorCode
from market_agent.core.logging import get_logger
from market_agent.llm.factory import build_structured_llm
from market_agent.tools.report import render_markdown

log = get_logger(__name__)


class AnalysisService:
    """Owns the compiled graph; turns requests into jobs and jobs into results."""

    def __init__(self, registry: JobRegistry, settings: Settings) -> None:
        self.registry = registry
        self.settings = settings
        self.llm = build_structured_llm(settings)
        self.nodes = AgentNodes(llm=self.llm, settings=settings)
        self.graph = build_graph(self.nodes, settings)
        self._tasks: dict[str, asyncio.Task] = {}

    async def start(
        self,
        product: str,
        *,
        language: str = "fr",
        analyses: list[str] | None = None,
        platforms: list[str] | None = None,
    ) -> Job:
        job = self.registry.create(
            {"product": product, "language": language, "analyses": analyses, "platforms": platforms}
        )
        self._tasks[job.id] = asyncio.create_task(self._run(job))
        return job

    async def wait(self, job_id: str) -> Job:
        task = self._tasks.get(job_id)
        if task is None:
            raise KeyError(job_id)
        await task
        return self.registry.get(job_id)  # type: ignore[return-value]

    async def _run(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        started = time.monotonic()
        state: dict = {}
        try:
            initial = make_initial_state(
                job.request["product"],
                language=job.request["language"],
                analyses=job.request["analyses"],
                platforms=job.request["platforms"],
            )
            state = await asyncio.wait_for(
                self._run_graph(initial), timeout=self.settings.analysis_timeout_s
            )
        except TimeoutError:
            state.setdefault("errors", []).append(
                AnalysisError(
                    code=ErrorCode.TIMEOUT,
                    source="analysis",
                    message=f"analysis exceeded {self.settings.analysis_timeout_s:g}s timeout",
                    recoverable=False,
                )
            )
        except Exception as exc:  # noqa: BLE001 - any orchestration crash becomes a failed job
            log.error(
                "analysis crashed",
                exc_info=True,
                extra={"ctx": {"job": job.id, "error": str(exc)}},
            )
            state.setdefault("errors", []).append(
                AnalysisError(
                    code=ErrorCode.LLM_FAILURE,
                    source="analysis",
                    message=str(exc) or exc.__class__.__name__,
                    recoverable=False,
                )
            )
        finally:
            self._finalize(job, state, started)

    async def _run_graph(self, initial: dict) -> dict:
        """Run the compiled graph to completion and return the final state."""
        return await self.graph.ainvoke(initial)

    def _finalize(self, job: Job, state: dict, started: float) -> None:
        """Build result + meta and set the terminal status. Must never raise:
        a job must always reach a terminal state."""
        duration_ms = int((time.monotonic() - started) * 1000)
        report = state.get("report")
        try:
            job.result = {
                "report": report.model_dump() if report else None,
                "report_markdown": render_markdown(report) if report else None,
                "plan": state["plan"].model_dump() if state.get("plan") else None,
                "judge": state["judge"].model_dump() if state.get("judge") else None,
                "errors": [e.model_dump() for e in state.get("errors", [])],
            }
            job.meta = self._build_meta(state, duration_ms)
        except Exception as exc:  # noqa: BLE001 - finalization must not strand the job
            log.error(
                "finalization failed",
                exc_info=True,
                extra={"ctx": {"job": job.id, "error": str(exc)}},
            )
            report = None
            job.result = {
                "report": None,
                "report_markdown": None,
                "plan": None,
                "judge": None,
                "errors": [
                    {
                        "code": "LLM_FAILURE",
                        "source": "finalize",
                        "message": str(exc),
                        "recoverable": False,
                    }
                ],
            }
            job.meta = {"provider": self.settings.llm_provider, "degraded": True}
        job.status = JobStatus.DONE if report else JobStatus.FAILED
        self._save_run(job)

    def _save_run(self, job: Job) -> None:
        """Archive the finished run to disk (JSON resource + Markdown report).

        Small files, written synchronously at the end of the run; a failure to
        archive is logged and never fails the analysis itself."""
        if not self.settings.runs_dir:
            return
        try:
            product = str(job.request.get("product", ""))
            slug = re.sub(r"[^a-z0-9]+", "-", product.lower()).strip("-")[:40] or "analysis"
            stamp = job.created_at.strftime("%Y%m%d-%H%M%S")
            run_dir = Path(self.settings.runs_dir) / f"{stamp}-{slug}-{job.id[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            resource = {
                "id": job.id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "request": job.request,
                "result": job.result,
                "meta": job.meta,
            }
            (run_dir / "analysis.json").write_text(
                json.dumps(resource, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            markdown = (job.result or {}).get("report_markdown")
            if markdown:
                (run_dir / "report.md").write_text(markdown, encoding="utf-8")
            log.info("run archived", extra={"ctx": {"analysis_id": job.id, "path": str(run_dir)}})
        except Exception as exc:  # noqa: BLE001 - archiving must never fail the job
            log.warning(
                "run archiving failed",
                extra={"ctx": {"analysis_id": job.id, "error": str(exc)}},
            )

    def _build_meta(self, state: dict, duration_ms: int) -> dict:
        usage = state.get("usage", [])
        input_tokens = sum(u.input_tokens for u in usage)
        output_tokens = sum(u.output_tokens for u in usage)
        judge = state.get("judge")
        plan = state.get("plan")
        planned = list(plan.analyses) if plan else []
        missing = [
            k
            for k in planned
            if (k == "sentiment" and state.get("sentiment") is None)
            or (k == "trends" and state.get("trends") is None)
        ]
        return {
            "provider": self.settings.llm_provider,
            "model": getattr(self.llm, "model_name", self.settings.llm_provider),
            "duration_ms": duration_ms,
            "llm_calls": len(usage),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "judge_passed": judge.passed if judge else None,
            "judge_criteria": (
                {
                    "grounding": judge.grounding.passed,
                    "completeness": judge.completeness.passed,
                    "actionability": judge.actionability.passed,
                }
                if judge
                else None
            ),
            "revised": state.get("revision_count", 0) > 1,
            "degraded": bool(state.get("errors")) or bool(missing),
        }
