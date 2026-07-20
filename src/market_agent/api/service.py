import asyncio
import time

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
        query: str,
        *,
        language: str = "fr",
        analyses: list[str] | None = None,
        platforms: list[str] | None = None,
    ) -> Job:
        job = self.registry.create(
            {"query": query, "language": language, "analyses": analyses, "platforms": platforms}
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
        self.registry.publish(job.id, {"type": "analysis_started"})
        started = time.monotonic()
        state: dict = {}
        try:
            initial = make_initial_state(
                job.request["query"],
                language=job.request["language"],
                analyses=job.request["analyses"],
                platforms=job.request["platforms"],
            )
            state = await asyncio.wait_for(
                self._stream_run(job, initial), timeout=self.settings.analysis_timeout_s
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

    def _finalize(self, job: Job, state: dict, started: float) -> None:
        """Build result + meta and emit the terminal event. Must never raise:
        a job must always reach a terminal state with its completion event."""
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
        self.registry.publish(job.id, {"type": "analysis_completed", "status": job.status.value})

    async def _stream_run(self, job: Job, initial: dict) -> dict:
        """Stream the graph; emit node_completed events; return final state."""
        final: dict = dict(initial)
        last = time.monotonic()
        async for chunk in self.graph.astream(
            initial, stream_mode=["updates", "values"], version="v2"
        ):
            if chunk["type"] == "updates":
                now = time.monotonic()
                for node_name in chunk["data"]:
                    elapsed_ms = int((now - last) * 1000)
                    self.registry.publish(
                        job.id,
                        {"type": "node_completed", "node": node_name, "elapsed_ms": elapsed_ms},
                    )
                    log.info(
                        "node completed",
                        extra={
                            "ctx": {
                                "analysis_id": job.id,
                                "node": node_name,
                                "elapsed_ms": elapsed_ms,
                            }
                        },
                    )
                last = now
            elif chunk["type"] == "values":
                final = chunk["data"]
        return final

    def _build_meta(self, state: dict, duration_ms: int) -> dict:
        usage = state.get("usage", [])
        input_tokens = sum(u.input_tokens for u in usage)
        output_tokens = sum(u.output_tokens for u in usage)
        cost = None
        if (
            self.settings.llm_price_in_per_mtok is not None
            and self.settings.llm_price_out_per_mtok is not None
        ):
            cost = round(
                input_tokens / 1e6 * self.settings.llm_price_in_per_mtok
                + output_tokens / 1e6 * self.settings.llm_price_out_per_mtok,
                6,
            )
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
            "cost_estimate_usd": cost,
            "judge_score": judge.score if judge else None,
            "revised": state.get("revision_count", 0) > 1,
            "degraded": bool(state.get("errors")) or bool(missing),
        }
