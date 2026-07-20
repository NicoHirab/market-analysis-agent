import asyncio
from collections.abc import Callable

from market_agent.agent.state import AnalysisPlan, AnalysisState
from market_agent.core.config import Settings
from market_agent.core.errors import AnalysisError, ErrorCode
from market_agent.core.logging import get_logger
from market_agent.llm.base import StructuredLLM
from market_agent.tools.models import CollectedData
from market_agent.tools.scraper.adapters import KNOWN_PLATFORMS, get_adapters
from market_agent.tools.scraper.base import PlatformAdapter

log = get_logger(__name__)

PLANNER_SYSTEM = (
    "You are the planning module of an e-commerce market-analysis agent. "
    "Given a user request, decide which analyses are relevant and on which platforms. "
    "Available analyses: 'sentiment' (customer review insights), 'trends' (price/popularity "
    "statistics). Available platforms: {platforms}. "
    "Request only what the user's question needs. Explain your choice briefly."
)

PLANNER_USER_TEMPLATE = (
    "User request: {query}\n"
    "Constraints: analyses={requested_analyses}, platforms={requested_platforms}\n"
    "If a constraint is not null you MUST respect it exactly; plan the rest yourself."
)


class AgentNodes:
    """Graph node implementations. Each method returns a partial state update."""

    def __init__(
        self,
        llm: StructuredLLM,
        settings: Settings,
        adapters_factory: Callable[[list[str] | None], list[PlatformAdapter]] = get_adapters,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.adapters_factory = adapters_factory

    async def planner(self, state: AnalysisState) -> dict:
        requested_analyses = state.get("requested_analyses")
        requested_platforms = state.get("requested_platforms")
        plan, usage = await self.llm.generate(
            AnalysisPlan,
            system=PLANNER_SYSTEM.format(platforms=", ".join(KNOWN_PLATFORMS)),
            user=PLANNER_USER_TEMPLATE.format(
                query=state["query"],
                requested_analyses=requested_analyses,
                requested_platforms=requested_platforms,
            ),
            context={
                "query": state["query"],
                "requested_analyses": requested_analyses,
                "requested_platforms": requested_platforms,
            },
            purpose="planner",
        )
        # Hard-enforce user constraints even if the LLM ignored them.
        if requested_analyses is not None:
            plan.analyses = [a for a in requested_analyses if a in ("sentiment", "trends")]  # type: ignore[misc]
        if requested_platforms:
            plan.platforms = list(requested_platforms)
        if not plan.platforms:
            plan.platforms = list(KNOWN_PLATFORMS)
        log.info(
            "plan ready",
            extra={"ctx": {"analyses": plan.analyses, "platforms": plan.platforms}},
        )
        return {"plan": plan, "usage": [usage]}

    async def collect(self, state: AnalysisState) -> dict:
        plan = state["plan"]
        assert plan is not None
        try:
            adapters = self.adapters_factory(plan.platforms)
        except ValueError as exc:
            return {
                "collected": None,
                "errors": [
                    AnalysisError(
                        code=ErrorCode.VALIDATION_FAILURE,
                        source="collect",
                        message=str(exc),
                        recoverable=False,
                    )
                ],
            }

        results = await asyncio.gather(
            *(asyncio.to_thread(a.fetch, state["query"]) for a in adapters),
            return_exceptions=True,
        )
        platforms, errors = [], []
        for adapter, result in zip(adapters, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    AnalysisError(
                        code=ErrorCode.ADAPTER_FAILURE, source=adapter.name, message=str(result)
                    )
                )
            else:
                platforms.append(result)

        collected = CollectedData(query=state["query"], platforms=platforms) if platforms else None
        return {"collected": collected, "errors": errors}
