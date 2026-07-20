import asyncio
from collections.abc import Callable

from market_agent.agent.state import AnalysisPlan, AnalysisState, JudgeVerdict
from market_agent.core.config import Settings
from market_agent.core.errors import AnalysisError, ErrorCode
from market_agent.core.logging import get_logger
from market_agent.llm.base import StructuredLLM
from market_agent.tools.models import CollectedData
from market_agent.tools.report import MarketReport
from market_agent.tools.scraper.adapters import KNOWN_PLATFORMS, get_adapters
from market_agent.tools.scraper.base import PlatformAdapter
from market_agent.tools.sentiment import analyze_sentiment
from market_agent.tools.trends import (
    TrendInsights,
    TrendInterpretation,
    compute_trend_stats,
)

log = get_logger(__name__)

PLANNER_SYSTEM = (
    "You are the planning module of an e-commerce market-analysis agent. "
    "The input is a product name. Plan the COMPLETE analysis for it — every "
    "available analysis, on all platforms — unless a caller constraint narrows "
    "the scope, in which case respect the constraint exactly. "
    "Available analyses: 'sentiment' (customer review insights), 'trends' "
    "(price/popularity statistics). Available platforms: {platforms}. "
    "Explain your choice briefly."
)

PLANNER_USER_TEMPLATE = (
    "Product: {query}\nCaller constraints: {constraints}\nProduce the analysis plan."
)

SYNTHESIS_SYSTEM = (
    "You are a senior e-commerce market analyst writing for a business audience. "
    "Write the report in {language}. Ground every claim in the provided data — never "
    "invent numbers. If some analyses are missing, say so in the caveats and lower "
    "your confidence accordingly. Recommendations must be concrete and prioritized."
)

SYNTHESIS_USER_TEMPLATE = (
    "Request: {query}\n\nCollected data summary:\n{data_summary}\n\n"
    "Sentiment insights: {sentiment}\nTrend insights: {trends}\n"
    "Known issues during collection/analysis: {errors}\n{critique_block}"
    "Produce the market report."
)

JUDGE_SYSTEM = (
    "You are a strict quality reviewer for market-analysis reports. Score the report "
    "from 0 to 1 on: grounding (claims match the data), completeness (covers what the "
    "plan requested), actionability (recommendations are concrete). Give a short, "
    "specific critique when the score is below {threshold}."
)

JUDGE_USER_TEMPLATE = (
    "Plan: {plan}\nData available: sentiment={has_sentiment}, trends={has_trends}\n"
    "Report:\n{report_json}\n\nScore this report."
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
        # Only mention constraints that exist: rendering Python's `None` into the
        # prompt reads as "no analyses requested" to some models.
        parts = []
        if requested_analyses is not None:
            parts.append(f"analyses MUST be exactly {list(requested_analyses)}")
        if requested_platforms:
            parts.append(f"platforms MUST be exactly {list(requested_platforms)}")
        constraints = "; ".join(parts) or "none — apply the default policy (complete analysis)"
        plan, usage = await self.llm.generate(
            AnalysisPlan,
            system=PLANNER_SYSTEM.format(platforms=", ".join(KNOWN_PLATFORMS)),
            user=PLANNER_USER_TEMPLATE.format(query=state["query"], constraints=constraints),
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

    async def sentiment(self, state: AnalysisState) -> dict:
        collected = state.get("collected")
        reviews = collected.all_reviews() if collected else []
        try:
            insights, usage = await analyze_sentiment(
                reviews, self.llm, query=state["query"], language=state.get("language", "fr")
            )
            return {"sentiment": insights, "usage": [usage]}
        except Exception as exc:  # degraded, not fatal
            log.warning("sentiment failed", extra={"ctx": {"error": str(exc)}})
            return {
                "sentiment": None,
                "errors": [
                    AnalysisError(code=ErrorCode.LLM_FAILURE, source="sentiment", message=str(exc))
                ],
            }

    async def trends(self, state: AnalysisState) -> dict:
        collected = state.get("collected")
        try:
            assert collected is not None
            stats = compute_trend_stats(collected)
            interp, usage = await self.llm.generate(
                TrendInterpretation,
                system="You are a pricing analyst. One or two sentences, in "
                f"{state.get('language', 'fr')}, interpreting the statistics.",
                user=stats.model_dump_json(),
                context={"query": state["query"], **stats.model_dump()},
                purpose="trends",
            )
            return {
                "trends": TrendInsights(stats=stats, interpretation=interp.interpretation),
                "usage": [usage],
            }
        except Exception as exc:
            log.warning("trends failed", extra={"ctx": {"error": str(exc)}})
            return {
                "trends": None,
                "errors": [
                    AnalysisError(code=ErrorCode.LLM_FAILURE, source="trends", message=str(exc))
                ],
            }

    def _data_summary(self, state: AnalysisState) -> str:
        collected = state.get("collected")
        if not collected:
            return "NO DATA COLLECTED"
        lines = []
        for p in collected.platforms:
            prices = [o.price for o in p.offers]
            price_range = f"{min(prices):.2f}-{max(prices):.2f} EUR" if prices else "aucune offre"
            lines.append(
                f"- {p.platform}: {len(p.offers)} offers "
                f"({price_range}), {len(p.reviews)} reviews, "
                f"popularity {p.popularity_score}/100"
            )
        return "\n".join(lines)

    async def synthesize(self, state: AnalysisState) -> dict:
        judge = state.get("judge")
        critique = judge.critique if judge and not judge.passed else None
        sentiment, trends = state.get("sentiment"), state.get("trends")
        plan = state.get("plan")
        caveats = [
            f"Source '{e.source}' indisponible : {e.message}" for e in state.get("errors", [])
        ]
        missing = [
            kind
            for kind in (plan.analyses if plan else [])
            if (kind == "sentiment" and sentiment is None) or (kind == "trends" and trends is None)
        ]
        avg_price = None
        if trends is not None:
            avg_price = trends.stats.avg_price
        elif state.get("collected"):
            offers = state["collected"].all_offers()
            if offers:
                avg_price = round(sum(o.price for o in offers) / len(offers), 2)

        report, usage = await self.llm.generate(
            MarketReport,
            system=SYNTHESIS_SYSTEM.format(language=state.get("language", "fr")),
            user=SYNTHESIS_USER_TEMPLATE.format(
                query=state["query"],
                data_summary=self._data_summary(state),
                sentiment=sentiment.model_dump_json() if sentiment else "unavailable",
                trends=trends.model_dump_json() if trends else "unavailable",
                errors="; ".join(c for c in caveats) or "none",
                critique_block=(
                    f"\nA reviewer rejected the previous version with this critique — fix it:\n"
                    f"{critique}\n\n"
                    if critique
                    else ""
                ),
            ),
            context={
                "query": state["query"],
                "language": state.get("language", "fr"),
                "avg_price": avg_price,
                "sentiment_summary": sentiment.summary if sentiment else None,
                "trend_summary": trends.interpretation if trends else None,
                "degraded": bool(caveats or missing),
                "caveats": caveats + [f"Analyse '{m}' indisponible." for m in missing],
                "critique": critique,
            },
            purpose="synthesize",
        )
        return {
            "report": report,
            "usage": [usage],
            "revision_count": state.get("revision_count", 0) + 1,
        }

    async def judge(self, state: AnalysisState) -> dict:
        report = state.get("report")
        assert report is not None
        plan = state.get("plan")
        try:
            verdict, usage = await self.llm.generate(
                JudgeVerdict,
                system=JUDGE_SYSTEM.format(threshold=self.settings.judge_threshold),
                user=JUDGE_USER_TEMPLATE.format(
                    plan=plan.model_dump_json() if plan else "{}",
                    has_sentiment=state.get("sentiment") is not None,
                    has_trends=state.get("trends") is not None,
                    report_json=report.model_dump_json(),
                ),
                context={
                    "query": state["query"],
                    "revision_count": state.get("revision_count", 0) - 1,
                    "threshold": self.settings.judge_threshold,
                },
                purpose="judge",
            )
            # Graph enforces the quality gate on the numeric score, not the LLM's
            # self-reported `passed` flag — configuration governs, not model whim.
            verdict.passed = verdict.score >= self.settings.judge_threshold
            return {"judge": verdict, "usage": [usage]}
        except Exception as exc:  # judge failure must never kill a finished report
            log.warning("judge failed", extra={"ctx": {"error": str(exc)}})
            return {
                "judge": JudgeVerdict(score=1.0, passed=True, critique="judge unavailable"),
                "errors": [
                    AnalysisError(code=ErrorCode.LLM_FAILURE, source="judge", message=str(exc))
                ],
            }
