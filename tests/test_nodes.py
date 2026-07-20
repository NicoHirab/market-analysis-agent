from market_agent.agent.nodes import AgentNodes
from market_agent.agent.state import AnalysisPlan, AnalysisState, JudgeVerdict
from market_agent.core.config import Settings
from market_agent.core.errors import ErrorCode
from market_agent.llm.mock import MockStructuredLLM
from market_agent.tools.models import CollectedData
from market_agent.tools.report import MarketReport
from market_agent.tools.scraper.adapters import get_adapters
from market_agent.tools.scraper.base import AdapterError, PlatformAdapter
from market_agent.tools.scraper.mock_data import generate_platform_data


def _nodes(**kw) -> AgentNodes:
    return AgentNodes(
        llm=kw.get("llm", MockStructuredLLM()),
        settings=Settings(_env_file=None),
        adapters_factory=kw.get("adapters_factory", get_adapters),
    )


def _state(**kw) -> AnalysisState:
    base: AnalysisState = {
        "query": "iPhone 16",
        "language": "fr",
        "requested_analyses": None,
        "requested_platforms": None,
        "revision_count": 0,
        "errors": [],
        "usage": [],
    }
    base.update(kw)  # type: ignore[typeddict-item]
    return base


async def test_planner_produces_plan_and_usage():
    update = await _nodes().planner(_state())
    plan = update["plan"]
    assert isinstance(plan, AnalysisPlan)
    assert set(plan.analyses) <= {"sentiment", "trends"}
    assert plan.platforms
    assert update["usage"][0].purpose == "planner"


async def test_planner_respects_explicit_constraints():
    update = await _nodes().planner(
        _state(requested_analyses=["trends"], requested_platforms=["bestbuy"])
    )
    assert update["plan"].analyses == ["trends"]
    assert update["plan"].platforms == ["bestbuy"]


async def test_collect_aggregates_planned_platforms():
    state = _state(
        plan=AnalysisPlan(analyses=["trends"], platforms=["amazon", "bestbuy"], rationale="r")
    )
    update = await _nodes().collect(state)
    collected = update["collected"]
    assert {p.platform for p in collected.platforms} == {"amazon", "bestbuy"}
    assert update.get("errors", []) == []


class _BoomAdapter(PlatformAdapter):
    name = "amazon"

    def fetch(self, query):
        raise AdapterError("amazon", "connection refused")


async def test_collect_partial_failure_degrades_not_crashes():
    def factory(platforms):
        return [_BoomAdapter(), *get_adapters(["bestbuy"])]

    state = _state(plan=AnalysisPlan(analyses=[], platforms=["amazon", "bestbuy"], rationale="r"))
    update = await _nodes(adapters_factory=factory).collect(state)
    assert {p.platform for p in update["collected"].platforms} == {"bestbuy"}
    assert update["errors"][0].code == ErrorCode.ADAPTER_FAILURE
    assert update["errors"][0].source == "amazon"


async def test_collect_total_failure_yields_none_and_errors():
    def factory(platforms):
        return [_BoomAdapter()]

    state = _state(plan=AnalysisPlan(analyses=[], platforms=["amazon"], rationale="r"))
    update = await _nodes(adapters_factory=factory).collect(state)
    assert update["collected"] is None
    assert len(update["errors"]) == 1


def _collected(query: str = "iPhone 16") -> CollectedData:
    return CollectedData(
        query=query,
        platforms=[generate_platform_data(query, p) for p in ("amazon", "bestbuy")],
    )


async def test_sentiment_node_produces_insights():
    update = await _nodes().sentiment(_state(collected=_collected()))
    assert update["sentiment"] is not None
    assert update["usage"][0].purpose == "sentiment"


async def test_sentiment_node_degrades_on_empty_reviews():
    empty = CollectedData(query="x", platforms=[])
    update = await _nodes().sentiment(_state(collected=empty))
    assert update["sentiment"] is None
    assert update["errors"][0].source == "sentiment"


async def test_trends_node_produces_insights():
    update = await _nodes().trends(_state(collected=_collected()))
    assert update["trends"].stats.avg_price > 0
    assert update["trends"].interpretation


async def test_synthesize_produces_report_with_caveats_when_degraded():
    from market_agent.core.errors import AnalysisError, ErrorCode

    state = _state(
        collected=_collected(),
        sentiment=None,
        trends=None,
        errors=[AnalysisError(code=ErrorCode.ADAPTER_FAILURE, source="walmart", message="down")],
        plan=AnalysisPlan(analyses=["sentiment"], platforms=["amazon", "bestbuy"], rationale="r"),
    )
    update = await _nodes().synthesize(state)
    report = update["report"]
    assert isinstance(report, MarketReport)
    assert report.caveats  # degraded sources surfaced
    assert update["revision_count"] == 1


async def test_judge_passes_normal_report():
    state = _state(report=_mock_report(), revision_count=1)
    update = await _nodes().judge(state)
    assert isinstance(update["judge"], JudgeVerdict)
    assert update["judge"].passed is True


async def test_judge_fails_then_flags_revision():
    state = _state(query="iPhone 16 force-revision", report=_mock_report(), revision_count=1)
    update = await _nodes().judge(state)
    assert update["judge"].passed is False
    assert update["judge"].critique


def _mock_report() -> MarketReport:
    return MarketReport(
        product="iPhone 16", executive_summary="s", price_analysis="p", confidence=0.8
    )


async def test_synthesize_handles_platform_with_no_offers():
    # A real adapter can return a platform with zero offers (product not found);
    # synthesis must degrade gracefully, not crash on min()/max() of empty prices.
    from market_agent.tools.models import PlatformData

    empty = PlatformData(
        platform="amazon", offers=[], reviews=[], price_history=[], popularity_score=10.0
    )
    state = _state(
        collected=CollectedData(query="Obscure Item", platforms=[empty]),
        plan=AnalysisPlan(analyses=[], platforms=["amazon"], rationale="r"),
    )
    update = await _nodes().synthesize(state)
    assert isinstance(update["report"], MarketReport)


async def test_judge_enforces_threshold_over_llm_passed_flag():
    # A model may return passed=True with a sub-threshold score; the node must
    # enforce the configured threshold rather than trust the self-report.
    from market_agent.agent.state import JudgeVerdict
    from market_agent.llm.base import LLMUsage

    class _InconsistentLLM:
        async def generate(self, schema, *, system, user, context=None, purpose=""):
            return (
                JudgeVerdict(score=0.3, passed=True, critique=""),
                LLMUsage(purpose=purpose, model="stub"),
            )

    nodes = AgentNodes(llm=_InconsistentLLM(), settings=Settings(_env_file=None))  # threshold 0.7
    update = await nodes.judge(_state(report=_mock_report(), revision_count=1))
    assert update["judge"].score == 0.3
    assert update["judge"].passed is False  # enforced: 0.3 < 0.7
