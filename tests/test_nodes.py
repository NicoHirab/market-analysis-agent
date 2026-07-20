from market_agent.agent.nodes import AgentNodes
from market_agent.agent.state import AnalysisPlan, AnalysisState
from market_agent.core.config import Settings
from market_agent.core.errors import ErrorCode
from market_agent.llm.mock import MockStructuredLLM
from market_agent.tools.scraper.adapters import get_adapters
from market_agent.tools.scraper.base import AdapterError, PlatformAdapter


def _nodes(**kw) -> AgentNodes:
    return AgentNodes(
        llm=kw.get("llm", MockStructuredLLM()),
        settings=Settings(_env_file=None),
        adapters_factory=kw.get("adapters_factory", get_adapters),
    )


def _state(**kw) -> AnalysisState:
    base: AnalysisState = {
        "query": "iPhone 16", "language": "fr",
        "requested_analyses": None, "requested_platforms": None,
        "revision_count": 0, "errors": [], "usage": [],
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
        _state(requested_analyses=["trends"], requested_platforms=["fnac"])
    )
    assert update["plan"].analyses == ["trends"]
    assert update["plan"].platforms == ["fnac"]


async def test_collect_aggregates_planned_platforms():
    state = _state(
        plan=AnalysisPlan(analyses=["trends"], platforms=["amazon", "fnac"], rationale="r")
    )
    update = await _nodes().collect(state)
    collected = update["collected"]
    assert {p.platform for p in collected.platforms} == {"amazon", "fnac"}
    assert update.get("errors", []) == []


class _BoomAdapter(PlatformAdapter):
    name = "amazon"

    def fetch(self, query):
        raise AdapterError("amazon", "connection refused")


async def test_collect_partial_failure_degrades_not_crashes():
    def factory(platforms):
        return [_BoomAdapter(), *get_adapters(["fnac"])]

    state = _state(plan=AnalysisPlan(analyses=[], platforms=["amazon", "fnac"], rationale="r"))
    update = await _nodes(adapters_factory=factory).collect(state)
    assert {p.platform for p in update["collected"].platforms} == {"fnac"}
    assert update["errors"][0].code == ErrorCode.ADAPTER_FAILURE
    assert update["errors"][0].source == "amazon"


async def test_collect_total_failure_yields_none_and_errors():
    def factory(platforms):
        return [_BoomAdapter()]

    state = _state(plan=AnalysisPlan(analyses=[], platforms=["amazon"], rationale="r"))
    update = await _nodes(adapters_factory=factory).collect(state)
    assert update["collected"] is None
    assert len(update["errors"]) == 1
