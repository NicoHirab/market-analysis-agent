from market_agent.agent.graph import build_graph, make_initial_state
from market_agent.agent.nodes import AgentNodes
from market_agent.core.config import Settings
from market_agent.llm.mock import MockStructuredLLM
from market_agent.tools.scraper.base import AdapterError, PlatformAdapter


def _graph(settings: Settings | None = None, adapters_factory=None):
    settings = settings or Settings(_env_file=None)
    kwargs = {}
    if adapters_factory is not None:
        kwargs["adapters_factory"] = adapters_factory
    nodes = AgentNodes(llm=MockStructuredLLM(), settings=settings, **kwargs)
    return build_graph(nodes, settings)


async def test_full_run_produces_judged_report():
    result = await _graph().ainvoke(make_initial_state("iPhone 16"))
    assert result["report"] is not None
    assert result["judge"].passed is True
    assert result["revision_count"] == 1
    purposes = [u.purpose for u in result["usage"]]
    assert {"planner", "synthesize", "judge"} <= set(purposes)


async def test_parallel_branches_both_run_on_auto():
    result = await _graph().ainvoke(make_initial_state("Nike Air Max avis clients et prix"))
    assert result["sentiment"] is not None
    assert result["trends"] is not None


async def test_explicit_trends_only_skips_sentiment():
    result = await _graph().ainvoke(make_initial_state("PS5", analyses=["trends"]))
    assert result["trends"] is not None
    assert result.get("sentiment") is None
    assert "sentiment" not in [u.purpose for u in result["usage"]]


async def test_judge_revision_loop_runs_exactly_once():
    result = await _graph().ainvoke(make_initial_state("iPhone 16 force-revision"))
    assert result["revision_count"] == 2  # initial synthesis + one revision
    assert result["judge"].passed is True
    assert [u.purpose for u in result["usage"]].count("synthesize") == 2


async def test_total_collect_failure_ends_without_report():
    class Boom(PlatformAdapter):
        name = "amazon"

        def fetch(self, query):
            raise AdapterError("amazon", "down")

    result = await _graph(adapters_factory=lambda p: [Boom()]).ainvoke(
        make_initial_state("iPhone 16")
    )
    assert result.get("report") is None
    assert result["errors"]


async def test_judge_disabled_skips_judge_node():
    settings = Settings(_env_file=None, judge_enabled=False)
    result = await _graph(settings=settings).ainvoke(make_initial_state("iPhone 16"))
    assert result["report"] is not None
    assert result.get("judge") is None
