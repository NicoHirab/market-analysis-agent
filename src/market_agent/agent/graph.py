from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from market_agent.agent.nodes import AgentNodes
from market_agent.agent.state import AnalysisState
from market_agent.core.config import Settings

MAX_SYNTHESIS_PASSES = 2  # initial synthesis + at most one judge-driven revision


def make_initial_state(
    query: str,
    *,
    language: str = "fr",
    analyses: list[str] | None = None,
    platforms: list[str] | None = None,
) -> AnalysisState:
    return {
        "query": query,
        "language": language,
        "requested_analyses": analyses,
        "requested_platforms": platforms,
        "plan": None,
        "collected": None,
        "sentiment": None,
        "trends": None,
        "report": None,
        "judge": None,
        "revision_count": 0,
        "errors": [],
        "usage": [],
    }


def build_graph(nodes: AgentNodes, settings: Settings) -> CompiledStateGraph:
    """Hybrid orchestration: the planner LLM decides *what* runs; the graph
    guarantees *how* — parallel fan-out, fan-in, bounded revision loop."""

    def route_after_collect(state: AnalysisState) -> list[str]:
        if state.get("collected") is None:
            return [END]  # hard failure: nothing to analyze
        plan = state.get("plan")
        branches = [a for a in (plan.analyses if plan else [])]
        return branches or ["synthesize"]

    def route_after_synthesize(state: AnalysisState) -> str:
        return "judge" if settings.judge_enabled else END

    def route_after_judge(state: AnalysisState) -> str:
        verdict = state.get("judge")
        if verdict and not verdict.passed and state.get("revision_count", 0) < MAX_SYNTHESIS_PASSES:
            return "synthesize"
        return END

    g = StateGraph(AnalysisState)
    g.add_node("planner", nodes.planner)
    g.add_node("collect", nodes.collect)
    g.add_node("sentiment", nodes.sentiment)
    g.add_node("trends", nodes.trends)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("judge", nodes.judge)

    g.add_edge(START, "planner")
    g.add_edge("planner", "collect")
    g.add_conditional_edges(
        "collect", route_after_collect, ["sentiment", "trends", "synthesize", END]
    )
    g.add_edge("sentiment", "synthesize")  # fan-in: waits for all activated branches
    g.add_edge("trends", "synthesize")
    g.add_conditional_edges("synthesize", route_after_synthesize, ["judge", END])
    g.add_conditional_edges("judge", route_after_judge, ["synthesize", END])
    return g.compile()
