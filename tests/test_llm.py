import pytest

from market_agent.llm.base import LLMUsage
from market_agent.llm.mock import LLMGenerationError, MockStructuredLLM
from market_agent.tools.trends import TrendInterpretation


async def test_mock_generates_registered_schema_deterministically():
    llm = MockStructuredLLM()
    ctx = {"query": "iPhone 16", "trend_direction": "rising"}
    out1, usage = await llm.generate(
        TrendInterpretation, system="s", user="u", context=ctx, purpose="trends"
    )
    out2, _ = await llm.generate(TrendInterpretation, system="s", user="u", context=ctx)
    assert isinstance(out1, TrendInterpretation)
    assert out1 == out2
    assert "hausse" in out1.interpretation  # rising → French wording
    assert isinstance(usage, LLMUsage)
    assert usage.model == "mock"
    assert usage.purpose == "trends"


async def test_mock_raises_for_unknown_schema():
    class Unknown(TrendInterpretation):
        pass

    llm = MockStructuredLLM()
    with pytest.raises(LLMGenerationError, match="no mock builder"):
        await llm.generate(Unknown, system="s", user="u", context={})


async def test_mock_market_report_builder_semantics():
    from market_agent.tools.report import MarketReport

    llm = MockStructuredLLM()
    ctx = {
        "query": "iPhone 16",
        "language": "fr",
        "avg_price": 899.0,
        "degraded": True,
        "caveats": ["Source 'cdiscount' indisponible : down"],
        "critique": "Manque de précision",
    }
    report, _ = await llm.generate(MarketReport, system="s", user="u", context=ctx)
    assert isinstance(report, MarketReport)
    assert report.product == "iPhone 16"
    assert report.confidence == 0.55  # degraded path
    assert report.caveats == ["Source 'cdiscount' indisponible : down"]
    assert report.executive_summary.endswith("(Version révisée suite au contrôle qualité.)")
    assert "899.00€" in report.executive_summary

    clean, _ = await llm.generate(
        MarketReport, system="s", user="u", context={"query": "iPhone 16"}
    )
    assert clean.confidence == 0.85
    assert clean.caveats == []
