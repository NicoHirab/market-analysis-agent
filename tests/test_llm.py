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
