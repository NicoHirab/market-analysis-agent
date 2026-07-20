import pytest

from market_agent.core.config import Settings
from market_agent.llm.base import LLMUsage
from market_agent.llm.factory import build_structured_llm
from market_agent.llm.langchain_impl import LangChainStructuredLLM
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


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_factory_mock_provider():
    assert isinstance(build_structured_llm(_settings(llm_provider="mock")), MockStructuredLLM)


def test_factory_openai_compatible_presets():
    llm = build_structured_llm(
        _settings(llm_provider="groq", llm_model="llama-3.3-70b-versatile", llm_api_key="k")
    )
    assert isinstance(llm, LangChainStructuredLLM)
    base = str(llm.model.openai_api_base)
    assert "api.groq.com" in base
    assert llm.model.max_retries == 2


def test_factory_anthropic_native_config():
    llm = build_structured_llm(_settings(llm_provider="anthropic", llm_api_key="k"))
    assert isinstance(llm, LangChainStructuredLLM)
    assert llm.model_name == "claude-haiku-4-5"  # DEFAULT_MODELS fallback
    assert llm.model.__class__.__name__ == "ChatAnthropic"
    assert llm.model.max_retries == 2


def test_factory_custom_base_url_overrides():
    llm = build_structured_llm(
        _settings(llm_provider="custom", llm_model="m", llm_api_key="k",
                  llm_base_url="https://my-gateway.local/v1")
    )
    assert isinstance(llm, LangChainStructuredLLM)
    assert "my-gateway.local" in str(llm.model.openai_api_base)


def test_factory_unknown_provider_without_base_url_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_structured_llm(_settings(llm_provider="nope", llm_api_key="k"))


async def test_langchain_impl_parses_and_retries(monkeypatch):
    """Drive LangChainStructuredLLM with a stubbed chain: first invalid, then valid."""
    from market_agent.tools.trends import TrendInterpretation

    class FakeStructuredRunnable:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"raw": _FakeRaw(), "parsed": None,
                        "parsing_error": ValueError("bad json")}
            return {"raw": _FakeRaw(), "parsed": TrendInterpretation(interpretation="ok"),
                    "parsing_error": None}

    class _FakeRaw:
        usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    class FakeModel:
        def with_structured_output(self, schema, include_raw=False):
            return fake_runnable

    fake_runnable = FakeStructuredRunnable()
    impl = LangChainStructuredLLM(FakeModel(), model_name="fake-model")
    out, usage = await impl.generate(TrendInterpretation, system="s", user="u", purpose="p")
    assert out.interpretation == "ok"
    assert fake_runnable.calls == 2  # retried once
    assert usage.input_tokens == 20  # both calls counted
    assert usage.model == "fake-model"
