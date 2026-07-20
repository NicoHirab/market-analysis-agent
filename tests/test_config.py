from market_agent.core.config import Settings
from market_agent.core.errors import AnalysisError, ErrorCode


def test_settings_defaults_to_mock_provider():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.judge_enabled is True
    assert 0 < s.judge_threshold < 1


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("JUDGE_ENABLED", "false")
    s = Settings(_env_file=None)
    assert s.llm_provider == "groq"
    assert s.llm_model == "llama-3.3-70b-versatile"
    assert s.judge_enabled is False


def test_analysis_error_serializes():
    e = AnalysisError(code=ErrorCode.ADAPTER_FAILURE, source="amazon", message="boom")
    assert e.model_dump()["code"] == "ADAPTER_FAILURE"
    assert e.recoverable is True
