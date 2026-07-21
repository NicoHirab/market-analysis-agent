from market_agent.core.config import Settings
from market_agent.core.errors import AnalysisError, ErrorCode


def test_settings_defaults_to_mock_provider():
    s = Settings(_env_file=None)
    assert s.llm_provider == "mock"
    assert s.judge_enabled is True


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


def test_env_file_loading(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=deepseek\nLLM_MODEL=deepseek-chat\n")
    s = Settings(_env_file=env)
    assert s.llm_provider == "deepseek"
    assert s.llm_model == "deepseek-chat"


def test_json_logging_emits_parseable_lines_with_context(capsys):
    import json
    import logging as stdlib_logging

    from market_agent.core.logging import get_logger, setup_logging

    setup_logging("INFO")
    get_logger("t").info("hello", extra={"ctx": {"analysis_id": "abc"}})
    stdlib_logging.getLogger().handlers = []  # detach from captured stdout
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["analysis_id"] == "abc"
    assert payload["level"] == "INFO"
