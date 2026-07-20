from market_agent.core.config import Settings
from market_agent.llm.base import StructuredLLM
from market_agent.llm.langchain_impl import LangChainStructuredLLM
from market_agent.llm.mock import MockStructuredLLM

OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "deepseek": "deepseek-chat",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "ollama": "llama3.2",
}


def build_structured_llm(settings: Settings) -> StructuredLLM:
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockStructuredLLM()

    model_name = settings.llm_model or DEFAULT_MODELS.get(provider, "")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # max_retries: the underlying SDK retries transient errors (429/5xx)
        # with exponential backoff.
        model = ChatAnthropic(
            model=model_name,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=2,
        )
        return LangChainStructuredLLM(
            model, model_name=model_name, timeout_s=settings.llm_timeout_s
        )

    base_url = settings.llm_base_url or OPENAI_COMPATIBLE_BASE_URLS.get(provider)
    if provider != "openai" and base_url is None:
        raise ValueError(
            f"unknown LLM provider '{provider}' — set LLM_BASE_URL for OpenAI-compatible endpoints"
        )

    from langchain_openai import ChatOpenAI

    api_key = settings.llm_api_key or ("ollama" if provider == "ollama" else "")
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_s,
        max_retries=2,
    )
    return LangChainStructuredLLM(model, model_name=model_name, timeout_s=settings.llm_timeout_s)


def validate_llm_settings(settings: Settings) -> None:
    """Fail fast at startup instead of mid-analysis."""
    provider = settings.llm_provider.lower()
    if provider in ("mock", "ollama"):
        return
    if not settings.llm_api_key:
        raise RuntimeError(
            f"LLM provider '{provider}' requires LLM_API_KEY to be set "
            "(or use LLM_PROVIDER=mock for a zero-key demo)."
        )
