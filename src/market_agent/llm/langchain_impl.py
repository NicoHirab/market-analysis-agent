from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from market_agent.llm.base import LLMGenerationError, LLMUsage

T = TypeVar("T", bound=BaseModel)


class LangChainStructuredLLM:
    """StructuredLLM backed by any LangChain chat model via with_structured_output."""

    def __init__(self, model: BaseChatModel, *, model_name: str, timeout_s: float = 60.0) -> None:
        self.model = model
        self.model_name = model_name
        self.timeout_s = timeout_s

    async def generate(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        context: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> tuple[T, LLMUsage]:
        chain = self.model.with_structured_output(schema, include_raw=True)
        usage = LLMUsage(purpose=purpose, model=self.model_name)
        messages: list[tuple[str, str]] = [("system", system), ("user", user)]

        for attempt in range(2):
            result = await chain.ainvoke(messages)
            raw = result.get("raw")
            meta = getattr(raw, "usage_metadata", None) or {}
            usage.input_tokens += int(meta.get("input_tokens", 0))
            usage.output_tokens += int(meta.get("output_tokens", 0))

            parsed = result.get("parsed")
            if parsed is not None:
                return parsed, usage

            error = result.get("parsing_error")
            if attempt == 0:
                messages.append(
                    ("user",
                     f"Your previous answer was not valid for the required schema "
                     f"({error}). Answer again, strictly matching the schema.")
                )
        raise LLMGenerationError(f"invalid structured output for {schema.__name__}: {error}")
