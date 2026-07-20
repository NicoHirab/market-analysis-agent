from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMUsage(BaseModel):
    purpose: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class LLMGenerationError(Exception):
    """The LLM could not produce a valid structured output."""


class StructuredLLM(Protocol):
    """Single seam for every LLM call: schema in, validated object + usage out.

    `context` carries structured facts (query, stats, ...) so implementations
    that don't parse prose (the mock) can still produce grounded output.
    """

    async def generate(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        context: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> tuple[T, LLMUsage]: ...
