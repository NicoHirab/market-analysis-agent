from enum import StrEnum

from pydantic import BaseModel


class ErrorCode(StrEnum):
    ADAPTER_FAILURE = "ADAPTER_FAILURE"
    LLM_FAILURE = "LLM_FAILURE"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    TIMEOUT = "TIMEOUT"


class AnalysisError(BaseModel):
    """A non-fatal error captured during an analysis run."""

    code: ErrorCode
    source: str
    message: str
    recoverable: bool = True
