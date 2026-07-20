from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from market_agent.api.registry import Job, JobStatus


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    platforms: list[str] | None = None
    analyses: Literal["auto"] | list[Literal["sentiment", "trends"]] = "auto"
    language: Literal["fr", "en"] = "fr"


class AnalysisResource(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    request: dict
    result: dict | None = None
    meta: dict | None = None

    @classmethod
    def from_job(cls, job: Job) -> "AnalysisResource":
        return cls(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            request=job.request,
            result=job.result,
            meta=job.meta,
        )


class AnalysisSummary(BaseModel):
    id: str
    status: JobStatus
    query: str
    created_at: datetime

    @classmethod
    def from_job(cls, job: Job) -> "AnalysisSummary":
        return cls(
            id=job.id,
            status=job.status,
            query=job.request.get("query", ""),
            created_at=job.created_at,
        )
