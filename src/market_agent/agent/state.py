import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from market_agent.core.errors import AnalysisError
from market_agent.llm.base import LLMUsage
from market_agent.tools.models import CollectedData
from market_agent.tools.report import MarketReport
from market_agent.tools.sentiment import SentimentInsights
from market_agent.tools.trends import TrendInsights

AnalysisKind = Literal["sentiment", "trends"]


class AnalysisPlan(BaseModel):
    analyses: list[AnalysisKind]
    platforms: list[str]
    rationale: str


class CriterionResult(BaseModel):
    passed: bool
    comment: str = ""


class JudgeVerdict(BaseModel):
    """Criteria-based quality verdict: the report is good only if EVERY
    criterion passes. The aggregation (`passed` = conjunction) is enforced in
    code by the judge node — never trusted from the model."""

    grounding: CriterionResult
    completeness: CriterionResult
    actionability: CriterionResult
    passed: bool
    critique: str = ""

    def failed_criteria(self) -> list[str]:
        return [
            name
            for name in ("grounding", "completeness", "actionability")
            if not getattr(self, name).passed
        ]


class AnalysisState(TypedDict, total=False):
    query: str
    language: str
    requested_analyses: list[str] | None
    requested_platforms: list[str] | None
    plan: AnalysisPlan | None
    collected: CollectedData | None
    sentiment: SentimentInsights | None
    trends: TrendInsights | None
    report: MarketReport | None
    judge: JudgeVerdict | None
    revision_count: int
    errors: Annotated[list[AnalysisError], operator.add]
    usage: Annotated[list[LLMUsage], operator.add]
