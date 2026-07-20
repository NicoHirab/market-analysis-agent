from pydantic import BaseModel, Field

from market_agent.llm.base import LLMUsage, StructuredLLM
from market_agent.tools.models import Review

SENTIMENT_SYSTEM = (
    "You are an e-commerce customer-review analyst. You extract faithful, "
    "actionable insights from raw reviews. Never invent complaints or praises "
    "that are not grounded in the provided reviews. Answer in {language}."
)

SENTIMENT_USER_TEMPLATE = (
    "Product: {query}\n"
    "Reviews ({count}):\n{reviews}\n\n"
    "Analyze these reviews: overall sentiment distribution, top praises, top "
    "complaints, recurring themes, and 2-3 short representative quotes."
)


class SentimentDistribution(BaseModel):
    positive: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    negative: float = Field(ge=0, le=1)


class SentimentInsights(BaseModel):
    distribution: SentimentDistribution
    top_praises: list[str]
    top_complaints: list[str]
    themes: list[str]
    representative_quotes: list[str]
    summary: str


async def analyze_sentiment(
    reviews: list[Review],
    llm: StructuredLLM,
    *,
    query: str,
    language: str = "fr",
) -> tuple[SentimentInsights, LLMUsage]:
    if not reviews:
        raise ValueError("no reviews to analyze")
    rendered = "\n".join(f"- [{r.rating}/5] {r.text}" for r in reviews)
    ratings = [r.rating for r in reviews]
    context = {
        "query": query,
        "language": language,
        "avg_rating": sum(ratings) / len(ratings),
        "positive_share": sum(1 for r in ratings if r >= 4) / len(ratings),
        "negative_share": sum(1 for r in ratings if r < 2.5) / len(ratings),
    }
    return await llm.generate(
        SentimentInsights,
        system=SENTIMENT_SYSTEM.format(language=language),
        user=SENTIMENT_USER_TEMPLATE.format(query=query, count=len(reviews), reviews=rendered),
        context=context,
        purpose="sentiment",
    )
