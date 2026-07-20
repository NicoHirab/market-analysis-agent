from datetime import date

import pytest

from market_agent.llm.mock import MockStructuredLLM
from market_agent.tools.models import Review
from market_agent.tools.sentiment import SentimentInsights, analyze_sentiment


def _reviews() -> list[Review]:
    return [
        Review(
            author="a",
            rating=5,
            text="Excellent produit, je recommande.",
            date=date(2026, 7, 1),
        ),
        Review(
            author="b",
            rating=2,
            text="Déçu, batterie trop faible.",
            date=date(2026, 7, 2),
        ),
        Review(
            author="c",
            rating=4,
            text="Très bon rapport qualité/prix.",
            date=date(2026, 7, 3),
        ),
    ]


async def test_analyze_sentiment_returns_insights():
    insights, usage = await analyze_sentiment(
        _reviews(), MockStructuredLLM(), query="iPhone 16", language="fr"
    )
    assert isinstance(insights, SentimentInsights)
    total = (
        insights.distribution.positive
        + insights.distribution.neutral
        + insights.distribution.negative
    )
    assert total == pytest.approx(1.0, abs=0.01)
    assert insights.summary
    assert usage.purpose == "sentiment"


async def test_analyze_sentiment_empty_reviews_raises():
    with pytest.raises(ValueError, match="no reviews"):
        await analyze_sentiment([], MockStructuredLLM(), query="x")
