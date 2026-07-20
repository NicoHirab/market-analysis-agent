from datetime import date, timedelta

import pytest

from market_agent.tools.models import CollectedData, PlatformData, PricePoint
from market_agent.tools.trends import compute_trend_stats


def _platform(name: str, prices: list[float]) -> PlatformData:
    start = date(2026, 6, 1)
    return PlatformData(
        platform=name,
        offers=[],
        reviews=[],
        price_history=[
            PricePoint(date=start + timedelta(days=i), price=p) for i, p in enumerate(prices)
        ],
        popularity_score=50.0,
    )


def test_rising_trend_detected():
    prices = [100 + i for i in range(10)]  # +1%/day at start
    stats = compute_trend_stats(CollectedData(query="x", platforms=[_platform("amazon", prices)]))
    assert stats.trend_direction == "rising"
    assert stats.trend_slope_pct_per_day > 0.1
    assert stats.min_price == 100
    assert stats.max_price == 109


def test_stable_trend_and_platform_comparison():
    flat = [100.0] * 10
    cheap = [90.0] * 10
    stats = compute_trend_stats(
        CollectedData(
            query="x",
            platforms=[_platform("amazon", flat), _platform("cdiscount", cheap)],
        )
    )
    assert stats.trend_direction == "stable"
    assert stats.cheapest_platform == "cdiscount"
    assert stats.priciest_platform == "amazon"
    assert stats.competitor_gap_pct == pytest.approx(11.11, abs=0.1)


def test_empty_history_raises():
    with pytest.raises(ValueError, match="no price history"):
        compute_trend_stats(CollectedData(query="x", platforms=[]))
