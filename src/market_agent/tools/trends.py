from statistics import mean, pstdev
from typing import Literal

from pydantic import BaseModel

from market_agent.tools.models import CollectedData

STABLE_SLOPE_PCT = 0.05  # |slope| below this (%/day) counts as stable


class TrendStats(BaseModel):
    avg_price: float
    min_price: float
    max_price: float
    volatility_pct: float
    trend_slope_pct_per_day: float
    trend_direction: Literal["rising", "falling", "stable"]
    cheapest_platform: str
    priciest_platform: str
    competitor_gap_pct: float
    avg_popularity: float


class TrendInterpretation(BaseModel):
    """LLM output schema for the trends node."""

    interpretation: str


class TrendInsights(BaseModel):
    stats: TrendStats
    interpretation: str


def _slope_pct_per_day(prices: list[float]) -> float:
    """Least-squares slope, as % of mean price per day."""
    n = len(prices)
    if n < 2:
        return 0.0
    xs = range(n)
    x_mean, y_mean = (n - 1) / 2, mean(prices)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, prices, strict=True))
    den = sum((x - x_mean) ** 2 for x in xs)
    return (num / den) / y_mean * 100 if den and y_mean else 0.0


def compute_trend_stats(collected: CollectedData) -> TrendStats:
    all_points = [pt.price for p in collected.platforms for pt in p.price_history]
    if not all_points:
        raise ValueError("no price history to analyze")

    by_day: dict[str, list[float]] = {}
    for p in collected.platforms:
        for pt in p.price_history:
            by_day.setdefault(pt.date.isoformat(), []).append(pt.price)
    daily_avg = [mean(v) for _, v in sorted(by_day.items())]

    slope = round(_slope_pct_per_day(daily_avg), 3)
    direction: Literal["rising", "falling", "stable"]
    if slope > STABLE_SLOPE_PCT:
        direction = "rising"
    elif slope < -STABLE_SLOPE_PCT:
        direction = "falling"
    else:
        direction = "stable"

    platform_avgs = {
        p.platform: mean(pt.price for pt in p.price_history)
        for p in collected.platforms
        if p.price_history
    }
    cheapest = min(platform_avgs, key=platform_avgs.get)
    priciest = max(platform_avgs, key=platform_avgs.get)
    gap = (platform_avgs[priciest] - platform_avgs[cheapest]) / platform_avgs[cheapest] * 100

    return TrendStats(
        avg_price=round(mean(all_points), 2),
        min_price=round(min(all_points), 2),
        max_price=round(max(all_points), 2),
        volatility_pct=round(pstdev(all_points) / mean(all_points) * 100, 2),
        trend_slope_pct_per_day=slope,
        trend_direction=direction,
        cheapest_platform=cheapest,
        priciest_platform=priciest,
        competitor_gap_pct=round(gap, 2),
        avg_popularity=round(mean(p.popularity_score for p in collected.platforms), 1),
    )
