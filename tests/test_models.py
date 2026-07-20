from datetime import date

import pytest
from pydantic import ValidationError

from market_agent.tools.models import CollectedData, Offer, PlatformData, PricePoint, Review


def _platform(name: str) -> PlatformData:
    return PlatformData(
        platform=name,
        offers=[
            Offer(
                platform=name, title="X", price=99.9, rating=4.2, review_count=10, url="https://x"
            )
        ],
        reviews=[Review(author="a", rating=4, text="good", date=date(2026, 7, 1))],
        price_history=[PricePoint(date=date(2026, 7, 1), price=99.9)],
        popularity_score=55.0,
    )


def test_collected_data_aggregates_across_platforms():
    c = CollectedData(query="x", platforms=[_platform("amazon"), _platform("bestbuy")])
    assert len(c.all_offers()) == 2
    assert len(c.all_reviews()) == 2
    assert {o.platform for o in c.all_offers()} == {"amazon", "bestbuy"}


def test_field_bounds_rejected():
    base_offer = {
        "platform": "amazon",
        "title": "X",
        "price": 10.0,
        "rating": 4.0,
        "review_count": 1,
        "url": "https://x",
    }
    for patch in ({"price": 0}, {"rating": 6}, {"review_count": -1}):
        with pytest.raises(ValidationError):
            Offer.model_validate({**base_offer, **patch})
    with pytest.raises(ValidationError):
        PlatformData.model_validate(
            {
                "platform": "a",
                "offers": [],
                "reviews": [],
                "price_history": [],
                "popularity_score": 101,
            }
        )
    with pytest.raises(ValidationError):
        PricePoint.model_validate({"date": "2026-07-01", "price": 0})
