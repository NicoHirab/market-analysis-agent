from datetime import date

from market_agent.tools.models import CollectedData, Offer, PlatformData, PricePoint, Review


def _platform(name: str) -> PlatformData:
    return PlatformData(
        platform=name,
        offers=[Offer(platform=name, title="X", price=99.9, rating=4.2, review_count=10, url="https://x")],
        reviews=[Review(author="a", rating=4, text="good", date=date(2026, 7, 1))],
        price_history=[PricePoint(date=date(2026, 7, 1), price=99.9)],
        popularity_score=55.0,
    )


def test_collected_data_aggregates_across_platforms():
    c = CollectedData(query="x", platforms=[_platform("amazon"), _platform("fnac")])
    assert len(c.all_offers()) == 2
    assert len(c.all_reviews()) == 2
    assert {o.platform for o in c.all_offers()} == {"amazon", "fnac"}
