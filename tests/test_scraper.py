import pytest

from market_agent.tools.scraper.adapters import MockAmazonAdapter, get_adapters
from market_agent.tools.scraper.mock_data import generate_platform_data


def test_generation_is_deterministic_per_query_and_platform():
    a = generate_platform_data("iPhone 16", "amazon")
    b = generate_platform_data("iPhone 16", "amazon")
    c = generate_platform_data("iPhone 16", "fnac")
    assert a == b
    assert a != c


def test_generated_data_is_realistic():
    data = generate_platform_data("Nike Air Max", "amazon")
    assert 3 <= len(data.offers) <= 6
    assert 8 <= len(data.reviews) <= 15
    assert len(data.price_history) == 30
    prices = [p.price for p in data.price_history]
    assert max(prices) / min(prices) < 1.5  # no absurd swings
    assert 0 <= data.popularity_score <= 100


def test_adapter_fetch_returns_platform_data():
    data = MockAmazonAdapter().fetch("PS5")
    assert data.platform == "amazon"
    assert data.offers


def test_get_adapters_default_and_unknown():
    assert {a.name for a in get_adapters(None)} == {"amazon", "cdiscount", "fnac"}
    assert [a.name for a in get_adapters(["fnac"])] == ["fnac"]
    with pytest.raises(ValueError, match="unknown platform"):
        get_adapters(["ebay"])
