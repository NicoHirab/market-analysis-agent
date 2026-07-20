from market_agent.tools.models import PlatformData
from market_agent.tools.scraper.base import PlatformAdapter
from market_agent.tools.scraper.mock_data import generate_platform_data


class _MockAdapter(PlatformAdapter):
    def fetch(self, query: str) -> PlatformData:
        return generate_platform_data(query, self.name)


class MockAmazonAdapter(_MockAdapter):
    name = "amazon"


class MockBestBuyAdapter(_MockAdapter):
    name = "bestbuy"


class MockWalmartAdapter(_MockAdapter):
    name = "walmart"


KNOWN_PLATFORMS: dict[str, type[PlatformAdapter]] = {
    "amazon": MockAmazonAdapter,
    "bestbuy": MockBestBuyAdapter,
    "walmart": MockWalmartAdapter,
}


def get_adapters(platforms: list[str] | None) -> list[PlatformAdapter]:
    names = platforms or list(KNOWN_PLATFORMS)
    unknown = [n for n in names if n not in KNOWN_PLATFORMS]
    if unknown:
        raise ValueError(f"unknown platform(s): {', '.join(unknown)}")
    return [KNOWN_PLATFORMS[n]() for n in names]
