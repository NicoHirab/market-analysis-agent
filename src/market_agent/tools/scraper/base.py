from abc import ABC, abstractmethod

from market_agent.tools.models import PlatformData


class AdapterError(Exception):
    """Raised when a platform adapter cannot fetch data."""

    def __init__(self, platform: str, message: str) -> None:
        self.platform = platform
        super().__init__(f"[{platform}] {message}")


class PlatformAdapter(ABC):
    """One e-commerce platform data source. Real adapters implement the same contract."""

    name: str

    @abstractmethod
    def fetch(self, query: str) -> PlatformData: ...
