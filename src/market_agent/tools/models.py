from datetime import date

from pydantic import BaseModel, Field


class Review(BaseModel):
    author: str
    rating: float = Field(ge=1, le=5)
    text: str
    date: date


class Offer(BaseModel):
    platform: str
    title: str
    price: float = Field(gt=0)
    currency: str = "EUR"
    rating: float | None = Field(default=None, ge=1, le=5)
    review_count: int = Field(ge=0)
    url: str


class PricePoint(BaseModel):
    date: date
    price: float = Field(gt=0)


class PlatformData(BaseModel):
    platform: str
    offers: list[Offer]
    reviews: list[Review]
    price_history: list[PricePoint]
    popularity_score: float = Field(ge=0, le=100)


class CollectedData(BaseModel):
    query: str
    platforms: list[PlatformData]

    def all_offers(self) -> list[Offer]:
        return [o for p in self.platforms for o in p.offers]

    def all_reviews(self) -> list[Review]:
        return [r for p in self.platforms for r in p.reviews]
