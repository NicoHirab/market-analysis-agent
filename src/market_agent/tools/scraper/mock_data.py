import random
from datetime import date, timedelta

from market_agent.tools.models import Offer, PlatformData, PricePoint, Review

_POSITIVE = [
    "Excellent produit, je recommande vivement.",
    "Très bonne qualité, conforme à la description.",
    "Livraison rapide et produit impeccable.",
    "Super rapport qualité/prix.",
]
_NEUTRAL = [
    "Correct sans plus, fait le travail.",
    "Produit moyen, quelques défauts mineurs.",
]
_NEGATIVE = [
    "Déçu, la batterie faiblit trop vite.",
    "Prix trop élevé pour la qualité reçue.",
    "Service client injoignable, produit fragile.",
]
_AUTHORS = ["Marie L.", "Julien P.", "Sophie D.", "Karim B.", "Claire M.", "Antoine R.", "Léa T."]

# Per-platform price flavor: (multiplier, popularity bonus)
_PLATFORM_FLAVOR = {"amazon": (1.0, 10.0), "cdiscount": (0.93, -5.0), "fnac": (1.06, 0.0)}

_ANCHOR_DATE = date(2026, 7, 18)


def _base_price(query: str) -> float:
    seed = random.Random(query.strip().lower())
    return round(seed.uniform(40, 1400), 2)


def generate_platform_data(query: str, platform: str, *, days: int = 30) -> PlatformData:
    """Deterministic pseudo-realistic data, seeded by (query, platform)."""
    rng = random.Random(f"{query.strip().lower()}::{platform}")
    mult, pop_bonus = _PLATFORM_FLAVOR.get(platform, (1.0, 0.0))
    center = _base_price(query) * mult

    offers = [
        Offer(
            platform=platform,
            title=f"{query} — offre {i + 1}",
            price=round(center * rng.uniform(0.92, 1.12), 2),
            rating=round(rng.uniform(3.2, 4.8), 1),
            review_count=rng.randint(15, 900),
            url=f"https://{platform}.example/{query.lower().replace(' ', '-')}/{i + 1}",
        )
        for i in range(rng.randint(3, 6))
    ]

    positivity = rng.uniform(0.35, 0.8)  # per-product sentiment mix
    reviews = []
    for _ in range(rng.randint(8, 15)):
        roll = rng.random()
        if roll < positivity:
            text, rating = rng.choice(_POSITIVE), rng.uniform(4, 5)
        elif roll < positivity + 0.2:
            text, rating = rng.choice(_NEUTRAL), rng.uniform(2.5, 3.9)
        else:
            text, rating = rng.choice(_NEGATIVE), rng.uniform(1, 2.4)
        reviews.append(
            Review(
                author=rng.choice(_AUTHORS),
                rating=round(rating, 1),
                text=text,
                date=_ANCHOR_DATE - timedelta(days=rng.randint(0, 90)),
            )
        )

    drift = rng.uniform(-0.004, 0.004)  # daily trend
    price = center * rng.uniform(0.95, 1.05)
    history = []
    start = _ANCHOR_DATE - timedelta(days=days - 1)
    floor, cap = center * 0.85, center * 1.25  # bounds keep max/min ratio < 1.5 for any horizon
    for i in range(days):
        price = min(cap, max(floor, price * (1 + drift + rng.uniform(-0.01, 0.01))))
        history.append(PricePoint(date=start + timedelta(days=i), price=round(price, 2)))

    popularity = min(100.0, max(0.0, rng.uniform(20, 90) + pop_bonus))
    return PlatformData(
        platform=platform,
        offers=offers,
        reviews=reviews,
        price_history=history,
        popularity_score=round(popularity, 1),
    )
