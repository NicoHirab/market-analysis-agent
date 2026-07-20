import random
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from market_agent.llm.base import LLMGenerationError, LLMUsage
from market_agent.tools.report import MarketReport, Recommendation
from market_agent.tools.sentiment import SentimentDistribution, SentimentInsights
from market_agent.tools.trends import TrendInterpretation

T = TypeVar("T", bound=BaseModel)


def _rng(context: dict[str, Any]) -> random.Random:
    return random.Random(str(context.get("query", "")).strip().lower())


def _build_trend_interpretation(context: dict[str, Any]) -> TrendInterpretation:
    direction = context.get("trend_direction", "stable")
    wording = {
        "rising": "Les prix sont en hausse sur la période",
        "falling": "Les prix sont en baisse sur la période",
        "stable": "Les prix sont stables sur la période",
    }[direction]
    gap = context.get("competitor_gap_pct", 0)
    return TrendInterpretation(
        interpretation=(
            f"{wording}, avec un écart de {gap:.1f}% entre la plateforme la moins chère "
            f"({context.get('cheapest_platform', 'n/a')}) et la plus chère "
            f"({context.get('priciest_platform', 'n/a')})."
        )
    )


def _build_sentiment_insights(context: dict[str, Any]) -> SentimentInsights:
    pos = round(float(context.get("positive_share", 0.6)), 2)
    neg = round(float(context.get("negative_share", 0.2)), 2)
    neu = round(max(0.0, 1.0 - pos - neg), 2)
    return SentimentInsights(
        distribution=SentimentDistribution(positive=pos, neutral=neu, negative=neg),
        top_praises=["Qualité perçue", "Rapport qualité/prix"],
        top_complaints=["Autonomie de la batterie", "Prix jugé élevé"],
        themes=["qualité", "prix", "livraison"],
        representative_quotes=["Excellent produit, je recommande.", "Déçu, batterie trop faible."],
        summary=(
            f"Sentiment majoritairement {'positif' if pos >= 0.5 else 'mitigé'} "
            f"({pos:.0%} positif, {neg:.0%} négatif) pour {context.get('query', 'le produit')}."
        ),
    )


def _build_market_report(context: dict[str, Any]) -> MarketReport:
    rng = _rng(context)
    query = context.get("query", "produit")
    avg = context.get("avg_price")
    critique = context.get("critique")
    recs = [
        Recommendation(
            title=f"Ajuster le positionnement prix de {query}",
            rationale="L'écart constaté entre plateformes laisse une marge de manœuvre tarifaire.",
            priority="high",
        ),
        Recommendation(
            title="Surveiller les avis négatifs récurrents",
            rationale="Les thèmes de plaintes identifiés sont adressables à court terme.",
            priority="medium",
        ),
    ]
    summary = (
        f"Analyse de marché pour {query}. "
        + (f"Prix moyen constaté : {avg:.2f}€. " if isinstance(avg, int | float) else "")
        + f"Position concurrentielle {'favorable' if rng.random() > 0.4 else 'à consolider'}."
    )
    if critique:
        summary += " (Version révisée suite au contrôle qualité.)"
    return MarketReport(
        product=query,
        language=context.get("language", "fr"),
        executive_summary=summary,
        price_analysis=context.get(
            "price_analysis_hint", "Analyse des prix multi-plateformes effectuée."
        ),
        sentiment_summary=context.get("sentiment_summary"),
        trend_summary=context.get("trend_summary"),
        recommendations=recs,
        confidence=0.55 if context.get("degraded") else 0.85,
        caveats=list(context.get("caveats", [])),
    )


def _build_analysis_plan(context: dict[str, Any]) -> BaseModel:
    from market_agent.agent.state import AnalysisPlan

    query = str(context.get("query", "")).lower()
    requested = context.get("requested_analyses")
    if requested is not None:
        analyses = [a for a in requested if a in ("sentiment", "trends")]
    else:
        analyses = ["sentiment", "trends"]
        if any(w in query for w in ("prix", "price", "comparer", "compare")) and not any(
            w in query for w in ("avis", "review", "sentiment", "client")
        ):
            analyses = ["trends"]
    platforms = context.get("requested_platforms") or ["amazon", "cdiscount", "fnac"]
    return AnalysisPlan(
        analyses=analyses,
        platforms=list(platforms),
        rationale=f"Mock plan derived from query keywords: {analyses}.",
    )


class MockStructuredLLM:
    """Deterministic, schema-dispatched fake LLM. Registered builders map
    schema type -> callable(context) -> instance."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[dict[str, Any]], BaseModel]] = {
            "TrendInterpretation": _build_trend_interpretation,
            "SentimentInsights": _build_sentiment_insights,
            "MarketReport": _build_market_report,
        }
        self._builders["AnalysisPlan"] = _build_analysis_plan

    def register(self, schema_name: str, builder: Callable[[dict[str, Any]], BaseModel]) -> None:
        self._builders[schema_name] = builder

    async def generate(
        self,
        schema: type[T],
        *,
        system: str,
        user: str,
        context: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> tuple[T, LLMUsage]:
        builder = self._builders.get(schema.__name__)
        if builder is None:
            raise LLMGenerationError(f"no mock builder for schema {schema.__name__}")
        obj = builder(context or {})
        usage = LLMUsage(
            purpose=purpose,
            model="mock",
            input_tokens=len(system) // 4 + len(user) // 4,
            output_tokens=64,
        )
        return obj, usage  # type: ignore[return-value]
