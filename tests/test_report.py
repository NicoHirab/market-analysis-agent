import pytest
from pydantic import ValidationError

from market_agent.tools.report import MarketReport, Recommendation, render_markdown


def _report() -> MarketReport:
    return MarketReport(
        product="iPhone 16",
        language="fr",
        executive_summary="Résumé.",
        price_analysis="Prix moyen 899 $ CAD.",
        sentiment_summary="Avis globalement positifs.",
        trend_summary=None,
        recommendations=[
            Recommendation(title="Baisser le prix", rationale="Écart de 5%", priority="high"),
        ],
        confidence=0.8,
        caveats=["Analyse de tendances indisponible."],
    )


def test_render_markdown_contains_sections_and_caveats():
    md = render_markdown(_report())
    assert "# Rapport d'analyse" in md
    assert "iPhone 16" in md
    assert "Baisser le prix" in md
    assert "Analyse de tendances indisponible." in md
    assert "## Tendances" not in md  # trend_summary is None → section omitted


def test_confidence_bounds_enforced():
    with pytest.raises(ValueError):
        MarketReport.model_validate({**_report().model_dump(), "confidence": 1.5})


def test_invalid_priority_rejected():
    with pytest.raises(ValidationError):
        Recommendation.model_validate({"title": "t", "rationale": "r", "priority": "urgent"})
