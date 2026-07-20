from typing import Literal

from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    title: str
    rationale: str
    priority: Literal["high", "medium", "low"]


class MarketReport(BaseModel):
    product: str
    language: str = "fr"
    executive_summary: str
    price_analysis: str
    sentiment_summary: str | None = None
    trend_summary: str | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    caveats: list[str] = Field(default_factory=list)


_PRIORITY_LABEL = {"high": "Haute", "medium": "Moyenne", "low": "Basse"}


def render_markdown(report: MarketReport) -> str:
    lines = [
        f"# Rapport d'analyse de marché — {report.product}",
        "",
        "## Synthèse",
        report.executive_summary,
        "",
        "## Analyse des prix",
        report.price_analysis,
    ]
    if report.sentiment_summary:
        lines += ["", "## Sentiment client", report.sentiment_summary]
    if report.trend_summary:
        lines += ["", "## Tendances", report.trend_summary]
    if report.recommendations:
        lines += ["", "## Recommandations"]
        for rec in report.recommendations:
            priority_label = _PRIORITY_LABEL[rec.priority]
            lines.append(f"- **{rec.title}** (priorité : {priority_label}) — {rec.rationale}")
    lines += ["", f"_Indice de confiance : {report.confidence:.0%}_"]
    if report.caveats:
        lines += ["", "## Limites"]
        lines += [f"- {c}" for c in report.caveats]
    return "\n".join(lines) + "\n"
