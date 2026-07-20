from contextlib import asynccontextmanager

from fastapi import FastAPI

from market_agent.api.registry import JobRegistry
from market_agent.api.routes import router
from market_agent.api.service import AnalysisService
from market_agent.core.config import Settings, get_settings
from market_agent.core.logging import setup_logging
from market_agent.llm.factory import validate_llm_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(app_settings.log_level)
        validate_llm_settings(app_settings)
        app.state.registry = JobRegistry()
        app.state.service = AnalysisService(app.state.registry, app_settings)
        yield

    app = FastAPI(
        title="Market Analysis Agent",
        description="Agent d'analyse de marché e-commerce — LangGraph orchestration",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "provider": app_settings.llm_provider,
            "model": app_settings.llm_model or "(default)",
        }

    return app


app = create_app()
