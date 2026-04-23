"""
main.py
-------
FastAPI application factory.

Lifespan context manager handles:
  - ML model loading (once at startup)
  - Database pool initialisation (once at startup)
  - Clean shutdown

Middleware stack (applied in reverse order):
  1. CORS
  2. SecurityMiddleware (API key + rate limiting)
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import init_db
from app.middleware.security import SecurityMiddleware
from app.models.ml_model import MLModels
from app.routes import predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Initialise DB pool + schema (once)
    init_db()

    # Load ML models into app state (once)
    ml = MLModels()
    ml.load()
    app.state.ml_models = ml

    logger.info("Startup complete — ready to serve requests")
    yield
    logger.info("Shutdown complete")


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered wearable health monitoring API. "
            "Provides heart-rate prediction, anomaly detection, "
            "and multi-disease risk assessment."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Security — API key + rate limiting
    app.add_middleware(
        SecurityMiddleware,
        exempt_paths=["/", "/health", "/docs", "/redoc", "/openapi.json",
                      "/api/health"],
    )

    # Routers
    app.include_router(predict.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "name":    settings.app_name,
            "version": settings.app_version,
            "docs":    "/docs",
        }

    return app


app = create_app()
