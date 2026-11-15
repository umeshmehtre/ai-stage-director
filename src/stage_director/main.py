"""Application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stage_director.api.routes import router
from stage_director.config import get_settings
from stage_director.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """App factory: builds a fully configured app."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=settings.app_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    logger.info("%s v%s started", settings.app_name, settings.app_version)
    return app


app = create_app()