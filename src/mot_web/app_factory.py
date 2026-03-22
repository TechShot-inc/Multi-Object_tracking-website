from __future__ import annotations

import os
import importlib.metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from .config import load_settings

from .web.routes.index import router as index_router
from .web.routes.health import router as health_router
from .web.routes.video import router as video_router
from .web.routes.realtime import router as realtime_router
from .metrics import install_metrics


class _NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        # Dev UX: avoid stale JS/CSS from browser cache when iterating.
        response.headers["Cache-Control"] = "no-store"
        return response


def create_app() -> FastAPI:
    settings = load_settings()

    app = FastAPI(title="MOT Web")

    install_metrics(app, service="mot-web")

    # Ensure runtime dirs exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    # Store settings + templating on app.state
    app.state.settings = settings
    app.state.max_content_length = settings.max_upload_mb * 1024 * 1024

    package_root = Path(__file__).resolve().parent
    templates_dir = package_root / "web" / "templates"
    static_dir = package_root / "web" / "static"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    static_version = os.getenv("STATIC_VERSION", "").strip() or os.getenv("GITHUB_SHA", "").strip()
    if static_version:
        static_version = static_version[:12]
    else:
        try:
            static_version = importlib.metadata.version("mot-web")
        except Exception:
            static_version = "dev"
    app.state.templates.env.globals["static_version"] = static_version

    static_cls = _NoCacheStaticFiles if settings.environment == "dev" else StaticFiles
    app.mount("/static", static_cls(directory=str(static_dir)), name="static")

    # Include routers (preserve existing URL contract)
    app.include_router(health_router)
    app.include_router(index_router)
    app.include_router(video_router)
    app.include_router(realtime_router)

    return app
