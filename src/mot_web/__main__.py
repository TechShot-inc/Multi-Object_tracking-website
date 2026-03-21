from __future__ import annotations

from .app_factory import create_app
from .config import load_settings
import uvicorn


def main() -> None:
    settings = load_settings()
    debug = settings.environment == "dev"

    if debug:
        # Uvicorn reload requires an import string (not an app object).
        uvicorn.run(
            "mot_web.app_factory:create_app",
            factory=True,
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level="debug",
        )
        return

    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
