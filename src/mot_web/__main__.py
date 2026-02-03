from __future__ import annotations
from .app_factory import create_app

def main() -> None:
    app = create_app()
    settings = app.config["SETTINGS"]
    debug = settings.environment == "dev"
    app.run(host=settings.host, port=settings.port, debug=debug)

if __name__ == "__main__":
    main()
