from __future__ import annotations

from flask import Flask
from .config import load_settings
from .web.routes.index import bp as index_bp
from .web.routes.health import bp as health_bp
from .web.routes.video import bp as video_bp
from .web.routes.realtime import bp as realtime_bp



def create_app() -> Flask:
    settings = load_settings()

    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static",
    )

    # Core config
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024

    # Ensure runtime dirs exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    # Store settings on app (simple pattern; you can later use flask.g or app.extensions)
    app.config["SETTINGS"] = settings

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(index_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(realtime_bp)
    
    return app
