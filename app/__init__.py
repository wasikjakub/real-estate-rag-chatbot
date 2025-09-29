import os

from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from .config import get_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_app() -> Flask:
    """Application factory to create and configure the Flask app."""
    load_dotenv()
    config = get_config()

    app = Flask(
        __name__,
        template_folder=os.path.join(os.getcwd(), "templates"),
    )

    CORS(app, origins=[config.frontend_origin], supports_credentials=True)

    # Register blueprints
    from .routes.chat import chat_bp

    app.register_blueprint(chat_bp)

    # Attach config to app for easy access in routes/services
    app.config["APP_CONFIG"] = config

    return app


