from dataclasses import dataclass
import os


@dataclass
class AppConfig:
    openai_api_key: str
    pdf_path: str
    frontend_origin: str
    openai_model: str


def get_config() -> AppConfig:
    """Load configuration from environment variables with sensible defaults."""
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        pdf_path=os.getenv("PDF_PATH", "data/nova_przestrzen.pdf"),
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    )


