# backend/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os

# Get absolute path to the folder where THIS file lives
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # --- Required Keys ---
    GROQ_API_KEY: str
    TEXT_MODEL_NAME: str
    VISION_MODEL_NAME: str

    # --- Optional ---
    DEBUG: bool = False

    # Force Pydantic to load .env from backend/core/.env
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
