# backend/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Safe configuration loader for all environment variables.
    Prevents FastAPI startup failures by providing fallback defaults.
    """

    # --- Required Keys but with safe defaults ---
    GROQ_API_KEY: str = ""
    OPENWEATHER_API_KEY: str = ""   # ✅ ADD
    TEXT_MODEL_NAME: str = "llama-3.1-70b-versatile"
    VISION_MODEL_NAME: str = "llama-3.2-vision-preview"

    # --- Optional ---
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()

# ----------------------------------------------------------------
# EXTRA SAFETY CHECKS (do NOT break FastAPI if missing)
# ----------------------------------------------------------------
if not settings.GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY is missing. Image/text models may fail.")

if not settings.TEXT_MODEL_NAME:
    print("⚠️ WARNING: TEXT_MODEL_NAME missing → using fallback.")

if not settings.VISION_MODEL_NAME:
    print("⚠️ WARNING: VISION_MODEL_NAME missing → using fallback.")

if not settings.OPENWEATHER_API_KEY:
    print("⚠️ WARNING: OPENWEATHER_API_KEY missing.")
