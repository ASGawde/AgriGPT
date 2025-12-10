from fastapi import APIRouter
from datetime import datetime, timedelta
import time

from backend.core.config import settings
from backend.core.llm_client import get_llm

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()

def _format_uptime(seconds: int) -> str:
    """Convert seconds to HH:MM:SS."""
    return str(timedelta(seconds=seconds))


@router.get("/")
async def health_check():


    uptime_sec = int(time.time() - START_TIME)

    try:
        llm = get_llm()
        model_ok = bool(settings.TEXT_MODEL_NAME) and bool(settings.GROQ_API_KEY)
        groq_status = "configured" if model_ok else "not_configured"
    except Exception as e:
        groq_status = f"error: {str(e)}"

    return {
        "status": "OK",
        "service": "AgriGPT Backend",
        "timestamp_utc": str(datetime.utcnow().isoformat()),
        "uptime_seconds": int(uptime_sec),
        "uptime_hhmmss": str(_format_uptime(uptime_sec)),

        "models": {
            "text_model": str(settings.TEXT_MODEL_NAME or ""),
            "vision_model": str(settings.VISION_MODEL_NAME or ""),
        },

        "dependencies": {
            "groq_api": str(groq_status),
        },

        "notes": "Health OK",
    }
