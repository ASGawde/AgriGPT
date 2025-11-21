# backend/routes/health_router.py

from fastapi import APIRouter
from datetime import datetime
import time

from backend.core.config import settings
from backend.core.llm_client import get_llm

router = APIRouter(prefix="/health", tags=["Health"])

# Track server uptime
START_TIME = time.time()


@router.get("/")
async def health_check():
    """
    Health check endpoint.
    Provides status, uptime, and checks if Groq API is reachable.
    """

    uptime_sec = int(time.time() - START_TIME)
    groq_status = "unknown"

    # Check Groq LLM connectivity (safe and fast)
    try:
        llm = get_llm()
        _ = llm.model_name  # access field to confirm object
        groq_status = "reachable"
    except Exception:
        groq_status = "unreachable"

    return {
        "status": "OK",
        "service": "AgriGPT Backend",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_sec,
        "models": {
            "text_model": settings.TEXT_MODEL_NAME,
            "vision_model": settings.VISION_MODEL_NAME,
        },
        "dependencies": {
            "groq_api": groq_status
        },
        "message": "AgriGPT backend is alive 🚜"
    }
