# backend/routes/health_router.py

from fastapi import APIRouter
from datetime import datetime
import time

from backend.core.config import settings
from backend.core.llm_client import get_llm

router = APIRouter(prefix="/health", tags=["Health"])

START_TIME = time.time()


@router.get("/")
async def health_check():
    """
    Health check endpoint with REAL Groq connectivity test.
    """

    uptime_sec = int(time.time() - START_TIME)

    # ---------------------------------------------------
    # Real-time GROQ API test
    # ---------------------------------------------------
    try:
        llm = get_llm()

        # Send a tiny message to confirm the API actually works
        _ = llm.invoke("ping")   # <-- REAL API CALL

        groq_status = "reachable"
    except Exception as e:
        groq_status = f"unreachable: {str(e)}"

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
