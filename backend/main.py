# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import ask_router, health_router


# ------------------------------------------------------
# Create FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="AgriGPT Backend (Groq + LLaMA-4 Scout)",
    description="Unified multimodal backend for AgriGPT — text + image intelligent farming assistant.",
    version="1.0.0",
)


# ------------------------------------------------------
# CORS (Important for frontend apps)
# ------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# Routers
# ------------------------------------------------------
app.include_router(ask_router.router)
app.include_router(health_router.router)


# ------------------------------------------------------
# Root Endpoint
# ------------------------------------------------------
@app.get("/")
def root():
    """Quick check to verify API availability."""
    return {
        "message": "🌾 AgriGPT Backend running successfully with Groq API and multimodal intelligence 🚀",
        "available_endpoints": ["/ask/text", "/ask/image", "/health", "/docs"]
    }


# ------------------------------------------------------
# Optional: Startup & Shutdown Hooks
# ------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print(" AgriGPT Backend Started: Ready to accept queries.")


@app.on_event("shutdown")
async def shutdown_event():
    print(" AgriGPT Backend Shutting down.")
