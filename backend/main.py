# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AgriGPT Backend",
    description="Multimodal AI farming assistant with Groq",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.routes.health_router import router as health_router
from backend.routes.ask_router import router as ask_router

from backend.routes.weather_router import router as weather_router

app.include_router(weather_router)

app.include_router(health_router)
app.include_router(ask_router)

@app.get("/")
def root():
    return {
        "message": "🌾 AgriGPT Backend running successfully",
        "endpoints": ["/ask/text", "/ask/image", "/ask/chat", "/health", "/docs"]
    }

@app.on_event("startup")
async def startup_event():
    print("✅ AgriGPT Backend Started: Ready to accept queries")

@app.on_event("shutdown")
async def shutdown_event():
    print("✅ AgriGPT Backend Shutting down")