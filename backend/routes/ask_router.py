# backend/routes/ask_router.py

from fastapi import (
    APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile, os, time, uuid, imghdr

from backend.agents.master_agent import route_query

router = APIRouter(prefix="/ask", tags=["Unified Multimodal Query"])

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB


# ----------------------------------------------------
# Swagger Response Model
# ----------------------------------------------------
class AskResponse(BaseModel):
    request_id: str
    status: str
    elapsed_ms: int
    input: dict
    analysis: str


# ----------------------------------------------------
# 1️⃣ TEXT-ONLY ENDPOINT
# ----------------------------------------------------
@router.post("/text", response_model=AskResponse)
async def ask_text(
    query: str = Form(..., description="Example: 'How to increase maize yield?'")
):
    """
    Handles text-only queries using multi-agent smart routing.
    """

    start = time.time()
    request_id = str(uuid.uuid4())

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid text query.")

    try:
        response = route_query(query=query, image_path=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AgriGPT processing error: {e}")

    return AskResponse(
        request_id=request_id,
        status="success",
        elapsed_ms=int((time.time() - start) * 1000),
        input={"query": query, "image_uploaded": False},
        analysis=response,
    )


# ----------------------------------------------------
# 2️⃣ IMAGE-ONLY ENDPOINT
# ----------------------------------------------------
@router.post("/image", response_model=AskResponse)
async def ask_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload a crop photo (JPEG or PNG).")
):
    """
    Handles image-only pest/disease diagnosis.
    """

    start = time.time()
    request_id = str(uuid.uuid4())
    tmp_path = None

    # Validate MIME type
    if file.content_type not in ALLOWED_IMAGE_MIME:
        raise HTTPException(415, "Unsupported file format. Only JPEG/PNG allowed.")

    try:
        # Read file bytes
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded image is empty.")

        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (maximum 8 MB).")

        # Preserve correct extension
        ext = ".jpg" if file.content_type == "image/jpeg" else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Route to master agent
        response = route_query(query=None, image_path=tmp_path)

        # Cleanup after background task finishes
        background_tasks.add_task(os.remove, tmp_path)

        return AskResponse(
            request_id=request_id,
            status="success",
            elapsed_ms=int((time.time() - start) * 1000),
            input={"query": None, "image_uploaded": True},
            analysis=response,
        )

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            background_tasks.add_task(os.remove, tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


# ----------------------------------------------------
# 3️⃣ OPTIONAL: COMBINED TEXT + IMAGE ENDPOINT
# ----------------------------------------------------
@router.post("/chat", response_model=AskResponse)
async def ask_chat(
    background_tasks: BackgroundTasks,
    query: str = Form(..., description="Farmer's question."),
    file: UploadFile = File(None)
):
    """
    Accepts both text query + image → triggers multimodal fusion.
    """

    start = time.time()
    request_id = str(uuid.uuid4())

    tmp_path = None

    # If no image, fallback to text-only
    if not file:
        response = route_query(query=query, image_path=None)
        return AskResponse(
            request_id=request_id,
            status="success",
            elapsed_ms=int((time.time() - start) * 1000),
            input={"query": query, "image_uploaded": False},
            analysis=response,
        )

    # Handle image upload like the previous endpoint
    if file.content_type not in ALLOWED_IMAGE_MIME:
        raise HTTPException(415, "Unsupported image type. Only JPEG/PNG allowed.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (maximum 8 MB).")

    ext = ".jpg" if file.content_type == "image/jpeg" else ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    response = route_query(query=query, image_path=tmp_path)
    background_tasks.add_task(os.remove, tmp_path)

    return AskResponse(
        request_id=request_id,
        status="success",
        elapsed_ms=int((time.time() - start) * 1000),
        input={"query": query, "image_uploaded": True},
        analysis=response,
    )
