# backend/routes/ask_router.py

from fastapi import (
    APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
)
from pydantic import BaseModel
import tempfile, os, time, uuid, imghdr

from backend.agents.master_agent import route_query

router = APIRouter(prefix="/ask", tags=["Unified Multimodal Query"])

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB
MAX_QUERY_CHARS = 2000              # 🔐 NEW: Prevent extremely long text input


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
# 1️⃣ TEXT-ONLY
# ----------------------------------------------------
@router.post("/text", response_model=AskResponse)
async def ask_text(query: str = Form(...)):
    start = time.time()
    request_id = str(uuid.uuid4())

    # Clean + validate
    if not query or not query.strip():
        raise HTTPException(400, "Please provide a valid text query.")

    query = query.strip()

    # 🔐 NEW — Length limit
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(
            413, f"Query too long. Maximum allowed is {MAX_QUERY_CHARS} characters."
        )

    try:
        response = route_query(query=query, image_path=None)
    except Exception as e:
        raise HTTPException(500, f"AgriGPT processing error: {e}")

    return AskResponse(
        request_id=request_id,
        status="success",
        elapsed_ms=int((time.time() - start) * 1000),
        input={"query": query, "image_uploaded": False},
        analysis=response,
    )


# ----------------------------------------------------
# 2️⃣ IMAGE-ONLY
# ----------------------------------------------------
@router.post("/image", response_model=AskResponse)
async def ask_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    start = time.time()
    request_id = str(uuid.uuid4())
    tmp_path = ""

    # Declared MIME check
    if file.content_type not in ALLOWED_IMAGE_MIME:
        raise HTTPException(415, "Unsupported file format. Only JPEG/PNG allowed.")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded image is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (maximum 8 MB).")

        ext = ".jpg" if file.content_type == "image/jpeg" else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # Real MIME verification
        real_type = imghdr.what(tmp_path)
        if real_type not in ("jpeg", "png"):
            background_tasks.add_task(os.remove, tmp_path)
            raise HTTPException(400, "Invalid or corrupted image file.")

        # Process only image
        response = route_query(query=None, image_path=tmp_path)

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
        raise HTTPException(500, f"Error processing image: {e}")


# ----------------------------------------------------
# 3️⃣ MULTIMODAL (TEXT + IMAGE)
# ----------------------------------------------------
@router.post("/chat", response_model=AskResponse)
async def ask_chat(
    background_tasks: BackgroundTasks,
    query: str = Form(""),
    file: UploadFile = File(None)
):
    start = time.time()
    request_id = str(uuid.uuid4())
    tmp_path = ""

    query_clean = (query or "").strip()

    # 🔐 NEW — Only check length if text exists
    if query_clean and len(query_clean) > MAX_QUERY_CHARS:
        raise HTTPException(
            413, f"Query too long. Maximum allowed is {MAX_QUERY_CHARS} characters."
        )

    # ------------------------------------------
    # A. No image → TEXT-ONLY
    # ------------------------------------------
    if not file:
        response = route_query(query=query_clean or None, image_path=None)
        return AskResponse(
            request_id=request_id,
            status="success",
            elapsed_ms=int((time.time() - start) * 1000),
            input={"query": query_clean, "image_uploaded": False},
            analysis=response,
        )

    # ------------------------------------------
    # B. Validate Image MIME
    # ------------------------------------------
    if file.content_type not in ALLOWED_IMAGE_MIME:
        raise HTTPException(415, "Unsupported image type. Only JPEG/PNG allowed.")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded image is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (maximum 8 MB).")

        ext = ".jpg" if file.content_type == "image/jpeg" else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # REAL MIME VALIDATION
        real_type = imghdr.what(tmp_path)
        if real_type not in ("jpeg", "png"):
            background_tasks.add_task(os.remove, tmp_path)
            raise HTTPException(400, "Invalid or corrupted image file.")

        # Decide routing
        if not query_clean:
            response = route_query(query=None, image_path=tmp_path)
        else:
            response = route_query(query=query_clean, image_path=tmp_path)

        background_tasks.add_task(os.remove, tmp_path)

        return AskResponse(
            request_id=request_id,
            status="success",
            elapsed_ms=int((time.time() - start) * 1000),
            input={"query": query_clean, "image_uploaded": True},
            analysis=response,
        )

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            background_tasks.add_task(os.remove, tmp_path)
        raise HTTPException(500, f"Error processing chat request: {e}")
