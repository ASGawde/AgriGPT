# backend/services/vision_service.py
"""
Vision Service (Unified & Production-Safe)
------------------------------------------
Fixes:
- Retry logic for 429/500 Groq failures
- Timeout protection
- Consistent temperature + max_tokens
- Sanitized inputs (Tamil/emojis safe)
- Validation for unsupported formats
- Normalized output formatting
"""

from __future__ import annotations
import base64
import imghdr
import os
import time

from groq import Groq
from backend.core.config import settings

MAX_RETRIES = 3
BACKOFF = [1, 2, 4]   # exponential retry delays


# ---------------------------------------------------------
# Utility: Image MIME Detection
# ---------------------------------------------------------
def _detect_mime(image_path: str) -> str:
    detected = imghdr.what(image_path)

    if detected == "png":
        return "image/png"
    if detected in ("jpeg", "jpg"):
        return "image/jpeg"

    # Unsupported images (HEIC, WEBP, GIF, etc.)
    supported = ["png", "jpeg", "jpg"]
    if detected not in supported:
        raise ValueError(
            f"Unsupported image format '{detected}'. Please upload PNG or JPG."
        )

    return "image/jpeg"


# ---------------------------------------------------------
# Utility: Flatten Output
# ---------------------------------------------------------
def _normalize_output(o):
    if isinstance(o, str):
        return o.strip()

    if isinstance(o, list):
        return "\n".join(_normalize_output(x) for x in o)

    if isinstance(o, dict):
        return "\n".join(f"{k}: {v}" for k, v in o.items())

    return str(o).strip()


# ---------------------------------------------------------
# Vision Query Function
# ---------------------------------------------------------
def query_groq_image(image_path: str, prompt: str) -> str:
    """
    Robust image+prompt call to Groq Vision with:
    - sanitization
    - retries
    - validation
    - consistent model settings
    """

    # Validate file exists
    if not os.path.exists(image_path):
        return "Error: Image file does not exist."

    # File size guard (8MB max)
    if os.path.getsize(image_path) > 8 * 1024 * 1024:
        return "Error: Image too large. Please upload an image smaller than 8 MB."

    # Validate MIME
    try:
        mime = _detect_mime(image_path)
    except ValueError as e:
        return str(e)

    # Clean prompt (Tamil/emojis_safe)
    prompt = prompt.strip() if isinstance(prompt, str) else ""

    # Read & encode image
    try:
        with open(image_path, "rb") as f:
            raw = f.read()
    except Exception:
        return "Error reading image file."

    if not raw:
        return "Error: Image file is empty."

    b64 = base64.b64encode(raw).decode("utf-8")
    image_data_url = f"data:{mime};base64,{b64}"

    # Retry loop
    for attempt in range(MAX_RETRIES):
        try:
            client = Groq(
                api_key=settings.GROQ_API_KEY,
                timeout=30  # timeout safeguard
            )

            completion = client.chat.completions.create(
                model=settings.VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are AgriGPT Vision, an expert crop disease classifier."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    },
                ],
                temperature=0.4,
                max_tokens=900,
                top_p=1.0,
                stream=False,
            )

            msg = (
                completion.choices[0].message.content
                if completion
                   and completion.choices
                   and completion.choices[0].message
                   and completion.choices[0].message.content
                else None
            )

            return _normalize_output(msg) or "I could not analyze the image."

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF[attempt])
                continue

            return f"Groq vision model error: {str(e)}"
