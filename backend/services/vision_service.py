# backend/services/vision_service.py

import base64
import imghdr
from groq import Groq
from backend.core.config import settings


def get_vision_client() -> Groq:
    """
    Always create a fresh Groq client for each call to avoid
    concurrency errors under multiple simultaneous image queries.
    """
    return Groq(api_key=settings.GROQ_API_KEY)


def _detect_mime(image_path: str) -> str:
    """
    Detect MIME type safely. Fallback to JPEG.
    """
    detected = imghdr.what(image_path)
    if detected == "png":
        return "image/png"
    if detected == "jpeg" or detected == "jpg":
        return "image/jpeg"
    return "image/jpeg"  # default


def _normalize_output(output):
    """
    Groq sometimes returns content as str, list, dict.
    Make sure everything is converted to clean text.
    """
    if isinstance(output, str):
        return output.strip()

    if isinstance(output, list):
        return "\n".join([_normalize_output(o) for o in output])

    if isinstance(output, dict):
        return str(output)

    return str(output).strip()


def query_groq_image(image_path: str, prompt: str) -> str:
    """
    Send an image (base64-encoded) + prompt to the Groq Vision model.

    Returns:
        Clean, normalized, simple-language diagnosis.
    """

    try:
        # ----------------------------------------------------
        # 1. Validate + detect MIME type
        # ----------------------------------------------------
        mime_type = _detect_mime(image_path)

        # ----------------------------------------------------
        # 2. Read and base64 encode image
        # ----------------------------------------------------
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        if not img_bytes:
            return "Error: Uploaded image is empty."

        b64_image = base64.b64encode(img_bytes).decode("utf-8")

        # ----------------------------------------------------
        # 3. Create fresh Groq client (thread-safe)
        # ----------------------------------------------------
        client = get_vision_client()

        # ----------------------------------------------------
        # 4. Call Groq Vision model
        # ----------------------------------------------------
        completion = client.chat.completions.create(
            model=settings.VISION_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are AgriGPT Vision, an expert crop image analyst."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.6,
            max_tokens=700,     # FIXED: use max_tokens (correct Groq param)
            top_p=1.0,
            stream=False,
        )

        # ----------------------------------------------------
        # 5. Defensive extraction
        # ----------------------------------------------------
        if (
            not completion
            or not completion.choices
            or not completion.choices[0].message
            or not completion.choices[0].message.content
        ):
            return "I could not analyze the image. Please try again with a clearer photo."

        raw = completion.choices[0].message.content
        return _normalize_output(raw)

    except FileNotFoundError:
        return "Error: Image file not found."

    except Exception as e:
        return f"Groq vision model error: {str(e)}"
