from __future__ import annotations
import base64
import os
import time
from typing import Any

from groq import Groq
from backend.core.config import settings

MAX_RETRIES = 3
RETRY_BACKOFF = (1, 2, 4)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_VISION_PROMPT_CHARS = 2000

def _detect_mime(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            header = f.read(10)
    except Exception:
        return "unknown"

    if header.startswith(b"\x89PNG"):
        return "image/png"

    if header.startswith(b"\xFF\xD8"):
        return "image/jpeg"

    return "unknown"


def _normalize_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, list):
        return "\n".join(_normalize_output(x) for x in output)
    if isinstance(output, dict):
        return "\n".join(f"{k}: {v}" for k, v in output.items())
    return str(output).strip()


def query_groq_image(image_path: str, prompt: str) -> str:

    if not image_path or not os.path.exists(image_path):
        return "The image file was not found."

    if os.path.getsize(image_path) > MAX_IMAGE_BYTES:
        return "The image is too large. Please upload an image under 8MB."

    mime = _detect_mime(image_path)
    if mime not in ("image/png", "image/jpeg"):
        return "Unsupported image format. Please upload a PNG or JPG image."

    try:
        with open(image_path, "rb") as f:
            raw_bytes = f.read()
    except Exception:
        return "The image could not be read."

    if not raw_bytes:
        return "The image file appears to be empty."

    if not isinstance(prompt, str):
        prompt = ""

    if len(prompt) > MAX_VISION_PROMPT_CHARS:
        prompt = prompt[:MAX_VISION_PROMPT_CHARS] + " [Prompt truncated]"

    image_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    image_url = f"data:{mime};base64,{image_b64}"

    client = Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=30,
    )

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=settings.VISION_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are AgriGPT Vision, a multimodal agricultural image "
                            "observation assistant.\n\n"
                            "Your task is to describe ONLY what is clearly visible in the image.\n\n"
                            "STRICT RULES:\n"
                            "- Do NOT guess pests, diseases, or causes\n"
                            "- Do NOT hallucinate unseen symptoms\n"
                            "- Do NOT infer crop stage or health unless clearly visible\n"
                            "- If uncertain, say so clearly\n\n"
                            "ALLOWED:\n"
                            "- Describe visible spots, discoloration, holes, insects, mold, wilting\n"
                            "- Mention blur, poor lighting, or unclear image quality\n\n"
                            "OUTPUT STYLE:\n"
                            "- Bullet points\n"
                            "- Simple, farmer-friendly language\n"
                            "- No technical jargon unless unavoidable"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                max_tokens=900,
                temperature=0.3,
                top_p=1.0,
            )

            result = _normalize_output(
                completion.choices[0].message.content
            )

            if not result or len(result) < 5:
                return (
                    "The image could not be analyzed clearly. "
                    "Please upload a clearer image."
                )

            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
                continue

            return (
                "The image could not be analyzed at this time. "
                "Please try again later."
            )
