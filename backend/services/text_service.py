# backend/services/text_service.py
"""
Unified Text Service (Production-Stable)
----------------------------------------
Fixes:
- Retry logic for Groq 429/500/timeout errors
- Consistent system message delivery
- Uses unified ChatGroq client from llm_client.py
- max_tokens=1500 everywhere for consistency
- Safe normalization for all responses
- NEW: Prompt length limit + auto-truncate
"""

from __future__ import annotations
import time
from typing import Optional

from backend.core.llm_client import get_llm
from backend.core.config import settings

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]

# 🔐 NEW — Prevent sending huge prompts to Groq
MAX_PROMPT_CHARS = 4000


def _normalize_output(output: Optional[str]) -> str:
    """Safely normalize model output into clean text."""
    if not output:
        return ""
    if isinstance(output, str):
        return output.strip()
    try:
        return str(output).strip()
    except Exception:
        return ""


def _is_retryable_error(error: Exception) -> bool:
    """Detects if an error should trigger a retry."""
    msg = str(error).lower()
    return any(
        key in msg
        for key in [
            "429",
            "rate limit",
            "too many requests",
            "500",
            "internal server error",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "service unavailable",
            "gateway timeout",
        ]
    )


def query_groq_text(
    prompt: str,
    system_msg: str = "You are AgriGPT.",
) -> str:
    """
    Production-safe LLM wrapper:
    - retries
    - consistent system+user message formatting
    - max_tokens=1500
    - NEW: prompt-size control
    """

    # 🔐 NEW: truncate very long prompts
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = (
            prompt[:MAX_PROMPT_CHARS]
            + "\n\n[Note: Input was truncated for safety because it exceeded "
              f"{MAX_PROMPT_CHARS} characters]"
        )

    llm = get_llm()

    for attempt in range(MAX_RETRIES):
        try:
            resp = llm.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.2,
            )

            raw = getattr(resp, "content", None)
            cleaned = _normalize_output(raw)

            return cleaned or "I could not generate a response at this moment. Please try again."

        except Exception as e:
            if attempt < MAX_RETRIES - 1 and _is_retryable_error(e):
                time.sleep(RETRY_BACKOFF[attempt])
                continue

            return f"Error generating response: {str(e)}"
