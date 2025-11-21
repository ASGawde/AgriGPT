# backend/core/llm_client.py

"""
llm_client.py
--------------
Central place to configure and cache the LangChain ChatGroq client.
All agents share the same LLM instance to ensure:
- consistent formatting
- reduced latency
- reduced API overhead
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

from langchain_groq import ChatGroq

from backend.core.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Build and cache the LangChain ChatGroq LLM client.

    Notes:
    - Uses TEXT_MODEL_NAME from settings.
    - Cached to avoid repeated client initialization.
    - Low temperature ensures consistent routing.
    """

    api_key: Final[str] = settings.GROQ_API_KEY
    model: Final[str] = settings.TEXT_MODEL_NAME  # THE CORRECT FIELD

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in your environment variables.")

    if not model:
        raise RuntimeError("TEXT_MODEL_NAME is missing in your environment variables.")

    return ChatGroq(
        api_key=api_key,       # FIX: correct arg name
        model=model,           # FIX: correct arg name
        temperature=0.2,       # more deterministic routing
        max_tokens=1500,       # agents often generate long guidance
    )
