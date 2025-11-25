# backend/core/llm_client.py
"""
Unified Groq LLM Client (Production-Safe)
-----------------------------------------
Fixes:
- Removes dangerous @lru_cache
- Adds REAL heartbeat ping (not attribute check)
- Auto-reconnect on stale/broken LLM connection
- Ensures router + agents use SAME LLM instance
"""

from __future__ import annotations
from typing import Optional
import time

from langchain_groq import ChatGroq
from backend.core.config import settings

# Cached instance
_cached_llm: Optional[ChatGroq] = None
_last_heartbeat: float = 0
_HEARTBEAT_INTERVAL = 300  # 5 minutes


def _build_fresh_llm() -> ChatGroq:
    """
    Always build a new, clean ChatGroq client.
    """
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.TEXT_MODEL_NAME,
        temperature=0.2,  # deterministic for router + agents
        max_tokens=1500,
    )


def _llm_alive(llm: ChatGroq) -> bool:
    """
    REAL heartbeat check — sends a tiny ping request.
    This detects:
    - expired API key
    - lost network
    - dead file descriptor
    - Groq downtime
    - connection reset
    """
    try:
        # This is a very small and cheap request
        llm.invoke("ping")
        return True
    except Exception:
        return False


def get_llm() -> ChatGroq:
    """
    Unified LLM client with:
    - Auto-reconnect
    - REAL heartbeat (ping)
    - Heartbeat-based refresh every 5 minutes
    - Router + agents stay consistent
    """
    global _cached_llm, _last_heartbeat

    now = time.time()

    # --------------------------------------------------------
    # CASE 1 — First use
    # --------------------------------------------------------
    if _cached_llm is None:
        _cached_llm = _build_fresh_llm()
        _last_heartbeat = now
        return _cached_llm

    # --------------------------------------------------------
    # CASE 2 — Heartbeat interval passed → perform a ping test
    # --------------------------------------------------------
    if now - _last_heartbeat > _HEARTBEAT_INTERVAL:
        if not _llm_alive(_cached_llm):
            # Cached client is stale or dead → rebuild
            _cached_llm = _build_fresh_llm()
        _last_heartbeat = now
        return _cached_llm

    # --------------------------------------------------------
    # CASE 3 — Cached client exists but is invalid BEFORE interval
    # (router requests or text service failures will hit this)
    # --------------------------------------------------------
    if not _llm_alive(_cached_llm):
        _cached_llm = _build_fresh_llm()
        _last_heartbeat = now

    return _cached_llm
