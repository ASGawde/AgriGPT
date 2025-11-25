# backend/agents/agri_agent_base.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from backend.services.history_service import log_interaction


class AgriAgentBase(ABC):
    name: str = "AgriAgentBase"

    # --------------------------------------------------
    # Every agent must follow this signature
    # --------------------------------------------------
    @abstractmethod
    def handle_query(
        self,
        query: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> str:
        pass

    # --------------------------------------------------
    # Detect type for logging (FIXED)
    # --------------------------------------------------
    @staticmethod
    def _normalize_query(q: Optional[str]) -> str:
        """Normalize query by stripping whitespace safely."""
        if q is None:
            return ""
        return q.strip()

    @staticmethod
    def _detect_query_type(query, image_path):
        """
        FIXES:
        - query="" + image → image-only
        - only image → image
        - only query → text
        - both non-empty → multimodal
        """

        normalized_query = AgriAgentBase._normalize_query(query)

        if normalized_query and image_path:
            return "multimodal"
        if image_path:
            return "image"
        return "text"

    # --------------------------------------------------
    # Actual logging (FIXED to include image_path)
    # --------------------------------------------------
    def record(self, query, response, query_type, image_path=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "query": query or "",
            "response": response[:5000],
            "type": query_type,
        }

        # NEW → include image path/filename for debugging
        if image_path:
            entry["image_path"] = image_path

        try:
            log_interaction(entry)
        except Exception:
            pass  # Never break execution because of logging

    # --------------------------------------------------
    # Wrapper used by ALL agents (UPDATED)
    # --------------------------------------------------
    def respond_and_record(self, query, response, image_path=None):
        query_type = self._detect_query_type(query, image_path)
        self.record(query, response, query_type, image_path=image_path)
        return response
