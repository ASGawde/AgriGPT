# backend/agents/agri_agent_base.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from backend.services.history_service import log_interaction


class AgriAgentBase(ABC):
    name: str = "AgriAgentBase"

    # Every agent must follow this signature
    @abstractmethod
    def handle_query(
        self,
        query: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> str:
        pass

    # Detect type for logging
    @staticmethod
    def _detect_query_type(query, image_path):
        if query and image_path:
            return "multimodal"
        if image_path:
            return "image"
        return "text"

    # Store logs
    def record(self, query, response, query_type):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "query": query or "",
            "response": response[:5000],
            "type": query_type,
        }
        try:
            log_interaction(entry)
        except Exception:
            pass

    # Standard wrapper
    def respond_and_record(self, query, response, image_path=None):
        query_type = self._detect_query_type(query, image_path)
        self.record(query, response, query_type)
        return response
