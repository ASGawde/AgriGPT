# backend/agents/crop_agent.py

from backend.services.text_service import query_groq_text
from backend.agents.agri_agent_base import AgriAgentBase


class CropAgent(AgriAgentBase):
    """Provides general crop management, fertilizer, soil, and cultivation advice."""

    name = "CropAgent"

    def handle_query(self, query: str = None, image_path: str = None) -> str:
        """
        Handles crop-related text questions:
        - Fertilizer dosage
        - Soil preparation
        - Crop growth timing
        - Yield improvement
        (Image input is ignored for this agent)
        """

        # Validate text input
        if not query or not query.strip():
            response = (
                "Please ask a crop-related question. Example:\n"
                "- 'How to improve my rice yield?'\n"
                "- 'What fertilizer should I use for tomatoes?'"
            )
            return self.respond_and_record(
                query="No query provided",
                response=response,
                image_path=image_path
            )

        # --- Construct AI prompt ---
        prompt = f"""
        You are **AgriGPT – Crop Management Specialist**.

        The farmer asks:
        "{query}"

        Provide clear, simple, farmer-friendly advice.
        MUST include:
        - Fertilizer type + dosage
        - Soil preparation tips
        - Growth-stage guidance or yield improvement tips
        - Specific actionable steps (not theory)
        - Tamil Nadu/Kharif relevance if applicable

        Keep it short, practical, and supportive.
        """

        # --- Query Groq API ---
        try:
            response = query_groq_text(prompt)
        except Exception as e:
            response = f"Error generating crop advice: {e}"

        # --- Log + return ---
        return self.respond_and_record(
            query=query,
            response=response,
            image_path=image_path
        )
