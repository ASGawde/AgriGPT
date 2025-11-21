# backend/agents/irrigation_agent.py

from backend.services.text_service import query_groq_text
from backend.agents.agri_agent_base import AgriAgentBase


class IrrigationAgent(AgriAgentBase):
    """
    Irrigation Agent
    -----------------
    Provides expert guidance on:
    - Watering intervals
    - Soil moisture balance
    - Drip/sprinkler operation
    - Water-saving practices
    """

    name = "IrrigationAgent"

    def handle_query(self, query: str = None, image_path: str = None) -> str:
        """
        Handle irrigation-related queries.
        Image input is ignored for this agent.
        """

        # ------------------------------------------------------
        # CASE 0 — Missing query
        # ------------------------------------------------------
        if not query or not query.strip():
            response = (
                "Please ask an irrigation-related question such as:\n"
                "- 'How often should I irrigate onions?'\n"
                "- 'How to save water in drip irrigation?'\n"
                "- 'How to adjust irrigation during summer?'\n"
            )
            return self.respond_and_record(
                "No query provided",
                response,
                image_path=image_path
            )

        query_clean = query.strip()

        # ------------------------------------------------------
        # CASE 1 — Build LLM prompt
        # ------------------------------------------------------
        prompt = f"""
        You are **AgriGPT – Irrigation Expert**.

        The farmer asked:
        \"{query_clean}\"

        Provide clear irrigation guidance covering:
        - Correct watering intervals (daily / weekly / stage-based)
        - Soil moisture management (how to check & maintain)
        - Drip, sprinkler, and flood irrigation best practices
        - Water-saving techniques (mulching, scheduling, pressure control)
        - How to adjust irrigation during rainfall or extreme heat
        - Soil-type adjustments (sandy, clay, loam)

        Use:
        - Short sentences
        - Bullet points
        - Very simple farmer-friendly tone
        """

        # ------------------------------------------------------
        # CASE 2 — Query Groq safely
        # ------------------------------------------------------
        try:
            result = query_groq_text(prompt)
        except Exception as e:
            result = f"Error generating irrigation advice: {e}"

        # ------------------------------------------------------
        # CASE 3 — Log & Return
        # ------------------------------------------------------
        return self.respond_and_record(
            query_clean,
            result,
            image_path=image_path
        )
