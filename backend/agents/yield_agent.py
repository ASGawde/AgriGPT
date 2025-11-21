# backend/agents/yield_agent.py

from backend.services.text_service import query_groq_text
from backend.agents.agri_agent_base import AgriAgentBase


class YieldAgent(AgriAgentBase):
    """
    Yield Optimization Agent
    ------------------------
    Provides farmers with:
    - Expected yield ranges
    - Diagnosis of low-yield causes
    - Step-by-step improvement actions
    - Crop-stage-specific advice
    """

    name = "YieldAgent"

    def handle_query(self, query: str = None, image_path: str = None) -> str:
        """
        Handles yield-related queries such as:
        - "How to increase rice yield?"
        - "Why is my maize yield low?"
        - "What reduces tomato productivity?"
        """

        # ------------------------------------------------------
        # CASE 0 — Missing query
        # ------------------------------------------------------
        if not query or not query.strip():
            msg = (
                "Please describe your crop and the yield problem. Examples:\n"
                "- 'My rice yield is low this season'\n"
                "- 'Maize only giving 2 tons per hectare'\n"
                "- 'Tomato plants giving fewer fruits'"
            )
            return self.respond_and_record(
                "No query provided",
                msg,
                image_path=image_path
            )

        query_clean = query.strip()

        # ------------------------------------------------------
        # CASE 1 — Build prompt for Groq
        # ------------------------------------------------------
        prompt = f"""
        You are AgriGPT, a specialist in crop yield improvement.

        The farmer asked:
        \"\"\"{query_clean}\"\"\"

        Provide a clear, simple explanation including:

        **1. Expected Yield Range**
        - Typical yield in India/global range for that crop (approximate).

        **2. Causes of Low Yield**
        Cover key categories:
        - Soil quality or nutrient imbalance
        - Fertilizer gaps (wrong type, timing, under-application)
        - Water stress (too little/too much)
        - Pest or disease pressure
        - Seed variety issues
        - Weather or planting time

        **3. Actionable Yield Improvement Steps**
        - Fertilizer schedule (stage-wise)
        - Irrigation intervals
        - Soil improvement tips
        - Pest/disease prevention
        - Seed/variety recommendations

        **4. Simple, farmer-friendly language**
        - Short sentences
        - Bullet points
        - No technical jargon

        Keep the output practical and easy to follow.
        """

        # ------------------------------------------------------
        # CASE 2 — Query Groq safely
        # ------------------------------------------------------
        try:
            result = query_groq_text(prompt)
        except Exception as e:
            result = f"Error generating yield advice: {e}"

        # ------------------------------------------------------
        # RETURN + LOG
        # ------------------------------------------------------
        return self.respond_and_record(
            query_clean,
            result,
            image_path=image_path
        )
