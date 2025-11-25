# backend/agents/pest_agent.py

from backend.services.text_service import query_groq_text
from backend.services.vision_service import query_groq_image
from backend.agents.agri_agent_base import AgriAgentBase


class PestAgent(AgriAgentBase):
    """
    Pest & Disease Agent
    Handles both image-based and text-based diagnosis.
    """

    name = "PestAgent"

    def handle_query(self, query: str = None, image_path: str = None) -> str:
        """
        FIXED:
        - If image_path is provided, DO NOT run text-based diagnosis.
        - Prevents duplicate PestAgent execution in multimodal mode.
        """

        # ------------------------------------------------------
        # CASE 0 — No input at all
        # ------------------------------------------------------
        if not query and not image_path:
            msg = (
                "Please upload a crop image or describe symptoms like "
                "yellow leaves, white powder, brown patches, insects, or wilting."
            )
            return self.respond_and_record(
                "No input provided",
                msg,
                image_path=image_path
            )

        # ------------------------------------------------------
        # CASE 1 — IMAGE ALWAYS TAKES PRIORITY (FIX)
        # ------------------------------------------------------
        if image_path:
            # Ignore text completely if image exists
            # -> prevents duplicate calls from master_agent
            vision_prompt = """
            You are AgriGPT Vision — a crop pest and disease detection expert.

            Analyze this crop image and return:

            1. Likely problem/pest/disease name
            2. Key visual symptoms visible in the photo
            3. Organic control options (neem, soap, traps, pruning, etc.)
            4. Chemical control (last resort) — only category name
            5. Preventive steps for next season

            Use:
            - Simple language
            - Bullet points
            - Short sentences
            """

            try:
                result = query_groq_image(image_path, vision_prompt)
            except Exception as e:
                result = f"Error analyzing crop image: {e}"

            return self.respond_and_record(
                "Image-based pest analysis",
                result,
                image_path=image_path
            )

        # ------------------------------------------------------
        # CASE 2 — TEXT-BASED DIAGNOSIS (Only when NO image)
        # ------------------------------------------------------
        query_clean = query.strip()

        text_prompt = f"""
        You are AgriGPT Pest Advisor.

        The farmer described:
        \"\"\"{query_clean}\"\"\"

        Provide:
        - Most likely pest/disease/deficiency
        - Confirming symptoms
        - Organic treatments (neem, soap, pruning, traps, bio-control)
        - Chemical treatment (only category, with caution)
        - Prevention tips

        Use bullet points and simple language.
        """

        try:
            result = query_groq_text(text_prompt)
        except Exception as e:
            result = f"Error generating pest diagnosis: {e}"

        return self.respond_and_record(
            query_clean,
            result,
            image_path=image_path
        )
