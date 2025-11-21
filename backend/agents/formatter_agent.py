# backend/agents/formatter_agent.py

from backend.services.text_service import query_groq_text
from backend.agents.agri_agent_base import AgriAgentBase


class FormatterAgent(AgriAgentBase):
    name = "FormatterAgent"

    # FIX: standardized signature
    def handle_query(self, query: str = None, image_path: str = None) -> str:

        if not query or not query.strip():
            response = "No text was provided for formatting."
            return self.respond_and_record("Empty text", response, image_path)

        prompt = f"""
        You are AgriGPT Formatter.
        Improve the following content:

        \"\"\"{query}\"\"\"

        Rules:
        - Add a 3–6 word title
        - Use clear bullet points
        - Short sentences
        - Simple farmer-friendly tone
        - End with a 1-line summary
        - Remove fluff
        """

        try:
            formatted = query_groq_text(prompt)
        except Exception as e:
            formatted = f"Error formatting text: {e}"

        return self.respond_and_record(query, formatted, image_path)
