# backend/agents/subsidy_agent.py

from backend.services.text_service import query_groq_text
from backend.agents.agri_agent_base import AgriAgentBase
from backend.services.rag_service import rag_service
import unicodedata


class SubsidyAgent(AgriAgentBase):
    """
    Subsidy & Government Scheme Agent
    ---------------------------------
    Handles:
    - Central/State schemes
    - Irrigation subsidies
    - Fertilizer/seed/machinery support
    - PM-Kisan, KCC loan details
    """

    name = "SubsidyAgent"

    def _sanitize_query(self, text: str) -> str:
        """
        Normalize Unicode and strip unsafe characters.
        Prevents FAISS and embedding errors.
        """
        if not text:
            return ""

        # Normalize Tamil/Hindi/Unicode properly
        text = unicodedata.normalize("NFKC", text)

        # Remove hidden null chars or control chars
        text = text.replace("\x00", "").replace("\u200c", "")

        return text.strip()

    def handle_query(self, query: str = None, image_path: str = None) -> str:
        """
        Handles subsidy-related queries.
        Image is ignored for this agent.
        """

        # ------------------------------------------------------
        # CASE 0 — No query
        # ------------------------------------------------------
        if not query or not query.strip():
            msg = (
                "Please ask about a specific subsidy or government scheme — for example:\n"
                "- 'Drip irrigation subsidy in Tamil Nadu'\n"
                "- 'PM-Kisan eligibility'\n"
                "- 'Kisan Credit Card loan details'\n"
                "- 'Fertilizer subsidy amount'\n"
            )
            return self.respond_and_record("No query provided", msg, image_path)

        # ------------------------------------------------------
        # CASE 1 — Sanitize input (FIXED)
        # ------------------------------------------------------
        query_clean = self._sanitize_query(query)

        # ------------------------------------------------------
        # CASE 2 — RAG Retrieval (FIXED with logging)
        # ------------------------------------------------------
        context_str = ""

        try:
            retrieved_docs = rag_service.retrieve(query_clean)
        except Exception as e:
            # Log error but continue gracefully
            rag_error_msg = f"[RAG ERROR] Failed to retrieve documents: {e}"
            context_str += f"\n\n{rag_error_msg}\n"
            retrieved_docs = []

        if retrieved_docs:
            context_str += "\n\n**Retrieved Official Information:**\n"
            for i, doc in enumerate(retrieved_docs, 1):
                context_str += (
                    f"Scheme {i}: {doc['scheme_name']}\n"
                    f"- Eligibility: {doc['eligibility']}\n"
                    f"- Benefits: {doc['benefits']}\n"
                    f"- Application: {doc['application_steps']}\n"
                    f"- Documents: {doc['documents']}\n"
                    f"- Notes: {doc['notes']}\n\n"
                )
        else:
            context_str += "\n\n(No matching official scheme found in RAG search.)\n"

        # ------------------------------------------------------
        # CASE 3 — Build LLM prompt
        # ------------------------------------------------------
        prompt = f"""
        You are AgriGPT, an expert on Indian agricultural subsidy and scheme information.

        The farmer asked:
        \"\"\"{query_clean}\"\"\"

        {context_str}

        If the Retrieved Official Information matches the user's query, USE IT as the primary source.
        If the RAG information is incomplete, rely on general knowledge (without exact numbers).

        Provide:
        1. Scheme name
        2. Central or State government
        3. Eligibility
        4. Financial benefits
        5. Application process
        6. Important notes

        Use:
        - Bullet points
        - Clear, simple language
        - Short sentences
        """

        # ------------------------------------------------------
        # CASE 4 — Query Groq LLM safely
        # ------------------------------------------------------
        try:
            result = query_groq_text(prompt)
        except Exception as e:
            result = f"Error retrieving subsidy details: {e}"

        # ------------------------------------------------------
        # CASE 5 — Log & Return
        # ------------------------------------------------------
        return self.respond_and_record(query_clean, result, image_path)
