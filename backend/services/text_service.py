# backend/services/text_service.py

from groq import Groq
from backend.core.config import settings


def get_text_client() -> Groq:
    """
    Create a fresh Groq client for every call.
    This avoids concurrency issues when multiple agents call the model simultaneously.
    """
    return Groq(api_key=settings.GROQ_API_KEY)


def _normalize_output(output):
    """
    Normalize different output types into clean text.
    Groq sometimes returns arrays or objects depending on content.
    """
    if isinstance(output, str):
        return output.strip()

    if isinstance(output, list):
        return "\n".join([_normalize_output(o) for o in output])

    if isinstance(output, dict):
        return str(output)

    return str(output).strip()


def query_groq_text(prompt: str, system_msg: str = "You are AgriGPT.") -> str:
    """
    Send a robust text request to Groq with defensive failure-handling.

    - Uses correct model: settings.TEXT_MODEL_NAME
    - Safe against missing choices[] or empty content
    - Returns clean, trimmed output
    - Accepts system message override (for formatting, routing, etc.)
    """

    try:
        client = get_text_client()

        completion = client.chat.completions.create(
            model=settings.TEXT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,          # Good balance for creativity + accuracy
            max_tokens=600,           # FIXED: Groq expects "max_tokens"
            top_p=1.0,
            stream=False,
        )

        # Defensive extraction
        if (
            not completion
            or not completion.choices
            or not completion.choices[0].message
            or not completion.choices[0].message.content
        ):
            return "I could not generate a response at this moment. Please try again."

        raw = completion.choices[0].message.content
        return _normalize_output(raw)

    except Exception as e:
        return f"Error while generating response: {str(e)}"
