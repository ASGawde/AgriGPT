# backend/agents/master_agent.py

"""
Master Agent Router (v5.1 - FINAL + OPTIMIZED)
-----------------------------------------------
Stable + VERIFIED FEATURES:
- FormatterAgent runs only at the END (one LLM call)
- Zero double-execution of PestAgent (image + multimodal fix)
- Text-only, image-only, multimodal pipelines cleanly separated
- Router failures fall back safely to CropAgent
- Combines multimodal outputs before formatting (correct)
"""

from __future__ import annotations
from typing import Optional, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel

from backend.core.langchain_tools import AGENT_DESCRIPTIONS, AGENT_REGISTRY
from backend.core.llm_client import get_llm
from backend.services.text_service import query_groq_text

router_llm = get_llm()

# 🔐 NEW — prevent 100KB input from hitting LLM
MAX_QUERY_CHARS = 2000


# ---------------------------------------------------------------
# ROUTING SCHEMA
# ---------------------------------------------------------------
class RoutingResult(BaseModel):
    agents: List[str]
    reason: str


def _format_agent_descriptions() -> str:
    """Generate router description text."""
    return "\n".join(
        f"- {a['name']}: {a['description']}"
        for a in AGENT_DESCRIPTIONS
    )


# ---------------------------------------------------------------
# MAIN ROUTER ENTRY
# ---------------------------------------------------------------
def route_query(query: Optional[str] = None, image_path: Optional[str] = None):
    """
    Universal routing entry point:
    - Multimodal → text + image
    - Image-only → PestAgent + formatter
    - Text-only → LangChain router + formatter
    """

    # Normalize early
    if query is not None:
        query = query.strip()
        if query == "":
            query = None

    # 🔐 NEW — Length limit (safe for all paths)
    if query and len(query) > MAX_QUERY_CHARS:
        return (
            f"Your query is too long. Maximum allowed is {MAX_QUERY_CHARS} characters.\n"
            f"Please shorten your question and try again."
        )

    # ============================================================
    # 1️⃣ MULTIMODAL (text + image)
    # ============================================================
    if image_path and query:
        pest_agent = AGENT_REGISTRY["PestAgent"]
        pest_output = pest_agent.handle_query(query=query, image_path=image_path)

        # Router should only choose text agents
        agent_names, _ = _choose_agent_via_langchain(query)

        text_outputs = []
        for name in agent_names:
            if name == "PestAgent":   # prevent double-run
                continue

            agent = AGENT_REGISTRY.get(name)
            if agent:
                text_outputs.append(agent.handle_query(query=query))

        merged_text = "\n\n---\n\n".join(text_outputs).strip() or \
                       "No additional text-based expert advice."

        # Merge multimodal result
        combined_prompt = f"""
        You are AgriGPT, a multimodal agriculture expert.

        Farmer question:
        "{query}"

        Image-based diagnosis:
        {pest_output}

        Expert text guidance:
        {merged_text}

        Combine into one simple, farmer-friendly final answer.
        """.strip()

        try:
            combined = query_groq_text(combined_prompt)
        except Exception as e:
            combined = f"Error generating final answer: {e}"

        formatter = AGENT_REGISTRY["FormatterAgent"]
        return formatter.handle_query(query=combined)

    # ============================================================
    # 2️⃣ IMAGE ONLY
    # ============================================================
    if image_path:
        pest_agent = AGENT_REGISTRY["PestAgent"]
        resp = pest_agent.handle_query(query="", image_path=image_path)

        formatter = AGENT_REGISTRY["FormatterAgent"]
        return formatter.handle_query(query=resp)

    # ============================================================
    # 3️⃣ TEXT EMPTY
    # ============================================================
    if not query:
        return "Please provide a text query or image."

    # ============================================================
    # 4️⃣ TEXT ONLY
    # ============================================================
    return _run_langchain_text_agent(query)


# ---------------------------------------------------------------
# TEXT-ONLY LOGIC
# ---------------------------------------------------------------
def _run_langchain_text_agent(query: str) -> str:
    agent_names, reasoning = _choose_agent_via_langchain(query)

    outputs = []
    for name in agent_names:
        agent = AGENT_REGISTRY.get(name)
        if agent:
            outputs.append(agent.handle_query(query=query))

    if not outputs:
        outputs.append(AGENT_REGISTRY["CropAgent"].handle_query(query=query))

    merged = "\n\n---\n\n".join(outputs)

    formatter = AGENT_REGISTRY["FormatterAgent"]
    final = formatter.handle_query(query=merged)

    if reasoning:
        final += f"\n\n_(Routed to {agent_names} because: {reasoning})_"

    return final


# ---------------------------------------------------------------
# ROUTER CHAIN
# ---------------------------------------------------------------
def _build_router_chain() -> RunnableSequence:
    agent_descriptions = _format_agent_descriptions()

    template = """
    You are AgriGPT Router.
    Choose 1 or more agents based on the query.

    Available agents:
    {agent_descriptions}

    Return ONLY valid JSON:
    {{
        "agents": ["Agent1", "Agent2"],
        "reason": "short explanation"
    }}

    {format_instructions}

    USER QUERY:
    {query}
    """

    parser = JsonOutputParser(pydantic_object=RoutingResult)

    prompt = PromptTemplate(
        template=template,
        input_variables=["query"],
        partial_variables={
            "agent_descriptions": agent_descriptions,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    return prompt | router_llm | parser


# ---------------------------------------------------------------
# ROUTER SELECTION (SAFE + CLEANED)
# ---------------------------------------------------------------
def _choose_agent_via_langchain(query: str) -> tuple[list[str], str]:
    try:
        chain = _build_router_chain()
        result = chain.invoke({"query": query})  # parsed JSON → dict

        agents = result.get("agents", [])
        reason = result.get("reason", "No reason given.")

        # Normalize string → list
        if isinstance(agents, str):
            agents = [agents]

        from backend.core.langchain_tools import NON_ROUTABLE_AGENTS

        # Filter out formatter + unknown names
        agents = [
            a for a in agents
            if a in AGENT_REGISTRY and a not in NON_ROUTABLE_AGENTS
        ]

        if not agents:
            return ["CropAgent"], "Fallback: no valid agents selected."

        return agents, reason

    except Exception as e:
        return ["CropAgent"], f"Routing failed: {e}"
