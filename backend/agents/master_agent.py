# backend/agents/master_agent.py

"""
Master Agent Router (v2)
------------------------
Central orchestrator for routing text and image queries to multiple AgriGPT agents.

Supports:
- Image-only
- Text-only
- Combined multimodal (text + image)
- Multi-agent routing via LangChain
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.core.langchain_tools import AGENT_DESCRIPTIONS, AGENT_REGISTRY
from backend.core.llm_client import get_llm
from backend.services.text_service import query_groq_text

router_llm = get_llm()


# ---------------------------------------------------------------------------
# MAIN ROUTER
# ---------------------------------------------------------------------------
def route_query(query: Optional[str] = None, image_path: Optional[str] = None) -> str:
    """
    Entry point for routing any user request.
    """

    # ---------------------------------
    # Case 1 — Combined text + image
    # ---------------------------------
    if image_path and query:

        # 1. Image → always PestAgent
        pest_agent = AGENT_REGISTRY["PestAgent"]
        pest_output = pest_agent.handle_query(query=query, image_path=image_path)

        # 2. Text → multi-agent routing
        agent_names, _ = _choose_agent_via_langchain(query)

        text_outputs = []
        for name in agent_names:
            if name == "PestAgent":
                continue
            agent = AGENT_REGISTRY.get(name)
            if agent:
                text_outputs.append(agent.handle_query(query=query))

        merged_text = "\n\n---\n\n".join(text_outputs)

        # 3. Merge both in LLM
        combined_prompt = f"""
        You are AgriGPT, a multimodal agriculture expert.

        Farmer question:
        "{query}"

        Image-based diagnosis:
        {pest_output}

        Expert text guidance:
        {merged_text}

        Combine everything into ONE clean, simple, farmer-friendly answer.
        Use bullet points and short sentences.
        """

        try:
            combined = query_groq_text(combined_prompt)
        except Exception as e:
            combined = f"Error generating multimodal final answer: {e}"

        # 4. FormatterAgent final pass
        formatter = AGENT_REGISTRY["FormatterAgent"]
        return formatter.handle_query(query=combined)

    # ---------------------------------
    # Case 2 — Image-only
    # ---------------------------------
    if image_path:
        pest_agent = AGENT_REGISTRY["PestAgent"]
        resp = pest_agent.handle_query(query="", image_path=image_path)

        formatter = AGENT_REGISTRY["FormatterAgent"]
        return formatter.handle_query(query=resp)

    # ---------------------------------
    # Case 3 — Invalid (no query)
    # ---------------------------------
    if not query:
        return "Please provide a text query or image."

    # ---------------------------------
    # Case 4 — Text-only
    # ---------------------------------
    return _run_langchain_text_agent(query)


# ---------------------------------------------------------------------------
# TEXT ROUTING
# ---------------------------------------------------------------------------
def _run_langchain_text_agent(query: str) -> str:
    agent_names, reasoning = _choose_agent_via_langchain(query)

    outputs = []
    for name in agent_names:
        agent = AGENT_REGISTRY.get(name)
        if agent:
            outputs.append(agent.handle_query(query=query))

    merged = "\n\n---\n\n".join(outputs)

    formatter = AGENT_REGISTRY["FormatterAgent"]
    final = formatter.handle_query(query=merged)

    if reasoning:
        final += f"\n\n_(Routed to {agent_names} because: {reasoning})_"

    return final


# ---------------------------------------------------------------------------
# AGENT SELECTION (LangChain)
# ---------------------------------------------------------------------------
def _choose_agent_via_langchain(query: str) -> tuple[list[str], str]:

    agent_list_text = "\n".join(
        f"- {a['name']}: {a['description']}" for a in AGENT_DESCRIPTIONS
    )

    system_prompt = f"""
You are AgriGPT Router.

Your job:
Analyze the farmer query and select ALL relevant agents.

Return ONLY JSON:
{{
  "agents": ["Agent1", "Agent2"],
  "reason": "Short explanation"
}}

Routing rules:
- Insects, larvae, pests, spots → PestAgent
- Fungus, white powder, brown patches → PestAgent
- Fertilizer, soil, nutrients → CropAgent
- Low yield, increase yield → YieldAgent
- Watering interval → IrrigationAgent
- Government schemes and subsidies → SubsidyAgent
- If unsure → include CropAgent
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    try:
        ai_message = router_llm.invoke(messages)
        raw = _extract_text(ai_message.content)
        data = json.loads(raw)

        agents = data.get("agents", ["CropAgent"])
        reason = data.get("reason", "")

    except Exception:
        agents = ["CropAgent"]
        reason = "Failed to parse routing output."

    agents = [a for a in agents if a in AGENT_REGISTRY]
    if not agents:
        agents = ["CropAgent"]

    return agents, reason.strip()


# ---------------------------------------------------------------------------
# OUTPUT NORMALIZATION
# ---------------------------------------------------------------------------
def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return str(first["text"]).strip()
        return str(first).strip()
    return str(content).strip()
