# backend/agents/master_agent.py

"""
Master Agent Router (v4.1 - Stable LangChain 1.x)
-------------------------------------------------
Central orchestrator for routing text and image queries to multiple AgriGPT agents.
Updated to use a pure Runnable Sequence routing approach compatible with LangChain 1.x+.

Supports:
- Image-only
- Text-only
- Combined multimodal (text + image)
- Multi-agent routing via semantic LLM classification
"""

from __future__ import annotations

from typing import Optional, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from backend.core.langchain_tools import AGENT_DESCRIPTIONS, AGENT_REGISTRY
from backend.core.llm_client import get_llm
from backend.services.text_service import query_groq_text

router_llm = get_llm()


# ---------------------------------------------------------------------------
# ROUTING DATA STRUCTURES
# ---------------------------------------------------------------------------

class RoutingResult(BaseModel):
    """Structured output for the routing decision."""
    agents: List[str] = Field(
        description="List of agent names to route the query to. Must be from the available registry."
    )
    reason: str = Field(
        description="A brief explanation of the routing logic."
    )


def _format_agent_descriptions() -> str:
    """Formats agent descriptions for the prompt."""
    return "\n".join(
        f"- {a['name']}: {a['description']}" for a in AGENT_DESCRIPTIONS
    )


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
            # Skip PestAgent for text if we already ran it for image (avoid redundancy),
            # UNLESS the router specifically requested it for text reasons too.
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

    # Fallback if no agents ran (shouldn't happen due to default CropAgent)
    if not outputs:
        crop_agent = AGENT_REGISTRY["CropAgent"]
        outputs.append(crop_agent.handle_query(query=query))

    merged = "\n\n---\n\n".join(outputs)

    formatter = AGENT_REGISTRY["FormatterAgent"]
    final = formatter.handle_query(query=merged)

    if reasoning:
        final += f"\n\n_(Routed to {agent_names} because: {reasoning})_"

    return final


# ---------------------------------------------------------------------------
# AGENT SELECTION (LangChain 1.x+ Semantic Router)
# ---------------------------------------------------------------------------

def _build_router_chain() -> RunnableSequence:
    """
    Builds the semantic router chain using LangChain 1.x+ components.
    Returns a Runnable that takes {"query": "..."} and returns a parsed dict.
    """
    agent_descriptions = _format_agent_descriptions()

    template = """You are AgriGPT Router.
Analyze the farmer's query using SEMANTIC understanding.

Available agents:
{agent_descriptions}

INSTRUCTIONS:
1. Analyze the user query carefully.
2. Select ONE or MORE agents from the list above that can best answer the query.
   - "My leaves have white powder and pests" -> ["PestAgent", "CropAgent"]
   - "Why is my rice yield low?" -> ["YieldAgent"]
   - "Subsidies for irrigation" -> ["SubsidyAgent", "IrrigationAgent"]
3. If uncertain, default to ["CropAgent"].

Return JSON:
{{
   "agents": ["Agent1","Agent2"],
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

    # LCEL: Prompt -> LLM -> Parser
    return prompt | router_llm | parser


def _choose_agent_via_langchain(query: str) -> tuple[list[str], str]:
    """
    Uses a semantic router chain to classify the query and select agents.
    """
    try:
        chain = _build_router_chain()
        # invoke() returns a dict because of JsonOutputParser
        result_dict = chain.invoke({"query": query})
        
        # Safely access fields (result_dict is a dict, not pydantic object directly yet unless typed)
        # JsonOutputParser returns a dict matching the schema.
        
        # Validate against schema manually if needed or trust parser
        agents = result_dict.get("agents", [])
        reason = result_dict.get("reason", "No reason provided.")
        
        # Ensure agents is a list
        if isinstance(agents, str):
            agents = [agents]
            
        # Filter against registry
        final_agents = [
            a for a in agents
            if a in AGENT_REGISTRY
        ]

        if not final_agents:
            return ["CropAgent"], "Fallback: invalid agent output or no matching agents."

        return final_agents, reason

    except Exception as e:
        # Fallback logic
        return ["CropAgent"], f"Routing failed: {str(e)}"
