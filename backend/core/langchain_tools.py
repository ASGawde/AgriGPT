# backend/core/langchain_tools.py

"""
langchain_tools.py
------------------
Central registry for:
- All AgriGPT agent instances (singletons)
- Descriptions used by LangChain Router for multi-agent selection
"""

from __future__ import annotations
from typing import Dict, List

# Import agent classes
from backend.agents.crop_agent import CropAgent
from backend.agents.irrigation_agent import IrrigationAgent
from backend.agents.pest_agent import PestAgent
from backend.agents.subsidy_agent import SubsidyAgent
from backend.agents.yield_agent import YieldAgent
from backend.agents.formatter_agent import FormatterAgent


# ----------------------------------------------------------------------
# Create ONE shared instance per agent (singleton pattern)
# ----------------------------------------------------------------------
CROP_AGENT = CropAgent()
PEST_AGENT = PestAgent()
IRRIGATION_AGENT = IrrigationAgent()
SUBSIDY_AGENT = SubsidyAgent()
YIELD_AGENT = YieldAgent()
FORMATTER_AGENT = FormatterAgent()       # <-- Used only AFTER routing


# ----------------------------------------------------------------------
# Global registry — master lookup for all agents
# Used directly by master_agent router
# ----------------------------------------------------------------------
AGENT_REGISTRY: Dict[str, object] = {
    "CropAgent": CROP_AGENT,
    "PestAgent": PEST_AGENT,
    "IrrigationAgent": IRRIGATION_AGENT,
    "SubsidyAgent": SUBSIDY_AGENT,
    "YieldAgent": YIELD_AGENT,
    "FormatterAgent": FORMATTER_AGENT,    # <-- Still available for router finalization
}


# ----------------------------------------------------------------------
# Descriptions for LangChain routing (VERY IMPORTANT)
# FormatterAgent MUST NOT be here.
# ----------------------------------------------------------------------
AGENT_DESCRIPTIONS: List[Dict[str, str]] = [
    {
        "name": CROP_AGENT.name,
        "description": (
            "General crop management: fertilizer schedules, soil preparation, "
            "planting techniques, growth stages, and best farming practices."
        ),
    },
    {
        "name": PEST_AGENT.name,
        "description": (
            "Pest and disease detection: insects, fungi, leaf spots, larvae, "
            "and nutrient deficiency signals from text or images."
        ),
    },
    {
        "name": IRRIGATION_AGENT.name,
        "description": (
            "Water management: irrigation intervals, soil moisture issues, "
            "drip/sprinkler guidance, and water-saving methods."
        ),
    },
    {
        "name": YIELD_AGENT.name,
        "description": (
            "Yield optimization: diagnosing low productivity and suggesting "
            "practical ways to increase harvest output."
        ),
    },
    {
        "name": SUBSIDY_AGENT.name,
        "description": (
            "Government schemes: subsidies, loans, micro-irrigation programs, "
            "PM-Kisan, machinery subsidies, and eligibility rules."
        ),
    }
]

