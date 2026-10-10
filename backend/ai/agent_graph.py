import logging
from typing import Literal
from langgraph.graph import StateGraph, START, END

from backend.ai.graph_state import AgentPipelineState
from backend.ai.agent_nodes import (
    transcribe_node,
    extract_node,
    valuation_match_node,
    response_node
)

logger = logging.getLogger(__name__)


# ==========================================
# CONDITIONAL EDGE ROUTER
# ==========================================
def should_continue_after_extraction(state: AgentPipelineState) -> Literal["valuation_match", "__end__"]:
    """
    Inspects state after Node 2 (extract_node).
    If extraction failed or normalized data is absent, diverts straight to END.
    """
    if state.get("error") or not state.get("normalized_lead"):
        print("[!] Router: Extraction failed or input invalid. Routing state directly to END.")
        return END

    print("[+] Router: Valid lead identified. Routing state forward to Valuation & Matchmaking.")
    return "valuation_match"


def build_pipeline_graph() -> StateGraph:
    """Assembles nodes and directional edges into a cohesive multi-agent state machine."""
    workflow = StateGraph(AgentPipelineState)

    # 1. Register Functional Worker Nodes
    workflow.add_node("transcribe", transcribe_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("valuation_match", valuation_match_node)
    workflow.add_node("response", response_node)

    # 2. Build Execution Edges
    # START -> Node 1 (transcribe) -> Node 2 (extract)
    workflow.add_edge(START, "transcribe")
    workflow.add_edge("transcribe", "extract")

    # Conditional Branching after Node 2:
    workflow.add_conditional_edges(
        "extract",
        should_continue_after_extraction,
        {
            "valuation_match": "valuation_match",
            END: END
        }
    )

    # Node 3 (valuation_match) -> Node 4 (response) -> END
    workflow.add_edge("valuation_match", "response")
    workflow.add_edge("response", END)

    return workflow


# Compile the graph into an executable state machine
agent_pipeline = build_pipeline_graph().compile()