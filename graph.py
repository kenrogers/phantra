from langgraph.graph import StateGraph, END
from models import AgentState
from nodes import (
    prepare_rag_if_needed,
    analyze_voice,
    extract_insights,
    writer_agent
)


def create_graph():
    """Create the workflow graph"""
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("prepare_rag", prepare_rag_if_needed)
    workflow.add_node("analyze_voice", analyze_voice)
    workflow.add_node("extract_insights", extract_insights)
    workflow.add_node("writer", writer_agent)

    # Define edges
    workflow.add_edge("prepare_rag", "analyze_voice")
    workflow.add_edge("analyze_voice", "extract_insights")
    workflow.add_edge("extract_insights", "writer")
    workflow.add_edge("writer", END)

    # Set entry point
    workflow.set_entry_point("prepare_rag")

    # Compile the graph before returning
    return workflow.compile()
