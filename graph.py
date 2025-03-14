from langgraph.graph import StateGraph, END
from models import AgentState
from nodes import (
    prepare_rag_if_needed,
    analyze_voice,
    extract_insights,
    writer_agent,
    editor_agent
)


def create_graph():
    """Create the workflow graph"""
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("prepare_rag", prepare_rag_if_needed)
    workflow.add_node("analyze_voice", analyze_voice)
    workflow.add_node("extract_insights", extract_insights)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("editor", editor_agent)

    # Define edges
    workflow.add_edge("prepare_rag", "analyze_voice")
    workflow.add_edge("analyze_voice", "extract_insights")
    workflow.add_edge("extract_insights", "writer")
    workflow.add_edge("writer", "editor")

    # Conditional routing based on editor feedback
    def should_continue_editing(state: AgentState) -> str:
        """Determine if we should continue editing or finish"""
        # Check if we have posts to edit
        if ("final_posts" not in state or not state["final_posts"]) and ("draft_posts" not in state or not state["draft_posts"]):
            # Add debug info
            if "debug_info" not in state:
                state["debug_info"] = ""
            state["debug_info"] += "No posts found, ending workflow. "
            return END

        # If we have draft_posts but no final_posts, copy them over
        if ("final_posts" not in state or not state["final_posts"]) and "draft_posts" in state and state["draft_posts"]:
            state["final_posts"] = state["draft_posts"].copy()
            if "debug_info" not in state:
                state["debug_info"] = ""
            state["debug_info"] += "Copied draft_posts to final_posts. "

        # Check if we should continue editing
        if state.get("continue_editing", False):
            return "writer"
        else:
            return END

    # Add conditional edge from editor
    workflow.add_conditional_edges(
        "editor",
        should_continue_editing,
        {
            "writer": "writer",  # Continue editing
            END: END             # Finish workflow
        }
    )

    # Set entry point
    workflow.set_entry_point("prepare_rag")

    # Compile the graph before returning
    return workflow.compile()
