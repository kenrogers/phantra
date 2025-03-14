from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# State definitions


class AgentState(TypedDict):
    transcript: str
    user_context: Dict[str, str]
    voice_analysis: str
    insights: List[Dict[str, Any]]
    draft_posts: List[str]
    final_posts: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    revision_count: Optional[int]
    status: Optional[Any]  # For Streamlit status updates
    next_step: Optional[str]  # Track current workflow step
    rag_info: Dict[str, Any]
    debug_info: str


# Model definitions
FAST_MODEL = "gpt-4o-mini"
SMART_MODEL = "gpt-4o"

# Configuration constants
MAX_EDITING_ITERATIONS = 1  # Set to 1 for now, can be easily changed in the future
