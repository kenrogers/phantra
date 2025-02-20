import os
import uuid
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
import random

# LangChain imports
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages

# State definitions


class AgentState(TypedDict):
    transcript: str
    user_context: Dict[str, str]
    voice_analysis: Dict[str, Any]
    insights: List[Dict[str, Any]]
    templates: List[Dict[str, Any]]
    draft_posts: List[Dict[str, Any]]
    final_posts: List[str]
    messages: Annotated[List[BaseMessage], add_messages]
    revision_count: Optional[int]
    status: Optional[Any]  # For Streamlit status updates
    next_step: Optional[str]  # Track current workflow step

# Agent definitions


FAST_MODEL = "gpt-4o-mini"
SMART_MODEL = "gpt-4o"  # For future use

# Load environment variables from .env
load_dotenv()

# Later, after setting Tavily API key
# os.environ["LANGSMITH_TRACING"] = "true"
# Use your actual project name
# os.environ["LANGSMITH_PROJECT"] = "pr-candid-verve-69"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


def voice_analyzer_agent(state: AgentState) -> AgentState:
    """Analyzes voice and style from transcript"""
    if "status" in state:
        state["status"].update(label="Analyzing speaker's voice and style...")
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    system_msg = SystemMessage(content="""
    You are a voice analysis expert. Analyze transcripts and extract speaker characteristics.
    Focus on tone, vocabulary, patterns, and style.
    """)

    human_msg = HumanMessage(content=f"""
    Analyze this transcript and extract voice characteristics:

    {state["transcript"]}

    Focus on:
    1. Tone (formal, casual, technical, etc.)
    2. Vocabulary level and complexity
    3. Speaking patterns and quirks
    4. Storytelling style
    5. Persuasion techniques
    """)

    response = llm.invoke([system_msg, human_msg])
    state["voice_analysis"] = response.content
    state["messages"].append(response)
    return state


def insight_extractor_agent(state: AgentState) -> AgentState:
    """Extracts insights based on user context"""
    if "status" in state:
        state["status"].update(label="Extracting key insights from content...")
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    system_msg = SystemMessage(content="""
    You are an insight extraction expert. Extract exactly 7 valuable insights from content
    that would resonate with a LinkedIn audience. Format as a numbered list.
    """)

    human_msg = HumanMessage(content=f"""
    Extract 7 key insights from this transcript considering the user's context:

    Transcript: {state["transcript"]}

    User Context:
    Business: {state["user_context"]["business"]}
    Teaching Focus: {state["user_context"]["teaching_focus"]}
    Primary CTA: {state["user_context"]["cta"]}

    Format your response as a numbered list of 7 insights.
    Each insight should be a complete, standalone thought.
    """)

    response = llm.invoke([system_msg, human_msg])

    # Ensure we have a list of insights
    insights = response.content.split("\n")
    insights = [i.strip() for i in insights if i.strip()
                and any(c.isdigit() for c in i[:3])]

    # Clean up numbering
    insights = [i.split(".", 1)[1].strip() if "." in i[:3]
                else i for i in insights]

    # Ensure exactly 7 insights
    while len(insights) < 7:
        insights.append(
            f"Additional insight about {state['user_context']['business']}")
    insights = insights[:7]

    state["insights"] = insights
    state["messages"].append(response)
    return state


def tavily_template_agent(insight: str) -> dict:
    """
    Searches for LinkedIn templates and matches them with the insight.
    Returns a dict with template details.
    """
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    # First search for LinkedIn post templates
    try:
        search_results = tavily_search_agent(
            "high performing LinkedIn post templates content structure format examples"
        )

        if not search_results:
            return get_fallback_template()

        # Extract templates from search results
        templates = []
        for result in search_results:
            content = result.get("content", "")
            # Look for content that resembles templates
            if "linkedin" in content.lower() and any(marker in content.lower() for marker in
                                                     ["template", "format", "structure", "example"]):
                templates.append({
                    "content": content,
                    "score": result.get("score", 0)
                })

        if not templates:
            return get_fallback_template()

        # Have LLM analyze templates and insight to find best match
        system_msg = SystemMessage(content="""
        You are a LinkedIn content strategist. Analyze the given insight and templates
        to determine the best template match. Consider:
        1. The insight's complexity and tone
        2. The template's structure and purpose
        3. How well the insight can be adapted to the template
        4. Engagement potential of the combination
        """)

        templates_text = "\n---\n".join(t["content"] for t in templates)
        human_msg = HumanMessage(content=f"""
        Insight to post about:
        {insight}

        Available templates:
        {templates_text}

        Select the best template and format it as a dict with:
        - type: The template style (story, lesson, question-based, etc.)
        - template: The actual template structure
        - best_for: When this template works best
        """)

        response = llm.invoke([system_msg, human_msg])

        # Parse response into template dict
        try:
            # Extract dict-like content from response
            template_str = response.content
            if "type" not in template_str:
                return get_fallback_template()

            # Basic string to dict conversion (you might want to add more robust parsing)
            template_dict = eval(template_str)
            return template_dict
        except:
            return get_fallback_template()

    except Exception:
        return get_fallback_template()


def get_fallback_template() -> dict:
    """Provides a fallback template when search or processing fails"""
    return {
        "type": "insight-focused",
        "template": """
        [Hook Question/Statement]

        [Context/Setup]

        [Main Insight]

        [Supporting Point or Example]

        [Takeaway or Application]

        [Engagement Question or CTA]
        """.strip(),
        "best_for": "General insights that need clear structure"
    }


def tavily_search_agent(query: str):
    """
    Example function demonstrating how to use TavilySearchResults.
    This performs a search and returns results you can process further.
    """
    search_tool = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True
    )
    results = search_tool.invoke({"query": query})
    return results


def template_matcher_agent(state: AgentState) -> AgentState:
    """Matches each insight with a Tavily-provided LinkedIn post template."""
    if "status" in state:
        state["status"].update(
            label="Fetching templates for each insight via Tavily...")

    final_templates = []
    for insight in state["insights"]:
        found_template = tavily_template_agent(insight)
        final_templates.append(found_template)

    state["templates"] = final_templates
    return state


def supervisor_agent(state: AgentState) -> AgentState:
    """Supervises the entire workflow and decides next steps"""
    if "status" in state:
        state["status"].update(label="Supervisor evaluating next steps...")

    llm = ChatOpenAI(model=SMART_MODEL, temperature=0.7)

    # Check current state and determine next step
    if not state.get('voice_analysis'):
        state['next_step'] = 'voice_analyzer'
        return state

    if not state.get('insights'):
        state['next_step'] = 'insight_extractor'
        return state

    if not state.get('templates'):
        state['next_step'] = 'template_matcher'
        return state

    if not state.get('draft_posts'):
        state['next_step'] = 'writer'
        return state

    if state.get('revision_count', 0) < 3 and state.get('draft_posts'):
        # Evaluate post quality
        system_msg = SystemMessage(content="""
        You are a LinkedIn post quality evaluator.
        Assess if the posts need further revision based on:
        1. Alignment with voice analysis
        2. Use of insights
        3. Professional tone
        4. Engagement potential
        Return only 'revise' or 'complete'.
        """)

        # Sample first 3 posts
        posts_summary = "\n---\n".join(state['draft_posts'][:3])
        human_msg = HumanMessage(content=f"""
        Voice Analysis: {state['voice_analysis']}
        Sample Posts:
        {posts_summary}

        Do these posts need revision? Answer with one word: 'revise' or 'complete'
        """)

        response = llm.invoke([system_msg, human_msg])
        needs_revision = 'revise' in response.content.lower()

        if needs_revision:
            state['next_step'] = 'editor'
            return state

    # If we reach here, we're done
    state['next_step'] = 'end'
    return state


def writer_agent(state: AgentState) -> AgentState:
    """Writes LinkedIn posts using insights, templates, and voice analysis"""
    if "status" in state:
        state["status"].update(label="Writing initial LinkedIn posts...")
    llm = ChatOpenAI(model=SMART_MODEL, temperature=0.8)

    draft_posts = []

    # Process each insight-template pair
    for insight, template in zip(state["insights"], state["templates"]):
        system_msg = SystemMessage(content="""
        You are a LinkedIn post writer. Create engaging, value-focused content that naturally
        demonstrates expertise without overtly promoting. Focus on the insight's value to the reader.

        Guidelines:
        - Keep under 1300 characters
        - No hashtags or emojis
        - Lead with value, not credentials
        - Make insights feel like organic discoveries
        - Focus on the "what" and "why" before any promotion
        - Let expertise show through knowledge, not statements about the business

        For CTAs:
        - Make them feel like a natural next step for someone interested in learning more
        - Vary between soft and direct approaches
        - Tie them to the specific value or insight shared
        - Focus on the reader's journey and goals
        - Avoid salesy language
        """)

        human_msg = HumanMessage(content=f"""
        Write a LinkedIn post using this insight and template. Use the business context to inform
        the perspective and depth, but keep the focus on the insight itself.

        INSIGHT:
        {insight}

        TEMPLATE TYPE: {template["type"]}
        TEMPLATE STRUCTURE:
        {template["template"]}

        VOICE ANALYSIS:
        {state["voice_analysis"]}

        CONTEXT (for perspective, not direct reference):
        Target Audience: People interested in {state["user_context"]["teaching_focus"]}
        Core Value Offer: {state["user_context"]["cta"]}

        The post should feel like organic knowledge sharing rather than business promotion.
        Create a CTA that naturally extends from the insight shared.
        """)

        response = llm.invoke([system_msg, human_msg])
        draft = response.content.strip()

        # Validate post length
        if len(draft) > 1300:
            draft = draft[:1250].rsplit('\n', 1)[0]

            cta_msg = HumanMessage(content=f"""
            Create a natural, non-promotional CTA (under 50 characters) that:
            1. Extends from this insight: {insight}
            2. Guides toward: {state["user_context"]["cta"]}
            Focus on the reader's learning journey.
            """)
            cta_response = llm.invoke(
                [SystemMessage(content="You write organic, value-focused CTAs."), cta_msg])
            draft += "\n\n" + cta_response.content.strip()

        draft_posts.append(draft)

    # Ensure exactly 7 posts
    while len(draft_posts) < 7:
        if len(state["insights"]) > len(draft_posts):
            insight = state["insights"][len(draft_posts)]
            # Create a more natural fallback post
            cta_msg = HumanMessage(content=f"""
            Create a brief, natural CTA that connects to learning about {state["user_context"]["teaching_focus"]}
            """)
            cta_response = llm.invoke(
                [SystemMessage(content="You write organic, value-focused CTAs."), cta_msg])
            draft_posts.append(
                f"A fascinating insight I wanted to share:\n\n{insight}\n\n{cta_response.content.strip()}")
        else:
            # Create a value-focused fallback post
            cta_msg = HumanMessage(content=f"""
            Create a brief, natural CTA about learning {state["user_context"]["teaching_focus"]}
            """)
            cta_response = llm.invoke(
                [SystemMessage(content="You write organic, value-focused CTAs."), cta_msg])
            draft_posts.append(
                f"An important insight about {state['user_context']['teaching_focus']}:\n\n" +
                f"Understanding this deeply changes how we approach problems.\n\n{cta_response.content.strip()}")

    state["draft_posts"] = draft_posts[:7]
    state["final_posts"] = draft_posts[:7]
    state["messages"].append(response)
    return state


def editor_agent(state: AgentState) -> dict:
    """Evaluates and refines posts"""
    if "status" in state:
        state["status"].update(
            label=f"Editing and refining posts (revision {state.get('revision_count', 0) + 1}/3)...")

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)
    state["revision_count"] = state.get("revision_count", 0) + 1

    # Validate input
    if not state["draft_posts"]:
        state["final_posts"] = ["Error: No draft posts to edit"]
        return {"state": state, "next": END}

    # Check revision limit
    if state["revision_count"] >= 3:
        state["final_posts"] = [post for post in state["draft_posts"] if post]
        return {"state": state, "next": END}

    system_msg = SystemMessage(content="""
    You are a LinkedIn post editor. Improve this post for clarity and impact.
    Keep it under 1300 characters and maintain a professional tone.
    Return only the improved post.
    """)

    improved_posts = []
    for draft in state["draft_posts"]:
        if not draft.strip():
            continue

        try:
            human_msg = HumanMessage(content=f"Improve this post:\n\n{draft}")
            response = llm.invoke([system_msg, human_msg])
            improved = response.content.strip()

            # Validate improved post
            if improved and len(improved) <= 1300:
                improved_posts.append(improved)
            else:
                # Keep original if validation fails
                improved_posts.append(draft)
        except Exception:
            improved_posts.append(draft)  # Keep original on error

    # Ensure we have posts
    if not improved_posts:
        state["final_posts"] = state["draft_posts"]
        return {"state": state, "next": END}

    state["draft_posts"] = improved_posts
    return {"state": state, "next": "editor"}

# Graph definition


def create_graph() -> StateGraph:
    """Creates the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("voice_analyzer", voice_analyzer_agent)
    workflow.add_node("insight_extractor", insight_extractor_agent)
    workflow.add_node("template_matcher", template_matcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("editor", editor_agent)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Define conditional edges based on supervisor's decision
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {
            "voice_analyzer": "voice_analyzer",
            "insight_extractor": "insight_extractor",
            "template_matcher": "template_matcher",
            "writer": "writer",
            "editor": "editor",
            "end": END
        }
    )

    # Add edges back to supervisor
    workflow.add_edge("voice_analyzer", "supervisor")
    workflow.add_edge("insight_extractor", "supervisor")
    workflow.add_edge("template_matcher", "supervisor")
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("editor", "supervisor")

    return workflow.compile()


# Create a simple app
st.title("Terrapin")
st.write("Turn your YouTube videos into engaging LinkedIn content. Automagically.")

# Sidebar for OpenAI API key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Sidebar for Tavily API key
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")
if not tavily_api_key:
    st.warning("Please enter your Tavily API key to proceed")
    st.stop()
else:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# Replace with this after the other API key checks
if "langsmith" not in st.secrets:
    st.warning("LangSmith configuration not found in secrets")
    st.stop()
else:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = st.secrets.langsmith.project
    os.environ["LANGSMITH_API_KEY"] = st.secrets.langsmith.api_key
    os.environ["LANGSMITH_ENDPOINT"] = st.secrets.langsmith.endpoint

# User context inputs
if 'context_saved' not in st.session_state:
    st.session_state.context_saved = False

with st.form("user_context"):
    business = st.text_input(
        "What's your business about?",
        value="I run an educational company called AI Makerspace where we teach people how to become AI engineers"
    )
    teaching_focus = st.text_input(
        "What do you teach or share knowledge about?",
        value="AI Engineering concepts for developers, so we teach a lot about how AI works, do livestreams on new developments in AI, all with the goal of helping developer deeply understand how to buikd with AI"
    )
    cta = st.text_input(
        "What's your primary call-to-action?",
        value="To get people to sign up for our LLM Foundations email course at https://aimakerspace.io/self-paced-learning/llm-foundations/"
    )
    submitted = st.form_submit_button("Save Context")
    if submitted:
        st.session_state.context_saved = True
        # Store the values in session state
        st.session_state.business = business
        st.session_state.teaching_focus = teaching_focus
        st.session_state.cta = cta

# Show success message after form submission
if st.session_state.context_saved:
    st.success("Context saved successfully!")
    youtube_url = st.text_input(
        "Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
    # Add validation for youtube_url
    has_valid_url = youtube_url and (
        "youtube.com/watch?v=" in youtube_url or "youtu.be/" in youtube_url)
else:
    youtube_url = None
    has_valid_url = False

# Helper functions


def get_youtube_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if "youtube.com/watch?v=" in url:
        return url.split("youtube.com/watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def fetch_transcript(url: str) -> str:
    """Fetch transcript from YouTube video"""
    video_id = get_youtube_id(url)
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = " ".join([entry["text"] for entry in transcript_list])
    return transcript


def prepare_rag_if_needed(transcript: str) -> Dict[str, Any]:
    """Set up RAG pipeline if transcript is long enough"""
    token_count = count_tokens(transcript)

    # If transcript is short, don't use RAG
    if token_count < 4000:
        return {
            "use_rag": False,
            "message": f"Transcript is {token_count} tokens. Using direct processing."
        }

    # Otherwise set up RAG pipeline
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=count_tokens
        )
        chunks = text_splitter.split_text(transcript)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Qdrant.from_texts(
            texts=chunks,
            embedding=embeddings,
            location=":memory:"
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        return {
            "use_rag": True,
            "retriever": retriever,
            "chunk_count": len(chunks),
            "token_count": token_count,
            "message": f"Transcript is {token_count} tokens. Using RAG with {len(chunks)} chunks."
        }
    except Exception as e:
        return {
            "use_rag": False,
            "message": f"Error setting up RAG: {str(e)}. Falling back to direct processing."
        }


def extract_insights(transcript: str,
                     llm: ChatOpenAI,
                     rag_info: Dict[str, Any]) -> List[str]:
    """Extract key insights from transcript"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    # Use RAG for longer transcripts
    if rag_info.get("use_rag", False):
        retriever = rag_info.get("retriever")
        insights = []

        # Topics to extract insights about
        topics = [
            "main thesis or argument",
            "supporting evidence or data",
            "practical applications",
            "future implications",
            "industry trends",
            "challenges or obstacles",
            "success strategies"
        ]

        # RAG prompt template
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an insights extractor analyzing a YouTube video transcript.
            Use these retrieved transcript sections:

            {context}

            Extract a clear, valuable insight about: {query}

            Make the insight useful for a LinkedIn audience. Be specific and substantive.
            """
        )

        # Extract each insight using RAG
        for topic in topics:
            documents = retriever.get_relevant_documents(topic)
            context = "\n\n".join([doc.page_content for doc in documents])

            messages = rag_prompt.format_messages(
                context=context,
                query=topic
            )

            response = llm(messages)
            insights.append(response.content)

    # Direct processing for shorter transcripts
    else:
        direct_prompt = ChatPromptTemplate.from_template(
            """Extract 7 key insights from this YouTube video transcript:

            {transcript}

            Each insight should:
            1. Be specific and substantial
            2. Provide valuable information
            3. Be suitable for a LinkedIn post
            4. Cover different aspects of the content

            Format as a numbered list of insights.
            """
        )

        messages = direct_prompt.format_messages(transcript=transcript)
        response = llm(messages)

        # Split the response into individual insights
        raw_insights = response.content.split("\n")
        insights = [line for line in raw_insights if line.strip() and any(
            c.isdigit() for c in line[:3])]

        # Clean up numbering and formatting
        for i in range(len(insights)):
            if "." in insights[i][:4]:
                insights[i] = insights[i].split(".", 1)[1].strip()

    return insights[:7]  # Ensure we return at most 7 insights


def create_linkedin_posts(insights: List[str], llm: ChatOpenAI) -> List[str]:
    """Create LinkedIn posts from insights"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    posts = []

    post_prompt = ChatPromptTemplate.from_template(
        """Create an engaging LinkedIn post based on this insight:

        INSIGHT: {insight}

        The LinkedIn post should:
        1. Be under 1300 characters
        2. Not use emojis
        3. Have a compelling hook
        4. Not use hashtags
        5. End with a question or call to action
        6. Be professionally written

        Write only the post, no explanations.
        """
    )

    for insight in insights:
        messages = post_prompt.format_messages(insight=insight)
        response = llm(messages)
        posts.append(response.content)

    return posts


def edit_and_optimize_posts(posts: List[str], llm: ChatOpenAI) -> List[str]:
    """Edit and optimize LinkedIn posts"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    optimized_posts = []

    optimization_prompt = ChatPromptTemplate.from_template(
        """Optimize this LinkedIn post for maximum engagement:

        {post}

        Improve:
        1. Grammar and clarity
        2. Engagement potential
        3. Professional tone
        4. No emojis
        5. No hashtags
        6. Keep under 1300 characters

        Return only the optimized post.
        """
    )

    for post in posts:
        messages = optimization_prompt.format_messages(post=post)
        response = llm(messages)
        optimized_posts.append(response.content)

    return optimized_posts

# Main process function


def generate_linkedin_posts(url: str, user_context: Dict[str, str]) -> List[str]:
    """Generate LinkedIn posts from YouTube video"""
    with st.status("Generating LinkedIn posts...") as status:
        try:
            status.update(label="Fetching YouTube transcript...")
            transcript = fetch_transcript(url)

            # Initialize state with status object
            initial_state = AgentState(
                transcript=transcript,
                user_context=user_context,
                voice_analysis={},
                insights=[],
                templates=[],
                draft_posts=[],
                final_posts=[],
                messages=[],
                revision_count=0,
                status=status
            )

            workflow = create_graph()
            final_state = workflow.invoke(initial_state)

            return final_state["final_posts"]

        except Exception as e:
            status.error(f"An error occurred: {str(e)}")
            raise e


# Update the main app logic
if st.button("Generate LinkedIn Posts", disabled=not (has_valid_url and st.session_state.context_saved)):
    try:
        user_context = {
            "business": business,
            "teaching_focus": teaching_focus,
            "cta": cta
        }

        posts = generate_linkedin_posts(youtube_url, user_context)
        st.success(f"Generated {len(posts)} LinkedIn posts!")

        # Display posts in expandable sections
        for i, post in enumerate(posts, 1):
            with st.expander(f"LinkedIn Post #{i}"):
                st.markdown(post)
                st.download_button(
                    label=f"Download Post #{i}",
                    data=post,
                    file_name=f"linkedin_post_{i}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Show app info in sidebar
st.sidebar.markdown("## About")
st.sidebar.markdown("""
This app converts YouTube content into LinkedIn posts by:
1. Analyzing the speaker's voice and style
2. Extracting key insights based on your business context
3. Matching insights with optimal post templates
4. Writing engaging posts in the speaker's voice
5. Editing and refining for maximum impact
""")
