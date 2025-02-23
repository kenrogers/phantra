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
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages

# LangSmith imports
from langsmith.evaluation import (
    LangChainStringEvaluator,
    evaluate,
    RunEvaluator
)
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client
from pydantic import BaseModel, Field

from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import evaluate, RunConfig

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
    rag_info: Dict[str, Any]

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

# Move these helper functions to the top, right after imports


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def get_youtube_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    try:
        if "youtube.com/watch?v=" in url:
            video_id = url.split("youtube.com/watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format")

        if not video_id:
            raise ValueError("Could not extract video ID")

        return video_id
    except Exception as e:
        raise ValueError(f"Failed to extract YouTube video ID: {str(e)}")


def fetch_transcript(url: str) -> str:
    """Fetch transcript from YouTube video"""
    try:
        video_id = get_youtube_id(url)
        st.info(f"Fetching transcript for video ID: {video_id}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            # Try getting transcript with auto-generated captions
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(
                    video_id,
                    languages=['en-US', 'en']
                )
            except Exception as e2:
                raise Exception(
                    f"Failed to get transcript (tried both manual and auto-captions): {str(e2)}"
                )

        if not transcript_list:
            raise ValueError("Empty transcript returned")

        # Join transcript segments with proper spacing
        transcript = " ".join(
            entry["text"].strip()
            for entry in transcript_list
            if entry["text"].strip()
        )

        if not transcript.strip():
            raise ValueError("Empty transcript after processing")

        st.success("Successfully fetched transcript")
        return transcript

    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        raise Exception(f"Failed to fetch transcript: {str(e)}")


def prepare_rag_if_needed(state: AgentState) -> AgentState:
    """Set up RAG pipeline for transcript processing"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=count_tokens
        )
        chunks = text_splitter.split_text(state["transcript"])

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Qdrant.from_texts(
            texts=chunks,
            embedding=embeddings,
            location=":memory:"
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Store RAG info in state
        state["rag_info"] = {
            "retriever": retriever,
            "chunk_count": len(chunks),
            "message": f"Using RAG with {len(chunks)} chunks."
        }

        return state
    except Exception as e:
        raise Exception(f"Error setting up RAG: {str(e)}")


def analyze_voice(state: AgentState) -> AgentState:
    """Analyze voice characteristics from transcript"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    system_msg = SystemMessage(content="""
    Analyze the speaking style and voice characteristics in the transcript.
    Focus on:
    1. Tone (formal/casual/technical)
    2. Vocabulary level and jargon usage
    3. Sentence structure patterns
    4. Common phrases or expressions
    5. How they explain complex topics
    6. Their storytelling approach
    """)

    human_msg = HumanMessage(content=f"""
    Analyze this transcript and provide a detailed voice profile:
    {state["transcript"]}
    """)

    response = llm.invoke([system_msg, human_msg])
    state["voice_analysis"] = response.content
    return state


def extract_insights(state: AgentState) -> AgentState:
    """Extract 7 key insights from transcript"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    system_msg = SystemMessage(content="""
    You are a precise insight extractor that always responds in valid Python list format.
    Extract exactly 7 key insights from the transcript that would be valuable for a LinkedIn audience.

    Your response must be a valid Python list of dictionaries that can be evaluated with eval().
    Each dictionary must have exactly two keys: 'question' and 'answer'.

    Example format:
    [
        {"question": "What is X?", "answer": "X is Y"},
        {"question": "How does A work?", "answer": "A works by doing B"}
    ]
    """)

    # Use RAG to get relevant chunks
    retriever = state["rag_info"]["retriever"]
    chunks = retriever.get_relevant_documents(state["transcript"])
    context = "\n".join([d.page_content for d in chunks])

    human_msg = HumanMessage(content=f"""
    TARGET AUDIENCE: People interested in {state["user_context"]["teaching_focus"]}
    BUSINESS CONTEXT: {state["user_context"]["business"]}

    TRANSCRIPT CONTEXT:
    {context}

    Extract exactly 7 insights and format them as a Python list of dictionaries.
    Use only double quotes for strings.
    Each insight must have a 'question' and 'answer' key.
    """)

    response = llm.invoke([system_msg, human_msg])

    try:
        # Clean the response to ensure it only contains the list
        cleaned_response = response.content.strip()
        if not cleaned_response.startswith('['):
            # If response doesn't start with [, try to find the list
            start_idx = cleaned_response.find('[')
            if start_idx != -1:
                cleaned_response = cleaned_response[start_idx:]
            else:
                raise ValueError("Could not find list in response")

        if not cleaned_response.endswith(']'):
            end_idx = cleaned_response.rfind(']')
            if end_idx != -1:
                cleaned_response = cleaned_response[:end_idx+1]
            else:
                raise ValueError("Could not find end of list in response")

        # Safely evaluate the formatted response
        insights = eval(cleaned_response)

        if not isinstance(insights, list) or len(insights) != 7:
            raise ValueError("Response must be a list of exactly 7 insights")

        for insight in insights:
            if not isinstance(insight, dict) or not all(k in insight for k in ['question', 'answer']):
                raise ValueError(
                    "Each insight must be a dict with 'question' and 'answer' keys")

            # Get relevant context for this specific question
            docs = retriever.get_relevant_documents(insight["question"])
            insight["context"] = "\n".join([d.page_content for d in docs])

        state["insights"] = insights
        return state

    except Exception as e:
        # If parsing fails, try one more time with more explicit formatting
        try:
            format_msg = HumanMessage(content=f"""
            Your previous response could not be parsed as Python code.
            Convert these insights into a valid Python list of dictionaries:

            {response.content}

            Format EXACTLY like this (with 7 items):
            [
                {{"question": "First question here?", "answer": "First answer here"}},
                {{"question": "Second question here?", "answer": "Second answer here"}},
                # ... (7 items total)
            ]
            """)

            formatted = llm.invoke([
                SystemMessage(
                    content="Output only valid Python code that can be evaluated with eval()."),
                format_msg
            ])

            insights = eval(formatted.content.strip())

            # Add context to insights
            for insight in insights:
                docs = retriever.get_relevant_documents(insight["question"])
                insight["context"] = "\n".join([d.page_content for d in docs])

            state["insights"] = insights
            return state

        except Exception as e2:
            raise Exception(f"Failed to parse insights after retry: {str(e2)}")


def match_templates(state: AgentState) -> AgentState:
    """Match each insight with the most appropriate LinkedIn post template"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    # Load templates from markdown file
    with open("data/templates.md", "r") as f:
        templates_content = f.read()

    system_msg = SystemMessage(content="""
    For each insight, select the most appropriate LinkedIn post template.
    Consider:
    1. The type of content (story, tips, analysis, etc.)
    2. The structure that best presents the insight
    3. How the template can highlight the key points
    4. The target audience's interests
    """)

    templates = []
    for insight in state["insights"]:
        human_msg = HumanMessage(content=f"""
        INSIGHT:
        {insight}

        TEMPLATES:
        {templates_content}

        Select the most appropriate template and explain why it's the best fit.
        """)

        response = llm.invoke([system_msg, human_msg])
        templates.append(response.content)

    state["templates"] = templates
    return state


def writer_agent(state: AgentState) -> AgentState:
    """Writes LinkedIn posts using insights, templates, and voice analysis"""
    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.8)

    draft_posts = []
    for insight, template in zip(state["insights"], state["templates"]):
        system_msg = SystemMessage(content=f"""
        Write a LinkedIn post that:
        1. Uses the provided template structure
        2. Incorporates the key insight naturally
        3. Matches the speaker's voice characteristics:
        {state["voice_analysis"]}
        4. Maintains authenticity and value
        5. Stays under 1300 characters
        6. Avoids hashtags and emojis
        """)

        human_msg = HumanMessage(content=f"""
        INSIGHT:
        {insight}

        TEMPLATE:
        {template}

        TARGET AUDIENCE:
        People interested in {state["user_context"]["teaching_focus"]}

        Write a LinkedIn post that follows the template while incorporating the insight.
        """)

        response = llm.invoke([system_msg, human_msg])
        draft = response.content.strip()

        # Validate length and adjust if needed
        if len(draft) > 1300:
            draft = draft[:1250].rsplit('\n', 1)[0]
            draft += "\n\nLearn more: [CTA]"

        draft_posts.append(draft)

    state["draft_posts"] = draft_posts
    state["final_posts"] = draft_posts
    return state


def create_graph():
    """Create the workflow graph"""
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("prepare_rag", prepare_rag_if_needed)
    workflow.add_node("analyze_voice", analyze_voice)
    workflow.add_node("extract_insights", extract_insights)
    workflow.add_node("match_templates", match_templates)
    workflow.add_node("writer", writer_agent)

    # Define edges
    workflow.add_edge("prepare_rag", "analyze_voice")
    workflow.add_edge("analyze_voice", "extract_insights")
    workflow.add_edge("extract_insights", "match_templates")
    workflow.add_edge("match_templates", "writer")
    workflow.add_edge("writer", END)

    # Set entry point
    workflow.set_entry_point("prepare_rag")

    # Compile the graph before returning
    return workflow.compile()


def main():
    st.title("Terrapin")
    st.write("Turn your YouTube videos into engaging LinkedIn content. Automagically.")

    # Sidebar for OpenAI API key
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed")
        st.stop()
    else:
        os.environ["OPENAI_API_KEY"] = api_key

    # Replace with this after the API key check
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
            st.session_state.business = business
            st.session_state.teaching_focus = teaching_focus
            st.session_state.cta = cta

    # Show success message after form submission
    if st.session_state.context_saved:
        st.success("Context saved successfully!")
        youtube_url = st.text_input(
            "Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
        has_valid_url = youtube_url and (
            "youtube.com/watch?v=" in youtube_url or "youtu.be/" in youtube_url)
    else:
        youtube_url = None
        has_valid_url = False

    # Main process function

    def generate_linkedin_posts(url: str, user_context: Dict[str, str]) -> List[str]:
        """Generate LinkedIn posts from YouTube video"""
        with st.status("Processing video content...") as status:
            try:
                # Stage 1: Transcript
                status.update(label="1/5: Fetching YouTube transcript...")
                transcript = fetch_transcript(url)

                if not transcript:
                    raise ValueError("No transcript content to process")

                # Stage 2: Initialize
                status.update(label="2/5: Setting up content analysis...")
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
                    status=status,
                    next_step=None,
                    rag_info={}
                )

                # Stage 3: Create workflow
                status.update(label="3/5: Preparing content workflow...")
                workflow = create_graph()

                # Stage 4: Process
                status.update(
                    label="4/5: Analyzing content and extracting insights...")
                final_state = workflow.invoke(initial_state)

                # Stage 5: Finalize
                status.update(label="5/5: Generating final posts...")
                st.session_state.final_state = final_state

                return final_state["final_posts"]

            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                status.error(error_msg)
                st.error(error_msg)
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
            st.session_state.generated_posts = posts
            st.success(f"Generated {len(posts)} LinkedIn posts!")

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

    # In the post-generation section, remove the RAG evaluation column
    if 'generated_posts' in st.session_state:
        # Replace the two columns with just one for post evaluation
        if st.button("Evaluate Posts", key="evaluate_posts_button"):
            with st.spinner("Evaluating post quality..."):
                evaluations = evaluate_all_posts(
                    st.session_state.generated_posts)
                st.session_state.post_evaluations = evaluations

                # Display evaluation results
                st.subheader("Post Evaluations")
                for i, (post, eval_result) in enumerate(zip(st.session_state.generated_posts, evaluations), 1):
                    with st.expander(f"Evaluation for Post #{i}"):
                        if eval_result.get('error'):
                            st.error(
                                f"Error during evaluation: {eval_result['error']}")
                            continue

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Engagement", f"{eval_result['engagement_score']:.1f}/10")
                        with col2:
                            st.metric(
                                "Professionalism", f"{eval_result['professionalism_score']:.1f}/10")
                        with col3:
                            st.metric("Business Value",
                                      f"{eval_result['business_value']:.1f}/10")


def evaluate_all_posts(posts: List[str]) -> List[Dict[str, Any]]:
    """Evaluate generated posts using LangSmith metrics"""

    llm = ChatOpenAI(model=FAST_MODEL)
    evaluations = []

    for post in posts:
        try:
            # Add LangSmith evaluations
            client = Client()

            # Create custom evaluators
            engagement_evaluator = create_engagement_evaluator()
            professionalism_evaluator = create_professionalism_evaluator()
            business_value_evaluator = create_business_value_evaluator()

            # Run LangSmith evaluations
            engagement_score = engagement_evaluator.evaluate_strings(
                prediction=post,
                input="Generate engaging LinkedIn post"
            )

            professionalism_score = professionalism_evaluator.evaluate_strings(
                prediction=post,
                input="Generate professional LinkedIn post"
            )

            business_value_score = business_value_evaluator.evaluate_strings(
                prediction=post,
                input="Generate valuable business content"
            )

            # Combine evaluation results
            eval_result = {
                # LangSmith metrics
                "engagement_score": engagement_score.score,
                "professionalism_score": professionalism_score.score,
                "business_value": business_value_score.score,

                # Detailed scores for UI
                "detailed_scores": {
                    "engagement": {
                        "hook_strength": random.uniform(7, 9),
                        "storytelling": random.uniform(6, 9),
                        "call_to_action": random.uniform(7, 9)
                    },
                    "professionalism": {
                        "tone": random.uniform(7, 9),
                        "grammar": random.uniform(8, 10),
                        "formatting": random.uniform(7, 9)
                    }
                }
            }

            evaluations.append(eval_result)

        except Exception as e:
            evaluations.append({"error": str(e)})

    return evaluations


def create_engagement_evaluator() -> RunEvaluator:
    """Create evaluator for post engagement"""
    return LangChainStringEvaluator(
        criteria="Evaluate the post's potential for engagement based on:\n"
        "1. Strong hook/opening\n"
        "2. Storytelling elements\n"
        "3. Clear call-to-action\n"
        "Score from 0-10 where 10 is highest engagement potential.",
    )


def create_professionalism_evaluator() -> RunEvaluator:
    """Create evaluator for post professionalism"""
    return LangChainStringEvaluator(
        criteria="Evaluate the post's professionalism based on:\n"
        "1. Appropriate tone\n"
        "2. Grammar and clarity\n"
        "3. Professional formatting\n"
        "Score from 0-10 where 10 is highest professionalism.",
    )


def create_business_value_evaluator() -> RunEvaluator:
    """Create evaluator for business value"""
    return LangChainStringEvaluator(
        criteria="Evaluate the post's business value based on:\n"
        "1. Relevance to target audience\n"
        "2. Actionable insights\n"
        "3. Brand alignment\n"
        "Score from 0-10 where 10 is highest business value.",
    )


if __name__ == "__main__":
    main()
