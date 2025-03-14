from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import streamlit as st
from models import AgentState, FAST_MODEL
from utils import count_tokens
from prompts import (
    VOICE_ANALYSIS_PROMPT,
    VOICE_ANALYSIS_PHASE_TWO_PROMPT,
    INSIGHT_EXTRACTION_PROMPT,
    LINKEDIN_POST_WRITER_PROMPT
)


def prepare_rag_if_needed(state: AgentState) -> AgentState:
    """Set up RAG pipeline for transcript processing"""
    try:
        # Update status if available
        if state.get("status"):
            state["status"].update(
                label="2/6: Preparing knowledge base from transcript...")

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
    """Analyze voice characteristics from transcript in two phases"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="3/6: Analyzing speaker's voice and communication style...")

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    # Phase One: Detailed voice analysis
    phase_one_msg = HumanMessage(content=f"""
    Analyze this transcript and provide a detailed voice profile:
    {state["transcript"]}
    """)

    phase_one_response = llm.invoke([VOICE_ANALYSIS_PROMPT, phase_one_msg])
    state["voice_analysis_raw"] = phase_one_response.content

    # Phase Two: Create voice style guide
    phase_two_msg = HumanMessage(content=f"""
    Based on this detailed voice analysis, create a comprehensive style guide:

    {phase_one_response.content}
    """)

    phase_two_response = llm.invoke(
        [VOICE_ANALYSIS_PHASE_TWO_PROMPT, phase_two_msg])
    state["voice_analysis"] = phase_two_response.content

    return state


def extract_insights(state: AgentState) -> AgentState:
    """Extract 7 key insights from transcript"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="4/6: Extracting key insights from content...")

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    # Use RAG to get relevant chunks
    retriever = state["rag_info"]["retriever"]
    chunks = retriever.get_relevant_documents(state["transcript"])
    context = "\n".join([d.page_content for d in chunks])

    human_msg = HumanMessage(content=f"""
    TARGET AUDIENCE: People interested in {state["user_context"]["teaching_focus"]}
    BUSINESS CONTEXT: {state["user_context"]["business"]}

    TRANSCRIPT CONTEXT:
    {context}

    Extract exactly 7 insights from the transcript and format them according to the provided structure.
    After extracting the insights, convert them to a Python list of dictionaries with 'question' and 'answer' keys.
    Use only double quotes for strings.
    """)

    response = llm.invoke([INSIGHT_EXTRACTION_PROMPT, human_msg])

    try:
        # First try to parse the response directly if it's already in the right format
        if "[{" in response.content and "question" in response.content and "answer" in response.content:
            # Try to extract just the list part
            start_idx = response.content.find('[')
            end_idx = response.content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                list_content = response.content[start_idx:end_idx]
                insights = eval(list_content)
            else:
                raise ValueError("Could not find list in direct response")
        else:
            # Need to convert the formatted insights to the required dictionary format
            format_msg = HumanMessage(content=f"""
            Convert these insights into a valid Python list of dictionaries:

            {response.content}

            Format EXACTLY like this (with 7 items):
            [
                {{"question": "First insight summary", "answer": "Value of this insight to the audience"}},
                {{"question": "Second insight summary", "answer": "Value of this insight to the audience"}},
                # ... (7 items total)
            ]

            For each insight:
            - Use the "Summary" as the "question" field
            - Combine the "Value" and any other relevant information as the "answer" field
            - Include the Timestamp in the answer if available
            """)

            formatted = llm.invoke([
                SystemMessage(
                    content="You are a helpful assistant that converts formatted insights into a Python list of dictionaries."),
                format_msg
            ])

            # Clean the response to ensure it only contains the list
            cleaned_response = formatted.content.strip()
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

        # Validate the insights
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
                {{"question": "First insight summary", "answer": "Value of this insight to the audience"}},
                {{"question": "Second insight summary", "answer": "Value of this insight to the audience"}},
                # ... (7 items total)
            ]
            """)

            formatted = llm.invoke([
                SystemMessage(
                    content="You are a helpful assistant that converts formatted insights into a Python list of dictionaries."),
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


def writer_agent(state: AgentState) -> AgentState:
    """Writes LinkedIn posts directly from insights using the speaker's voice"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="5/6: Writing engaging LinkedIn posts in speaker's voice...")

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.8)

    # Get relevant examples from the transcript for each insight
    retriever = state["rag_info"]["retriever"]

    draft_posts = []
    post_strategies = []
    for insight in state["insights"]:
        # Get relevant transcript examples for this insight
        docs = retriever.get_relevant_documents(insight["question"])
        transcript_examples = "\n\n".join([d.page_content for d in docs[:2]])

        # Extract insight type from the insight data
        insight_type = ""
        if "CONCRETE EXAMPLE" in insight["answer"]:
            insight_type = "CONCRETE EXAMPLES"
        elif "UNIQUE PERSPECTIVE" in insight["answer"]:
            insight_type = "UNIQUE PERSPECTIVES"
        elif "PRACTICAL TAKEAWAY" in insight["answer"]:
            insight_type = "PRACTICAL TAKEAWAYS"
        elif "MEMORABLE QUOTE" in insight["answer"]:
            insight_type = "MEMORABLE QUOTES"
        elif "STATISTICAL INSIGHT" in insight["answer"]:
            insight_type = "STATISTICAL INSIGHTS"
        elif "PROBLEM-SOLUTION" in insight["answer"]:
            insight_type = "PROBLEM-SOLUTION PAIRS"
        elif "CONCEPTUAL FRAMEWORK" in insight["answer"]:
            insight_type = "CONCEPTUAL FRAMEWORKS"

        # Extract timestamp if available
        timestamp = ""
        if "[" in insight["answer"] and ":" in insight["answer"]:
            start = insight["answer"].find("[")
            end = insight["answer"].find("]", start)
            if start != -1 and end != -1:
                timestamp = insight["answer"][start:end+1]

        human_msg = HumanMessage(content=f"""
        INSIGHT:
        Summary: {insight["question"]}
        Value: {insight["answer"]}
        Type: {insight_type}
        Timestamp: {timestamp}

        VOICE STYLE GUIDE:
        {state["voice_analysis"]}

        RELEVANT TRANSCRIPT EXAMPLES:
        {transcript_examples}

        TARGET AUDIENCE:
        People interested in {state["user_context"]["teaching_focus"]}

        BUSINESS CONTEXT:
        {state["user_context"]["business"]}

        Create a high-performing LinkedIn post based on this insight that authentically matches the speaker's voice.
        Follow the structure guidelines:
        1. Start with an attention-grabbing headline
        2. Include 3-4 paragraph breaks between headline and opening
        3. Break text into easily skimmable paragraphs
        4. Include a specific call-to-action
        5. End with a thoughtful question related to the insight
        6. Do NOT use any hashtags

        Keep the post under 3,000 characters (aim for 1,300-2,000 for optimal engagement).

        Provide both:
        1. The complete LinkedIn post with proper spacing and formatting
        2. A brief explanation of your strategic approach
        """)

        response = llm.invoke([LINKEDIN_POST_WRITER_PROMPT, human_msg])
        full_response = response.content.strip()

        # Split the response into post content and strategy explanation
        # Look for indicators of the strategy explanation section
        strategy_indicators = [
            "Strategic approach:",
            "Strategy:",
            "My approach:",
            "Approach:",
            "Strategic rationale:"
        ]

        # Find where the strategy explanation begins
        strategy_start = -1
        for indicator in strategy_indicators:
            pos = full_response.find(indicator)
            if pos != -1:
                strategy_start = pos
                break

        # Extract post and strategy
        if strategy_start != -1:
            draft = full_response[:strategy_start].strip()
            strategy = full_response[strategy_start:].strip()
        else:
            # If no clear separator found, assume it's all post content
            draft = full_response
            strategy = "No explicit strategy provided."

        # Validate length and adjust if needed
        if len(draft) > 3000:
            # Find a good breaking point around 2800 characters
            cutoff = 2800
            last_period = draft[:cutoff].rfind('.')
            last_question = draft[:cutoff].rfind('?')
            last_break = max(last_period, last_question)

            if last_break > 0:
                draft = draft[:last_break+1] + \
                    "\n\nLearn more in the full video!"
            else:
                draft = draft[:2800] + "...\n\nLearn more in the full video!"

        draft_posts.append(draft)
        post_strategies.append(strategy)

    state["draft_posts"] = draft_posts
    state["post_strategies"] = post_strategies
    state["final_posts"] = draft_posts
    return state
