from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import streamlit as st
from models import AgentState, FAST_MODEL, SMART_MODEL, MAX_EDITING_ITERATIONS
from utils import count_tokens
from prompts import (
    VOICE_ANALYSIS_PROMPT,
    VOICE_ANALYSIS_PHASE_TWO_PROMPT,
    INSIGHT_EXTRACTION_PROMPT,
    LINKEDIN_POST_WRITER_PROMPT,
    CONTENT_EDITOR_PROMPT
)


def prepare_rag_if_needed(state: AgentState) -> AgentState:
    """Set up RAG pipeline for transcript processing"""
    try:
        # Update status if available
        if state.get("status"):
            state["status"].update(
                label="2/6: Preparing knowledge base from transcript...")

        # Initialize debug info if not present
        if "debug_info" not in state:
            state["debug_info"] = ""

        state["debug_info"] += "DEBUG: Starting RAG preparation. "

        # Debug: Check if transcript exists
        if "transcript" not in state or not state["transcript"]:
            state["debug_info"] = "ERROR: No transcript found in state"
            return state

        state["debug_info"] += f"DEBUG: Transcript found, length: {len(state['transcript'])} characters. "

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=count_tokens
        )
        chunks = text_splitter.split_text(state["transcript"])

        state["debug_info"] += f"DEBUG: Split transcript into {len(chunks)} chunks. "

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = Qdrant.from_texts(
            texts=chunks,
            embedding=embeddings,
            location=":memory:"
        )

        state["debug_info"] += "DEBUG: Created vector store. "

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        state["debug_info"] += "DEBUG: Created retriever. "

        # Store RAG info in state
        state["rag_info"] = {
            "retriever": retriever,
            "chunk_count": len(chunks),
            "message": f"Using RAG with {len(chunks)} chunks."
        }

        state["debug_info"] += f"DEBUG: Stored RAG info in state with {len(chunks)} chunks. "

        # Debug: Add success message
        if "debug_info" not in state:
            state["debug_info"] = ""
        state["debug_info"] += "RAG setup successful. "

        return state
    except Exception as e:
        state["debug_info"] = f"ERROR in RAG setup: {str(e)}"
        raise Exception(f"Error setting up RAG: {str(e)}")


def analyze_voice(state: AgentState) -> AgentState:
    """Analyze voice characteristics from transcript in two phases"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="3/6: Analyzing speaker's voice and communication style...")

    # Debug: Check if transcript exists
    if "transcript" not in state or not state["transcript"]:
        state["debug_info"] = "ERROR: No transcript found for voice analysis"
        return state

    # Initialize debug info if not present
    if "debug_info" not in state:
        state["debug_info"] = ""

    state["debug_info"] += "DEBUG: Starting voice analysis. "
    state["debug_info"] += f"DEBUG: Transcript length: {len(state['transcript'])} characters. "

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    try:
        # Phase One: Detailed voice analysis
        phase_one_msg = HumanMessage(content=f"""
        Analyze this transcript and provide a detailed voice profile:
        {state["transcript"]}
        """)

        state["debug_info"] += "DEBUG: Sending transcript for voice analysis phase one. "
        phase_one_response = llm.invoke([VOICE_ANALYSIS_PROMPT, phase_one_msg])
        state["voice_analysis_raw"] = phase_one_response.content
        state["debug_info"] += "DEBUG: Completed voice analysis phase one. "

        # Phase Two: Create voice style guide
        phase_two_msg = HumanMessage(content=f"""
        Based on this detailed voice analysis, create a comprehensive style guide:

        {phase_one_response.content}
        """)

        state["debug_info"] += "DEBUG: Starting voice analysis phase two. "
        phase_two_response = llm.invoke(
            [VOICE_ANALYSIS_PHASE_TWO_PROMPT, phase_two_msg])
        state["voice_analysis"] = phase_two_response.content
        state["debug_info"] += "DEBUG: Completed voice analysis phase two. "

        # Debug: Add success message
        if "debug_info" not in state:
            state["debug_info"] = ""
        state["debug_info"] += "Voice analysis successful. "

        return state
    except Exception as e:
        state["debug_info"] = f"ERROR in voice analysis: {str(e)}"
        raise Exception(f"Error in voice analysis: {str(e)}")


def extract_insights(state: AgentState) -> AgentState:
    """Extract 7 key insights from transcript"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="4/6: Extracting key insights from content...")

    # Debug: Check prerequisites
    if "rag_info" not in state or "retriever" not in state["rag_info"]:
        state["debug_info"] = "ERROR: RAG setup missing for insight extraction"
        return state

    if "user_context" not in state:
        state["debug_info"] = "ERROR: User context missing for insight extraction"
        return state

    # Initialize debug info if not present
    if "debug_info" not in state:
        state["debug_info"] = ""

    state["debug_info"] += "DEBUG: Starting insight extraction. "

    llm = ChatOpenAI(model=FAST_MODEL, temperature=0.7)

    try:
        # Use RAG to get relevant chunks
        retriever = state["rag_info"]["retriever"]
        chunks = retriever.get_relevant_documents(state["transcript"])
        context = "\n".join([d.page_content for d in chunks])

        state["debug_info"] += f"DEBUG: Retrieved {len(chunks)} chunks for insight extraction. "

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

        # Debug: Print the response
        state["debug_info"] += f"\nDEBUG: LLM Response for insights (first 200 chars):\n{response.content[:200]}...\n"

        # First try to parse the response directly if it's already in the right format
        if "[{" in response.content and "question" in response.content and "answer" in response.content:
            # Try to extract just the list part
            start_idx = response.content.find('[')
            end_idx = response.content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                list_content = response.content[start_idx:end_idx]
                insights = eval(list_content)
                state[
                    "debug_info"] += f"DEBUG: Successfully parsed insights directly. Found {len(insights)} insights.\n"
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

        # Debug: Print the insights
        state["debug_info"] += f"\nDEBUG: Final insights:\n"
        for i, insight in enumerate(insights):
            state["debug_info"] += f"  Insight {i+1}: {insight['question'][:50]}...\n"

        # Debug: Add success message
        if "debug_info" not in state:
            state["debug_info"] = ""
        state["debug_info"] += f"Extracted {len(insights)} insights successfully. "

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

            # Debug: Add success message after retry
            if "debug_info" not in state:
                state["debug_info"] = ""
            state["debug_info"] += f"Extracted {len(insights)} insights after retry. "

            return state

        except Exception as e2:
            state["debug_info"] = f"ERROR in insight extraction: {str(e2)}"
            raise Exception(f"Failed to parse insights after retry: {str(e2)}")


def writer_agent(state: AgentState) -> AgentState:
    """Writes LinkedIn posts directly from insights using the speaker's voice"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="5/6: Writing engaging LinkedIn posts in speaker's voice...")

    # Initialize debug info if not present
    if "debug_info" not in state:
        state["debug_info"] = ""

    # Initialize empty arrays for posts to prevent errors downstream
    if "draft_posts" not in state:
        state["draft_posts"] = []
    if "final_posts" not in state:
        state["final_posts"] = []

    # Debug: Check prerequisites
    if "insights" not in state or not state["insights"]:
        state["debug_info"] += "ERROR: No insights found for post writing. "
        return state

    if "voice_analysis" not in state:
        state["debug_info"] += "ERROR: No voice analysis found for post writing. "
        return state

    if "user_context" not in state:
        state["debug_info"] += "ERROR: No user context found for post writing. "
        return state

    if "rag_info" not in state or "retriever" not in state["rag_info"]:
        state["debug_info"] += "ERROR: RAG retriever not found for post writing. "
        return state

    # Add debug info about insights
    state["debug_info"] += f"DEBUG: Found {len(state['insights'])} insights. "

    # Print detailed insight information
    for i, insight in enumerate(state["insights"]):
        state["debug_info"] += f"\nDEBUG: Insight {i+1}:\n"
        state["debug_info"] += f"  Question: {insight.get('question', 'N/A')}\n"
        state["debug_info"] += f"  Answer: {insight.get('answer', 'N/A')[:100]}...\n"

    try:
        # Initialize LLM
        llm = ChatOpenAI(model=SMART_MODEL, temperature=0.8)

        # Get retriever from state
        retriever = state["rag_info"]["retriever"]

        # Initialize draft_posts if this is the first run
        if not state["draft_posts"]:
            state["debug_info"] += f"Starting to write {len(state['insights'])} posts. "

            draft_posts = []
            post_strategies = []

            # Debug: Log insights
            for i, insight in enumerate(state["insights"]):
                state["debug_info"] += f"DEBUG: Insight {i+1}: {insight['question'][:30]}... "

            # First time writing posts
            for i, insight in enumerate(state["insights"]):
                # Get relevant transcript examples for this insight
                docs = retriever.get_relevant_documents(insight["question"])
                transcript_examples = "\n\n".join(
                    [d.page_content for d in docs[:2]])

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
                        draft = draft[:2800] + \
                            "...\n\nLearn more in the full video!"

                draft_posts.append(draft)
                post_strategies.append(strategy)

                # Debug
                state["debug_info"] += f"Wrote post {i+1}. "

            # Store the posts in state
            state["draft_posts"] = draft_posts
            state["post_strategies"] = post_strategies
            # Make a copy to avoid reference issues
            state["final_posts"] = draft_posts.copy()

            # Debug
            state["debug_info"] += f"Completed writing {len(draft_posts)} posts. "

        # If we're revising based on editor feedback
        elif "editor_feedback" in state and state.get("continue_editing", False):
            state["debug_info"] += f"Revising {len(state['editor_feedback'])} posts based on feedback. "

            # Process each piece of feedback
            for feedback_item in state["editor_feedback"]:
                post_index = feedback_item["post_index"]
                feedback = feedback_item["feedback"]
                insight = state["insights"][post_index]

                # Get relevant transcript examples for this insight
                docs = retriever.get_relevant_documents(insight["question"])
                transcript_examples = "\n\n".join(
                    [d.page_content for d in docs[:2]])

                human_msg = HumanMessage(content=f"""
                ORIGINAL POST:
                {state["draft_posts"][post_index]}

                EDITOR FEEDBACK:
                {feedback}

                INSIGHT:
                Summary: {insight["question"]}
                Value: {insight["answer"]}

                VOICE STYLE GUIDE:
                {state["voice_analysis"]}

                RELEVANT TRANSCRIPT EXAMPLES:
                {transcript_examples}

                TARGET AUDIENCE:
                People interested in {state["user_context"]["teaching_focus"]}

                BUSINESS CONTEXT:
                {state["user_context"]["business"]}

                EDITING ITERATION: {state.get("editing_iterations", 1)} of {MAX_EDITING_ITERATIONS}

                Revise the LinkedIn post based on the editor's feedback while maintaining the speaker's authentic voice.
                Avoid clichés, buzzwords, and phrases like "game-changer", "delve", "imagine a world", etc.
                Keep the post under 3,000 characters (aim for 1,300-2,000 for optimal engagement).

                Provide only the revised post with proper spacing and formatting.
                """)

                response = llm.invoke([LINKEDIN_POST_WRITER_PROMPT, human_msg])
                revised_post = response.content.strip()

                # Update the draft post with the revised version
                state["draft_posts"][post_index] = revised_post
                # Also update final_posts
                state["final_posts"][post_index] = revised_post

                # Debug
                state["debug_info"] += f"Revised post {post_index+1}. "

        # Ensure final_posts exists and has content
        if not state["final_posts"] and state["draft_posts"]:
            state["final_posts"] = state["draft_posts"].copy()
            state["debug_info"] += "Created final_posts from draft_posts. "

        # Debug: Count posts
        state["debug_info"] += f"Final post count: {len(state['final_posts'])}. "

        return state

    except Exception as e:
        state["debug_info"] += f"ERROR in writer_agent: {str(e)}"
        return state


def editor_agent(state: AgentState) -> AgentState:
    """Reviews and edits LinkedIn posts to ensure they match the speaker's voice and avoid clichés"""
    # Update status if available
    if state.get("status"):
        state["status"].update(
            label="6/6: Editing posts to ensure authentic voice and quality...")

    # Debug info
    if "debug_info" not in state:
        state["debug_info"] = ""

    state["debug_info"] += "DEBUG: Starting editor agent. "

    # Ensure we have posts to edit
    if "draft_posts" not in state:
        state["draft_posts"] = []
    if "final_posts" not in state:
        state["final_posts"] = []

    # Check if we have any posts to edit
    if not state["draft_posts"] and not state["final_posts"]:
        # If no posts, set continue_editing to False and return
        state["continue_editing"] = False
        state["editing_note"] = "No posts found to edit."
        state["debug_info"] += "ERROR: No draft posts found to edit. "
        return state

    # If we have draft_posts but no final_posts, copy them over
    if not state["final_posts"] and state["draft_posts"]:
        state["final_posts"] = state["draft_posts"].copy()
        state["debug_info"] += "Copied draft_posts to final_posts. "

    state["debug_info"] += f"DEBUG: Found {len(state['draft_posts'])} draft posts to edit. "

    # Ensure we have insights
    if "insights" not in state or not state["insights"]:
        state["continue_editing"] = False
        state["editing_note"] = "No insights found for editing."
        state["debug_info"] += "ERROR: No insights found for editing. "
        return state

    state["debug_info"] += f"DEBUG: Found {len(state['insights'])} insights for editing. "

    # Ensure we have voice analysis
    if "voice_analysis" not in state:
        state["continue_editing"] = False
        state["editing_note"] = "No voice analysis found for editing."
        state["debug_info"] += "ERROR: No voice analysis found for editing. "
        return state

    state["debug_info"] += "DEBUG: Found voice analysis for editing. "

    # Debug
    state["debug_info"] += f"Editor starting to review {len(state['draft_posts'])} posts. "

    try:
        llm = ChatOpenAI(model=SMART_MODEL, temperature=0.3)

        # Initialize editing iteration counter if not present
        if "editing_iterations" not in state:
            state["editing_iterations"] = 0

        # Increment iteration counter
        state["editing_iterations"] += 1
        state["debug_info"] += f"Editing iteration {state['editing_iterations']} of {MAX_EDITING_ITERATIONS}. "

        # Track if any posts need revision
        posts_need_revision = False
        revision_feedback = []

        # Review each post
        for i, post in enumerate(state["draft_posts"]):
            # Skip if post is empty or None
            if not post:
                state["debug_info"] += f"Post {i+1} is empty, skipping. "
                continue

            # Skip if we don't have a corresponding insight
            if i >= len(state["insights"]):
                state["debug_info"] += f"No insight found for post {i+1}, skipping. "
                continue

            insight = state["insights"][i]

            human_msg = HumanMessage(content=f"""
            ORIGINAL POST:
            {post}

            SPEAKER'S VOICE STYLE GUIDE:
            {state["voice_analysis"]}

            INSIGHT THIS POST IS BASED ON:
            Summary: {insight["question"]}
            Value: {insight["answer"]}

            TARGET AUDIENCE:
            People interested in {state["user_context"]["teaching_focus"]}

            BUSINESS CONTEXT:
            {state["user_context"]["business"]}

            CURRENT EDITING ITERATION: {state["editing_iterations"]} of {MAX_EDITING_ITERATIONS}

            Review this LinkedIn post for quality and authenticity.
            """)

            response = llm.invoke([CONTENT_EDITOR_PROMPT, human_msg])
            editor_feedback = response.content.strip()

            # Check if the post needs revision
            if "NEEDS REVISION" in editor_feedback:
                posts_need_revision = True
                revision_feedback.append({
                    "post_index": i,
                    "feedback": editor_feedback
                })
                state["debug_info"] += f"Post {i+1} needs revision. "
            else:
                # If post is approved, update final version
                if "APPROVED" in editor_feedback:
                    # Extract the edited post if provided
                    if "EDITED POST:" in editor_feedback:
                        start_idx = editor_feedback.find("EDITED POST:")
                        if start_idx != -1:
                            edited_post = editor_feedback[start_idx +
                                                          len("EDITED POST:"):].strip()
                            # Update the final post with the edited version
                            state["final_posts"][i] = edited_post
                            state["debug_info"] += f"Post {i+1} approved with edits. "
                    else:
                        state["debug_info"] += f"Post {i+1} approved as is. "

        # Store feedback for writer to use
        state["editor_feedback"] = revision_feedback

        # Determine if we should continue editing or finish
        if posts_need_revision and state["editing_iterations"] < MAX_EDITING_ITERATIONS:
            state["continue_editing"] = True
            state["debug_info"] += f"{len(revision_feedback)} posts need revision, continuing to writer. "
        else:
            state["continue_editing"] = False

            # If we've reached max iterations but still have issues, add a note
            if posts_need_revision:
                state["editing_note"] = f"Reached maximum editing iterations ({MAX_EDITING_ITERATIONS}). Some posts may still need manual review."
                state["debug_info"] += f"Reached max iterations ({MAX_EDITING_ITERATIONS}) with posts still needing revision. "
            else:
                state["debug_info"] += "All posts approved or max iterations reached. "

        return state

    except Exception as e:
        state["debug_info"] += f"ERROR in editor_agent: {str(e)}"
        state["continue_editing"] = False
        return state
