import os
import uuid
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import from our modules
from models import AgentState
from utils import fetch_transcript
from graph import create_graph
from evaluation import evaluate_all_posts

# Load environment variables from .env
load_dotenv()


def main():
    st.title("Phantra")
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

    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    # User context inputs
    if 'context_saved' not in st.session_state:
        st.session_state.context_saved = False

    with st.form("user_context"):
        business = st.text_input(
            "What's your business about?",
            value="I run a community called Vibe Coders where I teach people how to code and build with AI and teach AI engineering"
        )
        teaching_focus = st.text_input(
            "What do you teach or share knowledge about?",
            value="AI engineering, vibe coding, building with AI"
        )
        cta = st.text_input(
            "What's your primary call-to-action?",
            value="To get people to join my community Vibe Coders, url is https://skool.com/vibecoders"
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
            "Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...", value="https://www.youtube.com/watch?v=9lAx0bcAD5A")
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
                status.update(label="1/6: Fetching YouTube transcript...")
                transcript = fetch_transcript(url)

                if not transcript:
                    raise ValueError("No transcript content to process")

                # Initialize state
                initial_state = AgentState(
                    transcript=transcript,
                    user_context=user_context,
                    voice_analysis="",
                    insights=[],
                    draft_posts=[],
                    final_posts=[],
                    messages=[],
                    revision_count=0,
                    status=status,
                    next_step=None,
                    rag_info={},
                    debug_info=""
                )

                # Create workflow
                workflow = create_graph()

                # Process - each agent will update the status
                final_state = workflow.invoke(initial_state)
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

            # Display debug information if enabled
            if debug_mode and "final_state" in st.session_state:
                with st.expander("Debug Information", expanded=True):
                    st.markdown("### Debug Log")
                    st.text(st.session_state.final_state.get(
                        "debug_info", "No debug info available"))

                    st.markdown("### State Information")
                    st.markdown(
                        f"- Insights: {len(st.session_state.final_state.get('insights', []))} extracted")
                    st.markdown(
                        f"- Draft Posts: {len(st.session_state.final_state.get('draft_posts', []))} created")
                    st.markdown(
                        f"- Final Posts: {len(st.session_state.final_state.get('final_posts', []))} generated")
                    st.markdown(
                        f"- Editing Iterations: {st.session_state.final_state.get('editing_iterations', 0)}")

                    if "editing_note" in st.session_state.final_state:
                        st.warning(
                            st.session_state.final_state["editing_note"])

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

            # Show debug info even on error if debug mode is enabled
            if debug_mode and "final_state" in st.session_state:
                with st.expander("Debug Information (Error State)", expanded=True):
                    st.text(st.session_state.final_state.get(
                        "debug_info", "No debug info available"))

    # Show app info in sidebar
    st.sidebar.markdown("## About")
    st.sidebar.markdown("""
    This app converts YouTube content into LinkedIn posts by:
    1. Analyzing the speaker's voice and style
    2. Extracting key insights based on your business context
    3. Writing engaging LinkedIn posts in the speaker's voice
    4. Editing and refining for maximum impact
    """)


if __name__ == "__main__":
    main()
