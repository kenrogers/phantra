import os
import uuid
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Create a simple app
st.title("Terrapin")
st.write("Turn your YouTube videos into engaging LinkedIn content. Automagically.")

# Sidebar for API key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed")
    st.stop()
else:
    os.environ["OPENAI_API_KEY"] = api_key

# Main input
youtube_url = st.text_input(
    "Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

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
    posts = []

    post_prompt = ChatPromptTemplate.from_template(
        """Create an engaging LinkedIn post based on this insight:

        INSIGHT: {insight}

        The LinkedIn post should:
        1. Be under 1300 characters
        2. Include appropriate emojis
        3. Have a compelling hook
        4. Include 2-3 relevant hashtags
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
    optimized_posts = []

    optimization_prompt = ChatPromptTemplate.from_template(
        """Optimize this LinkedIn post for maximum engagement:

        {post}

        Improve:
        1. Grammar and clarity
        2. Engagement potential
        3. Professional tone
        4. Appropriate use of emojis (don't overdo it)
        5. Make sure all hashtags are relevant and properly formatted
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


def generate_linkedin_posts(url: str) -> List[str]:
    """Generate LinkedIn posts from YouTube video"""
    with st.status("Generating LinkedIn posts...") as status:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        # Step 1: Fetch transcript
        status.update(label="Fetching YouTube transcript...")
        transcript = fetch_transcript(url)

        # Step 2: Determine processing method (RAG or direct)
        status.update(label="Analyzing transcript length...")
        rag_info = prepare_rag_if_needed(transcript)
        st.info(rag_info["message"])

        # Step 3: Extract insights
        status.update(label="Extracting key insights...")
        insights = extract_insights(transcript, llm, rag_info)

        # Step 4: Create LinkedIn posts
        status.update(label="Creating LinkedIn posts...")
        posts = create_linkedin_posts(insights, llm)

        # Step 5: Edit and optimize
        status.update(label="Optimizing posts for engagement...")
        final_posts = edit_and_optimize_posts(posts, llm)

        status.update(label="Done!", state="complete")

    return final_posts


# Run when button is clicked
if st.button("Generate LinkedIn Posts", disabled=not youtube_url):
    try:
        posts = generate_linkedin_posts(youtube_url)

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
1. Extracting the video transcript
2. Adaptively using direct processing or RAG based on length
3. Extracting key insights
4. Creating engaging posts
5. Optimizing for maximum engagement
""")
