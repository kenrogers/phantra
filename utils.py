import os
import requests
import tiktoken
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def get_youtube_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    try:
        # Handle various YouTube URL formats
        if "youtube.com/watch" in url and "v=" in url:
            # Standard watch URL: https://www.youtube.com/watch?v=VIDEO_ID
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            # Shortened URL: https://youtu.be/VIDEO_ID
            video_id = url.split("youtu.be/")[1].split("?")[0].split("#")[0]
        elif "youtube.com/embed/" in url:
            # Embed URL: https://www.youtube.com/embed/VIDEO_ID
            video_id = url.split(
                "youtube.com/embed/")[1].split("?")[0].split("#")[0]
        elif "youtube.com/v/" in url:
            # Old embed URL: https://www.youtube.com/v/VIDEO_ID
            video_id = url.split(
                "youtube.com/v/")[1].split("?")[0].split("#")[0]
        elif "youtube.com/shorts/" in url:
            # YouTube Shorts: https://www.youtube.com/shorts/VIDEO_ID
            video_id = url.split(
                "youtube.com/shorts/")[1].split("?")[0].split("#")[0]
        else:
            # Try direct ID if it looks like a YouTube video ID (11 characters)
            if len(url.strip()) == 11 and all(c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in url.strip()):
                return url.strip()
            raise ValueError(f"Unsupported YouTube URL format: {url}")

        if not video_id:
            raise ValueError("Could not extract video ID")

        # Clean up the video ID
        video_id = video_id.strip()

        # Log for debugging
        st.write(f"Extracted YouTube ID: {video_id}")

        return video_id
    except Exception as e:
        st.error(f"Failed to extract YouTube video ID: {str(e)}")
        raise ValueError(f"Failed to extract YouTube video ID: {str(e)}")


def fetch_youtube_metadata(video_id: str) -> dict:
    """Fetch metadata about a YouTube video as a fallback"""
    try:
        # Use YouTube's oEmbed API to get basic metadata
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url)

        if response.status_code == 200:
            return response.json()
        else:
            st.warning(
                f"Failed to fetch YouTube metadata: Status code {response.status_code}")
            return {}
    except Exception as e:
        st.warning(f"Error fetching YouTube metadata: {str(e)}")
        return {}


def fetch_transcript(url: str) -> str:
    """Fetch transcript from YouTube video using youtube-transcript-api"""
    try:
        # Extract video ID using our existing function
        video_id = get_youtube_id(url)
        st.info(f"Fetching transcript for video ID: {video_id}")

        try:
            # Get transcript using YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Format the transcript
            transcript_text = ""
            for segment in transcript_list:
                transcript_text += segment['text'] + " "

            # Try to get video metadata to enhance the transcript
            try:
                metadata = fetch_youtube_metadata(video_id)
                if metadata and 'title' in metadata:
                    video_info = f"Title: {metadata.get('title', '')}\n"
                    video_info += f"Author: {metadata.get('author_name', '')}\n\n"
                    transcript_text = f"{video_info}{transcript_text}"
            except Exception as meta_error:
                st.warning(f"Could not fetch metadata: {str(meta_error)}")

            st.success("Successfully fetched transcript")
            return transcript_text

        except Exception as e:
            st.error(f"Error fetching transcript: {str(e)}")

            # Fallback to metadata if transcript fails
            try:
                metadata = fetch_youtube_metadata(video_id)
                if metadata and 'title' in metadata:
                    fallback_text = f"Video Title: {metadata.get('title', '')}\n"
                    fallback_text += f"Author: {metadata.get('author_name', '')}\n"
                    fallback_text += f"Description: {metadata.get('description', 'No description available.')}\n"
                    fallback_text += "\nNOTE: This is metadata only as transcript could not be retrieved."

                    st.warning(
                        "Could not retrieve transcript. Using video metadata as fallback.")
                    return fallback_text
            except Exception as meta_error:
                st.warning(
                    f"Metadata fallback method failed: {str(meta_error)}")

            return f"Failed to retrieve transcript for video ID: {video_id}. This video may have disabled captions or requires authentication."

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return f"Error processing video: {str(e)}"
