import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai
import re

# Load environment variables
load_dotenv()

# Configure Google GenerativeAI
genai.configure(api_key="AIzaSyAW6gQppI6lizfFad9Tlia_3mFIPbeDDlU")

# Prompt for summarization
base_prompt = """Welcome, Video Summarizer! Your task is to distill the essence of a given YouTube video transcript into a concise summary. Your summary should capture the key points and essential information. Let's dive into the provided transcript and extract the vital details for our audience."""

# Function to extract video ID from YouTube URL using regex
def extract_video_id(youtube_video_url):
    # Regular expression to match video ID from different YouTube URL formats
    regex = r"(?:https?://(?:www\.)?youtube\.com/watch\?v=|https?://youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.match(regex, youtube_video_url)
    
    if match:
        return match.group(1)
    else:
        return None

# Function to extract transcript details from a YouTube video URL
def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID using regex
        video_id = extract_video_id(youtube_video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please check the URL format.")
            return None
        
        # Get transcript
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine transcript text into a single string
        transcript = ""
        for entry in transcript_text:
            transcript += " " + entry["text"]

        return transcript
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video. Please choose another video.")
        return None
    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None

# Function to generate summary using Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Streamlit UI
st.title("YouTube Transcript Summarizer")

# YouTube URL input
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        video_id = extract_video_id(youtube_link)
        if video_id:
            # Display the video thumbnail
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

# Dropdown for selecting summary percentage
summary_percentage = st.selectbox(
    "Select the length of the summary:",
    ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
)

# Button to trigger summary generation
if st.button("Get Detailed Notes"):
    if youtube_link:
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            # Adjust the prompt based on the selected summary percentage
            if summary_percentage == "10%":
                prompt = base_prompt + " Summarize the video to 10% of its original content, focusing only on the most essential points."
            elif summary_percentage == "20%":
                prompt = base_prompt + " Summarize the video to 20% of its original content, highlighting the key points."
            elif summary_percentage == "30%":
                prompt = base_prompt + " Summarize the video to 30% of its original content, covering the main takeaways."
            elif summary_percentage == "40%":
                prompt = base_prompt + " Summarize the video to 40% of its original content, providing the key points with some additional details."
            elif summary_percentage == "50%":
                prompt = base_prompt + " Summarize the video to 50% of its original content, with moderate detail and key points."
            elif summary_percentage == "60%":
                prompt = base_prompt + " Summarize the video to 60% of its original content, providing more detailed information."
            elif summary_percentage == "70%":
                prompt = base_prompt + " Summarize the video to 70% of its original content, including most of the content in detail."
            elif summary_percentage == "80%":
                prompt = base_prompt + " Summarize the video to 80% of its original content, covering most points with detailed information."
            elif summary_percentage == "90%":
                prompt = base_prompt + " Summarize the video to 90% of its original content, providing a thorough summary with nearly all points covered."
            elif summary_percentage == "100%":
                prompt = base_prompt + " Summarize the video to 100% of its original content, providing a full and detailed summary with as much information as possible."

            # Generate summary using Gemini Pro
            summary = generate_gemini_content(transcript_text, prompt)

            # Display the summary
            st.markdown("## Detailed Notes:")
            st.write(summary)
