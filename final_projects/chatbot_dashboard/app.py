#streamlit run app.py

import streamlit as st
import requests

# FastAPI server URL
BASE_URL = "http://localhost:8000"  

# Function to make requests to the FastAPI backend
def call_api(endpoint, data):
    try:
        response = requests.post(f"{BASE_URL}/{endpoint}", json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

st.title("AI Tool with FastAPI")

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Select Task",
    ["Text Summarization", "Next Word Prediction", "Sentiment Analysis", "Chatbot", "Question Answering", "Generate Image"]
)

# Text input for different tasks
input_text = st.text_area("Input Text", "Type your text here...")


if task == "Text Summarization":
    if st.button("Summarize"):
        data = {"text": input_text}
        result = call_api("summarize", data)
        if result:
            st.write("**Summary:**", result.get("summary", "No summary generated."))

elif task == "Next Word Prediction":
    if st.button("Predict Next Word"):
        data = {"text": input_text}
        result = call_api("predict_next_word", data)
        if result:
            st.write("**Prediction:**", result.get("prediction", "No prediction generated."))

elif task == "Sentiment Analysis":
    if st.button("Analyze Sentiment"):
        data = {"text": input_text}
        result = call_api("sentiment", data)
        if result:
            st.write("**Sentiment:**", result.get("sentiment", "No sentiment detected"))
            st.write("**Confidence:**", result.get("confidence", "No confidence score"))

elif task == "Chatbot":
    if st.button("Chat with Bot"):
        data = {"text": input_text}
        result = call_api("chat", data)
        if result:
            st.write("**Bot Response:**", result.get("response", "No response"))

elif task == "Question Answering":
    context = st.text_area("Context", "Provide context for your question...")
    if st.button("Get Answer"):
        if input_text and context:
            data = {"question": input_text, "context": context}
            result = call_api("qa", data)
            if result:
                st.write("**Answer:**", result.get("answer", "No answer found"))
        else:
            st.error("Please provide both question and context.")

elif task == "Generate Image":
    if st.button("Generate Image"):
        data = {"text": input_text}
        result = call_api("generate_image", data)
        if result:
            st.image(result.get("image_url", "https://via.placeholder.com/150"), caption="Generated Image")


#st.write("""
#     Use this app to interact with various AI tasks like summarization, sentiment analysis, and more.
#     Just type the text in the provided box and select a task from the sidebar.
# """)
