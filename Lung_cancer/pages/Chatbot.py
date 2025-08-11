import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Create chatbot model with system instructions
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction="""
        Your name is Kate.
        You are a lung cancer expert who has 15 years of experience. 
        You are to check the user's prompts for any symptoms they may be experiencing and give any insights on their chance of having Lung Cancer. 
        If the users prompt doesn't include any symptoms kindly ask them to provide symptoms.
        If the user asks a question that is not related to Lung Cancer decline politely and tell them you only deal with lung cancer issues.
    """
)

# Streamlit UI
st.set_page_config(page_title="Kate - Lung Cancer Expert", page_icon="💬")

st.title("💬 Kate - Lung Cancer Expert")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for user
if prompt := st.chat_input("How are you feeling today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get model response
    response = model.generate_content(prompt)
    bot_reply = response.text

    # Add bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
