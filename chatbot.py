import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
st.title("Recipe Guide Chatbot")
genai.configure(api_key=os.environ["API_KEY"])
client = genai.GenerativeModel("gemini-1.5-flash")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

# Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.generate_content(prompt,
            generation_config=genai.types.GenerationConfig(
            max_output_tokens=200,
            temperature=0.6,
            ),
            stream=True,
        )
        chat_message = ""
        for chunk in stream:
            chat_message +=chunk.text
            st.write(chat_message )
    st.session_state.messages.append({"role": "assistant", "content": chat_message})