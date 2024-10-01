import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time

# Load environment variables from .env file
load_dotenv()
st.title("🥙 Recipe Guide Chatbot 🍕")
genai.configure(api_key=os.environ["API_KEY"])
client = genai.GenerativeModel("gemini-1.5-flash")


# load stylesheet
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "assistant":
        st.markdown(
            f"<div class='bot-message'>{message['content']}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='user-message'>{message['content']}</div>",
            unsafe_allow_html=True,
        )

    # with st.chat_message(message["role"]):
    #     if
    #     st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(
            f"<div class='user-message'>{prompt}</div>",
            unsafe_allow_html=True,
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # max_output_tokens=200,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            ),
            stream=True,
        )

        # show the bot's response in chunks (streaming the answer)
        chat_message = ""
        stream.resolve()
        placeholder = st.empty()
        chunk_size = 10  # size of each chunk of characters to be displayed at a time
        buffer = ""

        for chunk in stream:
            buffer += chunk.text  # append chunk text to the buffer

            # split the buffer into chunks
            for i in range(0, len(buffer), chunk_size):
                # append the next 10 characters to the chat_message
                chat_message += buffer[i : i + chunk_size]

                placeholder.markdown(
                    f"<div class='bot-message'>{chat_message}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.01)  # delay between chunks

            buffer = ""

    st.session_state.messages.append({"role": "assistant", "content": chat_message})
