import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
import time
from audio_recorder_streamlit import audio_recorder
import io
from RAG_functions import *

# import cookGpt

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.environ["API_KEY"])
# client = genai.GenerativeModel("gemini-1.5-flash")
st.set_page_config(
    page_title="Recipe Guide Chatbot",
    page_icon="🍕",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("🥙 Recipe Guide Chatbot 🍕")


# load stylesheet
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("style.css")

with st.sidebar:
    st.markdown(
        """
        <h1 style='text-align: center;'>🍕 Recipe Guide Chatbot 🥙</h1>
        <p style='text-align: center;'>Ask me anything about recipes!</p>
        """,
        unsafe_allow_html=True,
    )

    model = st.selectbox("Select your model:", ["Gemini", "CookGpt", "Gpt-2"], index=0)

    def reset_conv():
        if "messages" in st.session_state and len(st.session_state.messages) > 0:
            st.session_state.pop("messages", None)

    st.button("Reset Conversation", on_click=reset_conv)
    st.divider()

    # handling audio
    audio_prompt = None
    if "prev_speech_hash" not in st.session_state:
        st.session_state.prev_speech_hash = None

    speech_input = audio_recorder(
        "Talk to the chatbot:",
        icon_size="2x",
        neutral_color="#2C6FC3",
    )

    if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
        st.session_state.prev_speech_hash = hash(speech_input)
        # transcript = client.audio.transcription.create(
        #     model="whisper-1",
        #     file=("audio.wav", speech_input),
        # )

        # audio_prompt = transcript.text
        audio_file_path = "audio_input.mp3"
        with open(audio_file_path, "wb") as f:
            f.write(speech_input)
        myfile = genai.upload_file(path=audio_file_path)

        # Create the prompt for speech-to-text conversion
        prompt = "Convert speech to text"

        # Pass the prompt and the uploaded file to Gemini for transcription
        response = assistant.generate_content([prompt, myfile])

        # Get the transcription result from the response
        audio_prompt = response.text

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
if prompt := st.chat_input("How can I help you?") or audio_prompt:
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt or audio_prompt}
    )
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(
            f"<div class='user-message'>{prompt}</div>",
            unsafe_allow_html=True,
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # if model == "CookGpt":
        #     chat_message = cookGpt.generate_response(
        #         cookGpt.pipe, cookGpt.messages, prompt
        #     )

        # stream = client.generate_content(
        #     prompt,
        #     generation_config=genai.types.GenerationConfig(
        #         # max_output_tokens=200,
        #         temperature=0.7,
        #         top_k=50,
        #         top_p=0.95,
        #     ),
        #     stream=True,
        # )

        stream = generate_answer(prompt)

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
