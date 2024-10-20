# Super-Powered AI Assistant - Enhanced Version

import streamlit as st
import os
import json
import openai
import replicate
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFilter
import moviepy.editor as mp
from pydub import AudioSegment
import asyncio
import threading
from streamlit_chat import message as st_message  # For chat interface
import base64
import tempfile

# -------------------- Configuration --------------------

# Supported file types
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'bmp']
SUPPORTED_VIDEO_TYPES = ['mp4', 'mov', 'avi', 'mkv']
SUPPORTED_AUDIO_TYPES = ['mp3', 'wav', 'ogg', 'flac']

# File storage directory
FILES_DIR = Path("files")
FILES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Helper Functions --------------------

def init_session_state():
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    if 'api_keys' not in st.session_state:
        st.session_state['api_keys'] = {
            'openai': '',
            'replicate': '',
            'stability': '',
            'luma': ''
        }
    if 'files' not in st.session_state:
        st.session_state['files'] = []
    if 'progress' not in st.session_state:
        st.session_state['progress'] = 0

def save_api_keys():
    """Save API keys securely."""
    try:
        encoded_keys = secure_api_keys(st.session_state['api_keys'])
        with open('api_keys.json', 'w') as f:
            json.dump(encoded_keys, f)
        st.sidebar.success("API keys saved securely!")
    except Exception as e:
        st.sidebar.error(f"Error saving API keys: {e}")

def load_api_keys():
    """Load API keys."""
    try:
        with open('api_keys.json', 'r') as f:
            encoded_keys = json.load(f)
        st.session_state['api_keys'] = decrypt_api_keys(encoded_keys)
        st.sidebar.success("API keys loaded!")
    except FileNotFoundError:
        st.sidebar.warning("No saved API keys found.")
    except Exception as e:
        st.sidebar.error(f"Error loading API keys: {e}")

def secure_api_keys(api_keys):
    """Encrypt API keys using base64 encoding."""
    encoded_keys = {}
    for key, value in api_keys.items():
        encoded_keys[key] = base64.b64encode(value.encode()).decode()
    return encoded_keys

def decrypt_api_keys(encoded_keys):
    """Decrypt API keys."""
    decoded_keys = {}
    for key, value in encoded_keys.items():
        decoded_keys[key] = base64.b64decode(value.encode()).decode()
    return decoded_keys

def save_file(content, file_name, file_type):
    """Save uploaded file."""
    try:
        file_path = FILES_DIR / file_name
        with open(file_path, "wb") as f:
            f.write(content.getbuffer())
        st.session_state['files'].append({
            "name": file_name,
            "path": str(file_path),
            "type": file_type
        })
        st.success(f"'{file_name}' uploaded successfully!")
    except Exception as e:
        st.error(f"Error uploading file '{file_name}': {e}")

def list_files():
    """Display uploaded files."""
    st.sidebar.subheader("Your Files")
    files = st.session_state['files']
    if not files:
        st.sidebar.info("No files uploaded yet.")
    else:
        for idx, file in enumerate(files):
            with st.sidebar.expander(f"{file['name']} ({file['type']})"):
                st.sidebar.download_button(
                    "Download",
                    data=open(file['path'], "rb").read(),
                    file_name=file['name']
                )
                if st.sidebar.button(f"Delete {file['name']}", key=f"delete_{idx}"):
                    try:
                        os.remove(file['path'])
                        st.session_state['files'].pop(idx)
                        st.experimental_rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error deleting file: {e}")

def add_to_conversation(role, content):
    """Add message to conversation history."""
    st.session_state['conversation'].append({"role": role, "content": content})

async def nlp_agent(user_input):
    """Process user input and generate assistant's response."""
    add_to_conversation("user", user_input)
    openai.api_key = st.session_state['api_keys']['openai']
    if not openai.api_key:
        return "Please set your OpenAI API key."

    try:
        response = await asyncio.to_thread(openai.ChatCompletion.create,
            model="gpt-4",
            messages=st.session_state['conversation'],
            max_tokens=500,
            temperature=0.7
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        add_to_conversation("assistant", assistant_reply)
        # Process assistant's reply to perform actions
        await process_assistant_reply(assistant_reply)
        return assistant_reply
    except Exception as e:
        st.error(f"Error: {e}")
        return "I'm sorry, something went wrong."

async def process_assistant_reply(assistant_reply):
    """Process the assistant's reply to perform media actions."""
    # Here, you can parse the assistant's reply and execute media editing functions accordingly
    # For example, if the assistant says "I have edited your image as per your request."
    # You can call the image editing function with the appropriate parameters
    # This is a placeholder implementation

    if "edit image" in assistant_reply.lower():
        # Extract instructions and perform image editing
        await perform_image_editing(assistant_reply)
    elif "edit video" in assistant_reply.lower():
        await perform_video_editing(assistant_reply)
    elif "edit audio" in assistant_reply.lower():
        await perform_audio_editing(assistant_reply)
    else:
        # Handle other types of replies or actions
        pass

async def perform_image_editing(instructions):
    """Perform image editing based on instructions."""
    # For demonstration purposes, let's assume we apply a filter
    # In practice, you would parse the instructions to determine the exact actions
    image_file = select_file("image")
    if image_file:
        image_path = image_file['path']
        img = Image.open(image_path)
        img = img.filter(ImageFilter.BLUR)
        edited_image_path = FILES_DIR / f"edited_{Path(image_path).name}"
        img.save(edited_image_path)
        st.success(f"Image edited and saved as {edited_image_path.name}")
        st.image(str(edited_image_path))
    else:
        st.warning("No image file selected for editing.")

async def perform_video_editing(instructions):
    """Perform video editing based on instructions."""
    # Implement video editing logic here
    pass

async def perform_audio_editing(instructions):
    """Perform audio editing based on instructions."""
    # Implement audio editing logic here
    pass

def select_file(file_type):
    """Select a file of the specified type."""
    files = [f for f in st.session_state['files'] if f['type'] == file_type]
    if not files:
        st.warning(f"No {file_type} files available.")
        return None
    file_names = [f['name'] for f in files]
    selected_file_name = st.selectbox(f"Select a {file_type} file:", file_names)
    for f in files:
        if f['name'] == selected_file_name:
            return f
    return None

# -------------------- User Interface --------------------

def main():
    st.set_page_config(page_title="Super-Powered AI Assistant", layout="wide")
    init_session_state()

    # Sidebar - API Key Management
    st.sidebar.title("🔑 API Keys")
    st.sidebar.info("Your API keys are stored securely.")
    api_keys = st.session_state['api_keys']
    api_keys['openai'] = st.sidebar.text_input("OpenAI API Key", type="password", value=api_keys.get('openai', ''))
    api_keys['replicate'] = st.sidebar.text_input("Replicate API Key", type="password", value=api_keys.get('replicate', ''))
    api_keys['stability'] = st.sidebar.text_input("Stability AI API Key", type="password", value=api_keys.get('stability', ''))
    api_keys['luma'] = st.sidebar.text_input("Luma AI API Key", type="password", value=api_keys.get('luma', ''))
    st.sidebar.button("Save API Keys", on_click=save_api_keys)
    st.sidebar.button("Load API Keys", on_click=load_api_keys)

    # Sidebar - File Upload
    st.sidebar.title("📁 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Images, Videos, or Audio files",
        type=SUPPORTED_IMAGE_TYPES + SUPPORTED_VIDEO_TYPES + SUPPORTED_AUDIO_TYPES,
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension in SUPPORTED_IMAGE_TYPES:
                file_type = "image"
            elif file_extension in SUPPORTED_VIDEO_TYPES:
                file_type = "video"
            elif file_extension in SUPPORTED_AUDIO_TYPES:
                file_type = "audio"
            else:
                st.sidebar.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            save_file(uploaded_file, uploaded_file.name, file_type)
    list_files()

    # Main Interface
    st.title("✨ Super-Powered AI Assistant")
    st.write("Interact with me to create and edit media files using advanced AI capabilities.")

    # Chat Interface
    st.markdown("### 💬 Chat with the Assistant")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['conversation']:
            if message['role'] == 'assistant':
                st_message(message['content'], is_user=False)
            else:
                st_message(message['content'], is_user=True)

    user_input = st.text_input("Type your message...", key="user_input")
    if st.button("Send", key="send_button"):
        if user_input.strip() != "":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with st.spinner("Processing..."):
                assistant_reply = loop.run_until_complete(nlp_agent(user_input))
                st_message(assistant_reply, is_user=False)
                loop.close()

# -------------------- Run the App --------------------

if __name__ == "__main__":
    main()
