import streamlit as st
import os
import json
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageFilter
import moviepy.editor as mp
from pydub import AudioSegment
import replicate
import asyncio
import anthropic  # For integrating with Anthropic's Claude model

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
            'luma': '',
            'anthropic': ''
        }
    if 'files' not in st.session_state:
        st.session_state['files'] = []

def save_api_keys():
    """Save API keys to a JSON file."""
    try:
        with open('api_keys.json', 'w') as f:
            json.dump(st.session_state['api_keys'], f)
        st.sidebar.success("API keys saved!")
    except Exception as e:
        st.sidebar.error(f"Error saving API keys: {e}")

def load_api_keys():
    """Load API keys from a JSON file."""
    try:
        with open('api_keys.json', 'r') as f:
            st.session_state['api_keys'] = json.load(f)
        st.sidebar.success("API keys loaded!")
    except FileNotFoundError:
        st.sidebar.warning("No saved API keys found.")
    except Exception as e:
        st.sidebar.error(f"Error loading API keys: {e}")

def save_file(content, file_name, file_type):
    """Save uploaded file to the filesystem."""
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
    anthropic_api_key = st.session_state['api_keys']['anthropic']
    if not anthropic_api_key:
        return "Please set your Anthropic API key."

    try:
        # Use Anthropic's Claude model for advanced understanding
        client = anthropic.Client(anthropic_api_key)
        response = await asyncio.to_thread(client.completion, prompt=anthropic.HUMAN_PROMPT + user_input + anthropic.AI_PROMPT)
        assistant_reply = response['completion'].strip()
        add_to_conversation("assistant", assistant_reply)
        # Process assistant's reply to perform actions
        await process_assistant_reply(assistant_reply)
        return assistant_reply
    except Exception as e:
        st.error(f"Error: {e}")
        return "I'm sorry, something went wrong."

async def process_assistant_reply(assistant_reply):
    """Process the assistant's reply to perform media actions."""
    # Parse the assistant's reply to extract actions
    actions = parse_actions(assistant_reply)
    for action in actions:
        if action['type'] == 'edit_image':
            await perform_image_editing(action)
        elif action['type'] == 'edit_video':
            await perform_video_editing(action)
        elif action['type'] == 'edit_audio':
            await perform_audio_editing(action)
        else:
            st.write("Action not recognized.")

def parse_actions(assistant_reply):
    """Parse assistant's reply to extract actions."""
    # Implement a parser that can understand the assistant's reply
    # For demonstration, we'll assume the assistant provides a JSON-like response
    # In practice, you might need a more robust NLP parsing method
    try:
        actions = json.loads(assistant_reply)
        return actions
    except json.JSONDecodeError:
        st.write("Could not parse assistant's reply.")
        return []

async def perform_image_editing(action):
    """Perform image editing based on action."""
    image_file = select_file("image")
    if image_file:
        image_path = image_file['path']
        img = Image.open(image_path)

        # Example: Apply style transfer using Replicate
        if 'style' in action:
            replicate_api_token = st.session_state['api_keys']['replicate']
            if not replicate_api_token:
                st.warning("Replicate API key is required for style transfer.")
                return
            replicate.Client(api_token=replicate_api_token)
            model = replicate.models.get("laion-ai/erlich")
            output = model.predict(image=open(image_path, "rb"), style=action['style'])
            img = Image.open(BytesIO(output))

        # Apply other transformations as per action
        # ...

        edited_image_path = FILES_DIR / f"edited_{Path(image_path).name}"
        img.save(edited_image_path)
        st.success(f"Image edited and saved as {edited_image_path.name}")
        st.image(str(edited_image_path))
    else:
        st.warning("No image file selected for editing.")

async def perform_video_editing(action):
    """Perform video editing based on action."""
    video_file = select_file("video")
    if video_file:
        video_path = video_file['path']
        clip = mp.VideoFileClip(video_path)

        # Example: Apply effects based on action
        if 'effect' in action:
            # Apply video effects here
            pass

        edited_video_path = FILES_DIR / f"edited_{Path(video_path).name}"
        clip.write_videofile(str(edited_video_path))
        st.success(f"Video edited and saved as {edited_video_path.name}")
        st.video(str(edited_video_path))
    else:
        st.warning("No video file selected for editing.")

async def perform_audio_editing(action):
    """Perform audio editing based on action."""
    audio_file = select_file("audio")
    if audio_file:
        audio_path = audio_file['path']
        audio = AudioSegment.from_file(audio_path)

        # Example: Apply transformations
        if 'transform' in action:
            # Apply audio transformations here
            pass

        edited_audio_path = FILES_DIR / f"edited_{Path(audio_path).name}"
        audio.export(edited_audio_path, format="mp3")
        st.success(f"Audio edited and saved as {edited_audio_path.name}")
        st.audio(str(edited_audio_path))
    else:
        st.warning("No audio file selected for editing.")

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
    st.title("✨ Super-Powered AI Assistant")
    st.write("Interact with me to create and edit media files using advanced AI capabilities.")

    init_session_state()

    # Sidebar - API Key Management
    st.sidebar.title("🔑 API Keys")
    st.sidebar.info("Enter your API keys. They will be saved securely.")

    api_keys = st.session_state['api_keys']
    api_keys['openai'] = st.sidebar.text_input("OpenAI API Key", type="password", value=api_keys.get('openai', ''))
    api_keys['replicate'] = st.sidebar.text_input("Replicate API Key", type="password", value=api_keys.get('replicate', ''))
    api_keys['stability'] = st.sidebar.text_input("Stability AI API Key", type="password", value=api_keys.get('stability', ''))
    api_keys['luma'] = st.sidebar.text_input("Luma AI API Key", type="password", value=api_keys.get('luma', ''))
    api_keys['anthropic'] = st.sidebar.text_input("Anthropic API Key", type="password", value=api_keys.get('anthropic', ''))

    if st.sidebar.button("Save API Keys"):
        save_api_keys()
    if st.sidebar.button("Load API Keys"):
        load_api_keys()

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

    # Chat Interface
    st.markdown("### 💬 Chat with the Assistant")
    chat_placeholder = st.empty()
    user_input = st.text_input("Type your message...", key="user_input")

    if st.button("Send", key="send_button"):
        if user_input.strip() != "":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with st.spinner("Processing..."):
                assistant_reply = loop.run_until_complete(nlp_agent(user_input))
                loop.close()
            # Display conversation
            conversation = st.session_state['conversation']
            for idx, message in enumerate(conversation):
                if message['role'] == 'user':
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Assistant:** {message['content']}")

# -------------------- Run the App --------------------

if __name__ == "__main__":
    main()
