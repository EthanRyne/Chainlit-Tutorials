"""
Chainlit + Ollama Chatbot Application (Python Script Version)
--------------------------------------------------------------
This script replicates the Chainlit + Ollama integration with clear structure, 
descriptions, and inline documentation.
"""

# 1️⃣ Importing Required Libraries
from operator import itemgetter
import os
import ollama
import subprocess
import threading
import requests
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

# 2️⃣ Load Environment Variables
load_dotenv()

# NOTE: These environment variables are hardcoded for demonstration.
# In production, keep them in your .env file for better security.
os.environ['CHAINLIT_AUTH_SECRET'] = "r>>aPxK9Iwl%KMjr,sjeIoP@I.kGOLb*kwriPYwtW$S9vJVR2HYFh.JUc_0J:PF."
os.environ['DATABASE_URL'] = "postgresql+asyncpg://chainlit_user:securepassword@localhost:5532/chainlit_db"

# 3️⃣ Ollama Server Initialization

def _ollama():
    """
    Private helper function to start Ollama server as a subprocess.
    """
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(['ollama', 'serve'])

def start_ollama():
    """
    Starts Ollama server in a separate daemon thread to avoid blocking main thread.
    """
    thread = threading.Thread(target=_ollama)
    thread.daemon = True
    thread.start()

# 4️⃣ Authentication Callback

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Simple password authentication callback for Chainlit.
    You can customize this for your actual authentication needs.
    """
    return cl.User(identifier=username)

# 5️⃣ Chat Session Initialization

@cl.on_chat_start
async def on_chat_start():
    """
    Called when a new chat session starts. It starts Ollama and initializes chat history.
    """
    start_ollama()
    cl.user_session.set('chat_history', [])

# 6️⃣ Database Layer for Chat History Persistence

@cl.data_layer
def get_data_layer():
    """
    Establish SQLAlchemy-based data layer for persisting chat history into PostgreSQL.
    """
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL"))

# 7️⃣ Resume Chat Session

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Reload previous chat history when resuming a chat thread.
    """
    start_ollama()
    cl.user_session.set("chat_history", [])
    
    for message in thread['steps']:
        if message['type'] == 'user_message':
            cl.user_session.get("chat_history").append(
                {'role':'user', 'content': message['output']}
            )
        elif message['type'] == 'assistant_message':
            cl.user_session.get("chat_history").append(
                {'role':'assistant', 'content': message['output']}
            )

# 8️⃣ Chat Message Handling Logic

@cl.on_message
async def on_message(message: cl.message):
    """
    Main chat handler which takes incoming user message, forwards it to Ollama LLM, 
    streams back response, and updates chat history.
    """
    chat_history = cl.user_session.get("chat_history")
    model = "qwen2.5:0.5b"  # Selected LLM model hosted in Ollama

    chat_history.append({'role':'user', 'content':message.content})

    cb = cl.Message(content="")
    await cb.send()

    def generate_chunks():
        return ollama.chat(
            model=model,
            messages=chat_history,
            stream=True,
            options={'stop': ['<|im_end|>']},
        )
    
    loop = asyncio.get_event_loop()
    stream = await loop.run_in_executor(None, generate_chunks)

    assistant_response = ''
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            assistant_response += content
            await cb.stream_token(content)

    chat_history.append({'role':'assistant', 'content':assistant_response})

    await cb.update()

# ✅ Application Setup Completed
