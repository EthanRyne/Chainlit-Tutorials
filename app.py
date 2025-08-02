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
import fitz
import time

from docx import Document
from dotenv import load_dotenv
from typing import Dict, Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Switch
from chainlit.types import ThreadDict

# 2️⃣ Load Environment Variables and Define Global Variables
load_dotenv()
models = ['qwen3:0.6b','qwen2.5:0.5b']

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

# 3️⃣ Document Reader Function

def read_documents(documents):
    """
    Recieves a list of chainlit file type, filters and opens them based on their extension
    Returns a text body containing al the file names along with their contents
    """
    text = ''
    for document in documents:
        if document.path.endswith('pdf'):
            doc = fitz.open(document.path)
            text += '\n\nFile: ' + document.name + '\n'
            for page in doc:
                text += page.get_text()
        elif document.path.endswith('docx'):
            doc = Document(document.path)
            text += '\n\nFile: ' + document.name + '\n' + '\n'.join([para.text for para in doc.paragraphs])
        elif document.path.endswith('txt'):
            with open(document.path, 'r') as f:
                text += '\n\nFile: ' + document.name + '\n' + f.read()
    return text

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
    Setting up a settings widget for model handling
    """
    start_ollama()
    cl.user_session.set('chat_history', [])
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label='Qwen3',
                values=['qwen3:0.6b','qwen2.5:0.5b'],
                initial_index=0
            ),
            Switch(id='Think', label="Enable thinking for qwen3 models", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)

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
    Setup a new settings widget each time you resume chat thread.
    """
    start_ollama()
    cl.user_session.set("chat_history", [])
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label='Qwen3',
                values=['qwen3:0.6b','qwen2.5:0.5b'],
                initial_index=0
            ),
            Switch(id='Think', label="Enable thinking for qwen3 models", initial=True)
        ]
    ).send()
    cl.user_session.set("settings", settings)
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
    streams back response, and updates chat history, 
    reads documents and displays chain of thought tokens in Thinking Step.
    """
    chat_history = cl.user_session.get("chat_history")
    settings = cl.user_session.get("settings")

    global models
    
    model = settings['Model']
    settings['Think'] = False if settings['Think'] and model == models[1] else settings['Think']

    # Reading Files
    files = [file for file in message.elements]
    if files:
        async with cl.Step('Reading Documents') as reading_documents:
            await reading_documents.send()
            loop = asyncio.get_event_loop()
            document_data = await loop.run_in_executor(None, read_documents, files)
            content = f'The user has uploaded the following files {document_data}, assist them in their queries if related to the uploaded files'
            print(content)
            chat_history.append({'role':'system', 'content':content})
            time.sleep(4) # Not required
            await reading_documents.remove()
            
    chat_history.append({'role':'user', 'content':message.content})
    
    stream = ollama.chat(
            model=model,
            messages=chat_history,
            stream=True,
            think=settings['Think'])
    
    thinking = False
    assistant_response = ''
    start = time.time() # Calculate time
    final_answer = cl.Message(content='')

    # Thinking Step
    if settings["Think"]:
        async with cl.Step(name='Thinking',type='llm') as thinking_step:
            for chunk in stream:
                think = chunk.get("message", {}).get("thinking", "")
                if think:
                    thinking = True
                    await thinking_step.stream_token(think)
                elif settings["Think"] and thinking:
                    print('Thinking stopped')
                    thinking = False
                    thought_duration = round(time.time()-start)
                    thinking_step.name = f"Thought for {thought_duration}s"
                    await thinking_step.update()
                    break
    
    # Final Answer
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            assistant_response += content
            await final_answer.stream_token(content)

    await final_answer.send()

    chat_history.append({'role':'assistant', 'content':assistant_response})

# ✅ Application Setup Completed
