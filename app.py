"""
NITRO P.A.W.S. - Personal Abstract Witty adviSor-Lite
Streamlit chatbot powered by Gemini + NumPy RAG + Google Sheets logging
"""

import os
import uuid
import time
import pathlib
import numpy as np
from datetime import datetime, timezone

import gspread
import streamlit as st
from google import genai
from google.genai import types
from google.oauth2 import service_account

# ── Config ────────────────────────────────────────────────────────────────────

KB_FILE            = pathlib.Path("knowledge_base.npz")
EMBED_MODEL        = "models/gemini-embedding-001"
GEN_MODEL          = "gemini-2.5-flash"
FALLBACK_MODEL     = "gemma-3-27b-it"
SYSTEM_PROMPT_FILE = pathlib.Path("PAWS_Gemini Markdown.md")
TOP_K              = 5
SHEET_NAME         = "PAWS Conversations"

# ── Initialisation ────────────────────────────────────────────────────────────

@st.cache_resource
def load_clients():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not set.")
        st.stop()
    return genai.Client(api_key=api_key)

@st.cache_resource
def load_knowledge_base():
    data = np.load(KB_FILE, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    # Pre-normalise rows for fast cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return embeddings, data["documents"].tolist(), data["sources"].tolist()

@st.cache_resource
def load_sheet():
    """Returns a gspread worksheet, or None if not configured."""
    try:
        sa_info = dict(st.secrets["firestore"])
        creds   = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://spreadsheets.google.com/feeds",
                    "https://www.googleapis.com/auth/drive"],
        )
        gc    = gspread.authorize(creds)
        return gc.open(SHEET_NAME).sheet1
    except Exception:
        return None

@st.cache_data
def load_system_prompt():
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")

# ── Google Sheets logging ─────────────────────────────────────────────────────

def log_turn(sheet, session_id: str, user_msg: str, paws_reply: str):
    if sheet is None:
        return
    try:
        sheet.append_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            session_id,
            user_msg,
            paws_reply,
            GEN_MODEL,
        ])
    except Exception:
        pass

# ── RAG retrieval ─────────────────────────────────────────────────────────────

def retrieve_context(gemini_client, query: str) -> str:
    emb_matrix, documents, sources = load_knowledge_base()

    query_emb = np.array(
        gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
        ).embeddings[0].values,
        dtype=np.float32,
    )
    norm = np.linalg.norm(query_emb)
    query_emb = query_emb / (norm if norm else 1)

    similarities = emb_matrix @ query_emb
    top_k_idx   = np.argsort(similarities)[-TOP_K:][::-1]

    return "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{documents[i]}" for i in top_k_idx
    )

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NITRO P.A.W.S. On the Go",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 NITRO P.A.W.S. On the Go")
st.caption("Personal Abstract Witty adviSor-Lite · Powered by TTSH NITRO")
st.info(
    "Have an idea for your writing but nowhere near a clean computer? Turn to **NITRO P.A.W.S. On The Go!** "
    "Try out your idea with us here, then continue once ready on **NITRO P.A.W.S.**, "
    "accessible through [pair.gov.sg](https://pair.gov.sg) on a clean computer!",
    icon="💡",
)
st.divider()

gemini_client = load_clients()
sheet         = load_sheet()
system_prompt = load_system_prompt()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome = (
            "Hello! I'm **NITRO P.A.W.S.** — your Personal Abstract Witty adviSor-Lite, "
            "built by TTSH NITRO to help researchers like you turn ideas into strong abstracts. "
            "I'm not here to judge a finished product — I'm here to have a conversation and work through what you need together.\n\n"
            "> Please keep everything you share **deidentified and unclassified**. "
            "This conversation may be captured to improve the product.\n\n"
            "So — what are you working on? 🐾"
        )
        st.markdown(welcome)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Tell me about your research idea..."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = retrieve_context(gemini_client, user_input)

    augmented_input = (
        f"Reference context retrieved from academic literature and example abstracts:\n"
        f"<context>\n{context}\n</context>\n\n"
        f"User message:\n{user_input}"
    )

    contents = []
    for msg in st.session_state.messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])],
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=augmented_input)],
    ))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply  = ""
        last_error  = ""

        for attempt in range(3):
            model = FALLBACK_MODEL if attempt == 2 else GEN_MODEL
            try:
                full_reply = ""
                if attempt == 2:
                    placeholder.markdown("_Switching to backup model…_")
                for chunk in gemini_client.models.generate_content_stream(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.7,
                    ),
                    contents=contents,
                ):
                    if chunk.text:
                        full_reply += chunk.text
                        placeholder.markdown(full_reply + "▌")
                last_error = ""
                break
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    placeholder.markdown(f"_Model busy, retrying… (attempt {attempt + 2}/3)_")
                    time.sleep(4)

        if last_error:
            placeholder.markdown(
                f"⚠️ The model is temporarily unavailable. Please wait a moment and try again.\n\n"
                f"<details><summary>Error detail</summary>{last_error}</details>",
                unsafe_allow_html=True,
            )
        else:
            placeholder.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
    log_turn(sheet, st.session_state.session_id, user_input, full_reply)
