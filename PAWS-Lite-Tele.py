"""
NITRO P.A.W.S.-Lite — Telegram Bot
Personal Abstract Witty adviSor-Lite by TTSH NITRO
"""

import os
import uuid
import time
import json
import pathlib
import logging
import numpy as np
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import gspread
from google import genai
from google.genai import types
from google.oauth2 import service_account
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

KB_FILE            = pathlib.Path("knowledge_base.npz")
EMBED_MODEL        = "models/gemini-embedding-001"
GEN_MODEL          = "gemini-3.5-flash"
FALLBACK_MODEL     = "gemma-3-27b-it"
SYSTEM_PROMPT_FILE = pathlib.Path("PAWS_Gemini Markdown.md")
TOP_K              = 5
SHEET_NAME         = "PAWS Conversations"

# In-memory conversation store: {chat_id: {"session_id": str, "messages": [...]}}
conversations: dict = {}

# ── Resource loading ──────────────────────────────────────────────────────────

def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)

def load_knowledge_base():
    data       = np.load(KB_FILE, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32)
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return embeddings, data["documents"].tolist(), data["sources"].tolist()

def load_sheet():
    """Returns a gspread worksheet, or None if credentials are unavailable."""
    try:
        # Accept credentials as a JSON string in env var (for hosted deployments)
        # or fall back to a local service_account.json file
        raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        sa_info = json.loads(raw) if raw else json.loads(
            pathlib.Path("service_account.json").read_text()
        )
        creds = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=[
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        gc = gspread.authorize(creds)
        return gc.open(SHEET_NAME).sheet1
    except Exception as e:
        logger.warning("Google Sheets unavailable: %s", e)
        return None

def load_system_prompt():
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")

# ── RAG retrieval ─────────────────────────────────────────────────────────────

def retrieve_context(query: str) -> str:
    query_emb = np.array(
        gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
        ).embeddings[0].values,
        dtype=np.float32,
    )
    norm      = np.linalg.norm(query_emb)
    query_emb = query_emb / (norm if norm else 1)

    similarities = emb_matrix @ query_emb
    top_k_idx    = np.argsort(similarities)[-TOP_K:][::-1]

    return "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{documents[i]}" for i in top_k_idx
    )

# ── Google Sheets logging ─────────────────────────────────────────────────────

def log_turn(session_id: str, user_msg: str, paws_reply: str):
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

# ── Generation with fallback ──────────────────────────────────────────────────

def generate_reply(contents: list) -> tuple[str, str]:
    """Returns (reply_text, error_message). One of them will be empty."""
    last_error = ""
    for attempt in range(3):
        model = FALLBACK_MODEL if attempt == 2 else GEN_MODEL
        try:
            response = gemini_client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                ),
                contents=contents,
            )
            return response.text, ""
        except Exception as e:
            last_error = str(e)
            if attempt < 2:
                time.sleep(4)
    return "", last_error

# ── Telegram handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversations[chat_id] = {
        "session_id": str(uuid.uuid4()),
        "messages":   [],
    }
    welcome = (
        "Hello\\! I'm *NITRO P\\.A\\.W\\.S\\.-Lite* — your Personal Abstract Witty adviSor\\-Lite, "
        "built by TTSH NITRO to help researchers like you turn ideas into strong abstracts\\.\n\n"
        "_Please keep everything you share_ *deidentified and unclassified*\\. "
        "_This conversation may be captured to improve the product\\._\n\n"
        "So — what are you working on? 🐾"
    )
    await update.message.reply_text(welcome, parse_mode="MarkdownV2")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    conversations[chat_id] = {
        "session_id": str(uuid.uuid4()),
        "messages":   [],
    }
    await update.message.reply_text(
        "Conversation cleared\\. What would you like to work on? 🐾",
        parse_mode="MarkdownV2",
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id    = update.effective_chat.id
    user_input = update.message.text.strip()

    if not user_input:
        return

    # Initialise session if user skipped /start
    if chat_id not in conversations:
        conversations[chat_id] = {
            "session_id": str(uuid.uuid4()),
            "messages":   [],
        }

    session = conversations[chat_id]
    session["messages"].append({"role": "user", "content": user_input})

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # RAG retrieval
    rag_context = retrieve_context(user_input)
    augmented   = (
        f"Reference context retrieved from academic literature and example abstracts:\n"
        f"<context>\n{rag_context}\n</context>\n\n"
        f"User message:\n{user_input}"
    )

    # Build full conversation history for API
    contents = []
    for msg in session["messages"][:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])],
        ))
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=augmented)],
    ))

    reply, error = generate_reply(contents)

    if error:
        await update.message.reply_text(
            "⚠️ The model is temporarily unavailable\\. Please try again in a moment\\.",
            parse_mode="MarkdownV2",
        )
        return

    session["messages"].append({"role": "assistant", "content": reply})
    log_turn(session["session_id"], user_input, reply)

    # Send reply — fall back to plain text if Markdown parsing fails
    try:
        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(reply)

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gemini_client               = load_gemini_client()
    emb_matrix, documents, sources = load_knowledge_base()
    sheet                       = load_sheet()
    system_prompt               = load_system_prompt()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set.")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("NITRO P.A.W.S.-Lite Telegram bot is running.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)
