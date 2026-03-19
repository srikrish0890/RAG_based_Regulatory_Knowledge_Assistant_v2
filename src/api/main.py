# =============================================================================
# Milestone 6 -- Multi-Turn Conversation with Context-Aware Query Rewriting
# File: src/api/main.py
# =============================================================================
# What this script does:
#   1. Exposes POST /ask and GET /health endpoints via FastAPI
#   2. Implements LLM Pool (gemini-3.1-flash-lite-preview → gemini-3-flash-preview)
#   3. Integrates RAG pipeline (FAISS retrieval + LLM generation)
#   4. Supports multi-turn conversations via client-side history (sliding window: 5 turns)
#   5. Rewrites each query using conversation history before FAISS search (Gemini Call 1)
#   6. Generates context-aware answers using retrieved chunks + history (Gemini Call 2)
#   7. Detects rate limit errors and returns a distinct error_type for the UI
#   8. Serves index.html web interface
#   9. Logs all queries, LLM routing decisions, and errors
# =============================================================================

import os
import sqlite3
import logging
import numpy as np
import faiss
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from sentence_transformers import SentenceTransformer
from google import genai

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

FAISS_INDEX_FILE  = "index/faiss_index.faiss"
METADATA_DB_FILE  = "index/metadata.db"
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
PRIMARY_MODEL     = "gemini-3.1-flash-lite-preview"
FALLBACK_MODEL    = "gemini-3-flash-preview"
TOP_K             = 3
HISTORY_WINDOW    = 5   # Number of past Q&A turns to include in each request

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rag_query.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FastAPI app + templates
# -----------------------------------------------------------------------------
app = FastAPI(title="RAG Regulatory Explainer", version="2.0.0")
templates = Jinja2Templates(directory="src/api/templates")

# -----------------------------------------------------------------------------
# Load FAISS index + embedding model at startup
# -----------------------------------------------------------------------------
logger.info("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
logger.info(f"FAISS index loaded: {faiss_index.ntotal} vectors")

logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded")

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# =============================================================================
# Request / Response models
# =============================================================================
class ConversationTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    question: str
    conversation_history: Optional[list[ConversationTurn]] = []


class SourceCitation(BaseModel):
    chunk_id: str
    document_name: str
    page_number: int
    text_preview: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    model_used: str
    sources: list[SourceCitation]
    timestamp: str
    error_type: Optional[str] = None   # "rate_limit" | "unavailable" | None


# =============================================================================
# Helper -- Detect rate limit errors from Gemini exception messages
# =============================================================================
def is_rate_limit_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return any(keyword in error_str for keyword in [
        "429", "quota", "rate limit", "resource exhausted", "too many requests"
    ])


# =============================================================================
# LLM Pool -- Try primary, fallback on failure
# Raises RateLimitError if both models hit quota
# Raises RuntimeError if both models fail for other reasons
# =============================================================================
class RateLimitError(Exception):
    pass


def call_llm_pool(prompt: str) -> tuple[str, str]:
    """
    Returns (answer_text, model_used).
    Tries PRIMARY first, falls back to FALLBACK on any error.
    Raises RateLimitError if both models are quota-exhausted.
    Raises RuntimeError if both models fail for other reasons.
    """
    primary_rate_limited = False
    fallback_rate_limited = False

    # --- Try Primary ---
    try:
        logger.info(f"Calling PRIMARY model: {PRIMARY_MODEL}")
        response = gemini_client.models.generate_content(
            model=PRIMARY_MODEL,
            contents=prompt
        )
        logger.info("PRIMARY model responded successfully")
        return response.text, PRIMARY_MODEL

    except Exception as e:
        if is_rate_limit_error(e):
            logger.warning(f"PRIMARY model rate limited: {e}")
            primary_rate_limited = True
        else:
            logger.warning(f"PRIMARY model failed: {e}")
        logger.info(f"Routing to FALLBACK model: {FALLBACK_MODEL}")

    # --- Try Fallback ---
    try:
        response = gemini_client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt
        )
        logger.info("FALLBACK model responded successfully")
        return response.text, FALLBACK_MODEL

    except Exception as e:
        if is_rate_limit_error(e):
            logger.warning(f"FALLBACK model rate limited: {e}")
            fallback_rate_limited = True
        else:
            logger.error(f"FALLBACK model failed: {e}")

    # --- Both failed ---
    if primary_rate_limited and fallback_rate_limited:
        raise RateLimitError("Both models are rate limited. Please wait and try again.")

    raise RuntimeError("Both PRIMARY and FALLBACK models failed.")


# =============================================================================
# Milestone 6 -- Step 1: Rewrite query using conversation history (Gemini Call 1)
# =============================================================================
def rewrite_query(question: str, history: list[ConversationTurn]) -> str:
    """
    If there is no conversation history, return the original question unchanged.
    Otherwise, call Gemini to produce an enriched standalone search query
    that captures full context from the conversation so far.
    """
    if not history:
        logger.info("No conversation history — using original question for FAISS search")
        return question

    # Build conversation history string
    history_text = ""
    for turn in history:
        label = "User" if turn.role == "user" else "Assistant"
        history_text += f"{label}: {turn.content}\n"

    rewrite_prompt = f"""You are a search query optimizer for a regulatory document retrieval system.

Given the conversation history below and a new follow-up question, rewrite the question 
into a single complete standalone search query that captures the full context.
The rewritten query will be used to search a vector database of regulatory documents.
Return ONLY the rewritten query — no explanation, no preamble, no punctuation other than 
what is needed in the query itself.

CONVERSATION HISTORY:
{history_text}
NEW QUESTION:
{question}

REWRITTEN STANDALONE QUERY:"""

    try:
        rewritten, model_used = call_llm_pool(rewrite_prompt)
        rewritten = rewritten.strip()
        logger.info(f"Query rewritten using {model_used}: '{rewritten}'")
        return rewritten
    except RateLimitError:
        raise   # propagate so /ask endpoint can return rate_limit error_type
    except RuntimeError:
        logger.warning("Query rewrite failed — falling back to original question")
        return question


# =============================================================================
# RAG Pipeline -- Step 2: Retrieve top-k chunks from FAISS
# =============================================================================
def retrieve_chunks(query: str) -> list[dict]:
    logger.info(f"Retrieving chunks for query: {query}")

    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    distances, faiss_ids = faiss_index.search(query_vector, TOP_K)

    conn = sqlite3.connect(METADATA_DB_FILE)
    cursor = conn.cursor()

    chunks = []
    for faiss_id, distance in zip(faiss_ids[0], distances[0]):
        row = cursor.execute("""
            SELECT chunk_id, document_name, page_number, chunk_text, file_path
            FROM chunk_metadata
            WHERE faiss_id = ?
        """, (int(faiss_id),)).fetchone()

        if row:
            chunks.append({
                "chunk_id"      : row[0],
                "document_name" : row[1],
                "page_number"   : row[2],
                "chunk_text"    : row[3],
                "file_path"     : row[4],
                "distance"      : float(distance)
            })

    conn.close()
    logger.info(f"Retrieved {len(chunks)} chunks from FAISS")
    return chunks


# =============================================================================
# RAG Pipeline -- Step 3: Build prompt with chunks + conversation history
# =============================================================================
def build_prompt(question: str, chunks: list[dict], history: list[ConversationTurn]) -> str:

    # Context blocks from retrieved chunks
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}]\n"
            f"Document : {chunk['document_name']}\n"
            f"Page     : {chunk['page_number']}\n"
            f"Content  : {chunk['chunk_text']}\n"
        )
    context = "\n---\n".join(context_blocks)

    # Conversation history string (if any)
    history_section = ""
    if history:
        history_lines = ""
        for turn in history:
            label = "User" if turn.role == "user" else "Assistant"
            history_lines += f"{label}: {turn.content}\n"
        history_section = f"""
CONVERSATION HISTORY (for context only — do not repeat previous answers):
{history_lines}
"""

    prompt = f"""You are a regulatory expert assistant helping bank professionals understand regulatory documents.

Use ONLY the context provided below to answer the question. Do not use any external knowledge.
Always cite the source document and page number in your answer.
If the context does not contain enough information to answer, say so clearly.
{history_section}
CONTEXT:
{context}

CURRENT QUESTION:
{question}

ANSWER:
Provide a clear, professional explanation in 2-3 paragraphs.
At the end, list the sources you used as: Source: [Document Name], Page [X]
"""
    return prompt


# =============================================================================
# Endpoints
# =============================================================================

# --- Health check ---
@app.get("/health")
def health_check():
    return {
        "status"          : "ok",
        "faiss_vectors"   : faiss_index.ntotal,
        "embedding_model" : EMBEDDING_MODEL,
        "primary_llm"     : PRIMARY_MODEL,
        "fallback_llm"    : FALLBACK_MODEL,
        "history_window"  : HISTORY_WINDOW
    }


# --- Web interface ---
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# --- Main RAG endpoint ---
@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    question = payload.question.strip()
    # Apply sliding window — keep only the last HISTORY_WINDOW turns
    history  = (payload.conversation_history or [])[-HISTORY_WINDOW:]

    logger.info(f"Received question: {question} | History turns: {len(history)}")

    if not question:
        return AnswerResponse(
            question   = question,
            answer     = "Please enter a valid question.",
            model_used = "none",
            sources    = [],
            timestamp  = datetime.now().isoformat()
        )

    # ------------------------------------------------------------------
    # Step 1 -- Rewrite query using conversation history (Gemini Call 1)
    # ------------------------------------------------------------------
    try:
        enriched_query = rewrite_query(question, history)
    except RateLimitError:
        logger.warning("Rate limit hit during query rewrite")
        return AnswerResponse(
            question   = question,
            answer     = "Gemini rate limit reached. Please wait 1–2 minutes and try again.",
            model_used = "none",
            sources    = [],
            timestamp  = datetime.now().isoformat(),
            error_type = "rate_limit"
        )

    # ------------------------------------------------------------------
    # Step 2 -- Retrieve top-k chunks from FAISS using enriched query
    # ------------------------------------------------------------------
    chunks = retrieve_chunks(enriched_query)

    # ------------------------------------------------------------------
    # Step 3 -- Build prompt with chunks + conversation history
    # ------------------------------------------------------------------
    prompt = build_prompt(question, chunks, history)

    # ------------------------------------------------------------------
    # Step 4 -- Generate answer (Gemini Call 2)
    # ------------------------------------------------------------------
    try:
        answer, model_used = call_llm_pool(prompt)
    except RateLimitError:
        logger.warning("Rate limit hit during answer generation")
        return AnswerResponse(
            question   = question,
            answer     = "Gemini rate limit reached. Please wait 1–2 minutes and try again.",
            model_used = "none",
            sources    = [],
            timestamp  = datetime.now().isoformat(),
            error_type = "rate_limit"
        )
    except RuntimeError as e:
        logger.error(str(e))
        return AnswerResponse(
            question   = question,
            answer     = "Sorry, the system is currently unavailable. Please try again later.",
            model_used = "none",
            sources    = [],
            timestamp  = datetime.now().isoformat(),
            error_type = "unavailable"
        )

    # ------------------------------------------------------------------
    # Step 5 -- Build compact citations
    # ------------------------------------------------------------------
    sources = [
        SourceCitation(
            chunk_id      = c["chunk_id"],
            document_name = c["document_name"],
            page_number   = c["page_number"],
            text_preview  = c["chunk_text"][:200] + "..."
        )
        for c in chunks
    ]

    logger.info(f"Answer generated using: {model_used} | Enriched query: '{enriched_query}'")

    return AnswerResponse(
        question   = question,
        answer     = answer,
        model_used = model_used,
        sources    = sources,
        timestamp  = datetime.now().isoformat(),
        error_type = None
    )
