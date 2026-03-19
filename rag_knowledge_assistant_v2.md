# RAG-Based Regulatory Explainer -- Implementation Plan (FastAPI Version)

## Goal

Build a Retrieval-Augmented Generation (RAG) system that allows bank
users to ask questions about regulatory documents (Basel, RBI, internal
policies) and receive clear explanations with source citations.

Key capabilities:
- Ingest regulatory PDFs
- Convert text into embeddings
- Store embeddings in a vector database
- Retrieve relevant text
- Use an LLM to generate explanations
- Present answers through a FastAPI-based web application
- Support multiple LLM providers through an LLM pool
- Support multi-turn conversations with context-aware query rewriting

---

# Milestone 1 -- Project Setup & Environment ✅

### Task 1.1 -- Define Scope

- Identify document sources (Basel, RBI, internal regulatory documentation)
- Define target users (risk analysts, compliance, audit teams)
- Define output format (explanation + source citation)

### Task 1.2 -- Create Project Structure

    rag-regulatory-explainer/
    │
    ├── data/
    │
    ├── src/
    │   ├── api/
    │   │   └── main.py
    │   ├── ingestion/
    │   ├── embeddings/
    │   ├── retrieval/
    │   ├── llm_router/
    │   └── rag_pipeline/
    │
    ├── index/
    │
    ├── requirements.txt
    ├── .env
    └── README.md

### Task 1.3 -- Setup Python Environment

Recommended packages:

- fastapi
- uvicorn
- sentence-transformers
- faiss-cpu
- pypdf
- python-dotenv
- langchain (optional orchestration)

### Task 1.4 -- Configure API Keys

- Store API keys securely in `.env`
- Configure keys for multiple LLM providers

Example providers:
- Gemini (primary: `gemini-3.1-flash-lite-preview`, fallback: `gemini-3-flash-preview`)

---

# Milestone 2 -- Document Ingestion & Processing ✅

### Task 2.1 -- Collect Regulatory Documents

- Download Basel documents
- Download RBI circulars
- Organize files inside `data/regulatory_docs`

### Task 2.2 -- Parse PDFs

- Extract text from PDFs
- Handle page structures
- Convert content into raw text

Tools: PyPDF

### Task 2.3 -- Metadata Extraction

Extract and store metadata:
- document name
- page number
- section header

### Task 2.4 -- Text Cleaning

- Remove page headers
- Remove formatting artifacts
- Normalize whitespace

### Task 2.5 -- Incremental Ingestion (Registry-Based)

Only process new or changed documents on each run.

#### File Registry

Maintain a SQLite table `processed_files`:

    processed_files
    ├── filename          (name of the PDF file)
    ├── file_hash         (MD5/SHA256 hash of file content)
    ├── processed_at      (timestamp of last processing)
    ├── chunk_count       (number of chunks generated)
    └── status            (success / failed)

#### Ingestion Logic

1. Scan input folder for all PDF files
2. Compute file hash for each file
3. Compare against registry:
   - File not in registry → new file, process it
   - File in registry, hash matches → skip (already processed)
   - File in registry, hash changed → document updated, trigger rebuild
4. Write to registry only after successfully inserting into FAISS

#### Run Modes

    python ingest.py --incremental   ← process only new/changed files
    python ingest.py --rebuild       ← wipe index and reprocess all files

#### Notes

- Always use the same embedding model (`all-MiniLM-L6-v2`) across all runs
- FAISS does not support deletion of individual vectors — use `--rebuild` if documents are removed or updated
- After any ingestion run, re-upload `index.faiss` and `metadata.db` to the hosting environment

---

# Milestone 3 -- Chunking & Embedding Generation ✅

### Task 3.1 -- Chunk Documents

Recommended settings:
- Chunk size: 300–500 tokens
- Overlap: 50–100 tokens

**What overlap means:** Consecutive chunks share tokens at their boundary.
With 400-token chunks and 50-token overlap, Chunk 2 starts 350 tokens
after Chunk 1 begins — ensuring context at boundaries is never lost.

### Task 3.2 -- Attach Metadata

Each chunk includes:
- chunk text
- document source
- page number
- section name

### Task 3.3 -- Generate Embeddings

- Convert each chunk into a 384-dimension embedding vector
- Model: `all-MiniLM-L6-v2` (Sentence Transformers)

Output structure:

    chunk_text
    embedding_vector (384 dimensions)
    metadata

Output files:
- `chunks.json` — all text snippets with metadata
- `embeddings.npy` — matrix of shape [num_chunks × 384]
- `embeddings.pkl` — pickled embeddings for reuse

### Task 3.4 -- Batch Processing

- Process documents in batches
- Log embedding generation progress

---

# Milestone 4 -- Vector Database & Retrieval System ✅

### Task 4.1 -- Setup FAISS Index

FAISS serves two roles:
- **Build time:** Takes all chunk vectors and organizes them into a
  searchable index file (`faiss_index.faiss`) — one-time setup
- **Query time:** Loads the saved index and searches it to return the
  top-k closest vectors to a query vector in milliseconds

### Task 4.2 -- Persist Vector Index

- Save FAISS index to disk as `faiss_index.faiss`
- Reload at application startup

Output files:
- `index/faiss_index.faiss`
- `index/metadata.db` (chunk metadata for lookup after retrieval)

### Task 4.3 -- Query Retrieval Pipeline

1. Convert user question to embedding vector (384 dimensions)
2. FAISS searches index for top-k closest vectors
3. Retrieve chunk text from `metadata.db` using returned IDs

Recommended configuration: `top_k = 3`

### Task 4.4 -- Context Construction

- Combine retrieved chunks
- Prepare prompt context for LLM

---

# Milestone 5 -- FastAPI Application & LLM Pool ✅

### Task 5.1 -- FastAPI Backend

Endpoints:

    POST /ask
    GET /health

Single-turn user flow:
1. User sends question
2. System embeds question and performs FAISS retrieval
3. Top-k chunks sent to LLM with question
4. Answer + citations returned via API

### Task 5.2 -- LLM Pool (LLM Router)

    Primary LLM  →  gemini-3.1-flash-lite-preview
    Fallback LLM →  gemini-3-flash-preview

Routing behavior:
- Attempt primary LLM
- On failure or quota exhaustion, automatically route to fallback
- If both fail, return a user-friendly error message

### Task 5.3 -- RAG Pipeline Integration

    User Question
          ↓
    Query Embedding
          ↓
    FAISS Vector Search
          ↓
    Retrieve Top-3 Chunks
          ↓
    Send Context to LLM Pool
          ↓
    LLM Generates Explanation
          ↓
    Return Answer + Citations

### Task 5.4 -- Basic Web Interface

Single-turn interface:
- Textarea for question input
- Answer card displaying explanation
- Source citations (document name, page number, chunk preview)
- Sample question buttons
- Model badge showing which LLM responded

### Task 5.5 -- Logging & Monitoring

- Log user queries
- Track LLM usage and routing decisions
- Log API failures
- Output: `rag_query.log`

---

# Milestone 6 -- Multi-Turn Conversation with Context Rewriting ✅

### Overview

Upgrade the system to support multi-turn conversations where each
follow-up question is understood in the context of previous exchanges.
Uses a two-call Gemini strategy per user message: one call to rewrite
the query with context, one call to generate the answer.

### Task 6.1 -- Query Rewriting (Gemini Call 1)

Before searching FAISS, send the current question plus conversation
history to Gemini to produce an enriched standalone query.

    Prompt sent to Gemini (Call 1):
    "Given this conversation so far: [last 5 turns]
     And the new question: [current question]
     Rewrite this as a complete standalone search query
     that captures the full context."

The rewritten query is then used for FAISS search instead of the raw
question. Uses the same LLM pool (primary → fallback) as the answer call.

### Task 6.2 -- Context-Aware Answer Generation (Gemini Call 2)

After FAISS retrieval, send to Gemini:
- Rewritten query
- Retrieved top-3 chunks
- Full conversation history (last 5 turns)

Gemini generates an answer grounded in the retrieved chunks, with
awareness of what was discussed earlier in the conversation.

### Task 6.3 -- Conversation History Management

**Storage:** Client-side (browser JavaScript array)

Rationale:
- Hosting platforms (Render, PythonAnywhere) may restart or spin down
  servers, wiping server-side memory
- Client-side storage survives server restarts with zero configuration
- Works identically on any hosting platform

**Sliding window:** Last 5 Q&A turns sent with every request

- Conversation can continue indefinitely
- Only the most recent 5 turns are passed to Gemini each time
- Older turns are dropped from the window but remain visible in the UI

**History format sent in request body:**

    conversation_history: [
      { role: "user",      content: "What is market risk?" },
      { role: "assistant", content: "Market risk is..." },
      { role: "user",      content: "How is it measured?" },
      { role: "assistant", content: "It is measured using..." }
    ]

### Task 6.4 -- Updated API Contract

`POST /ask` updated request body:

    {
      "question": "string",
      "conversation_history": [ { "role": "string", "content": "string" } ]
    }

Response model unchanged.

### Task 6.5 -- Rate Limit Handling

Free tier limits for Gemini (approximate):
- 15 requests per minute
- 1 million tokens per day

With 2 Gemini calls per user question, effective throughput is halved.
For ~20 concurrent users this remains within free tier limits.

On rate limit error, the API returns a specific error response and the
UI displays:

    "Gemini rate limit reached. Please wait 1–2 minutes and try again."

### Task 6.6 -- ChatGPT-Style Frontend

Replace single-answer UI with a scrollable conversation thread:

- Chat thread area: messages bubble up as conversation grows
- User messages: right-aligned bubbles
- Assistant messages: left-aligned bubbles with answer text
- Compact citations: one line per source at the bottom of each
  assistant bubble (e.g. `📄 BIS Minimum Capital Requirements — Page 21`)
- Input box: pinned at bottom of screen
- Two-stage loading indicator:
  - Stage 1: "Understanding your question..."
  - Stage 2: "Generating answer..."
- "New Conversation" button: clears thread and resets history
- Rate limit error displayed inline in chat thread

### Task 6.7 -- Updated End-to-End Flow

    User sends message
          ↓
    Browser appends to conversation_history
          ↓
    POST /ask { question, last 5 turns of history }
          ↓
    Gemini Call 1 (LLM Pool) → Rewrite query with context
          ↓
    Embed rewritten query → FAISS search → Top-3 chunks
          ↓
    Gemini Call 2 (LLM Pool) → Answer using chunks + history
          ↓
    Response: answer + compact citations
          ↓
    Browser appends Q&A to history, displays in chat thread

---

# Hosting Notes

### Render / PythonAnywhere Deployment

- Use client-side conversation history (no server state required)
- After any document ingestion run, re-upload `faiss_index.faiss` and
  `metadata.db` to the hosting environment
- Set `GEMINI_API_KEY` as an environment variable in the hosting dashboard
- Use `uvicorn src.api.main:app` as the start command

---

# Final System Workflow

    Regulatory PDFs
          ↓
    Incremental Ingestion (registry + file hash check)
          ↓
    Text Extraction (new/changed files only)
          ↓
    Chunking (400 tokens, 50-token overlap)
          ↓
    Embedding Generation (all-MiniLM-L6-v2, 384 dimensions)
          ↓
    FAISS Vector Index (build once, query many times)
          ↓
    FastAPI Backend
          ↓
    User Question + Conversation History
          ↓
    Gemini Call 1: Query Rewriting (LLM Pool)
          ↓
    FAISS Search with Enriched Query → Top-3 Chunks
          ↓
    Gemini Call 2: Answer Generation (LLM Pool)
          ↓
    Answer + Compact Citations returned to Chat UI

---

# Expected Outcome

A production-style regulatory explainer system that:

- Indexes regulatory documents with incremental ingestion support
- Supports semantic search via FAISS
- Generates explanations using RAG
- Maintains multi-turn conversation context via query rewriting
- Provides compact source citations per response
- Runs through a FastAPI web service with a ChatGPT-style UI
- Uses an LLM pool (primary + fallback) to maintain reliability
- Handles free-tier rate limits gracefully with user-facing messages
- Deployable on Render or PythonAnywhere with minimal configuration
