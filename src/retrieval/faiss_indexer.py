# =============================================================================
# Milestone 4 -- FAISS Vector Database & Retrieval System
# File: src/retrieval/faiss_indexer.py
# =============================================================================
# What this script does:
#   1. Loads 1337 embedding vectors from embeddings_data/embeddings.npy
#   2. Loads chunk metadata from embeddings_data/chunks.json
#   3. Builds a FAISS index from the vectors
#   4. Saves the index to index/faiss_index.faiss
#   5. Saves metadata.db (SQLite) mapping FAISS positions to chunk text + metadata
#   6. Runs a sample query to verify retrieval works end-to-end
# =============================================================================

import os
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

EMBEDDINGS_FILE   = "embeddings_data/embeddings.npy"
CHUNKS_FILE       = "embeddings_data/chunks.json"
FAISS_INDEX_FILE  = "index/faiss_index.faiss"
METADATA_DB_FILE  = "index/metadata.db"
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K             = 3


# =============================================================================
# STEP 1 -- Load embeddings and chunks
# =============================================================================
def load_embeddings_and_chunks():
    print("\n[STEP 1] Loading embeddings and chunks...")

    # Load embedding vectors
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")
    print(f"  Embeddings loaded : {embeddings.shape[0]} vectors, dimension {embeddings.shape[1]}")

    # Load chunks
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    chunks = data["chunks"]
    print(f"  Chunks loaded     : {len(chunks)} chunks")

    # Sanity check
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} vectors vs {len(chunks)} chunks. "
            "Re-run chunking_embedding.py to regenerate consistent files."
        )

    print("  Sanity check      : vectors and chunks count match")
    return embeddings, chunks


# =============================================================================
# STEP 2 -- Build FAISS index
# =============================================================================
def build_faiss_index(embeddings):
    print("\n[STEP 2] Building FAISS index...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)   # Exact L2 distance search
    index.add(embeddings)

    print(f"  Index type        : IndexFlatL2")
    print(f"  Dimension         : {dimension}")
    print(f"  Vectors indexed   : {index.ntotal}")
    return index


# =============================================================================
# STEP 3 -- Save FAISS index to disk
# =============================================================================
def save_faiss_index(index):
    print("\n[STEP 3] Saving FAISS index to disk...")

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)
    size_kb = os.path.getsize(FAISS_INDEX_FILE) / 1024
    print(f"  Saved to          : {FAISS_INDEX_FILE}")
    print(f"  File size         : {size_kb:.1f} KB")


# =============================================================================
# STEP 4 -- Save metadata to SQLite
# =============================================================================
def save_metadata_db(chunks):
    print("\n[STEP 4] Saving metadata to SQLite...")

    os.makedirs("index", exist_ok=True)

    # Remove existing db to start fresh
    if os.path.exists(METADATA_DB_FILE):
        os.remove(METADATA_DB_FILE)
        print(f"  Existing metadata.db removed — rebuilding fresh")

    conn = sqlite3.connect(METADATA_DB_FILE)
    cursor = conn.cursor()

    # Create table
    # faiss_id = integer position in FAISS index (0 to N-1)
    cursor.execute("""
        CREATE TABLE chunk_metadata (
            faiss_id        INTEGER PRIMARY KEY,
            chunk_id        TEXT,
            chunk_text      TEXT,
            document_name   TEXT,
            file_path       TEXT,
            page_number     INTEGER,
            chunk_number    INTEGER
        )
    """)

    # Insert one row per chunk
    rows = []
    for faiss_id, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        rows.append((
            faiss_id,
            chunk.get("chunk_id", f"chunk_{faiss_id:06d}"),
            chunk.get("text", ""),
            meta.get("document_name", ""),
            meta.get("file_path", ""),
            meta.get("page_number", 0),
            meta.get("chunk_number", faiss_id)
        ))

    cursor.executemany("""
        INSERT INTO chunk_metadata
        (faiss_id, chunk_id, chunk_text, document_name, file_path, page_number, chunk_number)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()

    # Verify
    count = cursor.execute("SELECT COUNT(*) FROM chunk_metadata").fetchone()[0]
    conn.close()

    print(f"  Saved to          : {METADATA_DB_FILE}")
    print(f"  Rows inserted     : {count}")


# =============================================================================
# STEP 5 -- Sample query to verify end-to-end retrieval
# =============================================================================
def run_sample_query(query_text):
    print("\n[STEP 5] Running sample query to verify retrieval...")
    print(f"  Query             : \"{query_text}\"")

    # Load model
    print(f"  Loading model     : {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed the query
    query_vector = model.encode([query_text], convert_to_numpy=True).astype("float32")

    # Load FAISS index from disk
    index = faiss.read_index(FAISS_INDEX_FILE)

    # Search
    distances, faiss_ids = index.search(query_vector, TOP_K)

    # Retrieve chunks from metadata.db
    conn = sqlite3.connect(METADATA_DB_FILE)
    cursor = conn.cursor()

    print(f"\n  Top {TOP_K} results:\n")
    print("  " + "-" * 70)

    for rank, (faiss_id, distance) in enumerate(zip(faiss_ids[0], distances[0]), start=1):
        row = cursor.execute("""
            SELECT chunk_id, document_name, page_number, chunk_text
            FROM chunk_metadata
            WHERE faiss_id = ?
        """, (int(faiss_id),)).fetchone()

        if row:
            chunk_id, doc_name, page_num, chunk_text = row
            print(f"  Rank {rank}")
            print(f"  Chunk ID      : {chunk_id}")
            print(f"  Document      : {doc_name}")
            print(f"  Page          : {page_num}")
            print(f"  Distance      : {distance:.4f}")
            print(f"  Text preview  : {chunk_text[:200]}...")
            print("  " + "-" * 70)

    conn.close()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  Milestone 4 -- FAISS Vector Database & Retrieval System")
    print("=" * 70)

    # Step 1 -- Load
    embeddings, chunks = load_embeddings_and_chunks()

    # Step 2 -- Build index
    index = build_faiss_index(embeddings)

    # Step 3 -- Save FAISS index
    save_faiss_index(index)

    # Step 4 -- Save metadata
    save_metadata_db(chunks)

    # Step 5 -- Verify with sample query
    run_sample_query("What are the Basel III capital requirements?")

    print("\n[DONE] Milestone 4 complete.")
    print(f"  FAISS index : {FAISS_INDEX_FILE}")
    print(f"  Metadata DB : {METADATA_DB_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
