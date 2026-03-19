# Chunking & Embedding Generation Script - Usage Guide

## What This Script Does

The `chunking_embedding.py` script takes your processed PDF text and:
- ✓ Splits it into manageable chunks (~400 tokens each)
- ✓ Creates overlapping chunks (50 tokens) to preserve context
- ✓ Generates semantic embeddings using Sentence Transformers
- ✓ Saves chunks and embeddings ready for FAISS vector database

## Prerequisites

Make sure you have:
```bash
pip install sentence-transformers numpy
```

## Required Input

You must have already run `pdf_ingestion.py` which created:
```
processed_data/
└── processed_documents.json
```

## How to Run

```bash
# From your project root
python src/ingestion/chunking_embedding.py
```

Or if placed in root:
```bash
python chunking_embedding.py
```

## What It Does Step-by-Step

### 1. Loads Processed Data
Reads `processed_data/processed_documents.json`

### 2. Creates Chunks
- Splits text into ~400 token chunks
- 50 token overlap between chunks (preserves context)
- Tries to break at sentence boundaries
- Attaches metadata to each chunk

### 3. Generates Embeddings
- Uses `all-MiniLM-L6-v2` model
- Converts each chunk to 384-dimensional vector
- Shows progress bar during generation

### 4. Saves Output
Creates `embeddings_data/` folder with:
- `chunks.json` - All chunks with metadata
- `embeddings.npy` - Embeddings as numpy array
- `embeddings.pkl` - Embeddings as pickle file
- `chunking_report.txt` - Summary statistics

## Output Structure

### chunks.json
```json
{
  "metadata": {
    "total_chunks": 450,
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384,
    "chunk_size": 400,
    "chunk_overlap": 50
  },
  "chunks": [
    {
      "chunk_id": "chunk_000001",
      "text": "Basel III framework requires...",
      "metadata": {
        "document_name": "basel_framework",
        "page_number": 5,
        "chunk_number": 0,
        "chunk_tokens": 387
      }
    },
    ...
  ]
}
```

### embeddings.npy
Numpy array of shape: `(num_chunks, 384)`

Each row is the embedding vector for the corresponding chunk.

## Configuration Options

You can modify these parameters in the script:

```python
chunker = ChunkingAndEmbedding(
    input_file="processed_data/processed_documents.json",
    output_folder="embeddings_data",
    model_name="all-MiniLM-L6-v2",  # Change model here
    chunk_size=400,                  # Adjust chunk size
    chunk_overlap=50                 # Adjust overlap
)
```

### Alternative Embedding Models:
- `all-MiniLM-L6-v2` - Fast, 384 dimensions (default)
- `all-mpnet-base-v2` - Better quality, 768 dimensions
- `multi-qa-MiniLM-L6-cos-v1` - Optimized for Q&A

## Expected Performance

For your 165 pages (~468k characters):
- **Chunks created:** ~450-600 chunks
- **Processing time:** 2-5 minutes
- **Memory usage:** ~500 MB

## What Happens Next

After running this script, you'll have:
- ✓ Text split into semantic chunks
- ✓ Embeddings generated for each chunk
- ✓ Data ready for FAISS vector database

The next script will:
1. Create FAISS index
2. Insert embeddings
3. Enable similarity search

## Troubleshooting

**Model download slow:**
- First run downloads model (~80 MB)
- Subsequent runs use cached model

**Out of memory:**
- Reduce batch_size in `generate_embeddings()`
- Use smaller model: `all-MiniLM-L6-v2`

**Wrong input file:**
- Make sure `pdf_ingestion.py` ran successfully
- Check `processed_data/processed_documents.json` exists
