"""
Chunking & Embedding Generation Script for RAG Regulatory Explainer
Milestone 3: Chunking & Embedding Generation (OPTIMIZED VERSION)

This script:
- Loads processed PDF data from JSON
- Splits text into chunks (400 tokens with 50 token overlap)
- Attaches metadata to each chunk
- Generates embeddings using Sentence Transformers
- Saves chunks with embeddings for FAISS indexing

OPTIMIZATIONS:
- Better memory management
- Fixed infinite loop issues
- More frequent progress updates
- Smaller batch sizes for stability
- Early stopping on errors
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import gc  # Garbage collector for memory management


class ChunkingAndEmbedding:
    def __init__(
        self, 
        input_folder: str = "processed_data",
        output_folder: str = "embeddings_data",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50
    ):
        """
        Initialize chunking and embedding pipeline
        
        Args:
            input_folder: Folder containing per-document JSON files
            output_folder: Folder to save chunks and embeddings
            model_name: Name of sentence transformer model
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully (embedding dimension: {self.model.get_sentence_embedding_dimension()})\n")
        
        self.chunks = []
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation: 1 token ≈ 4 characters)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into overlapping chunks (OPTIMIZED)
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        chunk_size_chars = self.chunk_size * 4  # Approximate characters
        overlap_chars = self.chunk_overlap * 4
        
        # Validate overlap is not too large
        if overlap_chars >= chunk_size_chars:
            overlap_chars = chunk_size_chars // 2
            print(f"  ⚠ Overlap too large, reducing to {overlap_chars // 4} tokens")
        
        start_idx = 0
        chunk_num = 0
        max_chunks = 1000  # Safety limit per page
        
        while start_idx < len(text) and chunk_num < max_chunks:
            # Get chunk
            end_idx = min(start_idx + chunk_size_chars, len(text))
            
            # Safety check - ensure we're making progress
            if end_idx <= start_idx:
                break
            
            chunk_text = text[start_idx:end_idx]
            
            # Try to end at sentence boundary if possible
            if end_idx < len(text):
                # Look for sentence endings
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                # Only adjust if boundary is reasonable
                if boundary > len(chunk_text) * 0.5:  # Reduced from 0.7 for better splits
                    chunk_text = chunk_text[:boundary + 1]
                    end_idx = start_idx + boundary + 1
            
            # Skip very short chunks
            if len(chunk_text.strip()) < 50:
                # If we're at the end, break
                if end_idx >= len(text) - 100:
                    break
                # Otherwise skip this chunk and move forward
                start_idx = end_idx
                continue
            
            # Create chunk with metadata
            chunk_data = {
                "text": chunk_text.strip(),
                "metadata": {
                    **metadata,
                    "chunk_number": chunk_num,
                    "start_char": start_idx,
                    "end_char": end_idx,
                    "chunk_tokens": self.estimate_tokens(chunk_text)
                }
            }
            
            chunks.append(chunk_data)
            chunk_num += 1
            
            # Move to next chunk with overlap
            # CRITICAL FIX: Ensure we always move forward
            next_start = end_idx - overlap_chars
            if next_start <= start_idx:
                # If overlap would cause us to not progress, just move forward
                next_start = start_idx + chunk_size_chars // 2
            
            start_idx = next_start
        
        if chunk_num >= max_chunks:
            print(f"  ⚠ Hit max chunks limit ({max_chunks}) for page")
        
        return chunks

    def load_processed_data(self) -> List[Dict]:
        """
        Load processed documents from all JSON files in processed_data/ folder.
        Each file corresponds to one document (e.g. basel_iii.json)

        Returns:
            List of all processed pages across all documents
        """
        json_files = list(self.input_folder.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in: {self.input_folder}")

        print(f"Found {len(json_files)} document(s) in {self.input_folder}\n")

        all_pages = []
        total_chars = 0

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            pages = data['pages']
            all_pages.extend(pages)
            total_chars += data['summary']['total_characters']

            print(f"  ✓ Loaded: {json_file.name} — {data['summary']['total_pages']} pages")

        print(f"\n  • Total pages loaded : {len(all_pages)}")
        print(f"  • Total characters   : {total_chars:,}\n")

        return all_pages
    

    
    def create_chunks(self, pages: List[Dict]) -> List[Dict]:
        """
        Create chunks from all pages (OPTIMIZED with frequent updates)
        
        Args:
            pages: List of processed pages
            
        Returns:
            List of all chunks
        """
        print("=" * 60)
        print("CREATING CHUNKS")
        print("=" * 60)
        print(f"Processing {len(pages)} pages...\n")
        
        all_chunks = []
        
        for page_idx, page in enumerate(pages):
            text = page['text']
            metadata = page['metadata']
            
            # Create chunks for this page
            page_chunks = self.chunk_text(text, metadata)
            all_chunks.extend(page_chunks)
            
            # MORE FREQUENT PROGRESS UPDATES (every 10 pages instead of 20)
            if (page_idx + 1) % 10 == 0:
                print(f"  ✓ Processed {page_idx + 1}/{len(pages)} pages | {len(all_chunks)} chunks created so far...")
            
            # Memory cleanup every 50 pages
            if (page_idx + 1) % 50 == 0:
                gc.collect()
        
        print(f"\n{'=' * 60}")
        print(f"✓ CHUNKING COMPLETED")
        print(f"{'=' * 60}")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Total pages: {len(pages)}")
        print(f"  Average chunks per page: {len(all_chunks) / len(pages):.1f}\n")
        
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> Tuple[List[Dict], np.ndarray]:
        """
        Generate embeddings for all chunks (OPTIMIZED batch size)
        
        Args:
            chunks: List of chunks
            
        Returns:
            Tuple of (chunks with IDs, embeddings array)
        """
        print("=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Total chunks to embed: {len(chunks)}")
        print(f"Batch size: 16 (reduced for stability)\n")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in smaller batches for stability
        print("Generating embeddings... (this may take a few minutes)")
        print("Progress bar will appear below:\n")
        
        embeddings = self.model.encode(
            texts,
            batch_size=16,  # REDUCED from 32 for better stability
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # Can enable for cosine similarity later
        )
        
        # Add unique IDs to chunks
        for idx, chunk in enumerate(chunks):
            chunk['chunk_id'] = f"chunk_{idx:06d}"
        
        print(f"\n{'=' * 60}")
        print(f"✓ EMBEDDING GENERATION COMPLETED")
        print(f"{'=' * 60}")
        print(f"  Chunks embedded: {len(chunks)}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Embeddings array shape: {embeddings.shape}\n")
        
        return chunks, embeddings
    
    def save_chunks_and_embeddings(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Save chunks and embeddings to disk
        
        Args:
            chunks: List of chunks with metadata
            embeddings: Numpy array of embeddings
        """
        print("=" * 60)
        print("SAVING DATA")
        print("=" * 60)
        
        # Save chunks as JSON
        chunks_file = self.output_folder / "chunks.json"
        print(f"Saving chunks to JSON...")
        
        chunks_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "embedding_model": self.model_name,
                "embedding_dimension": embeddings.shape[1],
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "created_date": datetime.now().isoformat()
            },
            "chunks": chunks
        }
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved chunks to: {chunks_file}")
        
        # Save embeddings as numpy array
        embeddings_file = self.output_folder / "embeddings.npy"
        print(f"Saving embeddings...")
        np.save(embeddings_file, embeddings)
        
        print(f"✓ Saved embeddings to: {embeddings_file}")
        
        # Save embeddings as pickle (alternative format)
        embeddings_pkl = self.output_folder / "embeddings.pkl"
        with open(embeddings_pkl, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"✓ Saved embeddings (pickle) to: {embeddings_pkl}\n")
    
    def generate_report(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Generate summary report
        
        Args:
            chunks: List of chunks
            embeddings: Embeddings array
        """
        report_file = self.output_folder / "chunking_report.txt"
        
        # Calculate statistics
        chunk_lengths = [len(chunk['text']) for chunk in chunks]
        avg_length = np.mean(chunk_lengths)
        min_length = np.min(chunk_lengths)
        max_length = np.max(chunk_lengths)
        
        # Count documents
        doc_names = set(chunk['metadata']['document_name'] for chunk in chunks)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CHUNKING & EMBEDDING GENERATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Embedding Model: {self.model_name}\n")
            f.write(f"  Chunk Size: {self.chunk_size} tokens\n")
            f.write(f"  Chunk Overlap: {self.chunk_overlap} tokens\n")
            f.write(f"  Batch Size: 16\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Total Chunks: {len(chunks)}\n")
            f.write(f"  Documents Processed: {len(doc_names)}\n")
            f.write(f"  Embedding Dimension: {embeddings.shape[1]}\n\n")
            
            f.write("CHUNK LENGTH STATISTICS\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Average Length: {avg_length:.0f} characters\n")
            f.write(f"  Min Length: {min_length} characters\n")
            f.write(f"  Max Length: {max_length} characters\n\n")
            
            f.write("DOCUMENTS\n")
            f.write("-" * 60 + "\n")
            for doc_name in sorted(doc_names):
                doc_chunks = sum(1 for c in chunks if c['metadata']['document_name'] == doc_name)
                f.write(f"  • {doc_name}: {doc_chunks} chunks\n")
        
        print(f"✓ Generated report: {report_file}")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 60)
    print("RAG REGULATORY EXPLAINER - CHUNKING & EMBEDDING")
    print("(OPTIMIZED VERSION)")
    print("=" * 60 + "\n")
    
    try:
        # Initialize chunking and embedding
        chunker = ChunkingAndEmbedding(
            input_folder="processed_data",
            output_folder="embeddings_data",
            model_name="all-MiniLM-L6-v2",
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Load processed data
        pages = chunker.load_processed_data()
        
        # Create chunks
        chunks = chunker.create_chunks(pages)
        
        # Check if we got any chunks
        if not chunks:
            print("⚠ ERROR: No chunks were created!")
            return
        
        # Generate embeddings
        chunks_with_ids, embeddings = chunker.generate_embeddings(chunks)
        
        # Save everything
        chunker.save_chunks_and_embeddings(chunks_with_ids, embeddings)
        
        # Generate report
        chunker.generate_report(chunks_with_ids, embeddings)
        
        print("=" * 60)
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"  Total chunks created: {len(chunks_with_ids)}")
        print(f"  Embeddings generated: {embeddings.shape[0]}")
        print(f"  Output folder: {chunker.output_folder}")
        print(f"\n  Next step: Create FAISS vector database\n")
        
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ ERROR OCCURRED")
        print(f"{'=' * 60}")
        print(f"Error: {str(e)}")
        print(f"\nPlease report this error if it persists.\n")
        raise


if __name__ == "__main__":
    main()
