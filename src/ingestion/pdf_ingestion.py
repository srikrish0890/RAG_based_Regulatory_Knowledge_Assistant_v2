"""
PDF Ingestion Script for RAG Regulatory Explainer
Milestone 2: Document Ingestion & Processing

This script:
- Reads PDF files from data/ folder
- Checks registry to skip already-processed files (incremental ingestion)
- Extracts text page by page
- Extracts metadata (filename, page number)
- Cleans text (removes headers, formatting artifacts, normalizes whitespace)
- Saves each document as a separate JSON file in processed_data/
- Supports --rebuild flag to wipe registry and reprocess everything

Run modes:
    python pdf_ingestion.py              <- incremental (default)
    python pdf_ingestion.py --rebuild    <- wipe registry, reprocess all
"""

import os
import json
import re
import hashlib
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from datetime import datetime


class PDFIngestion:
    def __init__(self, data_folder: str = "data", output_folder: str = "processed_data", registry_folder: str = "registry", rebuild: bool = False):
        """
        Initialize PDF ingestion pipeline

        Args:
            data_folder: Folder containing PDF files
            output_folder: Folder to save processed documents
            registry_folder: Folder containing SQLite registry
            rebuild: If True, wipe registry and reprocess all files
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.registry_folder = Path(registry_folder)
        self.rebuild = rebuild

        # Create folders if they don't exist
        self.output_folder.mkdir(exist_ok=True)
        self.registry_folder.mkdir(exist_ok=True)

        # Registry DB path
        self.registry_path = self.registry_folder / "registry.db"

        # Setup registry
        self._setup_registry()

        # If rebuild, wipe registry and processed data
        if self.rebuild:
            self._wipe_registry()

        self.processed_documents = []
        self.skipped_documents = []

    # ------------------------------------------------------------------
    # Registry Methods
    # ------------------------------------------------------------------

    def _setup_registry(self):
        """
        Create registry table if it does not exist
        """
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_files (
                filename      TEXT PRIMARY KEY,
                file_hash     TEXT NOT NULL,
                processed_at  TEXT NOT NULL,
                chunk_count   INTEGER,
                status        TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _wipe_registry(self):
        """
        Wipe all records from registry (used in --rebuild mode)
        """
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM processed_files")
        conn.commit()
        conn.close()
        print("✓ Registry wiped. All files will be reprocessed.\n")

    def _get_registry_entry(self, filename: str) -> Optional[Dict]:
        """
        Fetch registry entry for a given filename

        Returns:
            Dict with registry row or None if not found
        """
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, file_hash, processed_at, chunk_count, status FROM processed_files WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "filename": row[0],
                "file_hash": row[1],
                "processed_at": row[2],
                "chunk_count": row[3],
                "status": row[4]
            }
        return None

    def _write_registry_entry(self, filename: str, file_hash: str, chunk_count: int, status: str):
        """
        Insert or update a registry entry after processing
        """
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO processed_files (filename, file_hash, processed_at, chunk_count, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                file_hash    = excluded.file_hash,
                processed_at = excluded.processed_at,
                chunk_count  = excluded.chunk_count,
                status       = excluded.status
        """, (filename, file_hash, datetime.now().isoformat(), chunk_count, status))
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def _compute_file_hash(self, pdf_path: Path) -> str:
        """
        Compute MD5 hash of a file to detect content changes

        Returns:
            MD5 hash string
        """
        hasher = hashlib.md5()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Text Processing (unchanged)
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing artifacts and normalizing whitespace
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\x00', '', text)
        text = re.sub(r'[\x0c\x0b]', '', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\nPage\s+\d+\s*\n', '\n', text)
        text = re.sub(r'[-_]{3,}', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        return text

    def extract_metadata(self, pdf_path: Path, page_num: int, total_pages: int) -> Dict:
        """
        Extract metadata for each page
        """
        return {
            "document_name": pdf_path.stem,
            "file_path": str(pdf_path),
            "page_number": page_num + 1,
            "total_pages": total_pages,
            "extracted_date": datetime.now().isoformat()
        }

    def parse_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Parse a single PDF file and extract text with metadata
        """
        pages_data = []

        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)

            print(f"Processing: {pdf_path.name} ({total_pages} pages)")

            for page_num, page in enumerate(reader.pages):
                raw_text = page.extract_text()

                if not raw_text or len(raw_text.strip()) < 10:
                    print(f"  ⚠ Page {page_num + 1}: No text or very short content, skipping")
                    continue

                cleaned_text = self.clean_text(raw_text)
                metadata = self.extract_metadata(pdf_path, page_num, total_pages)

                page_data = {
                    "text": cleaned_text,
                    "metadata": metadata,
                    "text_length": len(cleaned_text)
                }

                pages_data.append(page_data)
                print(f"  ✓ Page {page_num + 1}: Extracted {len(cleaned_text)} characters")

            print(f"✓ Completed: {pdf_path.name} - {len(pages_data)} pages processed\n")

        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}\n")

        return pages_data

    # ------------------------------------------------------------------
    # Incremental Ingestion
    # ------------------------------------------------------------------

    def ingest_all_pdfs(self) -> int:
        """
        Process all PDF files in the data folder.
        Skips files that are already in the registry with matching hash.

        Returns:
            Total number of pages processed
        """
        pdf_files = list(self.data_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"⚠ No PDF files found in {self.data_folder}")
            return 0

        print(f"Found {len(pdf_files)} PDF file(s) in {self.data_folder}\n")
        print("=" * 60)

        total_pages = 0

        for pdf_path in pdf_files:
            filename = pdf_path.name
            file_hash = self._compute_file_hash(pdf_path)
            registry_entry = self._get_registry_entry(filename)

            # --- Skip if already processed and unchanged ---
            if registry_entry and registry_entry["file_hash"] == file_hash:
                print(f"⏭ Skipping (unchanged): {filename}")
                self.skipped_documents.append(filename)
                continue

            # --- Log reason for processing ---
            if registry_entry and registry_entry["file_hash"] != file_hash:
                print(f"🔄 Re-processing (file changed): {filename}")
            else:
                print(f"🆕 New file detected: {filename}")

            # --- Parse PDF ---
            pages_data = self.parse_pdf(pdf_path)

            if pages_data:
                # Save per-document JSON
                self._save_document_json(pdf_path.stem, pages_data)

                # Write to registry only after successful save
                self._write_registry_entry(filename, file_hash, len(pages_data), "success")

                self.processed_documents.append({
                    "document_name": pdf_path.stem,
                    "pages_count": len(pages_data),
                    "file_path": str(pdf_path)
                })

                total_pages += len(pages_data)
            else:
                # Write failed status to registry
                self._write_registry_entry(filename, file_hash, 0, "failed")
                print(f"✗ No pages extracted from {filename}\n")

        return total_pages

    # ------------------------------------------------------------------
    # Save Output
    # ------------------------------------------------------------------

    def _save_document_json(self, document_name: str, pages_data: List[Dict]):
        """
        Save a single document's processed pages to its own JSON file

        Args:
            document_name: PDF filename without extension (used as JSON filename)
            pages_data: List of processed pages for this document
        """
        output_path = self.output_folder / f"{document_name}.json"

        summary = {
            "document_name": document_name,
            "total_pages": len(pages_data),
            "total_characters": sum(page["text_length"] for page in pages_data),
            "processed_date": datetime.now().isoformat()
        }

        output_data = {
            "summary": summary,
            "pages": pages_data
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved: {output_path}\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self):
        """
        Generate a summary report of ingestion run
        """
        report_path = self.output_folder / "ingestion_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PDF INGESTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'REBUILD' if self.rebuild else 'INCREMENTAL'}\n\n")

            f.write(f"Documents Processed: {len(self.processed_documents)}\n")
            f.write(f"Documents Skipped:   {len(self.skipped_documents)}\n")
            f.write("-" * 60 + "\n\n")

            if self.processed_documents:
                f.write("Processed:\n")
                for doc in self.processed_documents:
                    f.write(f"  • {doc['document_name']}\n")
                    f.write(f"    Pages: {doc['pages_count']}\n")
                    f.write(f"    Path: {doc['file_path']}\n\n")

            if self.skipped_documents:
                f.write("Skipped (unchanged):\n")
                for name in self.skipped_documents:
                    f.write(f"  • {name}\n")

        print(f"✓ Generated report: {report_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PDF Ingestion for RAG Regulatory Explainer")
    parser.add_argument("--rebuild", action="store_true", help="Wipe registry and reprocess all files")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RAG REGULATORY EXPLAINER - PDF INGESTION")
    mode = "REBUILD MODE" if args.rebuild else "INCREMENTAL MODE"
    print(f"Mode: {mode}")
    print("=" * 60 + "\n")

    ingestion = PDFIngestion(
        data_folder="data",
        output_folder="processed_data",
        registry_folder="registry",
        rebuild=args.rebuild
    )

    total_pages = ingestion.ingest_all_pdfs()

    if total_pages > 0 or ingestion.skipped_documents:
        ingestion.generate_report()

        print("\n" + "=" * 60)
        print(f"✓ Ingestion complete.")
        print(f"  • Processed : {len(ingestion.processed_documents)} document(s)")
        print(f"  • Skipped   : {len(ingestion.skipped_documents)} document(s)")
        print(f"  • Total pages processed: {total_pages}")
        print(f"\n  Next step: Run chunking and embedding generation")
        print("=" * 60 + "\n")
    else:
        print("\n⚠ No pages were processed. Please check your PDF files.\n")


if __name__ == "__main__":
    main()
