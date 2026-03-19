# PDF Ingestion Script - Usage Guide

## What This Script Does

The `pdf_ingestion.py` script processes regulatory PDF documents and prepares them for the RAG system.

**Features:**
- ✓ Reads all PDF files from `data/` folder
- ✓ Extracts text page-by-page
- ✓ Cleans text (removes headers, formatting artifacts)
- ✓ Captures metadata (filename, page numbers)
- ✓ Saves processed data as JSON

## Prerequisites

Make sure you have installed:
```bash
pip install pypdf
```

## Folder Structure

Ensure your project has this structure:
```
your-project/
├── data/                    # Put your PDF files here
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── pdf_ingestion.py         # This script
└── processed_data/          # Created automatically - output goes here
```

## How to Run

1. **Place your PDF files** in the `data/` folder

2. **Run the script:**
   ```bash
   python pdf_ingestion.py
   ```

3. **Check the output** in `processed_data/` folder:
   - `processed_documents.json` - All extracted text with metadata
   - `ingestion_report.txt` - Summary of what was processed

## Output Format

The script creates a JSON file with this structure:

```json
{
  "summary": {
    "total_documents": 3,
    "total_pages": 150,
    "total_characters": 450000,
    "documents": [...]
  },
  "pages": [
    {
      "text": "Cleaned text from page...",
      "metadata": {
        "document_name": "basel_framework",
        "page_number": 1,
        "total_pages": 50,
        "file_path": "data/basel_framework.pdf"
      },
      "text_length": 3000
    },
    ...
  ]
}
```

## What Happens Next

After running this script, you'll have:
- ✓ Clean text extracted from all PDFs
- ✓ Metadata attached to each page
- ✓ Data ready for chunking (next step)

The next script will:
1. Take this processed data
2. Split it into smaller chunks
3. Generate embeddings
4. Store in vector database

## Troubleshooting

**No PDF files found:**
- Make sure PDFs are in the `data/` folder
- Check file extensions are `.pdf` (lowercase)

**Empty pages:**
- Some PDFs may be scanned images without text
- Consider using OCR for scanned documents

**Encoding errors:**
- The script uses UTF-8 encoding
- Should handle most regulatory documents fine
