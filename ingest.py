"""
PAWS RAG Ingestion Script
Run once to build the ChromaDB knowledge base from PDFs and DOCX files.
Usage: python ingest.py
"""

import os
import sys
import time
import pathlib
import textwrap

import fitz                    # PyMuPDF - reads PDFs
import docx                    # python-docx - reads DOCX files
import chromadb                # vector database
from google import genai       # embeddings

# ── Config ────────────────────────────────────────────────────────────────────

DOCS_DIR      = pathlib.Path("Training AI Abstract Guide")
CHROMA_DIR    = pathlib.Path("chroma_db")
CHUNK_SIZE    = 500     # target words per chunk
CHUNK_OVERLAP = 50      # words of overlap between chunks
EMBED_MODEL   = "models/gemini-embedding-001"
COLLECTION    = "paws_knowledge"
SKIP_FILES    = {
    "SHBC Abstract Digital Natives vs Immigrants.docx",  # image-based, no extractable text
}

API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    sys.exit("Error: GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=API_KEY)

# ── Text extraction ───────────────────────────────────────────────────────────

def extract_pdf(path: pathlib.Path) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_docx(path: pathlib.Path) -> str:
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf(path)
    if path.suffix.lower() == ".docx":
        return extract_docx(path)
    return ""

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({
            "text": chunk,
            "id":   f"{source}__chunk{idx}",
            "meta": {"source": source},
        })
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, respecting the free-tier rate limit."""
    embeddings = []
    for i, text in enumerate(texts):
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        embeddings.append(result.embeddings[0].values)
        # Free tier: 1,500 req/day, ~1 req/s is safe
        if i % 10 == 9:
            time.sleep(1)
    return embeddings

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Collect all supported files
    files = sorted(
        p for p in DOCS_DIR.iterdir()
        if p.suffix.lower() in {".pdf", ".docx"}
        and not p.name.startswith("~")
        and p.name not in SKIP_FILES
    )

    if not files:
        sys.exit(f"No PDF or DOCX files found in {DOCS_DIR}")

    print(f"Found {len(files)} documents:")
    for f in files:
        print(f"  {f.name}")

    # Build chunks from all documents
    all_chunks = []
    for path in files:
        print(f"\nExtracting: {path.name}")
        text = extract(path)
        if not text.strip():
            print(f"  Warning: no text extracted, skipping.")
            continue
        chunks = chunk_text(text, path.name)
        print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # Embed all chunks
    print("Generating embeddings (this may take a few minutes)...")
    texts      = [c["text"] for c in all_chunks]
    embeddings = embed_batch(texts)

    # Store in ChromaDB
    print(f"\nStoring in ChromaDB at ./{CHROMA_DIR}/")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate collection for a clean rebuild
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids        = [c["id"]   for c in all_chunks],
        documents  = [c["text"] for c in all_chunks],
        embeddings = embeddings,
        metadatas  = [c["meta"] for c in all_chunks],
    )

    print(f"\nDone. {len(all_chunks)} chunks stored in collection '{COLLECTION}'.")
    print("ChromaDB is ready for the app.")

if __name__ == "__main__":
    main()
