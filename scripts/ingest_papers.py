"""
Ingestion pipeline: PDFs → chunks → embeddings → ChromaDB.

Pipeline steps:
1. Parse every PDF in the papers directory using PyMuPDF.
2. Split text into overlapping chunks (`CHUNK_SIZE` chars, `CHUNK_OVERLAP` overlap).
3. Generate embeddings with sentence-transformers (runs locally, no API cost).
4. Persist into a ChromaDB collection on disk.

Idempotency: the script clears and rebuilds the collection so re-running is safe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (  # noqa: E402
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PAPERS_DIR,
)


def extract_text(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF, one record per page.

    Returns a list of {page_number, text} dicts. Keeping per-page granularity
    lets us cite "[Author, p. 5]" later in the RAG response.
    """
    records: list[dict] = []
    try:
        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                records.append({"page_number": page_idx + 1, "text": text})
        doc.close()
    except Exception as exc:
        print(f"  ✗ Failed to parse {pdf_path.name}: {exc}")
    return records


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk a string into overlapping windows of characters.

    Character-based chunking is simple and language-agnostic. Word-aware or
    sentence-aware chunking would be slightly better but adds dependencies for
    marginal gain at this project's scale.
    """
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_corpus(papers_dir: Path) -> tuple[list[str], list[dict], list[str]]:
    """Walk every PDF and build parallel lists of (texts, metadatas, ids)."""
    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {papers_dir}. Run scripts/download_papers.py first."
        )

    print(f"→ Found {len(pdfs)} PDFs to ingest.\n")

    for pdf_path in pdfs:
        print(f"  • {pdf_path.name}")
        page_records = extract_text(pdf_path)
        if not page_records:
            print("      (no extractable text)")
            continue

        chunk_index = 0
        for record in page_records:
            for chunk in chunk_text(record["text"]):
                cid = f"{pdf_path.stem}__p{record['page_number']:03d}__c{chunk_index:04d}"
                texts.append(chunk)
                metadatas.append({
                    "source": pdf_path.name,
                    "page": record["page_number"],
                    "chunk_index": chunk_index,
                })
                ids.append(cid)
                chunk_index += 1

        print(f"      → {chunk_index} chunks from {len(page_records)} pages")

    return texts, metadatas, ids


def ingest() -> None:
    """Run the full ingestion pipeline."""
    # Build corpus first so that downloading the embedding model only happens
    # if we actually have something to ingest.
    texts, metadatas, ids = build_corpus(PAPERS_DIR)
    print(f"\n→ Total chunks to embed: {len(texts):,}")

    # Lazy import: sentence-transformers takes a few seconds to import.
    print("\n→ Loading embedding model (downloads on first run, ~80 MB)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  ✓ Loaded {EMBEDDING_MODEL}")

    print("\n→ Computing embeddings (CPU-friendly; will be much faster on GPU)...")
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print("\n→ Persisting into ChromaDB...")
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Reset the collection for clean ingestion (idempotency).
    try:
        client.delete_collection(name=CHROMA_COLLECTION)
        print(f"  • Cleared existing collection '{CHROMA_COLLECTION}'.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma has a soft batch limit; chunk our inserts to stay safe.
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        collection.add(
            documents=texts[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    print(f"  ✓ Stored {collection.count():,} chunks in '{CHROMA_COLLECTION}'.")
    print(f"\n→ Done. ChromaDB lives at {CHROMA_DB_PATH}.")
    print("→ Next: run `python scripts/query_demo.py \"what is a saccade?\"`")


if __name__ == "__main__":
    ingest()
