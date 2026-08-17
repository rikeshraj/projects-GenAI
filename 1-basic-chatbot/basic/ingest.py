"""
Ingest documents from ./data into a persistent Chroma collection.

Pure Python — NO LangChain / LlamaIndex / LangGraph. Uses:
  - sentence-transformers directly for embeddings (free, local)
  - chromadb directly for vector storage (free, local)
  - pypdf directly for PDF text extraction
  - a hand-written chunking function (no text-splitter library)

Usage:
    python ingest.py
"""

import os
import glob
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents():
    """Returns a list of (text, source_path) tuples for every supported file
    found under ./data."""
    documents = []
    for path in glob.glob(os.path.join(DATA_DIR, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        lower = path.lower()
        if lower.endswith((".txt", ".md")):
            documents.append((read_text_file(path), path))
        elif lower.endswith(".pdf"):
            documents.append((read_pdf_file(path), path))
    return documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Hand-written sliding-window chunker. Tries to break on a paragraph or
    sentence boundary near the target chunk_size, falling back to a hard
    character cut. This is a minimal stand-in for LangChain's
    RecursiveCharacterTextSplitter, written without any external library."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)
    return chunks


def main():
    print("Loading documents from ./data ...")
    documents = load_documents()
    if not documents:
        print("No documents found in ./data. Add .txt, .md, or .pdf files and re-run.")
        return
    print(f"Loaded {len(documents)} document(s).")

    print("Chunking documents...")
    all_chunks, all_metadatas, all_ids = [], [], []
    for text, source in documents:
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append(chunk)
            all_metadatas.append({"source": source, "chunk_index": i})
            all_ids.append(f"{source}::{i}")
    print(f"Created {len(all_chunks)} chunks.")

    print(f"Loading free local embedding model '{EMBEDDING_MODEL}' (downloads once)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    print("Writing to Chroma...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)  # start fresh on re-ingest
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )
    print(f"Done. {collection.count()} chunks indexed in ./{PERSIST_DIR}")


if __name__ == "__main__":
    main()
