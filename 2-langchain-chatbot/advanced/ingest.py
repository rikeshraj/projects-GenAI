"""
Ingest documents from ./data into:
  1. A persistent Chroma vector store (dense retrieval)
  2. A pickled chunk list (used to build the BM25 sparse retriever at runtime)

Uses a FREE, LOCAL embedding model (HuggingFace sentence-transformers) —
no API key, no cost.

Usage:
    python ingest.py
"""

import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def load_documents():
    docs = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.lower().endswith((".txt", ".md")):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif fname.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
    return docs


def main():
    print("Loading documents from ./data ...")
    documents = load_documents()
    if not documents:
        print("No documents found in ./data. Add .txt, .md, or .pdf files and re-run.")
        return

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunk list for BM25 retriever to {CHUNKS_PATH}")

    print(f"Loading free local embedding model '{EMBEDDING_MODEL}' "
          f"(downloads once, then cached)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building dense vector store...")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=PERSIST_DIR
    )
    # Recent chromadb versions persist to disk automatically (via the
    # underlying PersistentClient), and some current langchain-chroma
    # releases have removed the `.persist()` method entirely — calling it
    # unconditionally can raise AttributeError on newer installs. Guard it
    # so this script works across old and new versions alike.
    if hasattr(vectorstore, "persist"):
        vectorstore.persist()
    print(f"Done. Vector store persisted to ./{PERSIST_DIR}")


if __name__ == "__main__":
    main()
