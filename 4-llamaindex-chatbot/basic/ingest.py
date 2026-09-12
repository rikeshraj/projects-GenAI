"""
Ingest documents from ./data into a persistent Chroma-backed VectorStoreIndex.

Uses FREE, LOCAL resources — HuggingFace sentence-transformers embeddings and Chroma vector storage — no API key, no cost.

Usage:
    python ingest.py
"""

import os
from dotenv import load_dotenv
import chromadb

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def main():
    print("Loading documents from ./data ...")
    documents = SimpleDirectoryReader(
        DATA_DIR, required_exts=[".txt", ".md", ".pdf"]
    ).load_data()
    if not documents:
        print("No documents found in ./data. Add .txt, .md, or .pdf files and re-run.")
        return
    print(f"Loaded {len(documents)} document(s).")

    print(f"Loading free local embedding model '{EMBEDDING_MODEL}' (downloads once)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print("Building Chroma-backed vector index...")
    # A PersistentClient writes to disk automatically as data is added —
    # there's no separate vectorstore.persist() call needed with current
    # chromadb versions.
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)  # start fresh on re-ingest
    except Exception:
        pass
    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    # Persists the docstore/index-store metadata (node text, doc ids, etc.)
    # to disk — separate from the vector embeddings, which Chroma already
    # persists itself via the PersistentClient above.
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Done. Index persisted to ./{PERSIST_DIR}")


if __name__ == "__main__":
    main()
