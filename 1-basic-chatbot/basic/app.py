"""
Basic RAG Chatbot — NO FRAMEWORK (pure Python)
-------------------------------------------------
Terminal chatbot that retrieves relevant chunks from a Chroma collection using sentence-transformers embeddings, and asks a local Ollama model to answer using that context. No LangChain, no LlamaIndex, no LangGraph — every step (embedding, retrieval, prompt construction, the LLM call) is plain Python calling the underlying libraries/APIs directly.

Free resources used (no API key, no cost):
  - LLM:        Ollama, local, called via its REST API directly
                (default model: qwen2.5:0.5b)
  - Embeddings: HuggingFace sentence-transformers, local
                (default: all-MiniLM-L6-v2)
  - Vector DB:  Chroma, local

Setup:
    1. Install Ollama: https://ollama.com  and  ollama pull qwen2.5:0.5b
    2. pip install -r requirements.txt
    3. cp .env.example .env
    4. python ingest.py       (indexes ./data)
    5. python app.py           (start chatting)
"""

import os
import sys
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = """You are a helpful assistant. Use the following context to answer
the question. If the answer is not contained in the context, say you don't know —
do not make up information."""


def load_collection():
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run `python ingest.py` first.")
        sys.exit(1)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME)


def retrieve(collection, embed_model, query, k=TOP_K):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return list(zip(results["documents"][0], results["metadatas"][0]))


def call_ollama(user_prompt, system=SYSTEM_PROMPT):
    """Calls the local Ollama REST API directly — no client library, just
    a plain HTTP POST, to keep this project framework-free."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def build_prompt(query, retrieved):
    context = "\n\n---\n\n".join(doc for doc, _ in retrieved)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def main():
    print("=" * 60)
    print("Basic RAG Chatbot — no framework (pure Python)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    collection = load_collection()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            continue

        retrieved = retrieve(collection, embed_model, query)
        prompt = build_prompt(query, retrieved)

        try:
            answer = call_ollama(prompt)
        except requests.exceptions.ConnectionError:
            print("\nCould not reach Ollama. Is `ollama serve` running?")
            continue

        print(f"\nBot: {answer}")

        if retrieved:
            print("\nSources:")
            seen = set()
            for _, meta in retrieved:
                src = meta.get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    print(f"  - {src}")


if __name__ == "__main__":
    main()
