"""
Advanced RAG Chatbot — NO FRAMEWORK (pure Python)
------------------------------------------------------
Terminal chatbot with a more sophisticated, hand-built retrieval pipeline:

  - Hybrid retrieval: BM25 (keyword, via rank_bm25) + Chroma (dense/semantic),
    fused with Reciprocal Rank Fusion (RRF) — implemented from scratch,
    no framework retriever classes involved.
  - Cross-encoder re-ranking: the fused candidate set is re-scored with a
    sentence-transformers CrossEncoder for a precise final ordering
    (a local, free alternative to LLM-based re-ranking).
  - History-aware query rewriting: before retrieval, a local LLM call
    rewrites follow-up questions ("what about the Pro plan?") into
    standalone questions using recent chat history.
  - Multi-turn memory persisted to disk as JSON per session.

No LangChain / LlamaIndex / LangGraph — every piece (retrieval, fusion,
re-ranking, memory, the LLM call) is plain Python.

Setup:
    1. Install Ollama: https://ollama.com  and  ollama pull qwen2.5:0.5b
    2. pip install -r requirements.txt
    3. cp .env.example .env
    4. python ingest.py       (indexes ./data)
    5. python app.py           (start chatting — remembers earlier turns)
"""

import os
import sys
import json
import pickle
import re
from datetime import datetime
from collections import defaultdict

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
HISTORY_DIR = "chat_histories"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DENSE_K = int(os.getenv("DENSE_K", "8"))
BM25_K = int(os.getenv("BM25_K", "8"))
RRF_K = int(os.getenv("RRF_K", "60"))     # standard RRF smoothing constant
FINAL_K = int(os.getenv("FINAL_K", "4"))   # chunks kept after re-ranking

SYSTEM_PROMPT = """You are a helpful, precise assistant. Use ONLY the retrieved
context below to answer the user's question. If the answer isn't in the context,
say you don't know rather than making something up. Keep answers concise and
mention the source file name(s) when relevant."""

REWRITE_SYSTEM_PROMPT = """Given the recent chat history and a new user question,
rewrite the question to be a standalone question that can be understood without
the chat history. If it's already standalone, return it unchanged. Reply with
ONLY the rewritten question and nothing else."""


def call_ollama(messages, temperature=0.0):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_chunk_store():
    if not os.path.exists(CHUNKS_PATH):
        print("No indexed chunks found. Run `python ingest.py` first.")
        sys.exit(1)
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


class HybridRetriever:
    """Combines BM25 (sparse) and Chroma (dense) retrieval via Reciprocal
    Rank Fusion, then re-ranks the fused set with a cross-encoder."""

    def __init__(self):
        store = load_chunk_store()
        self.ids = store["ids"]
        self.documents = store["documents"]
        self.metadatas = store["metadatas"]
        self.id_to_index = {doc_id: i for i, doc_id in enumerate(self.ids)}

        print("Building BM25 index...")
        tokenized_corpus = [tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Loading cross-encoder re-ranker '{RERANKER_MODEL}'...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = client.get_collection(COLLECTION_NAME)

    def _bm25_ranked_ids(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.ids[i] for i in ranked_indices]

    def _dense_ranked_ids(self, query, k):
        query_embedding = self.embed_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        return results["ids"][0]

    @staticmethod
    def _rrf_fuse(ranked_lists, k=RRF_K):
        """Reciprocal Rank Fusion: combines multiple ranked ID lists into a
        single fused ranking. score(doc) = sum(1 / (k + rank)) across lists."""
        scores = defaultdict(float)
        for ranked_ids in ranked_lists:
            for rank, doc_id in enumerate(ranked_ids):
                scores[doc_id] += 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    def retrieve(self, query, final_k=FINAL_K):
        bm25_ids = self._bm25_ranked_ids(query, BM25_K)
        dense_ids = self._dense_ranked_ids(query, DENSE_K)
        fused_ids = self._rrf_fuse([bm25_ids, dense_ids])

        candidates = [
            (self.documents[self.id_to_index[doc_id]], self.metadatas[self.id_to_index[doc_id]])
            for doc_id in fused_ids
            if doc_id in self.id_to_index
        ]
        if not candidates:
            return []

        pairs = [(query, doc) for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [candidate for candidate, _ in reranked[:final_k]]


def rewrite_query(query, history):
    if not history:
        return query
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    messages = [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Chat history:\n{history_text}\n\nNew question: {query}"},
    ]
    return call_ollama(messages)


def build_answer_prompt(query, retrieved):
    context = "\n\n---\n\n".join(doc for doc, _ in retrieved)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def save_history(session_id, history):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main():
    print("=" * 60)
    print("Advanced RAG Chatbot — no framework (pure Python)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Hybrid retrieval (BM25 + dense, RRF) + cross-encoder re-rank + memory")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    retriever = HybridRetriever()
    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    history = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            save_history(session_id, history)
            print("Goodbye! Chat history saved to ./chat_histories/")
            break
        if not query:
            continue

        try:
            standalone_query = rewrite_query(query, history)
        except requests.exceptions.ConnectionError:
            print("\nCould not reach Ollama. Is `ollama serve` running?")
            continue

        retrieved = retriever.retrieve(standalone_query)
        prompt = build_answer_prompt(standalone_query, retrieved)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            answer = call_ollama(messages)
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

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        save_history(session_id, history)


if __name__ == "__main__":
    main()
