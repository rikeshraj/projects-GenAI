"""
Advanced RAG Chatbot using LlamaIndex — free & local stack
-----------------------------------------------------------------
Terminal chatbot with:
  - Hybrid retrieval + query expansion: LlamaIndex's QueryFusionRetriever
    combines a dense (Chroma) retriever and a BM25 (keyword) retriever,
    and also has the LLM generate a few alternate phrasings of the query
    before fusing every ranked list together (Reciprocal Rank Fusion).
  - Cross-encoder re-ranking: SentenceTransformerRerank re-scores the
    fused candidates with a local cross-encoder model for a more precise
    final ordering.
  - History-aware conversational memory: CondensePlusContextChatEngine
    rewrites follow-up questions using chat history before retrieving,
    and keeps a ChatMemoryBuffer across turns.
  - Persisted memory: chat history is saved to a local JSON file via
    LlamaIndex's SimpleChatStore, so it survives restarting the app.

Setup:
    1. Install Ollama: https://ollama.com  and  ollama pull qwen2.5:0.5b
    2. pip install -r requirements.txt
    3. cp .env.example .env
    4. python ingest.py       (indexes ./data)
    5. python app.py           (start chatting — remembers earlier turns)
"""

import os
import sys
from dotenv import load_dotenv
import chromadb

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHAT_STORE_PATH = "chat_histories/chat_store.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DENSE_K = int(os.getenv("DENSE_K", "6"))
BM25_K = int(os.getenv("BM25_K", "6"))
FUSION_K = int(os.getenv("FUSION_K", "6"))       # kept after RRF fusion
FINAL_K = int(os.getenv("FINAL_K", "4"))          # kept after re-ranking
NUM_QUERIES = int(os.getenv("NUM_QUERIES", "3"))   # query-expansion fan-out

SYSTEM_PROMPT = (
    "You are a helpful, precise assistant. Use ONLY the retrieved context "
    "to answer the user's question. If the answer isn't in the context, "
    "say you don't know rather than making something up. Mention which "
    "source file(s) your answer came from, if any."
)


def load_index():
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run `python ingest.py` first.")
        sys.exit(1)

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = Ollama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=180.0, temperature=0
    )

    chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR, vector_store=vector_store
    )
    return load_index_from_storage(storage_context)


def build_hybrid_retriever(index):
    vector_retriever = index.as_retriever(similarity_top_k=DENSE_K)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=BM25_K
    )
    # Fuses the dense + BM25 ranked lists with Reciprocal Rank Fusion, and
    # also asks the LLM to generate NUM_QUERIES alternate phrasings of the
    # question first, searching with each — hybrid retrieval and query
    # expansion in a single built-in retriever.
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=FUSION_K,
        num_queries=NUM_QUERIES,
        mode="reciprocal_rerank",
        use_async=False,
    )


def build_chat_engine(index):
    retriever = build_hybrid_retriever(index)
    reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=FINAL_K)

    os.makedirs(os.path.dirname(CHAT_STORE_PATH), exist_ok=True)
    if os.path.exists(CHAT_STORE_PATH):
        chat_store = SimpleChatStore.from_persist_path(CHAT_STORE_PATH)
    else:
        chat_store = SimpleChatStore()

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000, chat_store=chat_store, chat_store_key="session"
    )

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        llm=Settings.llm,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        node_postprocessors=[reranker],
    )
    return chat_engine, chat_store


def main():
    print("=" * 60)
    print("Advanced RAG Chatbot (LlamaIndex) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Hybrid retrieval + query expansion + re-ranking + memory")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    index = load_index()
    chat_engine, chat_store = build_chat_engine(index)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            chat_store.persist(CHAT_STORE_PATH)
            print("Goodbye! Chat history saved to ./chat_histories/")
            break
        if not query:
            continue

        response = chat_engine.chat(query)
        print(f"\nBot: {response}")

        sources = getattr(response, "source_nodes", [])
        if sources:
            print("\nSources:")
            seen = set()
            for node in sources:
                src = node.metadata.get("file_name") or node.metadata.get("file_path", "unknown")
                if src not in seen:
                    seen.add(src)
                    print(f"  - {src}")

        chat_store.persist(CHAT_STORE_PATH)


if __name__ == "__main__":
    main()
