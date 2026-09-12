"""
Basic RAG Chatbot using LlamaIndex — free & local stack
------------------------------------------------------------
Terminal chatbot built on a LlamaIndex VectorStoreIndex backed by Chroma.
Each question is answered independently via a query engine — single-turn
retrieval-augmented generation, no memory.

Free resources used (no API key, no cost, no rate limits):
  - LLM:        Ollama, local (default: qwen2.5:0.5b)
  - Embeddings: HuggingFace sentence-transformers, local
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
from dotenv import load_dotenv
import chromadb

from llama_index.core import VectorStoreIndex, PromptTemplate, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))

QA_PROMPT_TEMPLATE = PromptTemplate(
    "You are a helpful assistant. Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Using the context above, answer the question. If the answer isn't in "
    "the context, say you don't know rather than making something up.\n"
    "Question: {query_str}\n"
    "Answer: "
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
    return VectorStoreIndex.from_vector_store(vector_store)


def main():
    print("=" * 60)
    print("Basic RAG Chatbot (LlamaIndex) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=TOP_K)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT_TEMPLATE})

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

        response = query_engine.query(query)
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


if __name__ == "__main__":
    main()
