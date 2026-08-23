"""
Basic RAG Chatbot using LangChain — 100% FREE / LOCAL stack
--------------------------------------------------------------
A simple terminal chatbot that answers questions using documents indexed
from ./data. Single-turn retrieval-augmented generation: each question is
treated independently (no conversational memory).

This deliberately builds the pipeline with plain LCEL (prompt | llm) from
langchain_core rather than a langchain.chains.RetrievalQA chain: legacy
chain classes have moved or been removed across LangChain versions (e.g.
`from langchain.chains import RetrievalQA` raises `ModuleNotFoundError`
on current LangChain releases), while `langchain_core` primitives are the
stable, version-proof way to compose a retriever + prompt + LLM.

Free resources used (no API key, no cost, no rate limits):
  - LLM:        Ollama, running a free open-weight model locally
                (default: qwen2.5:0.5b — swap via OLLAMA_MODEL)
  - Embeddings: HuggingFace sentence-transformers, running locally
                (default: all-MiniLM-L6-v2)
  - Vector DB:  Chroma — free, open-source, stored on local disk

Setup:
    1. Install Ollama: https://ollama.com  (one-time, free)
    2. Pull a model:   ollama pull qwen2.5:0.5b
    3. pip install -r requirements.txt
    4. cp .env.example .env   (defaults already work — edit only if needed)
    5. python ingest.py       (indexes ./data)
    6. python app.py           (start chatting)
"""

import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))

QA_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use the following context to answer "
    "the question. If the answer is not contained in the context, say "
    "you don't know — do not make up information.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def load_vectorstore():
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run `python ingest.py` first to index your documents.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def build_pipeline():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    # LCEL pipeline: prompt -> llm. `retriever` is invoked separately in the
    # main loop (rather than piped straight into the chain) so we can also
    # print out which source documents were used for each answer.
    chain = QA_PROMPT | llm
    return retriever, chain


def format_context(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def main():
    print("=" * 60)
    print("Basic RAG Chatbot (LangChain) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    retriever, chain = build_pipeline()

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

        docs = retriever.invoke(query)
        context = format_context(docs)

        response = chain.invoke({"context": context, "question": query})
        print(f"\nBot: {response.content}")

        if docs:
            print("\nSources:")
            seen = set()
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                if src not in seen:
                    seen.add(src)
                    print(f"  - {src}")


if __name__ == "__main__":
    main()
