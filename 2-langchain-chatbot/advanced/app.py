"""
Advanced RAG Chatbot using LangChain — 100% FREE / LOCAL stack
-------------------------------------------------------------------
A terminal chatbot with a more sophisticated retrieval pipeline than the
basic version:

  - Hybrid retrieval: BM25 (sparse/keyword) + Chroma (dense/semantic),
    combined by merging and de-duplicating both result lists.
  - Re-ranking / trimming: the top results from the combined list are
    kept and stuffed into the answer prompt.
  - Conversational memory: follow-up questions ("what about the Pro
    plan?") are rewritten into standalone questions using the chat
    history, and the full conversation is persisted to disk.

Free resources used (no API key, no cost, no rate limits):
  - LLM:        Ollama, running a free open-weight model locally
                (default: qwen2.5:0.5b — swap via OLLAMA_MODEL)
  - Embeddings: HuggingFace sentence-transformers, running locally
                (default: all-MiniLM-L6-v2)
  - Vector DB:  Chroma — free, open-source, stored on local disk
  - Sparse retriever: BM25 (rank_bm25) — free, pure Python, no service needed

Setup:
    1. Install Ollama: https://ollama.com  (one-time, free)
    2. Pull a model:   ollama pull qwen2.5:0.5b
    3. pip install -r requirements.txt
    4. cp .env.example .env   (defaults already work — edit only if needed)
    5. python ingest.py       (indexes ./data)
    6. python app.py           (start chatting — remembers earlier turns)

Note: this pipeline makes two LLM calls per follow-up question (query
rewrite, then the final answer) and one for the very first question in a
conversation (no rewrite needed yet). On CPU-only Ollama this can take a
few seconds per call — if responses feel slow, try a smaller model (the
default qwen2.5:0.5b is already quite small), or a larger one like
qwen2.5:3b if you want stronger answers and can spare the RAM.
"""

import os
import sys
import json
import pickle
from datetime import datetime
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

PERSIST_DIR = "chroma_db"
CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
HISTORY_DIR = "chat_histories"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CONTEXTUALIZE_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

SYSTEM_PROMPT = """You are a helpful, precise assistant. Use ONLY the retrieved
context below to answer the user's question. If the answer isn't in the context,
say you don't know rather than making something up. Keep answers concise and
mention the source file name(s) when relevant.

Context:
{context}"""


def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        print("No indexed chunks found. Run `python ingest.py` first.")
        sys.exit(1)
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


def build_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    return dense_retriever, bm25_retriever


def hybrid_search(query, dense_retriever, bm25_retriever):
    dense_docs = dense_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    combined = []
    seen = set()

    # Give semantic results priority
    for doc in dense_docs + bm25_docs:
        key = (doc.metadata.get("source", ""), doc.page_content)

        if key not in seen:
            seen.add(key)
            combined.append(doc)

    return combined


def build_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )


def rewrite_query(llm, query, chat_history):
    """
    Convert a follow-up question into a standalone question, using the
    chat history so retrieval isn't run on an ambiguous fragment like
    "what about the Pro plan?".
    """

    if not chat_history:
        # Nothing to disambiguate against yet — the first question in a
        # conversation is already standalone, so skip the extra LLM call.
        return query

    # IMPORTANT: CONTEXTUALIZE_PROMPT is a plain instruction string with no
    # {chat_history}/{input} placeholders in it, so it must go in as a
    # *system* message via from_messages(), with the actual history and
    # question supplied separately via MessagesPlaceholder/"{input}".
    # (Using ChatPromptTemplate.from_template(CONTEXTUALIZE_PROMPT) here
    # would silently drop the chat_history/input variables, since
    # from_template only fills in variables that literally appear as
    # {...} in the template text — and this string has none.)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    messages = prompt.invoke({
        "chat_history": chat_history,
        "input": query
    })

    response = llm.invoke(messages)

    return response.content.strip()


def generate_answer(llm, query, context, chat_history):
    """
    Generate the final answer using retrieved context.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    messages = prompt.invoke({
        "chat_history": chat_history,
        "context": context,
        "input": query
    })

    response = llm.invoke(messages)

    return response.content


STORE = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


def save_history(session_id: str):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history = STORE.get(session_id)
    if not history:
        return
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    serializable = [{
            "type": message.type,
            "content": message.content
        } for message in history.messages
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def main():
    print("=" * 60)
    print("Advanced RAG Chatbot (LangChain) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Hybrid retrieval + query rewriting + memory")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    llm = build_llm()

    dense_retriever, bm25_retriever = build_retriever()

    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    history = get_session_history(session_id)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            save_history(session_id)
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            save_history(session_id)
            print("Goodbye! Chat history saved to ./chat_histories/")
            break
        if not query:
            continue

        # 1. Rewrite follow-up question
        standalone_query = rewrite_query(llm, query, history.messages)

        # 2. Hybrid retrieval
        documents = hybrid_search(standalone_query, dense_retriever, bm25_retriever)

        # Keep the top 6 results
        documents = documents[:6]

        # 3. Build context
        context = "\n\n".join(doc.page_content for doc in documents)

        # 4. Generate answer
        answer = generate_answer(llm, standalone_query, context, history.messages)

        print(f"\nBot: {answer}")

        # 5. Display sources
        if documents:
            print("\nSources:")
            seen = set()

            for doc in documents:
                src = doc.metadata.get(
                    "source",
                    "unknown"
                )
                if src not in seen:
                    seen.add(src)
                    print(
                        f"  - {src}"
                    )

        # 6. Update conversation history
        history.add_user_message(query)
        history.add_ai_message(answer)
        save_history(session_id)


if __name__ == "__main__":
    main()
