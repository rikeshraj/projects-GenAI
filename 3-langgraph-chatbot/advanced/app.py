"""
Advanced RAG Chatbot using LangGraph — free & local stack
------------------------------------------------------------
A terminal chatbot built as a LangGraph graph with real branching, unlike
the basic version's straight line:

    START -> classify -> (conditional) -> rewrite_query -> retrieve -> rerank -> generate -> END
                       \\-> (conditional) -> generate (skips retrieval entirely) -> END

  - classify: the LLM decides whether this question needs the knowledge
    base at all (e.g. "hi, how are you?" doesn't) — a conditional edge
    then routes to either the retrieval path or straight to generate.
  - rewrite_query: rewrites follow-up questions ("what about the Pro
    plan?") into standalone questions using chat history.
  - retrieve: hybrid retrieval — BM25 (keyword) + Chroma (dense/semantic)
    combined with LangChain's EnsembleRetriever.
  - rerank: a cross-encoder re-ranks the fused candidates for a precise
    final ordering (a local, free alternative to LLM-based re-ranking).
  - generate: produces the final answer using the (possibly empty) context.

Memory: this version uses LangGraph's SqliteSaver checkpointer, which
persists the full graph state (message history) to a local SQLite file —
so, unlike the basic version's in-memory-only checkpointer, conversations
survive restarting the app (as long as you reuse the same thread_id).

Free resources used (no API key, no cost, no rate limits):
  - LLM:        Ollama, local (default: qwen2.5:0.5b)
  - Embeddings: HuggingFace sentence-transformers, local
  - Vector DB:  Chroma, local
  - Sparse retriever: BM25 (rank_bm25), pure Python
  - Re-ranker:  sentence-transformers CrossEncoder, local
  - Memory:     LangGraph SqliteSaver — local SQLite file, free

Setup:
    1. Install Ollama: https://ollama.com  and  ollama pull qwen2.5:0.5b
    2. pip install -r requirements.txt
    3. cp .env.example .env
    4. python ingest.py       (indexes ./data)
    5. python app.py           (start chatting — remembers earlier turns,
                                 even across restarts)
"""

import os
import sys
import sqlite3
import pickle
from collections import defaultdict
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from sentence_transformers import CrossEncoder

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

PERSIST_DIR = "chroma_db"
CHUNKS_PATH = os.path.join(PERSIST_DIR, "chunks.pkl")
CHECKPOINT_DB = "checkpoints.sqlite"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DENSE_K = int(os.getenv("DENSE_K", "6"))
BM25_K = int(os.getenv("BM25_K", "6"))
RRF_K = int(os.getenv("RRF_K", "60"))     # standard RRF smoothing constant
FINAL_K = int(os.getenv("FINAL_K", "4"))
THREAD_ID = os.getenv("THREAD_ID", "terminal-session")

CLASSIFY_PROMPT = """Decide whether answering the user's latest message requires
searching a knowledge base of product documentation, or whether it can be
answered directly (e.g. greetings, thanks, general chit-chat, or something
already answered earlier in this conversation).

Reply with exactly one word: "retrieve" or "direct"."""

REWRITE_PROMPT = """Given the chat history and the latest user question, rewrite the
question to be a standalone question understandable without the chat history.
If it's already standalone, return it unchanged. Reply with ONLY the rewritten
question, nothing else."""

SYSTEM_PROMPT = """You are a helpful, precise assistant. If context from a knowledge
base is provided below, use ONLY that context to answer — if the answer isn't in
it, say you don't know rather than making something up, and mention the source
file name(s) when relevant. If no context is provided, answer directly and
concisely.

Context:
{context}"""


class ChatState(MessagesState):
    context: str
    route: str
    standalone_question: str


def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        print("No indexed chunks found. Run `python ingest.py` first.")
        sys.exit(1)
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


def rrf_fuse(ranked_lists, k=RRF_K):
    """Reciprocal Rank Fusion: combines multiple ranked document lists
    (e.g. from dense and BM25 retrievers) into one fused ranking, without
    depending on langchain.retrievers.EnsembleRetriever (which lives in
    the top-level `langchain` package and has moved/broken across recent
    LangChain releases). score(doc) = sum(1 / (k + rank)) across lists."""
    scores = defaultdict(float)
    doc_lookup = {}
    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = (doc.metadata.get("source", ""), doc.page_content)
            scores[key] += 1.0 / (k + rank + 1)
            doc_lookup[key] = doc
    fused_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_lookup[key] for key in fused_keys]


def build_graph():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run `python ingest.py` first.")
        sys.exit(1)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": DENSE_K})

    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    reranker = CrossEncoder(RERANKER_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    # --- Nodes ---------------------------------------------------------

    def classify(state: ChatState):
        last_message = state["messages"][-1]
        history = state["messages"][:-1][-4:]  # a little recent context helps
        messages = [SystemMessage(content=CLASSIFY_PROMPT)] + history + [last_message]
        response = llm.invoke(messages)
        route = "retrieve" if "retrieve" in response.content.lower() else "direct"
        return {"route": route}

    def route_after_classify(state: ChatState):
        return state["route"]

    def rewrite_query(state: ChatState):
        last_message = state["messages"][-1]
        history = state["messages"][:-1][-6:]
        if not history:
            return {"standalone_question": last_message.content}
        messages = [SystemMessage(content=REWRITE_PROMPT)] + history + [last_message]
        response = llm.invoke(messages)
        return {"standalone_question": response.content.strip()}

    def retrieve_and_rerank(state: ChatState):
        """Node: run hybrid retrieval — dense (Chroma) + BM25 (keyword),
        fused with hand-written Reciprocal Rank Fusion (rrf_fuse) rather
        than langchain.retrievers.EnsembleRetriever — then re-rank the
        fused candidates with a cross-encoder for a more precise final
        ordering. Combined into one node because LangGraph state only
        persists explicitly declared fields — keeping the raw candidate
        list out of state (it's just an intermediate value) is simpler
        than adding a field for it."""
        query = state["standalone_question"]
        dense_docs = dense_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        candidates = rrf_fuse([dense_docs, bm25_docs])
        if not candidates:
            return {"context": ""}
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked[:FINAL_K]]
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in top_docs
        )
        return {"context": context}

    def generate(state: ChatState):
        context = state.get("context", "")
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=context or "(none)"))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    # --- Graph -----------------------------------------------------------

    graph = StateGraph(ChatState)
    graph.add_node("classify", classify)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve_and_rerank", retrieve_and_rerank)
    graph.add_node("generate", generate)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"retrieve": "rewrite_query", "direct": "generate"},
    )
    graph.add_edge("rewrite_query", "retrieve_and_rerank")
    graph.add_edge("retrieve_and_rerank", "generate")
    graph.add_edge("generate", END)

    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)


def main():
    print("=" * 60)
    print("Advanced RAG Chatbot (LangGraph) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Conditional routing + query rewriting + hybrid retrieval + re-ranking")
    print(f"Memory persisted to ./{CHECKPOINT_DB} (thread: {THREAD_ID})")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    app = build_graph()
    config = {"configurable": {"thread_id": THREAD_ID}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if query.lower() in {"exit", "quit"}:
            print("Goodbye! Conversation saved — resume it by running app.py again.")
            break
        if not query:
            continue

        result = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        answer = result["messages"][-1].content
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()
