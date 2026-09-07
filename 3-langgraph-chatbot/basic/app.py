"""
Basic RAG Chatbot using LangGraph — free & local stack
-----------------------------------------------------------
A terminal chatbot built as a small LangGraph graph:

    START -> retrieve -> generate -> END

Unlike LangChain's chain abstractions (RetrievalQA, create_retrieval_chain),
LangGraph makes the control flow an explicit graph of nodes and edges over
a shared state object. This basic version's graph is a straight line (no
branching), but it already gets multi-turn conversational memory "for
free" via LangGraph's checkpointer, which persists the graph's state
(the message list) between invocations of the same thread_id — no manual
history bookkeeping required.

Free resources used (no API key, no cost, no rate limits):
  - LLM:        Ollama, running a free open-weight model locally
                (default: qwen2.5:0.5b — swap via OLLAMA_MODEL)
  - Embeddings: HuggingFace sentence-transformers, running locally
                (default: all-MiniLM-L6-v2)
  - Vector DB:  Chroma — free, open-source, stored on local disk
  - Memory:     LangGraph's built-in MemorySaver checkpointer (free,
                in-process — lasts for the current run)

Setup:
    1. Install Ollama: https://ollama.com  (one-time, free)
    2. Pull a model:   ollama pull qwen2.5:0.5b
    3. pip install -r requirements.txt
    4. cp .env.example .env   (defaults already work — edit only if needed)
    5. python ingest.py       (indexes ./data)
    6. python app.py           (start chatting — remembers earlier turns)
"""

import os
import sys
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = """You are a helpful assistant. Use the retrieved context below to
answer the user's question. If the answer isn't in the context, say you don't know
rather than making something up.

Context:
{context}"""


class ChatState(MessagesState):
    """LangGraph's built-in MessagesState already manages a `messages`
    list with automatic append semantics (new messages returned by a node
    are appended, not overwritten). We extend it with a `context` field
    used to pass retrieved chunks from the retrieve node to the generate
    node."""
    context: str


def load_vectorstore():
    if not os.path.exists(PERSIST_DIR):
        print("No vector store found. Run `python ingest.py` first.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def build_graph():
    vectorstore = load_vectorstore()
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    def retrieve(state: ChatState):
        """Node: embed the latest human message and search Chroma."""
        last_message = state["messages"][-1]
        docs = vectorstore.similarity_search(last_message.content, k=TOP_K)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        return {"context": context}

    def generate(state: ChatState):
        """Node: call the LLM with the system prompt (+ retrieved context)
        and the full message history, then append its reply."""
        system = SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def main():
    print("=" * 60)
    print("Basic RAG Chatbot (LangGraph) — free & local (Ollama + HF embeddings)")
    print(f"LLM: {OLLAMA_MODEL} via Ollama @ {OLLAMA_BASE_URL}")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print("Note: make sure `ollama serve` is running and the model is pulled")
    print(f"      (ollama pull {OLLAMA_MODEL}) before asking questions.\n")

    app = build_graph()
    # One conversation thread for this terminal session — LangGraph's
    # checkpointer keys memory by thread_id, so re-running with the same
    # id (or invoking with it again) would resume that conversation.
    config = {"configurable": {"thread_id": "terminal-session"}}

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

        result = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        answer = result["messages"][-1].content
        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()
