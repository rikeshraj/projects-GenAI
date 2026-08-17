# Terminal RAG Chatbots & Agents

A collection of terminal-based Retrieval-Augmented Generation (RAG) projects — chatbots and agents — built with three frameworks (LangChain, LangGraph, LlamaIndex), each in a **basic** and an **advanced** RAG variant. Every project ships as a self-contained folder/zip with its own `readme.md` (setup & usage) and `explanation.md` (architecture & design rationale).

## Free-by-default policy

Every project in this repo is built to run on **100% free, local resources** by default — no API keys, no cost, no rate limits:

| Layer | Default free choice |
|---|---|
| LLM | **Ollama** (local open-weight models, e.g. `llama3.2`) |
| Embeddings | **HuggingFace `sentence-transformers`** (local, e.g. `all-MiniLM-L6-v2`) |
| Vector DB | **Chroma** (open-source, local/embedded) |
| Sparse/keyword retrieval (advanced variants) | **BM25** (`rank_bm25`, pure Python) |

Every project's `explanation.md` also includes a **"switch to paid resources"** section with the exact code changes needed to swap in a hosted provider (OpenAI, Anthropic, Google, Cohere, Groq, etc.) if you want stronger models or don't want to run anything locally.

## Projects

Projects are listed in the order they were built. Each row is one downloadable zip containing both the basic and advanced RAG variant for that framework/type combination.

| # | Project | Type | Framework | RAG Variants | Core Tech Stack | Status |
|---|---|---|---|---|---|---|
| 1 | [`no_framework_chatbot`](./no_framework_chatbot) | Chatbot | **None (pure Python)** | Basic + Advanced | **Basic:** hand-written chunker, `sentence-transformers` embeddings, `chromadb` dense retrieval, Ollama called directly via REST (`requests`). **Advanced:** + BM25 (`rank_bm25`) hybrid search fused with hand-written Reciprocal Rank Fusion, cross-encoder re-ranking (`sentence-transformers` `CrossEncoder`), LLM-based history-aware query rewriting, JSON-persisted multi-turn memory | ✅ Completed |
| 2 | [`langchain_chatbot`](./langchain_chatbot) | Chatbot | LangChain | Basic + Advanced | **Basic:** Chroma dense retrieval, `RetrievalQA`, Ollama LLM, HF embeddings. **Advanced:** + BM25 hybrid search (`EnsembleRetriever`), LLM query expansion (`MultiQueryRetriever`), LLM re-ranking/compression (`ContextualCompressionRetriever`), history-aware retriever, persisted multi-turn memory (`RunnableWithMessageHistory`) | ✅ Completed |
| 3 | `langgraph_chatbot` | Chatbot | LangGraph | Basic + Advanced | Graph-based conversational RAG: `StateGraph` nodes for retrieve/generate, checkpointer-based memory (basic); planned advanced additions: conditional routing, query rewriting node, hybrid retrieval, re-ranking | ⏳ Planned |
| 4 | `llamaindex_chatbot` | Chatbot | LlamaIndex | Basic + Advanced | `VectorStoreIndex` over Chroma, `HuggingFaceEmbedding`, `Ollama` LLM, `ChatEngine` (basic); planned advanced additions: hybrid retriever, node post-processors/re-ranking, sub-question query engine | ⏳ Planned |
| 5 | [`no_framework_agent`](./no_framework_agent) | Agent | **None (pure Python)** | Basic + Advanced | **Basic:** hand-written tool-calling loop against Ollama's native `/api/chat` tools API, single tool (`retrieve_documents` over dense Chroma), no memory. **Advanced:** + 2 more tools (safe AST-based `calculator`, `current_datetime`), `retrieve_documents` upgraded to hybrid BM25+dense retrieval (RRF fusion) with cross-encoder re-ranking, multi-step tool chaining, full conversation persisted to disk | ✅ Completed |
| 6 | `langchain_agent` | Agent | LangChain | Basic + Advanced | Tool-using ReAct-style agent (`create_react_agent` / `AgentExecutor`) with a RAG-retriever tool, Ollama LLM (basic); planned advanced additions: multiple tools (calculator, web search), self-correction/reflection, hybrid retrieval tool | ⏳ Planned |
| 7 | `langgraph_agent` | Agent | LangGraph | Basic + Advanced | Graph-based tool-calling agent with explicit control flow and state (basic); planned advanced additions: multi-step planning, human-in-the-loop / interrupt points, persistent checkpointed memory across sessions | ⏳ Planned |
| 8 | `llamaindex_agent` | Agent | LlamaIndex | Basic + Advanced | `ReActAgent` / `FunctionAgent` with a query-engine-as-tool over the RAG index (basic); planned advanced additions: multiple tools, query planning, sub-question decomposition | ⏳ Planned |

> Stack details for "Planned" rows describe the intended design and will be finalized (and this table updated) as each project is delivered.

## Repository structure

```
.
├── README.md                    # this file
├── no_framework_chatbot/         # project 1 — pure Python, no framework
│   ├── readme.md
│   ├── explanation.md
│   ├── basic/
│   └── advanced/
├── langchain_chatbot/              # project 2
│   ├── readme.md
│   ├── explanation.md
│   ├── basic/
│   └── advanced/
├── langgraph_chatbot/               # project 3 (coming soon)
├── llamaindex_chatbot/              # project 4 (coming soon)
├── no_framework_agent/            # project 5 — pure Python, no framework
│   ├── readme.md
│   ├── explanation.md
│   ├── basic/
│   └── advanced/
├── langchain_agent/                  # project 6 (coming soon)
├── langgraph_agent/                   # project 7 (coming soon)
└── llamaindex_agent/                  # project 8 (coming soon)
```

> The two `no_framework_*` projects implement the exact same RAG patterns (chunking, dense/hybrid retrieval, re-ranking, memory) as their framework-based counterparts, but written directly against the underlying libraries (`chromadb`, `sentence-transformers`, `rank_bm25`, plain HTTP calls to Ollama) with no LangChain / LlamaIndex / LangGraph involved. They're a good place to see what a framework is actually doing under the hood before using the framework versions.

Each project folder is fully self-contained (its own `requirements.txt`, `.env.example`, sample `data/`, and docs) — you can copy any single project folder out on its own and run it independently.

## Basic vs Advanced, at a glance

- **Basic** — the minimum correct RAG loop for that framework: load → chunk → embed → store → retrieve → generate. Good for learning the framework's core API and for small, simple document sets.
- **Advanced** — adds the techniques that meaningfully improve real-world RAG quality: hybrid (keyword + semantic) retrieval, query expansion/rewriting, re-ranking/compression, and persistent multi-turn conversational memory. Costs more latency (more LLM calls)  in exchange for better answers on larger or more ambiguous document sets.

## Getting started

1. Pick a project folder (or download its zip).
2. Read that project's `readme.md` for exact setup steps.
3. Install Ollama once (https://ollama.com) and pull a model — this is
   shared across all projects in this repo:
   ```bash
   ollama pull llama3.2
   ```
4. `pip install -r requirements.txt` inside the `basic/` or `advanced/`
   folder you want to run, then follow that project's ingest → run steps.

## License

Use and adapt freely for learning and experimentation.
