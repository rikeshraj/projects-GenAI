# RAG Chatbot using LangGraph (Basic + Advanced)

Two terminal-based RAG chatbots built with **LangGraph** — LangChain's graph-based orchestration library. Instead of composing a fixed chain (as in the `langchain_chatbot` project), each chatbot here is expressed as an explicit graph of nodes and edges over a shared state object.

## ⚠️ This project currently runs on 100% free resources — no API keys, no cost, no rate limits

| Component | Choice | Why it's free |
|---|---|---|
| LLM | **Ollama** (local, default `qwen2.5:0.5b`) | Runs entirely on your machine, no account or API key |
| Embeddings | **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** | Open-source model, local |
| Vector DB | **Chroma** | Open-source, local disk storage |
| Sparse retriever (advanced only) | **BM25** (`rank_bm25`) | Pure Python, no external service |
| Re-ranker (advanced only) | **Cross-encoder** (`sentence-transformers`) | Local model, no LLM calls |
| Memory (basic) | **LangGraph `MemorySaver`** | Free, in-process checkpointer |
| Memory (advanced) | **LangGraph `SqliteSaver`** | Free, local SQLite file — persists across restarts |

### Switching to paid resources instead

Only the model/embeddings instantiation lines need to change — the graph structure (nodes, edges, conditional routing) is provider-agnostic. See the **"Switching to paid resources"** section of `explanation.md` for exact code; in short: install `langchain-openai`, swap `ChatOllama` → `ChatOpenAI` and `HuggingFaceEmbeddings` → `OpenAIEmbeddings`, add `OPENAI_API_KEY` to `.env`, and re-run `python ingest.py`.

```
langgraph_chatbot/
├── readme.md
├── explanation.md
├── basic/
│   ├── app.py            # StateGraph: retrieve -> generate, with checkpointer memory
│   ├── ingest.py           # builds the Chroma vector store from ./data
│   ├── requirements.txt
│   ├── .env.example
│   └── data/sample.md
└── advanced/
    ├── app.py            # StateGraph with conditional routing, rewriting, hybrid retrieval, re-ranking, and SQLite-persisted memory
    ├── ingest.py           # builds Chroma index + BM25 chunk cache
    ├── requirements.txt
    ├── .env.example
    └── data/sample.md
```

## One-time setup (shared by both versions)

1. Install Ollama (free): https://ollama.com
2. Pull a model:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
3. Make sure the Ollama server is running (`ollama serve`, or it starts automatically). Default: `http://localhost:11434`.

## Basic version

Graph: `START -> retrieve -> generate -> END`. Every turn retrieves from chroma unconditionally, same as the basic LangChain chatbot — but here multi-turn memory comes from LangGraph's checkpointer rather than a framework memory class, and the control flow is an explicit graph you can inspect and extend node by node.

```bash
cd basic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python app.py
```

## Advanced version

A graph with real branching:

```
START -> classify --(needs retrieval?)--> rewrite_query -> retrieve_and_rerank -> generate -> END
                    \--(no)---------------------------------------------------> generate -> END
```

- **classify** — the LLM decides whether the question needs the knowledge base at all (skips retrieval for greetings/chit-chat).
- **rewrite_query** — turns follow-up questions into standalone ones using chat history.
- **retrieve_and_rerank** — hybrid BM25 + dense retrieval (fused with hand-written Reciprocal Rank Fusion), then cross-encoder re-ranking.
- **generate** — answers using the retrieved context, or directly if the classify step routed around retrieval.
- **Memory** — persisted to a local `checkpoints.sqlite` file via LangGraph's `SqliteSaver`, so conversations survive restarting the app (reuse the same `THREAD_ID` in `.env` to resume one).

```bash
cd advanced
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python app.py
```

Try asking something conversational first ("hi, how's it going?") and then a knowledge-base question — you'll see in the retrieval quality (and the classify step's routing) that the graph is actually branching, not always taking the same path.

## Using your own documents

Drop `.txt`, `.md`, or `.pdf` files into `data/` (either project) and re-run `python ingest.py`.

## Notes
- `chroma_db/` and `checkpoints.sqlite` are created locally; delete them to reset state.
- See `explanation.md` for how the graph works, how LangGraph's approach compares to LangChain's chain-based `langchain_chatbot`, and the paid-resource swap instructions.
