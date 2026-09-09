# RAG Chatbot using LlamaIndex (Basic + Advanced)

Two terminal-based RAG chatbots built with **LlamaIndex**. Both read documents from a local `data/` folder, index them into a Chroma-backed `VectorStoreIndex`, and let you chat with them from the command line.

## ⚠️ This project currently runs on 100% free resources — no API keys, no cost, no rate limits

| Component | Choice | Why it's free |
|---|---|---|
| LLM | **Ollama** (local, default `qwen2.5:0.5b`) | Runs entirely on your machine, no account or API key |
| Embeddings | **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** | Open-source model, local |
| Vector DB | **Chroma** | Open-source, local disk storage |
| Sparse/keyword retrieval (advanced only) | **BM25** (`llama-index-retrievers-bm25`) | Local, no external service |
| Re-ranker (advanced only) | **Cross-encoder** (`SentenceTransformerRerank`) | Local model, no LLM calls |

### Switching to paid resources instead

Only the `Settings.llm` / `Settings.embed_model` assignments need to change — the index, retrieval pipeline, and chat engine are provider- agnostic. See the **"Switching to paid resources"** section of `EXPLANATION.md` for exact code; in short: install `llama-index-llms-openai` and `llama-index-embeddings-openai`, swap `Ollama` → `OpenAI` and `HuggingFaceEmbedding` → `OpenAIEmbedding`, add `OPENAI_API_KEY` to `.env`, and re-run `python ingest.py`.

```
llamaindex_chatbot/
├── README.md
├── EXPLANATION.md
├── basic/
│   ├── app.py            # query engine, single-turn, no memory
│   ├── ingest.py           # builds the Chroma-backed index from ./data
│   ├── requirements.txt
│   ├── .env.example
│   └── data/sample.md
└── advanced/
    ├── app.py            # chat engine: hybrid retrieval + rerank + memory
    ├── ingest.py           # builds the index + persists the docstore for BM25
    ├── requirements.txt
    ├── .env.example
    └── data/sample.md
```

## A note on package versions

This project deliberately does **not** pin exact package versions in `requirements.txt` — `llama-index`, `chromadb`, and the HuggingFace/ sentence-transformers ecosystem all ship frequent breaking changes, and a version pinned today may already be unavailable by the time you install it. Install with `pip install -r requirements.txt`; if you hit a version-compatibility error, try `pip install --upgrade -r requirements.txt` and check that project's error message against the [llama-index changelog](https://github.com/run-llama/llama_index/blob/main/CHANGELOG.md) if needed.

## One-time setup (shared by both versions)

1. Install Ollama (free): https://ollama.com
2. Pull the model:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
3. Make sure the Ollama server is running (`ollama serve`, or it starts
   automatically). Default: `http://localhost:11434`.

## Basic version

Loads documents with `SimpleDirectoryReader`, chunks them with `SentenceSplitter`, embeds with HuggingFace, and builds a `VectorStoreIndex` on top of Chroma. Answers each question independently via `index.as_query_engine()` — no memory.

```bash
cd basic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python app.py
```

## Advanced version

Uses a `CondensePlusContextChatEngine`, which:
- rewrites follow-up questions into standalone ones using chat history,
- retrieves with a `QueryFusionRetriever` that combines BM25 + dense retrieval **and** expands the query into a few alternate phrasings before fusing everything with Reciprocal Rank Fusion,
- re-ranks the fused candidates with a local cross-encoder (`SentenceTransformerRerank`),
- and keeps conversation memory in a `ChatMemoryBuffer` that's persisted to `chat_histories/chat_store.json` after every turn, so conversations survive restarting the app.

```bash
cd advanced
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python app.py
```

## Using your own documents

Drop `.txt`, `.md`, or `.pdf` files into `data/` (either project) and re-run `python ingest.py`.

## Notes
- `chroma_db/` and `chat_histories/` are created locally; delete them to reset state.
- See `EXPLANATION.md` for how the retrieval pipeline and chat engine work, how this compares to the LangChain and LangGraph chatbot projects, and the paid-resource swap instructions.
