# RAG Chatbot — No Framework (Basic + Advanced)

Two terminal-based RAG chatbots built with **plain Python** — no LangChain, no LlamaIndex, no LangGraph. Every step (loading documents, chunking, embedding, storing/searching vectors, calling the LLM, fusing retrieval results, re-ranking, memory) is written directly against the underlying libraries/APIs, so you can see exactly what a RAG framework is doing under the hood.

## ⚠️ This project currently runs on 100% free resources — no API keys, no cost, no rate limits

| Component | Choice | Why it's free |
|---|---|---|
| LLM | **Ollama** (local, default `qwen2.5:0.5b`), called via its REST API with `requests` | Runs entirely on your machine, no account or API key |
| Embeddings | **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** | Open-source model, downloaded once and run locally |
| Vector DB | **Chroma** (`chromadb`, embedded) | Open-source, local disk storage |
| Sparse/keyword retrieval (advanced only) | **BM25** (`rank_bm25`) | Pure Python, no external service |
| Re-ranker (advanced only) | **Cross-encoder** (`sentence-transformers` `CrossEncoder`) | Open-source model, runs locally, no LLM calls needed |

### Switching to paid resources instead

Because there's no framework abstraction layer, swapping to a paid provider means changing a small, explicit set of functions rather than a framework's constructor — see the **"Switching to paid resources"** section in `explanation.md` for exact code. In short:

1. Replace the `call_ollama()` function's `requests.post(...)` call with a call to your paid provider's chat completion endpoint (e.g. OpenAI's `POST /v1/chat/completions`), or use their official Python SDK.
2. Replace `SentenceTransformer(...).encode(...)` with a call to your paid provider's embeddings endpoint (e.g. OpenAI's `POST /v1/embeddings`).
3. Add the relevant API key to `.env` and read it with `os.getenv(...)`.
4. Re-run `python ingest.py` after changing embeddings — different embedding models are not compatible with an existing index.

```
1-basic-chatbot/
├── README.md
├── EXPLANATION.md
├── basic-rag/
│   ├── app.py            # terminal chat loop (single-turn RAG)
│   ├── ingest.py          # builds the Chroma collection from ./data
│   ├── requirements.txt
│   ├── .env.example
│   └── data/sample.md      # sample knowledge base — replace with your own
└── advanced-rag/
    ├── app.py            # terminal chat loop (hybrid retrieval + rerank + memory)
    ├── ingest.py          # builds Chroma collection + BM25 chunk store
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

Loads documents → chunks them with a hand-written splitter → embeds with `sentence-transformers` → stores/searches in Chroma → builds a prompt with the retrieved chunks → calls Ollama directly over HTTP. No memory (single-turn).

```bash
cd basic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py
python app.py
```

## Advanced version

Adds hybrid retrieval (BM25 + Chroma dense search fused with **Reciprocal Rank Fusion**, implemented from scratch), **cross-encoder re-ranking** of the fused candidates, an LLM call that rewrites follow-up questions into standalone questions using chat history, and multi-turn history persisted to disk.

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
- See `explanation.md` for how retrieval/fusion/re-ranking work here, how this compares to the framework-based version of the same chatbot (`langchain_chatbot`), and the paid-resource swap instructions.
