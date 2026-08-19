# RAG Chatbot using LangChain (Basic + Advanced)

Two terminal-based Retrieval-Augmented Generation (RAG) chatbots built with LangChain. Both read documents from a local `data/` folder, index them into a vector store, and let you chat with them from the command line.

## ⚠️ This project currently runs on 100% free resources — no API keys, no cost, no rate limits

| Component | Choice | Why it's free |
|---|---|---|
| LLM | **Ollama** running an open-weight model locally (default `qwen2.5:0.5b`) | Runs entirely on your machine, no account or API key |
| Embeddings | **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** | Open-source model, downloaded once and run locally |
| Vector DB | **Chroma** | Open-source, embedded, stored on local disk |
| Sparse retriever (advanced only) | **BM25 (`rank_bm25`)** | Pure Python, no external service |

Nothing in this project calls a paid API. The only one-time setup cost is installing Ollama and pulling a model, both free.

### Switching to paid resources instead

The pipeline is provider-agnostic — swapping in paid services only means changing the model/embeddings instantiation and adding an API key. You do not need to touch the retrieval logic. See the **"Switching to paid resources"** section of `EXPLANATION.md` for the exact code changes; in short:

1. `pip install langchain-openai` (or another provider's LangChain package)
2. In `app.py` (and `ingest.py` for embeddings), replace:
   - `ChatOllama(...)` → `ChatOpenAI(model="gpt-4o-mini", ...)`
   - `HuggingFaceEmbeddings(...)` → `OpenAIEmbeddings(...)`
3. Add `OPENAI_API_KEY=your_key_here` to `.env`
4. Re-run `python ingest.py` (embeddings changed, so the index must be rebuilt)

The same swap pattern works for Anthropic, Google, Cohere, or any other LangChain-supported provider — only the import and constructor change.

```
langchain_chatbot/
├── README.md
├── EXPLANATION.md
├── basic/
│   ├── app.py            # terminal chat loop (single-turn RAG)
│   ├── ingest.py          # builds the Chroma vector store from ./data
│   ├── requirements.txt
│   ├── .env.example
│   └── data/sample.md      # sample knowledge base — replace with your own
└── advanced/
    ├── app.py            # terminal chat loop (hybrid retrieval + memory)
    ├── ingest.py          # builds Chroma index + BM25 chunk cache
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
   `qwen2.5:0.5b` is already one of the smallest capable models — this is the default used throughout this project. If your machine can spare more RAM and you'd like stronger answers, try
   `ollama pull qwen2.5:3b` instead.
3. Make sure the Ollama server is running (it usually starts automatically, or run `ollama serve` manually). It listens on `http://localhost:11434` by default.

## Basic version

Single-turn Retrieval QA: retrieves the top-k most similar chunks with dense (embedding) search and stuffs them into the prompt.

```bash
cd basic
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env        # defaults already work; edit only if you use a different model
python ingest.py            # indexes documents in ./data (downloads the free embedding model once)
python app.py                # start chatting
```

## Advanced version

Multi-turn conversational RAG with a stronger retrieval pipeline: hybrid (keyword + semantic) search, LLM-based query expansion, LLM-based re-ranking/compression of retrieved chunks, and chat history that is persisted to disk.

```bash
cd advanced
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python ingest.py            # indexes documents in ./data, caches chunks for BM25
python app.py                # start chatting (remembers earlier turns)
```

Note: the advanced pipeline makes several LLM calls per question (query rewrite, multi-query expansion, compression, final answer). Even with the small `qwen2.5:0.5b` default this means a few sequential CPU-bound calls, so the advanced version will feel noticeably slower than the basic version — that's expected. If it feels too slow, you can trade quality for speed by lowering `k` values in `.env`, or trade speed for quality with a larger model like `qwen2.5:3b`.

## Using your own documents

Drop `.txt`, `.md`, or `.pdf` files into `data/` (either project) and re-run `python ingest.py`. The chatbot will only answer from what it has indexed.

## Notes
- The vector store (`chroma_db/`) and chat histories (`chat_histories/`) are created locally and are safe to delete to reset state.
- Want to swap in a different free model? Any model pulled into Ollama works — just change `OLLAMA_MODEL` in `.env`. To use a different free embedding model, change `EMBEDDING_MODEL` to any other `sentence-transformers` model name from the HuggingFace Hub.
- See `EXPLANATION.md` for how the retrieval pipelines work, why the advanced version differs from the basic one, the free-stack choices, and step-by-step instructions for switching to paid resources.
