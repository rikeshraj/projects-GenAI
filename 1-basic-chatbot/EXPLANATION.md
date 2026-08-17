# Explanation: RAG Chatbot — No Framework

## Why build this without a framework?

Frameworks like LangChain, LlamaIndex, and LangGraph package RAG patterns (retrievers, chains, agents) into reusable abstractions. That's convenient, but it can hide what's actually happening. This project implements the same basic and advanced RAG pipelines as the `langchain_chatbot` project using nothing but the underlying libraries directly — `chromadb`, `sentence-transformers`, `rank_bm25`, and plain HTTP calls to Ollama — so every step is visible and easy to reason about.

## Basic version — architecture

```
User question
     │
     ▼
sentence-transformers encodes the question into a vector
     │
     ▼
Chroma collection.query() — cosine similarity search, top-k chunks
     │
     ▼
Chunks concatenated into a plain-text prompt (manual string formatting)
     │
     ▼
requests.post() to Ollama's /api/chat REST endpoint — LLM generates the answer
```

No memory: every question is handled independently.

### Chunking, written by hand
`chunk_text()` in `ingest.py` is a small sliding-window splitter: it tries to end each chunk at a paragraph break or sentence boundary near the target size, and falls back to a hard cut otherwise, with a configurable overlap between consecutive chunks. This does what LangChain's `RecursiveCharacterTextSplitter` does, just written out explicitly.

## Advanced version — architecture

```
User question + chat history
     │
     ▼
call_ollama() with a rewrite prompt → standalone question
     │
     ├──────────────┬───────────────┐
     ▼               ▼
BM25Okapi.get_scores()   Chroma collection.query()
(keyword ranking)         (dense/semantic ranking)
     │               │
     └──────┬────────┘
            ▼
   Reciprocal Rank Fusion (hand-written _rrf_fuse)
            │
            ▼
   CrossEncoder.predict() re-ranks the fused candidates
            │
            ▼
   Top FINAL_K chunks → prompt → call_ollama() → answer
            │
            ▼
   history.append(...) → save_history() writes JSON to disk
```

### Why each piece is there

- **Query rewriting** — a follow-up like "what about the Pro plan?" is ambiguous on its own. Before retrieval, a small LLM call rewrites it into a standalone question using the last few turns of chat history, the same way LangChain's `create_history_aware_retriever` does — here it's just one explicit function (`rewrite_query`) and one HTTP call.

- **Hybrid retrieval via Reciprocal Rank Fusion (RRF)** — BM25 (sparse) is strong on exact keyword/term matches; dense embedding search is strong on semantic similarity but can miss exact wording. Rather than merging their raw scores (which live on different, incomparable scales), RRF combines their **rankings**: each retriever produces a ranked list, and a document's fused score is the sum of `1 / (k + rank)` across all the lists it appears in. This is the same fusion technique used inside many frameworks' "ensemble retriever" classes — here it's ~10 lines of plain Python (`_rrf_fuse`).

- **Cross-encoder re-ranking** — a **bi-encoder** (like the `all-MiniLM-L6-v2` embedding model) encodes the query and each document independently, then compares vectors — fast, but less precise. A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) feeds the query and a candidate chunk into the model *together*, letting it directly model how well they match — slower per pair, but much more accurate, and cheap enough to run on the small fused candidate set from RRF. This is a deliberately different choice from the `langchain_chatbot` advanced version, which re-ranks using LLM calls (`ContextualCompressionRetriever`) — the cross-encoder approach here is faster and doesn't consume any LLM calls at all.

- **Persisted chat history** — a plain Python list of `{"role", "content"}` dicts, appended to after every turn and written to `chat_histories/<session_id>.json` — no framework message-history class involved.

## Basic vs Advanced — trade-offs

| | Basic | Advanced |
|---|---|---|
| Retrieval | Dense only | Hybrid (BM25 + dense) fused with RRF |
| Re-ranking | None | Cross-encoder re-ranking of fused candidates |
| Memory | None (single-turn) | Multi-turn, persisted to disk |
| LLM calls per question | 1 | 2 (rewrite + answer) |
| Extra local model loaded | embedding model only | + cross-encoder re-ranker |
| Answer quality on ambiguous / follow-up questions | Weaker | Stronger |

Because re-ranking here uses a cross-encoder instead of extra LLM calls, the advanced version in this project is actually **cheaper in LLM calls** than the advanced `langchain_chatbot` (2 vs 4–6 per question), at the cost of loading one more local model into memory.

## Comparing to the framework version (`langchain_chatbot`)

| | No framework (this project) | LangChain version |
|---|---|---|
| Retrieval fusion | Hand-written RRF | `EnsembleRetriever` |
| Query expansion | Not included (query rewriting only) | `MultiQueryRetriever` |
| Re-ranking | Cross-encoder (local model, no LLM calls) | `ContextualCompressionRetriever` (LLM calls) |
| Memory | Plain list + manual JSON persistence | `RunnableWithMessageHistory` |
| Lines of orchestration code | More (everything explicit) | Fewer (framework handles wiring) |
| Flexibility to customize any single step | Total — it's all your code | High, but within the framework's abstractions |

Neither is "better" — the framework version is faster to build on and easier to extend with more framework features; this version is more transparent and has fewer moving dependencies.

## ⚠️ Currently free — how to switch to paid resources

This project is configured to run entirely on free, local resources. To use a paid hosted provider instead, only a few functions change — the retrieval logic (chunking, RRF fusion, re-ranking, memory) doesn't need to change at all.

### 1. Swap the LLM (e.g. to OpenAI)
Replace `call_ollama()`'s body with a call to OpenAI's chat completions endpoint:
```python
import openai
client = openai.OpenAI()  # reads OPENAI_API_KEY from env

def call_llm(messages, temperature=0.0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
```
(and update the call sites from `call_ollama(...)` to `call_llm(...)`).

### 2. Swap the embeddings (e.g. to OpenAI)
In both `ingest.py` and `app.py`, replace:
```python
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts).tolist()
```
with:
```python
import openai
client = openai.OpenAI()

def embed(texts):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]

embeddings = embed(texts)
```
**Important:** after switching embedding models, delete `chroma_db/` and re-run `python ingest.py` — embeddings from different models aren't comparable.

### 3. Add the API key
```
OPENAI_API_KEY=your_key_here
```

### Other providers
Any provider with a REST API or Python SDK works the same way — write a small `call_llm()` / `embed()` function for it (Anthropic, Google, Cohere, Groq, etc.) and swap the call sites. Because this project never depended on a framework's provider abstraction, there's no framework-side compatibility to worry about — you're just calling a different API.

## Extending this project
- Add a second sparse signal (e.g. TF-IDF) into the RRF fusion.
- Stream the LLM response by setting `"stream": True` in the Ollama request and reading the response line by line.
- Swap Chroma for FAISS (also free/local) if you don't need Chroma's metadata filtering.
- Add basic query classification (e.g. skip retrieval entirely for greetings/small talk) before running the retrieval pipeline.
