# Explanation: RAG Chatbot using LangChain

## What is RAG?
Retrieval-Augmented Generation (RAG) combines a retrieval step (finding relevant text from a knowledge base) with a generation step (an LLM writing an answer grounded in that text). Instead of relying purely on what the model memorized during training, the model is given fresh, specific context at answer time — which reduces hallucination and lets the chatbot answer questions about documents it has never seen before.

A minimal RAG pipeline has four stages:
1. **Load** — read raw documents (txt, md, pdf, etc.).
2. **Split** — break documents into overlapping chunks so each piece fits in the model's context window and stays topically focused.
3. **Embed & store** — convert each chunk into a vector (embedding) and store it in a vector database for similarity search.
4. **Retrieve & generate** — for a user question, find the most similar chunks and pass them to the LLM alongside the question to produce a grounded answer.

## Why this project uses a free / local stack

| Layer | Common paid option | Free option used here | Trade-off |
|---|---|---|---|
| LLM | OpenAI GPT-4o / Anthropic Claude via API | **Ollama** running an open-weight model (e.g. Llama 3.2) locally | No per-token cost or key, but needs local compute; smaller open models are somewhat weaker reasoners than frontier hosted models |
| Embeddings | OpenAI `text-embedding-3-small` | **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** locally | No cost or network call after the one-time download; slightly lower embedding quality than top hosted models, but very solid for most RAG use cases |
| Vector DB | Pinecone / managed Weaviate | **Chroma** (embedded, local) | No hosting cost or account; not built for massive multi-tenant scale, but ideal for a single-user terminal app |

This means the project runs completely offline after the initial model downloads — no signup, no API key, no usage limits, and no bill.

## Basic version — architecture

```
User question
     │
     ▼
Chroma similarity search (top-k dense retrieval, HF embeddings)
     │
     ▼
"Stuff" the retrieved chunks into a prompt template
     │
     ▼
Ollama (local LLM) generates the answer
```

Bui## Advanced version — architecture

```
User question + chat history
     │
     ▼
rewrite_query: local LLM rewrites follow-ups into standalone questions (skipped on the first turn, when there's no history to disambiguate against)
     │
     ▼
hybrid_search: runs the dense (Chroma) and BM25 retrievers separately, merges both result lists and de-duplicates, keeping dense results first
     │
     ▼
Top 6 merged results kept, concatenated into context
     │
     ▼
generate_answer: {context} + chat history + question → prompt → Ollama
     │
     ▼
Answer printed; chat history updated and saved to disk
```

This version is built entirely from plain Python functions and `langchain_core` primitives (`ChatPromptTemplate`, `MessagesPlaceholder`) rather than the higher-level `langchain.chains`/`langchain.retrievers` classes (`create_history_aware_retriever`, `create_retrieval_chain`, `EnsembleRetriever`, `MultiQueryRetriever`, `ContextualCompressionRetriever`) originally used here — those live in the top-level `langchain` package, whose `chains`/`retrievers` modules have been restructured and partly removed across recent LangChain releases (e.g. `ModuleNotFoundError: No module named 'langchain.chains'` on current installs). The manual version below is less "batteries included" but has no dependency on that fragile namespace.

### Why each piece is there
- **`rewrite_query`** — without it, a follow-up like "what about the Pro plan?" would be searched literally and miss the earlier context ("what about *storage limits*?"). The LLM first rewrites it into a standalone question ("What are the storage limits of the Pro plan?"), using `ChatPromptTemplate.from_messages([...])` with `MessagesPlaceholder("chat_history")` so the actual history and question reach the model (not just the static instruction text).
- **`hybrid_search` (dense + BM25)** — dense/embedding search is great at matching meaning but can miss exact keywords, product names, or codes. BM25 is the opposite: strong on exact terms, weak on paraphrase. Running both and merging their results (de-duplicated by source + content, dense results kept first) covers more cases than either alone. This is a simpler fusion than true Reciprocal Rank Fusion (no score-based re-ordering of the merged list) — see "Extending this project" below for how to upgrade it.
- **Persisted chat history** — `save_history()` writes the conversation to `chat_histories/<session_id>.json` after every turn (via a plain `ChatMessageHistory` object, not `RunnableWithMessageHistory`), so a session's conversation isn't lost if the terminal is closed.

## Basic vs Advanced — trade-offs

| | Basic | Advanced |
|---|---|---|
| Retrieval | Dense only | Hybrid (dense + BM25, merged) |
| Re-ranking | None | None (merge order only: dense results first) |
| Memory | None (single-turn) | Multi-turn, persisted to disk |
| LLM calls per question | 1 | 1 (first turn) or 2 (rewrite + answer, later turns) |
| Latency (on local/free LLM) | Low | Slightly higher on follow-up questions |
| Answer quality on ambiguous / follow-up questions | Weaker | Stronger |

The basic version is a good starting point or a low-latency option for small, well-structured document sets, and it's easier on CPU-only Ollama. The advanced version is worth the extra latency when documents are large, questions are conversational, or exact terminology matters (product names, codes, IDs) alongside semantic meaning.

## ⚠️ Currently free — how to switch to paid resources

**This project is currently configured to run entirely on free, local resources** (Ollama + HuggingFace embeddings + Chroma), as described above. If you'd rather use a paid hosted provider — for a stronger model, lower local hardware requirements, or faster responses — the swap only touches the model/embeddings instantiation. The retrieval logic (Chroma, BM25, hybrid search, memory) stays exactly the same.

### 1. Swap the LLM (e.g. to OpenAI)
```bash
pip install langchain-openai
```
In `app.py`, replace:
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
```
with:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # reads OPENAI_API_KEY from env
```

### 2. Swap the embeddings (e.g. to OpenAI)
In both `app.py` and `ingest.py`, replace:
```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```
with:
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
**Important:** embeddings from different models are not compatible with each other. After switching, delete `chroma_db/` and re-run `python ingest.py` to rebuild the index with the new embedding model.

### 3. Add the API key
Add to `.env`:
```
OPENAI_API_KEY=your_key_here
```

### 4. (Optional) Swap the vector DB to a managed service
`Chroma` can be swapped for a hosted vector DB (e.g. Pinecone, Qdrant Cloud, Weaviate Cloud) if you need multi-user access or scale beyond a single machine — install the corresponding `langchain-<provider>` package and replace the `Chroma(...)` calls with that provider's client. This is optional; Chroma remains free and sufficient for a single-user terminal chatbot of any size used here.

### Other providers
The same three-step pattern (install package → swap constructor → add API key) works for any LangChain-supported provider, e.g.:
- **Anthropic**: `langchain-anthropic`, `ChatAnthropic(model="claude-...")`
- **Google**: `langchain-google-genai`, `ChatGoogleGenerativeAI(...)`
- **Cohere**: `langchain-cohere`, `ChatCohere(...)` / `CohereEmbeddings(...)`
- **Groq** (fast, has a free tier): `langchain-groq`, `ChatGroq(...)`

## Extending this project
- Swap `Chroma` for another free local vector store (e.g. FAISS).
- Swap the Ollama model for any other model available in the free Ollama library (`ollama pull <model>`), or point `ChatOllama` at another OpenAI-compatible free/local server (e.g. LM Studio, vLLM).
- Upgrade `hybrid_search`'s simple merge into true Reciprocal Rank Fusion (score-weighted re-ordering instead of "dense results first") — see `no_framework_chatbot/advanced/app.py`'s hand-written `_rrf_fuse` for a reference implementation using the same idea.
- Add a cross-encoder re-ranker (e.g. `sentence-transformers` `CrossEncoder`, free, local) after `hybrid_search` to re-score and trim the merged results before they reach the answer prompt — see `langgraph_chatbot/advanced/app.py`'s `retrieve_and_rerank` node for a reference implementation.
- Add streaming output by using `.stream()` instead of `.invoke()`.
- If you later want to use a hosted API instead (e.g. for stronger reasoning), the only code that needs to change is the embeddings/LLM instantiation lines — the rest of the pipeline is provider-agnostic.lt with plain LCEL (`prompt | llm`) from `langchain_core`, `langchain_core` primitives are the stable, version-proof way to compose a retriever + prompt + LLM. Each question is independent — there's no memory of previous turns. This is the simplest correct RAG loop and a good baseline to compare against.
