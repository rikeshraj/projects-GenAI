# Explanation: RAG Chatbot using LlamaIndex

## LlamaIndex's core abstraction: the Index

Where LangChain centers around composable **chains** and LangGraph
centers around an explicit **graph**, LlamaIndex centers around the
**index** — a data structure built over your documents that knows how to
retrieve from itself. A `VectorStoreIndex` wraps a vector store (Chroma,
here) and exposes `.as_query_engine()` (single-turn Q&A) or
`.as_chat_engine()` (multi-turn conversation) directly, without needing
to hand-assemble a retriever + prompt + LLM chain yourself.

## Basic version — architecture

```
ingest.py:
  SimpleDirectoryReader -> SentenceSplitter -> HuggingFaceEmbedding
      -> VectorStoreIndex (backed by Chroma) -> persisted to disk

app.py:
  User question
       │
       ▼
  index.as_query_engine(similarity_top_k=TOP_K)
       │  (retrieves top-k chunks, stuffs them into a QA prompt template)
       ▼
  Ollama generates the answer
```

The query engine handles retrieval and prompt assembly internally; the
only customization here is swapping in a slightly more explicit prompt
via `query_engine.update_prompts(...)`, LlamaIndex's documented way to
override a query engine's internal prompt templates without
reconstructing it from scratch. No memory: every question is independent.

## Advanced version — architecture

```
User question + chat history (ChatMemoryBuffer)
       │
       ▼
CondensePlusContextChatEngine:
  1. Condense: rewrite the question into a standalone one using history
       │
       ▼
  2. Retrieve: QueryFusionRetriever
       ├─ generates NUM_QUERIES alternate phrasings of the question (LLM)
       ├─ runs each phrasing through both the dense (Chroma) and BM25
       │    retrievers
       └─ fuses all resulting ranked lists with Reciprocal Rank Fusion
       │
       ▼
  3. Post-process: SentenceTransformerRerank
       re-scores the fused candidates with a cross-encoder, keeps top FINAL_K
       │
       ▼
  4. Generate: Ollama answers using the reranked context + chat history
       │
       ▼
  ChatMemoryBuffer updated; SimpleChatStore.persist() writes it to disk
```

### Why each piece is there

- **`CondensePlusContextChatEngine`** — LlamaIndex ships this as a
  built-in chat engine specifically for the "rewrite follow-ups, retrieve
  context, keep memory" pattern, which is exactly what the LangChain and
  LangGraph advanced chatbots hand-assemble from smaller pieces
  (`create_history_aware_retriever` / a `rewrite_query` node). Here it's
  one framework class instead of several composed pieces.
- **`QueryFusionRetriever` does both hybrid retrieval *and* query
  expansion in one component** — this is a notable difference from the
  LangChain version, which uses two separate retrievers
  (`EnsembleRetriever` for hybrid fusion, `MultiQueryRetriever` for query
  expansion) chained together. LlamaIndex bundles both ideas into a
  single retriever class, since generating alternate query phrasings and
  fusing multiple ranked lists are complementary steps that are commonly
  used together.
- **`SentenceTransformerRerank`** — a cross-encoder re-ranking
  post-processor, the same underlying technique used by hand in the
  no-framework advanced chatbot, but here it's a one-line built-in
  `node_postprocessor` rather than custom code. This is a different
  re-ranking approach than the LangChain advanced chatbot's LLM-based
  `ContextualCompressionRetriever` — faster, and doesn't consume extra
  LLM calls.
- **`ChatMemoryBuffer` + `SimpleChatStore`** — LlamaIndex's memory
  abstraction, analogous to LangChain's `RunnableWithMessageHistory` or
  LangGraph's checkpointer. `SimpleChatStore.persist(path)` writes the
  full conversation to a JSON file after every turn, and
  `SimpleChatStore.from_persist_path(path)` reloads it — giving the same
  "survives restarting the app" behavior as the LangGraph advanced
  chatbot's `SqliteSaver`, just backed by a JSON file instead of SQLite.

## Basic vs Advanced — trade-offs

| | Basic | Advanced |
|---|---|---|
| Engine | `QueryEngine` (single-turn) | `CondensePlusContextChatEngine` (multi-turn) |
| Retrieval | Dense only | Hybrid (BM25 + dense) fused via RRF, plus query expansion |
| Re-ranking | None | Cross-encoder (`SentenceTransformerRerank`) |
| Memory | None | `ChatMemoryBuffer`, persisted to disk |
| LLM calls per question | 1 | 1 (condense) + up to `NUM_QUERIES` (query expansion) + 1 (answer) |
| Answer quality on ambiguous / follow-up questions | Weaker | Stronger |

## Comparing to the LangChain and LangGraph chatbots

| | LlamaIndex (this project) | LangChain (`langchain_chatbot`) | LangGraph (`langgraph_chatbot`) |
|---|---|---|---|
| Core abstraction | Index + (Query/Chat)Engine | Chain | Graph (nodes + edges) |
| Hybrid + query expansion | One retriever (`QueryFusionRetriever`) | Two retrievers chained (`EnsembleRetriever` + `MultiQueryRetriever`) | One retriever (`EnsembleRetriever`) — no built-in query expansion in that project |
| Re-ranking | Cross-encoder (`SentenceTransformerRerank`) | LLM-based (`ContextualCompressionRetriever`) | Cross-encoder (custom node) |
| Memory backend | `ChatMemoryBuffer` + `SimpleChatStore` (JSON) | `RunnableWithMessageHistory` (custom JSON) | Checkpointer (`SqliteSaver`) |
| Conditional routing (skip retrieval for small talk) | Not included in this version | Not included in this version | Included (`classify` node) |

All three advanced chatbots solve the same problem — better retrieval,
re-ranking, and memory than a naive RAG loop — but each framework's
"batteries-included" classes push you toward slightly different building
blocks by default.

## ⚠️ Currently free — how to switch to paid resources

This project runs entirely on free, local resources by default. Only the
`Settings.llm` and `Settings.embed_model` assignments need to change to
use a paid provider — the index, retrievers, and chat engine stay the
same.

### 1. Swap the LLM (e.g. to OpenAI)
```bash
pip install llama-index-llms-openai
```
Replace:
```python
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=180.0, temperature=0)
```
with:
```python
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)   # reads OPENAI_API_KEY from env
```

### 2. Swap the embeddings (e.g. to OpenAI)
In both `app.py` files and `ingest.py`, replace:
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
```
with:
```python
pip install llama-index-embeddings-openai
```
```python
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```
**Important:** after switching embeddings, delete `chroma_db/` and
re-run `python ingest.py` — embeddings from different models aren't
compatible with an existing index.

### 3. Add the API key
```
OPENAI_API_KEY=your_key_here
```

### Other providers
The same pattern (install a `llama-index-llms-<provider>` /
`llama-index-embeddings-<provider>` package → swap the `Settings`
assignment → add an API key) works for Anthropic
(`llama-index-llms-anthropic`), Google (`llama-index-llms-gemini`),
Cohere (`llama-index-llms-cohere`, which also offers a hosted reranker
you could swap in for `SentenceTransformerRerank`), or Groq
(`llama-index-llms-groq`).

## Extending this project
- Swap `SimpleDirectoryReader` for one of LlamaIndex's many data
  connectors (LlamaHub) to ingest from sources other than local files.
- Add a `SimilarityPostprocessor` to drop low-relevance nodes before
  they reach the reranker.
- Use `index.as_chat_engine(chat_mode="condense_plus_context")` as a
  shortcut instead of manually constructing `CondensePlusContextChatEngine`
  if you don't need the custom retriever/reranker.
- Stream the response with `chat_engine.stream_chat(query)` instead of
  `chat_engine.chat(query)` to print tokens as they arrive.
