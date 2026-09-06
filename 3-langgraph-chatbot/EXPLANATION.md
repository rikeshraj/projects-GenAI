# Explanation: RAG Chatbot using LangGraph

## What LangGraph adds over a fixed pipeline

LangChain's chain classes (`RetrievalQA`, `create_retrieval_chain`) — or, as in the current `langchain_chatbot` project, plain sequential Python calling `langchain_core` primitives directly — compose a **fixed sequence** of steps: retrieve, then generate, always in that order, every turn. LangGraph instead represents the chatbot as an explicit **graph**: a set of named nodes (plain Python functions that read and return pieces of a shared state object) connected by edges, where edges can be conditional on the state. This makes branching, loops, and multi-step control flow a first-class part of the app's structure rather than something bolted on with `if`/`else` around a fixed sequence.

For the basic version, that extra power isn't really needed — the graph is a straight line, same shape as `langchain_chatbot`'s basic pipeline. The advanced version is where it matters: it adds a real decision point (should this question even hit the knowledge base?) expressed as a first-class part of the graph, rather than custom control flow written around a fixed sequence of calls.

## Basic version — architecture

```
START
  │
  ▼
retrieve   — embed the latest message, similarity-search Chroma, store context
  │
  ▼
generate   — LLM answers using {context} + full message history
  │
  ▼
END
```

### State
```python
class ChatState(MessagesState):
    context: str
```
`MessagesState` is a small built-in LangGraph state type: it has a `messages` field whose updates are *appended* rather than overwritten (via a reducer function), so returning `{"messages": [response]}` from a node adds to the conversation instead of replacing it. We extend it with a plain `context` field to pass retrieved text from `retrieve` to `generate`.

### Memory
The graph is compiled with `checkpointer=MemorySaver()`. Every `app.invoke(...)` call is scoped to a `thread_id` (passed via `config`); LangGraph automatically loads that thread's saved state before running the graph and saves the updated state afterward — which is why the message history accumulates across calls in `main()`'s loop without any manual list-management. `MemorySaver` keeps this in memory only (lost when the process exits) — see the advanced version for disk persistence.

## Advanced version — architecture

```
START
  │
  ▼
classify ───────────────► conditional edge, based on state["route"]
  │                                     │
  │ "retrieve"                         │ "direct"
  ▼                                     │
rewrite_query                           │
  │                                     │
  ▼                                     │
retrieve_and_rerank                     │
  │                                     │
  └──────────────┬──────────────────────┘
                 ▼
              generate
                 │
                 ▼
                END
```

### Why each piece is there

- **`classify` + conditional edge** — `graph.add_conditional_edges( "classify", route_after_classify, {"retrieve": "rewrite_query",
  "direct": "generate"})` routes execution based on the LLM's own judgment about whether this question needs the knowledge base at all. This is the clearest example of something a LangChain *chain* can't express directly: a chain's shape is fixed at construction time, while a LangGraph conditional edge lets the *state* (here, the model's own classification) decide the path at run time.
- **`rewrite_query`** — same purpose as the history-aware retriever in the LangChain advanced chatbot: turn "what about the Pro plan?" into a standalone question using recent chat history, so retrieval doesn't search on an ambiguous fragment.
- **`retrieve_and_rerank`** — hybrid retrieval (BM25 + dense, combined with hand-written Reciprocal Rank Fusion, `rrf_fuse` — a deliberate choice over LangChain's `langchain.retrievers.EnsembleRetriever`, since that lives in the top-level `langchain` package, whose `retrievers` module has been restructured across recent releases; hand-written RRF has no dependency on that fragile namespace) followed by cross-encoder re-ranking of the fused candidates. Combining both steps in one node keeps the raw candidate list out of the graph's persisted state, since it's only an intermediate value the next node needs, not something worth checkpointing.
- **`generate`** — answers from `{context}` if retrieval happened, or directly if `classify` routed around it — the same node handles both cases, since `context` is simply empty on the direct path.
- **`SqliteSaver` checkpointer** — swaps `MemorySaver` for a SQLite-file- backed checkpointer, so the full graph state (including all messages) survives restarting `app.py`, as long as you reuse the same `THREAD_ID`. This is a different (and more production-realistic) way to get persisted memory than the JSON-file approach used in the no-framework and LangChain advanced chatbots — LangGraph treats persistence as a pluggable checkpointer rather than something the app code writes to disk manually.

## Basic vs Advanced — trade-offs

| | Basic | Advanced |
|---|---|---|
| Graph shape | Straight line (2 nodes) | Branching (4 nodes, 1 conditional edge) |
| Retrieval | Always runs, dense only | Conditional; hybrid (BM25+dense) + re-ranked when it runs |
| Query handling | Uses the raw latest message | Rewrites follow-ups into standalone questions |
| Memory | In-process only (`MemorySaver`) | Persisted to disk (`SqliteSaver`), survives restarts |
| LLM calls per question | 1 | 2–3 (classify, optionally rewrite, answer) |
| Handles small talk gracefully | No — always searches the knowledge base | Yes — routes around retrieval when it's not needed |

## Comparing to the LangChain version (`langchain_chatbot`)

| | LangGraph (this project) | LangChain (`langchain_chatbot`) |
|---|---|---|
| Control flow | Explicit graph (nodes + edges), inspectable and extensible | Plain Python functions calling `langchain_core` primitives directly (LCEL `prompt \| llm`, manual retrieval/rewrite/answer functions) |
| Conditional logic | First-class (`add_conditional_edges`) | Plain `if`/`else` in Python — nothing framework-specific needed for either project, since both avoid the higher-level chain/agent classes |
| Re-ranking approach | Cross-encoder (local model, no LLM calls) | None (advanced version merges dense + BM25 results, dense first) |
| Memory | Pluggable checkpointer (`MemorySaver` / `SqliteSaver`) | Plain `ChatMessageHistory`, manually saved to JSON |
| Best for | Multi-step or branching workflows, agents, anything with real decision points | Straightforward, mostly-linear RAG pipelines |

Both share the same underlying LangChain integration packages (`langchain-community`'s vectorstore/retriever/loader classes, `langchain-huggingface`, `langchain-ollama`) and both deliberately avoid the top-level `langchain` package's `chains`/`retrievers` classes (`RetrievalQA`, `EnsembleRetriever`, `create_retrieval_chain`, etc.), which have been restructured and partly removed across recent LangChain releases. The difference between the two projects is in how the *flow* between retrieval and generation is expressed and controlled — an explicit graph here, versus plain Python control flow in `langchain_chatbot`.

## ⚠️ Currently free — how to switch to paid resources

This project runs entirely on free, local resources by default. The graph structure doesn't need to change at all to use a paid provider — only the LLM and embeddings instantiation lines do.

### 1. Swap the LLM (e.g. to OpenAI)
```bash
pip install langchain-openai
```
Replace:
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
**Important:** after switching embeddings, delete `chroma_db/` and
re-run `python ingest.py` — embeddings from different models aren't
compatible with an existing index.

### 3. Add the API key
```
OPENAI_API_KEY=your_key_here
```

### 4. (Optional) Swap the checkpointer for a hosted/shared store
LangGraph supports other checkpointer backends (e.g. Postgres) if you need multi-process or multi-user persistence beyond a single local SQLite file — install `langgraph-checkpoint-postgres` and swap `SqliteSaver` for `PostgresSaver`. This is optional; SQLite is free and sufficient for a single-user terminal chatbot.

### Other providers
The same pattern (install package → swap `ChatOllama`/`HuggingFaceEmbeddings` → add API key) works for Anthropic (`langchain-anthropic`), Google (`langchain-google-genai`), Cohere (`langchain-cohere`), or Groq (`langchain-groq`).

## Extending this project
- Add a `rerank` fallback: if the cross-encoder returns low scores for every candidate, route back to `generate` with a "no good matches" note instead of forcing an answer from weak context.
- Add a loop: let `generate` route back to `retrieve_and_rerank` with a refined query if its own answer expresses low confidence (a simple form of self-correction, natural to express as a graph cycle).
- Add query expansion: generate a few alternate phrasings of the standalone question (one more LLM call) and run `rrf_fuse` across all of their retrieval results too — similar to what LlamaIndex's `QueryFusionRetriever` does in `llamaindex_chatbot`.
- Stream node outputs with `app.stream(...)` instead of `app.invoke(...)` to show intermediate steps (e.g. "retrieving...", "re-ranking...") in the terminal as they happen.
