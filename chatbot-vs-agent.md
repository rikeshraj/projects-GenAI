# Chatbot vs Agent — What's the difference?

This doc explains, concretely, how a **RAG chatbot** differs from a **RAG agent** in each of the four approaches used in this repo: no framework, LangChain, LangGraph, and LlamaIndex. It complements each project's own `explanation.md` — this file focuses specifically on the chatbot/agent distinction and how each framework expresses it.

## The core distinction (applies everywhere)

| | Chatbot | Agent |
|---|---|---|
| Retrieval | **Always** happens before answering — it's a fixed step in the pipeline | **Conditional** — the LLM decides whether retrieval (or any other tool) is needed at all |
| Control flow | Fixed, linear (or a fixed graph): retrieve → generate | Dynamic: the LLM can call zero, one, or many tools, in any order, across multiple steps, before answering |
| What the LLM decides | Only the wording of the final answer | Which tool(s) to use, with what arguments, how many times, and when to stop |
| Typical tools | None — "the knowledge base" is baked into the flow | Explicit, named tools (retrieval is just one of them; calculators, APIs, etc. are others) |
| Best for | Questions that are always about the indexed documents | Mixed workloads: some questions need the knowledge base, some need computation or other actions, some need neither |

A chatbot is a RAG *pipeline*. An agent is a RAG-capable *decision-maker* that has retrieval available as one option among several. Everything below is a variation on that same idea, expressed in each framework's own vocabulary.

---

## 1. No Framework (pure Python)

**Status:** both implemented (`no_framework_chatbot`, `no_framework_agent`).

**Chatbot** (`no_framework_chatbot`): a straight-line function pipeline. `retrieve()` always runs, its output is always stuffed into the prompt, and the LLM is called once (basic) or after a query-rewrite call (advanced) to produce the final answer. There's no branching — the code itself is the control flow.

**Agent** (`no_framework_agent`): a `while` loop (`run_agent()`) around Ollama's native tool-calling API. Each turn, the LLM is sent a list of tool schemas (`TOOLS_SCHEMA`) alongside the conversation; it can return either plain text (done) or `tool_calls` (run this Python function with these arguments, then show me the result and let me continue). The loop repeats until the model stops requesting tools or a round cap is hit. Retrieval (`retrieve_documents`) is just one tool in the list — the advanced agent also has `calculator` and `current_datetime`, and the LLM
picks freely among them per question.

**Key code difference:** the chatbot's `app.py` never gives the LLM a choice — it calls `retrieve()` unconditionally. The agent's `app.py` hands the LLM a *menu* (`TOOLS_SCHEMA`) and only acts on what the LLM requests, in a loop.

---

## 2. LangChain

**Status:** chatbot implemented (`langchain_chatbot`); agent planned (`langchain_agent`).

**Chatbot** (`langchain_chatbot`): built from `RetrievalQA` (basic) or `create_retrieval_chain` + a history-aware retriever (advanced). These are **chains** — fixed compositions of steps (`retriever | prompt | llm`, roughly). The retriever always runs as part of the chain; there's no point at which the chain "decides" not to retrieve.

**Agent** (`langchain_agent`, planned): built from `create_react_agent` / `AgentExecutor` (or the newer `create_tool_calling_agent`), wrapping the retriever as a `Tool` (typically via `create_retriever_tool`). Unlike a chain, an `AgentExecutor` runs an internal loop: it prompts the LLM with the available tools, executes whatever tool calls come back, feeds the results back in, and repeats until the LLM emits a final answer — conceptually identical to the no-framework agent's `run_agent()` loop, but implemented as a reusable LangChain class instead of a hand-written `while` loop.

**Key code difference:** `RetrievalQA`/`create_retrieval_chain` is a **chain** (fixed shape, retrieval is a required step). `AgentExecutor` is a **loop** around an LLM that has tools available, where retrieval is one optional tool among others (a calculator, a web search, etc., once added).

---

## 3. LangGraph

**Status:** planned (`langgraph_chatbot`, `langgraph_agent`).

LangGraph represents *both* chatbots and agents as a **graph of nodes and edges** over an explicit state object — the difference between the two is what that graph looks like, not whether a "graph" is used at all.

**Chatbot** (`langgraph_chatbot`, planned design): a graph with a fixed shape — typically `retrieve → generate` (basic), or with a few extra fixed nodes for the advanced version (e.g. a `rewrite_query` node before `retrieve`, a `rerank` node after it). Edges between these nodes are unconditional: execution always flows retrieve → generate, every turn. Conversation memory is handled by LangGraph's checkpointer, which persists the graph's `state` (including message history) between calls.

**Agent** (`langgraph_agent`, planned design): a graph with a **conditional edge** — typically an `agent` node (calls the LLM with tools bound) connected to a `tools` node (executes whatever tool calls came back) via an edge whose direction depends on the LLM's output: if the LLM requested tool calls, go to `tools` and then loop back to `agent`; if not, end. This is the same request/execute/repeat loop as the other two agent implementations, but made *visually explicit* as a cycle in the graph rather than a `while` loop in code or a framework's internal executor.

**Key code difference:** in the chatbot graph, every edge is unconditional (the path through the graph is the same every time). In the agent graph, at least one edge is conditional on the LLM's decision, and that conditional edge is what creates the loop — the graph can revisit the same node multiple times in one turn, which is exactly what lets the agent call tools repeatedly before answering.

---

## 4. LlamaIndex

**Status:** planned (`llamaindex_chatbot`, `llamaindex_agent`).

**Chatbot** (`llamaindex_chatbot`, planned design): built from a `ChatEngine` (e.g. `ContextChatEngine` or `CondenseQuestionChatEngine`) wrapping a `VectorStoreIndex`'s query engine. Like LangChain's chains, a chat engine always retrieves context for every turn — the "condense question" step (rewriting a follow-up into a standalone question, for the advanced version) still leads unconditionally into a retrieval step.

**Agent** (`llamaindex_agent`, planned design): built from a `ReActAgent` or `FunctionAgent`, given the index's query engine wrapped as a `QueryEngineTool` (plus other tools for the advanced version). As with the other frameworks' agents, the agent class runs its own internal loop: prompt the LLM with available tools, execute what it asks for, feed results back, repeat until a final answer. Retrieval happens only if the agent decides the `QueryEngineTool` is relevant to the question.

**Key code difference:** a `ChatEngine` is built directly on top of a query engine (retrieval is structural, not optional). An `Agent` is built on top of a *list of tools*, one of which happens to be a query engine — retrieval competes with other tools for the agent's choice rather than being guaranteed.

---

## Summary table

| Approach | Chatbot mechanism | Agent mechanism | Is retrieval guaranteed? (Chatbot / Agent) |
|---|---|---|---|
| No framework | Hand-written function pipeline | Hand-written `while` loop over Ollama's tool-calling API | Yes / No |
| LangChain | `RetrievalQA` / `create_retrieval_chain` (a chain) | `AgentExecutor` (a loop) with the retriever wrapped as a `Tool` | Yes / No |
| LangGraph | `StateGraph` with unconditional edges | `StateGraph` with a conditional edge creating a tool-call loop | Yes / No |
| LlamaIndex | `ChatEngine` over a query engine | `ReActAgent`/`FunctionAgent` with the query engine as a `QueryEngineTool` | Yes / No |

Across every approach, the same underlying shape repeats: a **chatbot is a fixed pipeline where retrieval is structural**, and an **agent is a loop where retrieval (and other tools) are optional actions the LLM chooses**. The frameworks differ in *how* they express that loop — a hand-written `while`, a framework's `AgentExecutor` class, an explicit graph cycle, or an agent class wrapping a list of tools — but the underlying idea is identical, and it's why an "agent" can naturally answer things a fixed RAG chatbot can't (like doing a calculation, or correctly saying "hi" without searching a knowledge base for it).
