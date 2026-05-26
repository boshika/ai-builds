# Build 02 — Memory Assistant

A stateful conversational AI that remembers facts about you across sessions.

## What It Does

- **Short-term memory** — holds the last 10 messages in the current session
- **Long-term memory** — extracts facts from conversations and persists them in SQLite
- **Semantic retrieval** — uses ChromaDB to find relevant past memories for each new message
- **No frameworks** — wired manually with the OpenAI API so every step is visible

## Architecture

```
User message
    │
    ▼
ChromaDB (find relevant past memories)
    │
    ▼
Build prompt: system + memories + conversation history
    │
    ▼
OpenAI LLM → response
    │
    ▼
Extract facts (gpt-4o-mini)
    │
    ├── SQLite (source of truth)
    └── ChromaDB (search index)
```

## Concepts Covered

- Short-term vs long-term memory
- Two-layer storage: relational DB + vector DB
- LLM-based fact extraction
- Semantic memory retrieval
- Context window management

## How to Run

```bash
cd memory-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your OpenAI API key to .env

cd src

# Gradio web UI (recommended)
python app.py

# CLI version
python assistant.py
```

Open http://localhost:7860 in your browser.

## Files

| File | What it does |
|------|-------------|
| `src/store.py` | SQLite layer — saves and loads memories |
| `src/retriever.py` | ChromaDB — semantic search over memories |
| `src/memory.py` | Short-term buffer + LLM fact extraction |
| `src/app.py` | Gradio frontend — chat UI + live memory panel |
| `src/assistant.py` | CLI version |

## What to Explore Next

- Swap SQLite for Postgres
- Add memory summarization (compress old memories)
- Add a `/memories` command to inspect what's stored
- Try different fact extraction prompts and compare results

## Related

- [System design notes](https://github.com/boshika/ai-products-systemdesign/tree/main/build-01-memory-assistant)
- [PM framing](https://github.com/boshika/product-interview-vault/blob/main/AI-Technical/agents-and-orchestration.md)
