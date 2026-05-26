# ai-builds

Hands-on AI learning projects — each folder is a self-contained build focused on a core AI engineering concept.

> **Companion repos:**
> - [ai-products-systemdesign](https://github.com/boshika/ai-products-systemdesign) — system design and architecture notes for these same topics
> - [product-interview-vault](https://github.com/boshika/product-interview-vault) — PM framing and interview answers

---

## Projects

| Project | Concept | Status |
|---------|---------|--------|
| [`rag-pipeline/`](rag-pipeline/) | Retrieval-Augmented Generation | 🚧 In progress |
| [`memory-assistant/`](memory-assistant/) | Stateful chat with memory | 🚧 In progress |
| [`agent-workflow/`](agent-workflow/) | Agentic task execution | 📋 Planned |
| [`eval-framework/`](eval-framework/) | LLM evaluation and quality | 📋 Planned |

---

## Structure

Each project is self-contained:

```
project-name/
├── README.md       — what it does, how to run it, what you learned
├── requirements.txt or pyproject.toml
├── .env.example    — required env vars (never commit .env)
└── src/            — source code
```

Shared utilities live in [`shared/`](shared/) and can be imported across projects.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/boshika/ai-builds.git
cd ai-builds

# Each project has its own venv
cd rag-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Related Repositories

- [ai-products-systemdesign](https://github.com/boshika/ai-products-systemdesign) — Architecture and system design depth
- [product-interview-vault](https://github.com/boshika/product-interview-vault) — PM interview prep and framing
- [ai-knowledge-graph](https://github.com/boshika/ai-knowledge-graph) — AI and Cloud concepts reference
