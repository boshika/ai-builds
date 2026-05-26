# Build 01 — RAG Pipeline

A simple retrieval-augmented generation pipeline: ingest documents, embed them, store in a vector DB, and answer questions grounded in the docs.

## Concepts Covered

- Document chunking and preprocessing
- Text embeddings (OpenAI / local models)
- Vector storage and similarity search (ChromaDB)
- Prompt construction with retrieved context
- Response grounding and citation

## Stack

- Python
- LangChain / LlamaIndex
- ChromaDB (local vector store)
- OpenAI API (embeddings + completion)

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env

python src/ingest.py    # chunk and embed documents
python src/query.py     # query the pipeline
```

## What to Build Next

- Add a reranker
- Swap ChromaDB for Pinecone
- Add streaming responses
- Evaluate retrieval quality

## Related

- [System design notes](https://github.com/boshika/ai-products-systemdesign/tree/main/rag_system_design)
- [PM framing](https://github.com/boshika/product-interview-vault/blob/main/AI-Technical/rag-and-knowledge-systems.md)
