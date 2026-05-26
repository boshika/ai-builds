"""
retriever.py — Semantic search over long-term memories using ChromaDB.

Why: SQLite stores everything, but we don't want to inject all memories
into every prompt — that wastes tokens and adds noise. ChromaDB lets us
find only the memories most relevant to the current conversation turn.

How it works:
- Each memory is embedded (converted to a vector) and stored in ChromaDB
- When the user sends a message, we embed that message and find the
  closest matching memories by cosine similarity
- Only those top-k memories get injected into the prompt
"""

import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "memories"
CHROMA_PATH = "chroma"

# Use OpenAI embeddings for semantic similarity
# ChromaDB handles the embedding calls internally
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)


def get_collection():
    """Return the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    return collection


def add_memory(memory_id: str, content: str):
    """
    Add a memory to ChromaDB.
    memory_id: unique string ID (we use the SQLite row id as string)
    content: the memory text to embed and store
    """
    collection = get_collection()

    # Check if this memory is already in ChromaDB to avoid duplicates
    existing = collection.get(ids=[memory_id])
    if existing["ids"]:
        return

    collection.add(
        documents=[content],
        ids=[memory_id]
    )


def search_memories(query: str, n_results: int = 3) -> list[str]:
    """
    Find the top-n memories most semantically similar to the query.
    Returns a list of memory strings to inject into the prompt.
    """
    collection = get_collection()

    # If the collection is empty, return nothing
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count())  # can't request more than we have
    )

    return results["documents"][0]  # list of matching memory strings


def seed_from_store(memories: list[tuple[int, str]]):
    """
    On startup, load all memories from SQLite into ChromaDB.
    This ensures ChromaDB is always in sync with the source of truth (SQLite).

    memories: list of (id, content) tuples from SQLite
    """
    for memory_id, content in memories:
        add_memory(str(memory_id), content)
