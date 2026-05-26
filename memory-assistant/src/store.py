"""
store.py — SQLite persistence layer for long-term memory.

Two tables:
- sessions: each time the user opens the assistant
- memories: facts extracted from conversations, linked to a session
"""

import sqlite3
from datetime import datetime

DB_PATH = "memory.db"


def get_connection():
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Create the database tables if they don't exist.
    Called once on startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # sessions table — one row per app session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL
        )
    """)

    # memories table — facts extracted from conversations
    # content: the actual memory string, e.g. "User prefers concise answers"
    # session_id: which session this came from
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()


def create_session() -> int:
    """
    Start a new session and return its ID.
    Called at the beginning of each conversation.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (created_at) VALUES (?)",
        (datetime.now().isoformat(),)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def save_memory(session_id: int, content: str):
    """
    Save a single extracted fact to the database.
    Called after the LLM extracts a fact worth remembering.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO memories (session_id, content, created_at) VALUES (?, ?, ?)",
        (session_id, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def load_all_memories() -> list[str]:
    """
    Load every memory ever stored.
    Used to seed ChromaDB on startup so semantic search works.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM memories ORDER BY created_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]
