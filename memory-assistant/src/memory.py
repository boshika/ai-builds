"""
memory.py — Short-term memory buffer + long-term fact extraction.

Short-term memory: the last N messages in the current session.
These go directly into the LLM prompt as conversation history.
Limited by MAX_MESSAGES to avoid overflowing the context window.

Long-term memory: facts extracted by the LLM from each exchange.
These are saved to SQLite and indexed in ChromaDB for future sessions.
"""

from openai import OpenAI

client = OpenAI()

# How many messages to keep in the short-term buffer.
# Older messages are dropped when the limit is hit.
MAX_MESSAGES = 10

EXTRACT_PROMPT = """
You are a memory extraction assistant. Given a conversation exchange,
extract any facts worth remembering about the user — preferences,
goals, background, things they've mentioned about themselves.

Rules:
- Only extract facts explicitly stated by the user
- Be concise — one sentence per fact
- If there's nothing worth remembering, return an empty list
- Return ONLY a Python list of strings, nothing else

Example output:
["User is learning Python", "User prefers short explanations"]

Conversation:
User: {user_message}
Assistant: {assistant_message}
"""


class ShortTermMemory:
    """
    In-session conversation buffer.
    Holds the last MAX_MESSAGES messages as a list of role/content dicts.
    These get passed directly to the OpenAI chat API.
    """

    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Drop oldest messages if we exceed the limit
        # Always keep at least the system message (index 0) if present
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]

    def get(self) -> list[dict]:
        return self.messages

    def clear(self):
        self.messages = []


def extract_facts(user_message: str, assistant_message: str) -> list[str]:
    """
    Ask the LLM to extract memorable facts from a single exchange.

    We use a cheap, fast model (gpt-4o-mini) for this — it runs after
    every message, so cost matters. The main conversation uses the model
    the user configured.
    """
    prompt = EXTRACT_PROMPT.format(
        user_message=user_message,
        assistant_message=assistant_message
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0  # deterministic — we want consistent extraction
    )

    raw = response.choices[0].message.content.strip()

    # Safely parse the list the LLM returns
    try:
        facts = eval(raw)  # LLM returns a Python list literal
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception:
        pass

    return []
