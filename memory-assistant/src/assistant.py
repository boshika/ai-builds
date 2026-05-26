"""
assistant.py — Main chat loop for the memory assistant.

Startup sequence:
1. Init SQLite database
2. Load all past memories from SQLite into ChromaDB (keeps them in sync)
3. Create a new session
4. Start the conversation loop

Each conversation turn:
1. Retrieve relevant memories from ChromaDB based on user's message
2. Build the prompt: system message + relevant memories + conversation history
3. Send to LLM, get response
4. Add both messages to short-term buffer
5. Extract facts from the exchange and save to SQLite + ChromaDB
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from store import init_db, create_session, save_memory, load_all_memories
from memory import ShortTermMemory, extract_facts
from retriever import seed_from_store, search_memories, add_memory

client = OpenAI()
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful personal assistant with memory.
You remember facts about the user from previous conversations and use
them to give more relevant, personalized responses.

When relevant memories are provided, use them naturally — don't
announce "I remember that..." unless it adds value.
"""


def build_system_message(relevant_memories: list[str]) -> str:
    """
    Combine the base system prompt with any relevant memories.
    Memories are injected here rather than as separate messages —
    this keeps the context clean and avoids confusing the model.
    """
    if not relevant_memories:
        return SYSTEM_PROMPT

    memory_block = "\n".join(f"- {m}" for m in relevant_memories)
    return f"{SYSTEM_PROMPT}\nWhat you remember about this user:\n{memory_block}"


def chat():
    print("Memory Assistant — type 'quit' to exit\n")

    # --- Startup ---
    init_db()

    # Load all SQLite memories into ChromaDB
    # We store (id, content) pairs so ChromaDB IDs match SQLite IDs
    all_memories = load_all_memories()  # returns list of (id, content)
    seed_from_store(all_memories)

    session_id = create_session()
    short_term = ShortTermMemory()
    memory_counter = len(all_memories)  # used to generate unique ChromaDB IDs

    print(f"Session started. {len(all_memories)} memories loaded from past sessions.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            print("Goodbye.")
            break

        # 1. Retrieve relevant long-term memories for this message
        relevant = search_memories(user_input, n_results=3)

        # 2. Build system message with relevant memories injected
        system_message = build_system_message(relevant)

        # 3. Assemble messages: system + conversation history + new user message
        messages = (
            [{"role": "system", "content": system_message}]
            + short_term.get()
            + [{"role": "user", "content": user_input}]
        )

        # 4. Call the LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        assistant_message = response.choices[0].message.content

        print(f"\nAssistant: {assistant_message}\n")

        # 5. Update short-term buffer
        short_term.add("user", user_input)
        short_term.add("assistant", assistant_message)

        # 6. Extract facts and save to long-term memory
        facts = extract_facts(user_input, assistant_message)
        for fact in facts:
            # Save to SQLite (source of truth)
            save_memory(session_id, fact)
            memory_counter += 1

            # Index in ChromaDB (for future semantic retrieval)
            add_memory(str(memory_counter), fact)

            print(f"  [memory saved: {fact}]")


if __name__ == "__main__":
    chat()
