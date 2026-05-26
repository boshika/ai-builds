"""
app.py — Gradio frontend for the memory assistant.

Layout: two-column
- Left:  chat interface
- Right: live memory panel (updates after every message)

Why gr.Blocks instead of gr.ChatInterface?
gr.ChatInterface is simpler but gives you one component.
gr.Blocks lets us compose multiple components — chat + memory viewer —
so you can actually watch memories being saved as you talk.
"""

import os
import sys
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Add src/ to path so imports work whether you run from src/ or root
sys.path.append(os.path.dirname(__file__))

from store import init_db, create_session, save_memory, load_all_memories
from memory import ShortTermMemory, extract_facts
from retriever import seed_from_store, search_memories, add_memory

client = OpenAI()
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful personal assistant with memory.
You remember facts about the user from previous conversations and use
them to give more relevant, personalized responses.
Use memories naturally — don't announce them unless it adds value.
"""

# --- Startup (runs once when the app loads) ---
init_db()
all_memories = load_all_memories()   # list of (id, content)
seed_from_store(all_memories)
session_id = create_session()

# Global state for this session
short_term = ShortTermMemory()
memory_counter = len(all_memories)
session_memories: list[str] = []     # tracks memories saved this session

print(f"Started. {len(all_memories)} memories loaded from past sessions.")


# --- Core logic ---

def build_system_message(relevant_memories: list[str]) -> str:
    """Inject relevant past memories into the system prompt."""
    if not relevant_memories:
        return SYSTEM_PROMPT
    memory_block = "\n".join(f"- {m}" for m in relevant_memories)
    return f"{SYSTEM_PROMPT}\nWhat you remember about this user:\n{memory_block}"


def chat(user_message: str, history: list):
    """
    Called by Gradio on every message submission.

    history: list of [user, assistant] pairs — managed by Gradio's Chatbot component.
    We use our own short_term buffer for the LLM prompt (same data, our control).

    Returns:
      - "" (clears the input box)
      - updated history (Gradio redraws the chat)
      - updated memory list (redraws the memory panel)
    """
    global memory_counter, session_memories

    if not user_message.strip():
        return "", history, format_memories()

    # 1. Find relevant past memories for this message
    relevant = search_memories(user_message, n_results=3)

    # 2. Build prompt
    system_message = build_system_message(relevant)
    messages = (
        [{"role": "system", "content": system_message}]
        + short_term.get()
        + [{"role": "user", "content": user_message}]
    )

    # 3. Call LLM
    response = client.chat.completions.create(model=MODEL, messages=messages)
    assistant_message = response.choices[0].message.content

    # 4. Update short-term buffer
    short_term.add("user", user_message)
    short_term.add("assistant", assistant_message)

    # 5. Extract facts and save to long-term memory
    facts = extract_facts(user_message, assistant_message)
    for fact in facts:
        save_memory(session_id, fact)
        memory_counter += 1
        add_memory(str(memory_counter), fact)
        session_memories.append(fact)

    # 6. Update Gradio chat history
    history.append([user_message, assistant_message])

    return "", history, format_memories()


def format_memories() -> str:
    """
    Format all memories for display in the memory panel.
    Past session memories + this session's new memories.
    """
    past = [m for _, m in all_memories] if all_memories else []
    lines = []

    if past:
        lines.append("**From past sessions:**")
        for m in past:
            lines.append(f"- {m}")

    if session_memories:
        lines.append("\n**This session:**")
        for m in session_memories:
            lines.append(f"- {m}")

    if not lines:
        lines.append("*No memories saved yet. Start chatting!*")

    return "\n".join(lines)


def clear_chat():
    """Reset the short-term buffer and chat history (keeps long-term memories)."""
    short_term.clear()
    return [], format_memories()


# --- Gradio UI ---

with gr.Blocks(title="Memory Assistant", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🧠 Memory Assistant")
    gr.Markdown("A conversational AI that remembers facts about you across sessions.")

    with gr.Row():

        # Left column — chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_label=False
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type a message and press Enter...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear chat (keeps memories)", size="sm")

        # Right column — live memory viewer
        with gr.Column(scale=1):
            gr.Markdown("### 💾 Saved Memories")
            gr.Markdown("Updates live as you chat.")
            memory_display = gr.Markdown(format_memories())

    # Wire up events
    # Submit on Enter or Send button
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, memory_display]
    )
    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, memory_display]
    )
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, memory_display]
    )


if __name__ == "__main__":
    demo.launch()
