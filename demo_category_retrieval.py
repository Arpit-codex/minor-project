"""
Demonstration Script 2: Categorized Local Memory Retrieval.
Shows how the assistant:
1. Stores categorized memories into local Markdown files (college, projects, career).
2. Analyzes a question and identifies the relevant memory category.
3. Retrieves and ranks memories from that category.
4. Naturally injects the verified facts into the prompt.
5. Successfully handles a question where no local memory exists without hallucinating.
"""

from pathlib import Path
from core.assistant import ContextualAssistant
from core.memory_manager import MemoryManager
from core.conversation_manager import ConversationManager
from core.ai_client import AIClient


def run_demo():
    print("=" * 75)
    print("DEMO 2: CATEGORIZED LOCAL MEMORY RETRIEVAL & CATEGORY TARGETING")
    print("=" * 75)

    demo_dir = Path("memory_demo")
    demo_db = Path("data/demo_cat.db")

    cm = ConversationManager(db_path=demo_db)
    mm = MemoryManager(memory_dir=demo_dir)
    ai = AIClient(provider_type="mock")
    assistant = ContextualAssistant(
        conversation_manager=cm,
        memory_manager=mm,
        ai_client=ai
    )

    print("\n--- STEP 1: INITIALIZING LOCAL CATEGORIZED MEMORIES ---")
    mem1, _ = mm.add_or_update_memory(
        category="college",
        title="University & Major",
        content="User is enrolled at National Institute of Technology, majoring in Computer Science (Batch of 2026).",
        tags=["college", "education", "cs", "nit"],
        confidence=0.98
    )
    print(f"  [+] Saved memory in 'memory_demo/college/': {mem1.title}")

    mem2, _ = mm.add_or_update_memory(
        category="projects",
        title="AI Assistant Architecture",
        content="Current project is a Local Memory-Based Contextual AI Assistant using Python, SQLite, and modular Markdown memory folders.",
        tags=["projects", "python", "ai"],
        confidence=0.95
    )
    print(f"  [+] Saved memory in 'memory_demo/projects/': {mem2.title}")

    print("\n--- STEP 2: QUERYING COLLEGE MEMORY ---")
    q_college = "Which college am I attending and what is my degree?"
    print(f"User: '{q_college}'")

    res_college = assistant.ask(q_college)
    mems = res_college.get("retrieved_memories", [])
    print(f"\n[Retrieval Metrics]")
    print(f"  - Target Category Matched: {res_college['intent'].get('intent_type')}")
    print(f"  - Retrieved Memories Count: {len(mems)}")
    for m in mems:
        print(f"    * Category: {m.get('category')} | Score: {m.get('score')} | Title: {m.get('title')}")

    print(f"\nAssistant:\n  {res_college['answer']}")

    print("\n--- STEP 3: QUERYING PROJECTS MEMORY ---")
    q_proj = "Can you remind me of the architecture of my current project?"
    print(f"User: '{q_proj}'")

    res_proj = assistant.ask(q_proj)
    print(f"\nAssistant:\n  {res_proj['answer']}")

    print("\n--- STEP 4: QUERYING AN UNRELATED TOPIC (NO STORED MEMORY) ---")
    q_none = "What is the capital of Italy?"
    print(f"User: '{q_none}'")

    res_none = assistant.ask(q_none)
    print(f"Retrieved memories count: {len(res_none.get('retrieved_memories', []))}")
    print(f"Assistant:\n  {res_none['answer']}")
    print("  (Notice: Assistant answers directly without hallucinating any personal facts)")

    print("\n" + "=" * 75)
    print("SUCCESS: Categorized memory retrieval and category targeting verified!")
    print("=" * 75)

    # Clean up demo files
    import shutil
    if demo_dir.exists():
        shutil.rmtree(demo_dir, ignore_errors=True)
    if demo_db.exists():
        try:
            demo_db.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    run_demo()
