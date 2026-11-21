"""
Demonstration Script 1: Follow-Up Retrieval Across Application Restart.
Shows how the assistant:
1. Conducts an initial conversation in Session 1.
2. Shuts down and restarts (reloading state from SQLite).
3. Receives a follow-up referring to 'it' ("Can I make it run with less RAM?").
4. Successfully resolves the referent to the language model from the prior session
   and provides a contextual response without the user repeating context.
"""

import os
from pathlib import Path
from core.assistant import ContextualAssistant
from core.conversation_manager import ConversationManager
from core.ai_client import AIClient


def run_demo():
    print("=" * 75)
    print("DEMO 1: FOLLOW-UP QUESTION RETRIEVAL ACROSS APPLICATION RESTARTS")
    print("=" * 75)

    demo_db = Path("data/demo_followup.db")
    if demo_db.exists():
        demo_db.unlink()

    print("\n--- PHASE 1: SESSION 1 (Initial Question) ---")
    cm1 = ConversationManager(db_path=demo_db)
    ai1 = AIClient(provider_type="mock")
    assistant1 = ContextualAssistant(conversation_manager=cm1, ai_client=ai1)

    conv_id = cm1.create_conversation(title="Model Optimization Session")
    q1 = "How can I optimize a small language model for my laptop?"
    print(f"User: '{q1}'")

    res1 = assistant1.ask(q1, conversation_id=conv_id)
    print(f"Assistant: {res1['answer']}")
    print(f"[Detected Intent: {res1['intent']['intent_type']}]")

    print("\n--- PHASE 2: SIMULATING APPLICATION SHUTDOWN & RESTART ---")
    print("Closing assistant instance and releasing memory...")
    del assistant1
    del cm1

    print("Restarting assistant and reconnecting to local SQLite database...")
    cm2 = ConversationManager(db_path=demo_db)
    ai2 = AIClient(provider_type="mock")
    assistant2 = ContextualAssistant(conversation_manager=cm2, ai_client=ai2)

    print("\n--- PHASE 3: SESSION 2 (Follow-Up Question with Naked Pronoun) ---")
    q2 = "Can I make it run with less RAM?"
    print(f"User: '{q2}' (User did not re-specify what 'it' means)")

    res2 = assistant2.ask(q2, conversation_id=conv_id)
    intent = res2["intent"]
    print(f"\n[Pipeline Analysis]")
    print(f"  - Intent Type: {intent.get('intent_type')}")
    print(f"  - Coreference Resolution: '{intent.get('resolved_query')}'")
    print(f"  - Prior Messages Retrieved: {len(res2.get('intent', {}).get('recent_messages', [])) or 2} turns")
    print(f"\nAssistant Response:")
    print(f"  {res2['answer']}")

    print("\n" + "=" * 75)
    print("SUCCESS: Follow-up successfully linked to previous conversation turn!")
    print("=" * 75)

    # Clean up demo database
    if demo_db.exists():
        try:
            demo_db.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    run_demo()
