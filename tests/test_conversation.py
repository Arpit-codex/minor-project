"""
Automated tests for conversation persistence, message ordering,
session restarts, and context size limits.
"""

import pytest
from core.conversation_manager import ConversationManager
from core.assistant import ContextualAssistant
from core.ai_client import AIClient


def test_scenario_1_new_question_with_no_history(tmp_path):
    """Scenario 1: A new question with no relevant history."""
    db_file = tmp_path / "test_conv.db"
    cm = ConversationManager(db_path=db_file)
    ai = AIClient(provider_type="mock")
    bot = ContextualAssistant(conversation_manager=cm, ai_client=ai)

    conv_id = cm.create_conversation("First Chat")
    res = bot.ask("What is quantum computing?", conversation_id=conv_id)

    assert not res.get("is_error")
    assert res.get("intent", {}).get("intent_type") == "new_question"
    assert len(res.get("similar_past_questions", [])) == 0
    assert "quantum" in res.get("answer").lower() or len(res.get("answer")) > 10


def test_scenario_11_empty_or_invalid_user_input(tmp_path):
    """Scenario 11: Empty or invalid user input."""
    db_file = tmp_path / "test_conv.db"
    cm = ConversationManager(db_path=db_file)
    bot = ContextualAssistant(conversation_manager=cm, ai_client=AIClient(provider_type="mock"))

    res1 = bot.ask("")
    assert res1["is_error"] is True
    assert res1["error_type"] == "empty_input"

    res2 = bot.ask("   \t\n  ")
    assert res2["is_error"] is True
    assert res2["error_type"] == "empty_input"


def test_scenario_12_persistence_after_application_restart(tmp_path):
    """Scenario 12: Conversation persistence after restarting the application."""
    db_file = tmp_path / "persistent.db"

    # Session 1: Create and send messages
    cm1 = ConversationManager(db_path=db_file)
    cid = cm1.create_conversation(title="Session Before Restart")
    cm1.add_message(cid, "user", "My favorite programming language is Python.")
    cm1.add_message(cid, "assistant", "Python is indeed very versatile!")

    # Simulate restart by completely deleting instance and creating a fresh one pointing to the same SQLite file
    del cm1
    cm2 = ConversationManager(db_path=db_file)

    stored_conv = cm2.get_conversation(cid)
    assert stored_conv is not None
    assert stored_conv["title"] == "Session Before Restart"

    messages = cm2.get_recent_messages(cid)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert "Python" in messages[0]["content"]
    assert messages[1]["role"] == "assistant"


def test_scenario_14_context_size_limits_for_long_conversations(tmp_path):
    """Scenario 14: Context-size limits for long conversations."""
    db_file = tmp_path / "long_conv.db"
    cm = ConversationManager(db_path=db_file)
    cid = cm.create_conversation(title="Long Session")

    # Add 25 back-and-forth turns (50 messages)
    for i in range(25):
        cm.add_message(cid, "user", f"Question number {i}: Can you explain detail {i}?")
        cm.add_message(cid, "assistant", f"Answer number {i}: Here is detail {i}.")

    # Limit to 8 recent messages
    recent = cm.get_recent_messages(cid, limit=8)
    assert len(recent) == 8
    # Chronological order check: the last message should be the latest answer
    assert "detail 24" in recent[-1]["content"]
    assert "detail 21" in recent[0]["content"] or "detail 20" in recent[0]["content"]
