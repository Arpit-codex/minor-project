"""
Automated end-to-end tests for the complete Assistant pipeline:
Follow-ups, pronoun resolution, and handling API errors (rate limits & network issues).
"""

import pytest
from core.assistant import ContextualAssistant
from core.conversation_manager import ConversationManager
from core.memory_manager import MemoryManager
from core.similarity_engine import SimilarityEngine
from core.ai_client import AIClient, MockAIProvider


def test_scenario_2_followup_question_referring_to_previous_message(tmp_path):
    """
    Scenario 2: A follow-up question referring to the previous message.
    User asks: 'How can I optimize a small language model for my laptop?'
    Follow-up: 'Can I make it run with less RAM?'
    Assistant should recognize 'it' refers to the model and provide a context-aware answer.
    """
    db_file = tmp_path / "conv_pipeline.db"
    mem_dir = tmp_path / "memories"

    cm = ConversationManager(db_path=db_file)
    mm = MemoryManager(memory_dir=mem_dir)
    se = SimilarityEngine()
    ai = AIClient(provider_type="mock")

    bot = ContextualAssistant(
        conversation_manager=cm,
        memory_manager=mm,
        similarity_engine=se,
        ai_client=ai
    )

    cid = cm.create_conversation("Model Optimization Chat")

    # Step 1: Initial question
    q1 = "How can I optimize a small language model for my laptop?"
    res1 = bot.ask(q1, conversation_id=cid)
    assert not res1["is_error"]
    assert res1["intent"]["intent_type"] == "new_question"

    # Step 2: Follow-up question with pronoun "it"
    q2 = "Can I make it run with less RAM?"
    res2 = bot.ask(q2, conversation_id=cid)
    assert not res2["is_error"]
    assert res2["intent"]["intent_type"] == "follow_up"
    # Resolved query should associate 'it' with the language model
    assert "language model" in res2["intent"]["resolved_query"].lower() or "it" in res2["intent"]["resolved_query"].lower()
    # Mock response incorporates RAM/quantization knowledge
    assert "quantization" in res2["answer"].lower() or "ram" in res2["answer"].lower()


def test_scenario_10_api_rate_limits_and_network_failures(tmp_path):
    """
    Scenario 10: API rate limits and network failures handled gracefully.
    System reports the problem clearly and does not crash.
    """
    db_file = tmp_path / "conv_errors.db"
    cm = ConversationManager(db_path=db_file)
    ai = AIClient(provider_type="mock")

    bot = ContextualAssistant(conversation_manager=cm, ai_client=ai)

    # 1. Simulate Rate Limit (HTTP 429)
    ai.mock_provider.simulate_rate_limit = True
    res_rate = bot.ask("What is machine learning?")
    assert res_rate["is_error"] is True
    assert res_rate["error_type"] == "rate_limit"
    assert "rate limit" in res_rate["answer"].lower() or "429" in res_rate["answer"]

    # 2. Simulate Network Failure
    ai.mock_provider.simulate_rate_limit = False
    ai.mock_provider.simulate_network_error = True
    res_net = bot.ask("What is deep learning?")
    assert res_net["is_error"] is True
    assert res_net["error_type"] == "network_error"
    assert "network" in res_net["answer"].lower() or "connection" in res_net["answer"].lower()

    # 3. Resume normal operations
    ai.mock_provider.simulate_network_error = False
    res_ok = bot.ask("What is reinforcement learning?")
    assert res_ok["is_error"] is False
