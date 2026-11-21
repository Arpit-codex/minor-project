"""
Integration tests verifying:
1. AI API Key Integration & Validation
2. Chat Isolation & Independent History per Conversation
3. Storage Persistence and Clean Resets
4. LLM Client Error Handling (No silent mock fallback in production)
"""

import os
import json
import pytest
from ai_efficiency.core.config import (
    AISettings,
    validate_configuration,
    test_connection as api_test_connection,
    save_settings,
    load_settings,
)
from ai_efficiency.core.chat_manager import ChatManager, Conversation, ChatMessage
from ai_efficiency.core.fidelity.llm_client import LLMClient


def test_missing_api_key_validation():
    # 1. Start application without an API key -> requests API configuration
    settings = AISettings(provider="openai", api_key="", model_name="gpt-4o")
    valid, msg = validate_configuration(settings)
    assert not valid
    assert "API Key is missing" in msg


def test_missing_model_name_validation():
    settings = AISettings(provider="openai", api_key="sk-test", model_name="")
    valid, msg = validate_configuration(settings)
    assert not valid
    assert "Model name is missing" in msg


def test_ollama_local_does_not_require_api_key():
    settings = AISettings(provider="ollama", api_key="", model_name="llama3")
    valid, msg = validate_configuration(settings)
    assert valid


def test_invalid_api_key_connection():
    # 2. Enter an invalid API key -> displays a meaningful authentication error
    settings = AISettings(provider="openai", api_key="sk-invalid-test-key-12345", model_name="gpt-4o")
    ok, err_msg = api_test_connection(settings)
    assert not ok
    assert len(err_msg) > 0


def test_no_silent_mock_fallback_on_invalid_key():
    # Production chat must NOT silently fall back to mock DeterministicFrontierEngine
    client = LLMClient(model_name="gpt-4o", api_key="sk-invalid-key-testing", base_url="https://api.openai.com/v1")
    resp = client.chat_completion(
        messages=[{"role": "user", "content": "What is 2 + 2?"}]
    )
    # Must NOT have returned mock/simulated text; must report api_error
    assert resp.api_error is not None
    assert "Authentication" in resp.api_error or "401" in resp.api_error or "API Error" in resp.api_error
    assert not resp.is_live_api or resp.text == ""


def test_chat_isolation_and_independent_history(tmp_path):
    storage_file = str(tmp_path / "test_conversations.json")
    cm = ChatManager(storage_file=storage_file)

    # 5. Create two separate conversations
    c1 = cm.create_conversation("Chat 1")
    c2 = cm.create_conversation("Chat 2")

    assert c1.id != c2.id
    assert len(c1.messages) == 0
    assert len(c2.messages) == 0

    # 4 & 5. Send two completely different questions in Chat 1
    cm.add_message(c1.id, "user", "How do I reverse a linked list in Python?")
    cm.add_message(c1.id, "assistant", "Use prev, curr, next pointers...")
    cm.add_message(c1.id, "user", "What is its time complexity?")
    cm.add_message(c1.id, "assistant", "It is O(N) time and O(1) auxiliary space.")

    # In Chat 2, send completely different topic
    cm.add_message(c2.id, "user", "What is the capital of France?")
    cm.add_message(c2.id, "assistant", "Paris is the capital of France.")

    # 6. Verify messages in Chat 1 do not leak to Chat 2
    llm_msgs_c1 = cm.get_messages_for_llm(c1.id)
    llm_msgs_c2 = cm.get_messages_for_llm(c2.id)

    assert len(llm_msgs_c1) == 4
    assert len(llm_msgs_c2) == 2

    c1_contents = " ".join([m["content"] for m in llm_msgs_c1])
    c2_contents = " ".join([m["content"] for m in llm_msgs_c2])

    assert "linked list" in c1_contents
    assert "Paris" not in c1_contents
    assert "Paris" in c2_contents
    assert "linked list" not in c2_contents


def test_conversation_persistence(tmp_path):
    storage_file = str(tmp_path / "test_persist.json")
    cm1 = ChatManager(storage_file=storage_file)
    conv = cm1.create_conversation()
    cm1.add_message(conv.id, "user", "Hello persistent world")

    # 7. Restart simulation by reloading from disk
    cm2 = ChatManager(storage_file=storage_file)
    loaded = cm2.get_conversation(conv.id)
    assert loaded is not None
    assert loaded.title == "Hello persistent world"
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content == "Hello persistent world"


def test_settings_persistence(tmp_path):
    cfg_file = str(tmp_path / "test_settings.json")
    s1 = AISettings(provider="groq", api_key="gsk-test", model_name="llama-3.3-70b-versatile")
    save_settings(s1, filepath=cfg_file)

    s2 = load_settings(filepath=cfg_file)
    assert s2.provider == "groq"
    assert s2.api_key == "gsk-test"
    assert s2.model_name == "llama-3.3-70b-versatile"
    assert s2.base_url == "https://api.groq.com/openai/v1"
