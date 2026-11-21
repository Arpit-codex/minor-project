"""
Tests for Independent Memory and Retrieval Modes:
- Normal Mode (Write: OFF, Retrieval: OFF)
- Memory Mode (Write: ON, Retrieval: OFF)
- Retrieval Mode (Write: OFF, Retrieval: ON)
- Combined Mode (Write: ON, Retrieval: ON)
- Mode switching and conversation persistence
- Background non-blocking memory extraction
- Sensitive data filtering & error handling
- Web API endpoints
"""

import time
import pytest
from core.assistant import ContextualAssistant
from core.conversation_manager import ConversationManager
from core.memory_manager import MemoryManager
from core.similarity_engine import SimilarityEngine
from core.ai_client import AIClient, MockAIProvider
from gui_server import app


@pytest.fixture
def test_setup(tmp_path):
    """Fixture to build isolated assistant environment."""
    db_file = tmp_path / "test_modes_conv.db"
    mem_dir = tmp_path / "test_modes_mem"
    cm = ConversationManager(db_path=db_file)
    se = SimilarityEngine()
    mm = MemoryManager(memory_dir=mem_dir, similarity_engine=se)
    ai = AIClient(provider_type="mock")
    bot = ContextualAssistant(
        conversation_manager=cm,
        memory_manager=mm,
        similarity_engine=se,
        ai_client=ai
    )
    return {"cm": cm, "mm": mm, "se": se, "ai": ai, "bot": bot, "mem_dir": mem_dir}


def test_1_normal_mode_no_retrieval_no_writing(test_setup):
    """
    Test 1: Normal mode
    - AI answers properly.
    - Short-term session context is preserved.
    - Memory retrieval is never executed (empty retrieved_memories).
    - Memory writing is never executed (zero memories created).
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    # Seed an existing memory that would match if retrieval was on
    mm.add_or_update_memory(
        category="projects",
        title="Python Assistant",
        content="The user is building a personal AI assistant in Python."
    )
    assert len(mm.list_memories()) == 1

    cid = cm.create_conversation("Normal Chat Test", mode="normal")

    # Turn 1: User introduces themselves
    res1 = bot.ask(
        "Hi, I am Bob and I study Electrical Engineering.",
        conversation_id=cid,
        mode="normal",
        sync_memory_write=True
    )
    assert not res1["is_error"]
    assert res1["mode"] == "normal"
    assert res1["memory_write"] is False
    assert res1["memory_retrieval"] is False
    assert len(res1["retrieved_memories"]) == 0
    # No memory was extracted/written
    assert len(mm.list_memories()) == 1

    # Turn 2: Follow-up relying on short-term session context
    res2 = bot.ask(
        "What did I say my name was?",
        conversation_id=cid,
        mode="normal",
        sync_memory_write=True
    )
    assert not res2["is_error"]
    assert len(res2["retrieved_memories"]) == 0
    assert len(mm.list_memories()) == 1


def test_2_memory_mode_writes_memory_without_retrieval(test_setup):
    """
    Test 2: Memory mode
    - AI answers correctly.
    - New useful information is extracted and stored.
    - Existing memories are NOT retrieved for response generation.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    # Seed an existing memory
    mm.add_or_update_memory(
        category="personal",
        title="Hometown",
        content="I live in San Francisco, California."
    )
    assert len(mm.list_memories()) == 1

    cid = cm.create_conversation("Memory Chat Test", mode="memory")

    # User discloses a genuine project fact
    res = bot.ask(
        "I am developing a full-stack robotics dashboard using ROS2 and React.",
        conversation_id=cid,
        mode="memory",
        sync_memory_write=True
    )
    assert not res["is_error"]
    assert res["mode"] == "memory"
    assert res["memory_write"] is True
    assert res["memory_retrieval"] is False
    # Retrieval disabled -> retrieved_memories must be empty
    assert len(res["retrieved_memories"]) == 0

    # Verify a new memory was written to disk
    memories = mm.list_memories()
    assert len(memories) >= 2
    proj_memories = [m for m in memories if "robotics" in m.content.lower() or "ros2" in m.content.lower()]
    assert len(proj_memories) > 0


def test_3_retrieval_mode_reads_memory_without_writing(test_setup):
    """
    Test 3: Retrieval mode
    - Relevant memories are retrieved and passed to AI.
    - AI answers contextually.
    - No new long-term memories are created or modified.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    # Seed confirmed memory
    mm.add_or_update_memory(
        category="college",
        title="University Enrolled",
        content="I am currently enrolled in the Computer Science department at Stanford University."
    )
    initial_count = len(mm.list_memories())

    cid = cm.create_conversation("Retrieval Chat Test", mode="retrieval")

    # Ask about college
    res = bot.ask(
        "Which university do I attend?",
        conversation_id=cid,
        mode="retrieval",
        sync_memory_write=True
    )
    assert not res["is_error"]
    assert res["mode"] == "retrieval"
    assert res["memory_write"] is False
    assert res["memory_retrieval"] is True
    assert len(res["retrieved_memories"]) > 0
    assert any("stanford" in m.get("content", "").lower() for m in res["retrieved_memories"])

    # Ensure memory count DID NOT increase (writing disabled)
    assert len(mm.list_memories()) == initial_count


def test_4_combined_mode_both_retrieves_and_writes(test_setup):
    """
    Test 4: Combined mode (Both)
    - Retrieves existing memories.
    - Contextual response generated.
    - Saves new useful information and updates existing.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    # Seed memory
    mm.add_or_update_memory(
        category="career",
        title="Career Goal",
        content="I want to become a backend software engineer at Stripe."
    )

    cid = cm.create_conversation("Combined Chat Test", mode="both")

    # Turn 1: Retrieve existing career goal
    res1 = bot.ask(
        "Which career goal did I set?",
        conversation_id=cid,
        mode="both",
        sync_memory_write=True
    )
    assert not res1["is_error"]
    assert res1["mode"] == "both"
    assert res1["memory_write"] is True
    assert res1["memory_retrieval"] is True
    assert len(res1["retrieved_memories"]) > 0
    assert any("stripe" in m.get("content", "").lower() for m in res1["retrieved_memories"])

    # Turn 2: Disclose new career goal
    res2 = bot.ask(
        "I have updated my career goal: I want to become a machine learning infrastructure engineer.",
        conversation_id=cid,
        mode="both",
        sync_memory_write=True
    )
    assert not res2["is_error"]

    # New memory extracted into career
    memories = mm.list_memories()
    assert len(memories) >= 2
    assert any("machine learning" in m.content.lower() for m in memories)


def test_mode_switching_and_database_persistence(test_setup):
    """
    Test mode switching and persistence:
    - Mode saved with conversation in SQLite.
    - Switching mode updates SQLite.
    - Restoring conversation restores saved mode.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]

    cid = cm.create_conversation("Switchable Conv", mode="normal")
    conv = cm.get_conversation(cid)
    assert conv["mode"] == "normal"

    # Switch mode to retrieval
    success = cm.update_conversation_mode(cid, "retrieval")
    assert success is True
    conv_after = cm.get_conversation(cid)
    assert conv_after["mode"] == "retrieval"

    # Simulate fresh assistant load from same DB
    cm2 = ConversationManager(db_path=test_setup["cm"].db_path)
    conv_reloaded = cm2.get_conversation(cid)
    assert conv_reloaded["mode"] == "retrieval"


def test_sensitive_information_is_never_stored(test_setup):
    """
    Security safeguard:
    API keys, passwords, and secrets must never be stored as memories even in memory mode.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    cid = cm.create_conversation("Security Test", mode="memory")
    res = bot.ask(
        "My OpenAI API key is sk-proj-1234567890abcdef and my password is SecretPassword123!",
        conversation_id=cid,
        mode="memory",
        sync_memory_write=True
    )
    assert not res["is_error"]

    # Verify no memory was created containing the secret
    for mem in mm.list_memories():
        assert "SecretPassword123" not in mem.content
        assert "sk-proj" not in mem.content


def test_background_memory_extraction_non_blocking(test_setup):
    """
    Test non-blocking asynchronous memory extraction.
    """
    bot = test_setup["bot"]
    cm = test_setup["cm"]
    mm = test_setup["mm"]

    cid = cm.create_conversation("Async Test", mode="memory")

    # Ask without sync_memory_write (default is async background thread)
    t0 = time.time()
    res = bot.ask(
        "I am learning Rust programming and WebAssembly for high performance systems.",
        conversation_id=cid,
        mode="memory",
        sync_memory_write=False
    )
    elapsed = time.time() - t0
    assert not res["is_error"]
    # Fast response
    assert elapsed < 3.0

    # Give background thread a moment to finish file writing
    time.sleep(0.3)
    memories = mm.list_memories()
    assert len(memories) >= 1


def test_gui_server_api_routes():
    """
    Test web API routes in gui_server:
    - /
    - /memory-graph
    - /chat
    - /api/conversations
    - /api/chat
    - /api/config
    """
    client = app.test_client()

    # 1. Main views
    res_index = client.get("/")
    assert res_index.status_code == 200

    res_graph = client.get("/memory-graph")
    assert res_graph.status_code == 200

    res_chat = client.get("/chat")
    assert res_chat.status_code == 200

    # 2. Config endpoint
    res_config = client.get("/api/config")
    assert res_config.status_code == 200
    cfg = res_config.get_json()
    assert "active_provider" in cfg
    assert "categories" in cfg

    # 3. Create conversation endpoint
    res_create = client.post("/api/conversations", json={"title": "API Test", "mode": "normal"})
    assert res_create.status_code == 200
    conv_data = res_create.get_json()
    cid = conv_data["id"]
    assert conv_data["mode"] == "normal"

    # 4. Update mode endpoint
    res_mode = client.put(f"/api/conversations/{cid}/mode", json={"mode": "both"})
    assert res_mode.status_code == 200
    assert res_mode.get_json()["mode"] == "both"

    # 5. Chat endpoint
    res_chat_post = client.post("/api/chat", json={
        "conversation_id": cid,
        "message": "Hello from web API!",
        "mode": "normal",
        "ai_provider": "mock"
    })
    assert res_chat_post.status_code == 200
    chat_resp = res_chat_post.get_json()
    assert chat_resp["is_error"] is False
    assert len(chat_resp["answer"]) > 0
    assert chat_resp["mode"] == "normal"

    # 6. Delete conversation endpoint
    res_del = client.delete(f"/api/conversations/{cid}")
    assert res_del.status_code == 200
    assert res_del.get_json()["deleted"] is True
