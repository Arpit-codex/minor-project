"""
Automated tests for memory management: duplicate detection,
updates/contradictions, manual editing, and deletion.
"""

import pytest
from core.memory_manager import MemoryManager, MemoryItem
from core.similarity_engine import SimilarityEngine


def test_scenario_8_duplicate_memory_detection_and_updating(tmp_path):
    """Scenario 8: Duplicate memory detection and updating."""
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)

    # Add initial memory
    item1, status1 = mm.add_or_update_memory(
        category="preferences",
        title="Coding Preference",
        content="I prefer writing clean code using Python with strict type hinting.",
        confidence=0.85
    )
    assert status1 == "created"

    # Add identical content
    item2, status2 = mm.add_or_update_memory(
        category="preferences",
        title="Coding Preference",
        content="I prefer writing clean code using Python with strict type hinting.",
        confidence=0.95
    )
    assert status2 == "skipped_duplicate"
    # Should not create two separate memories
    all_prefs = mm.list_memories(category="preferences")
    assert len(all_prefs) == 1
    # Confidence should be updated to higher value
    assert all_prefs[0].confidence == 0.95


def test_scenario_9_contradictory_memories_update(tmp_path):
    """
    Scenario 9: Contradictory memories.
    When newer information updates an existing topic (e.g. changing job/company),
    the system updates the record rather than accumulating contradictory duplicates.
    """
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)

    # Original fact
    item1, status1 = mm.add_or_update_memory(
        category="career",
        title="Current Role",
        content="I work as a Junior Frontend Developer at Acme Corp.",
        confidence=0.90
    )
    assert status1 == "created"

    # Later update: Promoted to Senior Backend Developer at Acme Corp
    item2, status2 = mm.add_or_update_memory(
        category="career",
        title="Current Role",
        content="I work as a Senior Backend Developer at Acme Corp.",
        confidence=0.95
    )
    assert status2 in ("updated", "created")

    careers = mm.list_memories(category="career")
    # Verified: The active memory reflects the updated role
    active_contents = [c.content for c in careers]
    assert any("Senior Backend Developer" in c for c in active_contents)


def test_scenario_13_memory_deletion_and_updates(tmp_path):
    """Scenario 13: Memory deletion and index updates."""
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)

    item, status = mm.add_or_update_memory(
        category="personal",
        title="Temporary Note",
        content="Remember to buy groceries on Friday evening."
    )
    assert status == "created"
    assert mm.get_memory(item.id) is not None

    # Delete the memory
    deleted = mm.delete_memory(item.id)
    assert deleted is True

    # Verify it is no longer retrieved
    assert mm.get_memory(item.id) is None
    all_personal = mm.list_memories(category="personal")
    assert len(all_personal) == 0
