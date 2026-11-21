"""
Automated tests for memory retrieval: category targeting,
multi-category retrieval, and handling queries with no matching memories.
"""

import pytest
from core.memory_manager import MemoryManager
from core.similarity_engine import SimilarityEngine
from core.memory_retriever import MemoryRetriever


def test_scenario_5_retrieval_from_correct_category(tmp_path):
    """Scenario 5: Retrieval from the correct memory category."""
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)
    se = SimilarityEngine()
    retriever = MemoryRetriever(mm, se)

    # Populate multiple categories
    mm.add_or_update_memory("college", "University", "I am studying at National Institute of Technology.")
    mm.add_or_update_memory("family", "Brother", "My brother's name is Alex and he lives in Boston.")
    mm.add_or_update_memory("career", "Job", "I work as a software engineer at TechCorp.")

    # Target college query
    results = retriever.search_memories("Where do I go to college?", categories=["college"])
    assert len(results) >= 1
    assert results[0]["category"] == "college"
    assert "National Institute of Technology" in results[0]["content"]


def test_scenario_6_retrieval_from_multiple_categories(tmp_path):
    """
    Scenario 6: Retrieval from multiple categories when needed.
    (e.g., A query touching upon academic projects involves both 'college' and 'projects').
    """
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)
    se = SimilarityEngine()
    retriever = MemoryRetriever(mm, se)

    mm.add_or_update_memory("college", "Graduation", "Final year B.Tech semester exams are in May.")
    mm.add_or_update_memory("projects", "Capstone Project", "Building a distributed database system for my final year degree capstone.")
    mm.add_or_update_memory("family", "Pet", "I have a golden retriever named Max.")

    # Query bridging college and project
    results = retriever.search_memories("Tell me about my final year degree capstone and exams", categories=["college", "projects"])
    assert len(results) >= 2
    matched_cats = {r["category"] for r in results}
    assert "college" in matched_cats
    assert "projects" in matched_cats


def test_scenario_7_no_relevant_memory_found(tmp_path):
    """
    Scenario 7: No relevant memory found.
    When querying about an unrelated topic, retriever returns empty list or below threshold.
    """
    mem_dir = tmp_path / "memory"
    mm = MemoryManager(memory_dir=mem_dir)
    se = SimilarityEngine()
    retriever = MemoryRetriever(mm, se)

    # Store some existing memories
    mm.add_or_update_memory("preferences", "Theme", "I like dark mode on my code editor.")

    # Query about completely unrelated topic
    results = retriever.search_memories("What is the recipe for chocolate chip cookies?", min_relevance=0.45)
    assert len(results) == 0
