"""
Automated tests for semantic question similarity, candidate ranking,
and distinguishing different intended meanings.
"""

import pytest
from core.similarity_engine import SimilarityEngine


def test_scenario_3_semantically_similar_question_different_wording():
    """Scenario 3: A semantically similar question with different wording."""
    engine = SimilarityEngine()

    q_stored = "How can I optimize a small language model for my laptop?"
    past_questions = [
        {"id": 1, "question": q_stored, "answer": "You can use 4-bit quantization."},
        {"id": 2, "question": "What is the best recipe for baking sourdough bread?", "answer": "Use active starter."},
        {"id": 3, "question": "Who was the first president of the United States?", "answer": "George Washington."}
    ]

    new_query = "How to run an LLM efficiently on a notebook computer?"
    similar = engine.find_similar_questions(new_query, past_questions, threshold=0.15, top_k=3)

    assert len(similar) >= 1
    # Top match must be the language model question, not bread or president
    top_match = similar[0]
    assert "language model" in top_match["question"] or "laptop" in top_match["question"]
    assert top_match["similarity_score"] > 0.15


def test_scenario_4_similar_questions_different_intended_meanings():
    """
    Scenario 4: Two similar questions with different intended meanings.
    (e.g., install vs uninstall, or same subject but opposite action).
    """
    engine = SimilarityEngine()

    q_install = "How do I install Docker desktop on Windows 11?"
    q_uninstall = "How do I uninstall and remove Docker desktop completely from Windows 11?"
    q_weather = "What is the weather in Tokyo?"

    candidates = [q_install, q_uninstall, q_weather]
    scores_for_install = engine.compute_similarity("I want to install Docker on Windows", candidates)

    # Score for install should be higher than uninstall, and both significantly higher than weather
    score_install = scores_for_install[0]
    score_uninstall = scores_for_install[1]
    score_weather = scores_for_install[2]

    assert score_install > score_uninstall
    assert score_uninstall > score_weather
    assert score_weather < 0.10
