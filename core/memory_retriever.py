"""
Memory Retriever module.
Implements the multi-stage retrieval pipeline:
1. Category prediction & keyword extraction.
2. Semantic memory search + exact term matching.
3. Ranking by semantic relevance, recency, confidence, and keyword overlap.
4. Redundancy elimination and context budget enforcement.
"""

import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from config import Config
from core.memory_manager import MemoryManager, MemoryItem
from core.similarity_engine import SimilarityEngine
from core.conversation_manager import ConversationManager
from core.memory_classifier import CATEGORY_KEYWORDS


class MemoryRetriever:
    """Retrieves and ranks relevant memories and historical conversation context."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        similarity_engine: SimilarityEngine,
        conversation_manager: Optional[ConversationManager] = None
    ):
        self.memory_manager = memory_manager
        self.similarity_engine = similarity_engine
        self.conversation_manager = conversation_manager

    def identify_relevant_categories(self, query: str) -> List[str]:
        """
        Identify memory categories most likely related to the query.
        Returns matching categories, or all categories if query is broad or ambiguous.
        """
        lower = query.lower()
        matched = []

        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", lower) for kw in keywords):
                matched.append(cat)

        # Broad personal queries ("about me", "know about", "my info") → search all
        broad_patterns = [r"\babout me\b", r"\bknow about\b", r"\bmy info\b", r"\btell me about me\b"]
        if any(re.search(p, lower) for p in broad_patterns):
            return list(Config.CATEGORIES)

        # If none specifically matched, return all active categories
        if not matched:
            return list(Config.CATEGORIES)

        # Always include personal and general as secondary fallbacks
        for fallback in ["personal", "preferences", "general"]:
            if fallback not in matched and fallback in Config.CATEGORIES:
                matched.append(fallback)

        return matched

    def extract_keywords(self, query: str) -> List[str]:
        """Extract key technical terms, acronyms, and proper nouns."""
        words = re.findall(r"\b[A-Za-z0-9_\-\.]{3,}\b", query)
        stopwords = {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "about", "above", "after", "again", "against", "all", "and", "any", "are",
            "can", "could", "did", "does", "doing", "down", "during", "each", "few",
            "for", "from", "further", "had", "has", "have", "having", "here", "how",
            "into", "just", "more", "most", "other", "our", "out", "over", "same",
            "should", "some", "such", "than", "that", "the", "their", "them", "then",
            "there", "these", "they", "this", "those", "through", "too", "under",
            "until", "very", "was", "were", "what", "when", "where", "which", "while",
            "with", "would", "you", "your", "tell", "explain", "give", "please"
        }
        return [w for w in words if w.lower() not in stopwords]

    def _compute_recency_score(self, iso_timestamp: str) -> float:
        """Compute a normalized score [0.0 - 1.0] representing recency."""
        try:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_ago = (now - dt).total_seconds() / 86400.0
            # Exponential decay over 30 days
            return float(max(0.1, 1.0 / (1.0 + (days_ago / 30.0))))
        except Exception:
            return 0.5

    def search_memories(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        min_relevance: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search across categorized memories.
        Combines semantic similarity, exact keyword matches, confidence, and recency.
        """
        k = top_k or Config.MAX_RETRIEVED_MEMORIES
        thresh = min_relevance if min_relevance is not None else Config.MEMORY_RELEVANCE_THRESHOLD

        target_categories = categories or self.identify_relevant_categories(query)

        # Collect candidate memories
        candidates: List[MemoryItem] = []
        for cat in target_categories:
            candidates.extend(self.memory_manager.list_memories(category=cat))

        # If candidates is empty and we restricted categories, expand to all
        if not candidates and categories is not None:
            candidates = self.memory_manager.list_memories()

        if not candidates or not query.strip():
            return []

        # Prepare text representations for semantic search including category and title
        candidate_texts = [f"[{m.category}] {m.title}. {m.content}. Tags: {' '.join(m.tags)}" for m in candidates]
        semantic_scores = self.similarity_engine.compute_similarity(query, candidate_texts)

        keywords = [kw.lower() for kw in self.extract_keywords(query)]

        ranked_results = []
        seen_contents: set = set()

        for mem, sem_score in zip(candidates, semantic_scores):
            # ── QUALITY FILTER: skip memories that are pure questions with no personal fact ──
            mem_content_stripped = mem.content.strip()
            is_question_memory = (
                mem_content_stripped.endswith("?")
                and not re.search(
                    r"\b(i am|i'm|my|i have|i live|i work|i study|i want|i enjoy)\b",
                    mem_content_stripped.lower()
                )
            )
            if is_question_memory:
                continue  # Don't surface question-type stored memories

            # Keyword exact matching score - includes category name and tags
            mem_text_lower = f"{mem.category} {mem.title} {mem.content} {' '.join(mem.tags)}".lower()
            kw_hits = sum(1 for kw in keywords if kw in mem_text_lower)
            kw_score = min(1.0, kw_hits / (len(keywords) or 1))

            # Recency & confidence scores
            recency = self._compute_recency_score(mem.updated_at)
            confidence = mem.confidence

            # Composite ranking score
            composite_score = (
                (0.50 * sem_score) +
                (0.30 * kw_score) +
                (0.12 * confidence) +
                (0.08 * recency)
            )

            # Accept if semantic is high OR strong keyword match
            if composite_score >= thresh or kw_score >= 0.5 or sem_score >= 0.40:
                # Deduplication check
                content_key = mem.content.strip().lower()
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)

                ranked_results.append({
                    "item": mem,
                    "score": round(composite_score, 4),
                    "semantic_score": round(sem_score, 4),
                    "keyword_score": round(kw_score, 4),
                    "confidence": mem.confidence,
                    "category": mem.category,
                    "title": mem.title,
                    "content": mem.content,
                    "source": mem.source_conversation,
                    "updated_at": mem.updated_at
                })

        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:k]

    def retrieve_full_context(
        self,
        query: str,
        conversation_id: str,
        max_tokens_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Orchestrate complete retrieval:
        1. Recent conversation history for this conversation.
        2. Similar past questions & answers from older conversations.
        3. Relevant categorized memories.
        Enforces a context budget to avoid prompt bloat.
        """
        budget = max_tokens_budget or Config.MAX_CONTEXT_TOKENS

        # 1. Recent conversation context
        recent_messages: List[Dict[str, Any]] = []
        if self.conversation_manager:
            recent_messages = self.conversation_manager.get_recent_messages(conversation_id)

        # 2. Similar past questions (from other conversations or earlier)
        similar_questions: List[Dict[str, Any]] = []
        if self.conversation_manager:
            past_qa = self.conversation_manager.get_all_user_questions(exclude_conversation_id=conversation_id)
            if past_qa:
                similar_questions = self.similarity_engine.find_similar_questions(
                    query, past_qa, threshold=Config.SIMILARITY_THRESHOLD, top_k=3
                )

        # 3. Categorized memories
        relevant_memories = self.search_memories(query)

        # Enforce character/token budget (~4 chars per token)
        char_budget = budget * 4
        current_char_count = sum(len(m.get("content", "")) for m in recent_messages)

        budgeted_memories = []
        for mem_dict in relevant_memories:
            cost = len(mem_dict["content"]) + len(mem_dict["title"]) + 50
            if current_char_count + cost <= char_budget:
                budgeted_memories.append(mem_dict)
                current_char_count += cost
            else:
                break

        return {
            "query": query,
            "conversation_id": conversation_id,
            "recent_messages": recent_messages,
            "similar_past_questions": similar_questions,
            "retrieved_memories": budgeted_memories,
            "matched_categories": list({m["category"] for m in budgeted_memories}),
            "keywords": self.extract_keywords(query)
        }
