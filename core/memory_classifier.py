"""
Memory Classifier module.
Extracts and classifies potential long-term memories from conversation turns.
Distinguishes between user self-disclosure and general inquiries.
Assigns categories, confidence scores, and detects sensitive information.
Uses online model when available, with a comprehensive heuristic fallback.
"""

import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from config import Config
from core.memory_manager import MemoryItem, MemoryManager


SENSITIVE_PATTERNS = [
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bapi[_\s]?key\b",
    r"\bcredit[_\s]?card\b",
    r"\bssn\b|\bsocial\s+security\b",
    r"\bdiagnosed with\b|\bmedical record\b",
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "college": [
        "college", "university", "campus", "professor", "semester", "cgpa",
        "degree", "btech", "mtech", "bachelor", "master", "phd", "syllabus",
        "assignment", "course", "curriculum", "nit", "iit", "department",
        "student", "studying", "enrolled", "graduate"
    ],
    "exams": [
        "exam", "test", "finals", "midterm", "gate", "gre", "toefl", "ielts",
        "quiz", "revision", "admit card", "previous year", "pyq", "preparation",
        "score", "rank", "result", "mock test", "practice test"
    ],
    "career": [
        "job", "career", "resume", "cv", "interview", "salary", "promotion",
        "company", "hired", "internship", "linkedin", "portfolio", "offer letter",
        "work", "profession", "goal", "engineer", "developer", "data scientist",
        "ai engineer", "become", "want to be", "aspire"
    ],
    "family": [
        "father", "mother", "sister", "brother", "parents", "family",
        "wife", "husband", "daughter", "son", "cousin", "uncle", "aunt",
        "grandfather", "grandmother", "sibling"
    ],
    "projects": [
        "project", "developing", "building an app", "repository", "architecture",
        "codebase", "stack", "frontend", "backend", "database schema", "github",
        "api", "microservice", "app", "website", "tool", "script"
    ],
    "preferences": [
        "i prefer", "i like concise", "always format", "my preference",
        "dont use", "always use", "response style", "dark mode",
        "format responses", "keep it short", "bullet points", "prefer markdown"
    ],
    "technical_knowledge": [
        "bug fix", "note:", "tip:", "remember:", "command to",
        "configuration note", "how to configure", "solution was",
        "the fix is", "workaround", "issue with", "error message"
    ],
    "personal": [
        "my goal", "my hobby", "i enjoy", "i am learning", "i live in",
        "my name is", "my birthday", "my passion", "i want to become",
        "i am", "years old", "age", "born", "from", "location", "city",
        "country", "background", "interested in"
    ],
}

# Questions are not memories — we never store these
QUESTION_PREFIXES = (
    "what is", "how do", "how can", "how does", "can you", "could you",
    "why does", "tell me about", "who is", "explain", "what are",
    "when does", "where is", "do you know", "what should", "which",
    "is it possible", "give me", "help me", "show me", "list",
)


class MemoryClassifier:
    """Classifies conversation interactions into potential long-term memories."""

    def __init__(self, memory_manager: MemoryManager, ai_client=None):
        self.memory_manager = memory_manager
        self.ai_client = ai_client

    def is_sensitive(self, text: str) -> bool:
        """Check if content mentions sensitive personal data."""
        lower = text.lower()
        for pat in SENSITIVE_PATTERNS:
            if re.search(pat, lower):
                return True
        return False

    def is_first_person_disclosure(self, text: str) -> bool:
        """
        Check if the text is genuinely an explicit disclosure about the user.
        Filters out hypothetical questions, general inquiries, or third-person mentions.
        """
        lower = text.lower().strip()

        # Reject if it starts with ANY question-type word (interrogative form)
        # This catches "What courses should I...", "How can I...", "Do I need..."
        interrogative_starters = (
            "what ", "where ", "when ", "which ", "who ", "why ", "how ",
            "do ", "does ", "did ", "can ", "could ", "should ", "would ",
            "is ", "are ", "was ", "were ", "have ", "has ", "will ",
        )
        if lower.startswith(interrogative_starters):
            return False

        # Also reject pure questions that end with '?' (even if they contain 'my')
        if lower.endswith("?"):
            return False

        # First-person disclosure markers
        first_person_markers = [
            r"\bi am\b", r"\bi'm\b", r"\bmy\b", r"\bi have\b", r"\bi live\b",
            r"\bi work\b", r"\bi study\b", r"\bi've been\b", r"\bi prefer\b",
            r"\bi like\b", r"\bi built\b", r"\bi am building\b", r"\bfor my project\b",
            r"\bi want to\b", r"\bi enjoy\b", r"\bi use\b", r"\bi need\b",
            r"\bmy name\b", r"\bmy goal\b", r"\bmy age\b",
        ]
        return any(re.search(pat, lower) for pat in first_person_markers)

    def _best_category(self, text: str) -> str:
        """Pick the most specific category based on keyword match count."""
        lower = text.lower()
        scores: Dict[str, int] = {}

        for cat, keywords in CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[cat] = score

        if not scores:
            return "general"

        # Prefer more specific categories over 'personal' when tied
        priority_order = [
            "college", "career", "exams", "projects",
            "technical_knowledge", "family", "preferences", "personal", "general"
        ]
        best = max(scores, key=lambda c: (scores[c], -priority_order.index(c) if c in priority_order else -99))
        return best

    def _generate_title(self, text: str, max_words: int = 8) -> str:
        """Generate a descriptive title from the memory content."""
        # Strip leading instruction words
        clean = re.sub(r"^(remember|note|store|save|keep in mind)[:\s]+", "", text.strip(), flags=re.IGNORECASE)
        words = clean.split()
        title = " ".join(words[:max_words])
        if len(words) > max_words:
            title += "..."
        return title or "User Note"

    def _extract_tags(self, text: str, category: str) -> List[str]:
        """Extract meaningful tags from text."""
        tags = [category]
        lower = text.lower()

        # Technical tags
        tech_terms = ["python", "javascript", "sql", "machine learning", "deep learning",
                      "nlp", "ai", "neural network", "api", "backend", "frontend",
                      "linux", "git", "docker", "llm", "transformer"]
        for term in tech_terms:
            if term in lower:
                tags.append(term)

        # Domain tags
        if any(w in lower for w in ["nit", "iit", "university", "college"]):
            tags.append("education")
        if any(w in lower for w in ["goal", "aspire", "want to become"]):
            tags.append("goal")

        return list(dict.fromkeys(tags))  # deduplicate preserving order

    def heuristic_classify(self, user_text: str, assistant_text: str = "") -> Optional[Dict[str, Any]]:
        """
        Rule-based classifier to extract memory without calling external APIs.
        Guarantees fast, privacy-preserving, zero-cost operation.
        """
        clean_user = user_text.strip()

        # 1. Check if user made a self-disclosure or explicit store instruction
        is_explicit_note = bool(re.search(r"^(remember|note|store|save|keep in mind)\b", clean_user.lower()))
        has_disclosure = self.is_first_person_disclosure(clean_user)

        if not has_disclosure and not is_explicit_note:
            return None

        # 2. Don't store very short fragments (less than 4 words) — likely noise
        if len(clean_user.split()) < 4 and not is_explicit_note:
            return None

        # 3. Don't store messages that are purely greetings or filler
        filler_patterns = [r"^(hi|hello|hey|ok|okay|thanks|thank you|yes|no|sure|cool|great)\b"]
        if any(re.match(p, clean_user.lower()) for p in filler_patterns):
            return None

        # 4. Determine best category
        best_category = self._best_category(clean_user)

        # 5. Generate title and tags
        title = self._generate_title(clean_user)
        tags = self._extract_tags(clean_user, best_category)

        confidence = 0.85 if has_disclosure else 0.72
        sensitive = self.is_sensitive(clean_user)

        return {
            "should_store": True,
            "category": best_category,
            "title": title,
            "content": clean_user,
            "tags": tags,
            "confidence": confidence,
            "sensitive": sensitive
        }

    def classify_and_extract(
        self,
        user_text: str,
        assistant_text: str = "",
        source_conversation: str = "direct"
    ) -> Optional[MemoryItem]:
        """
        Analyze conversation turn and optionally create/update a long-term memory.
        Returns the MemoryItem if stored/updated, or None if not suitable.
        """
        extraction: Optional[Dict[str, Any]] = None

        # Use AI-based extraction if available (calls extract_memory on AI client)
        if self.ai_client and hasattr(self.ai_client, "extract_memory") and Config.is_api_configured():
            try:
                extraction = self.ai_client.extract_memory(user_text, assistant_text)
            except Exception:
                extraction = None

        # Always fall back to heuristic if AI extraction is unavailable or failed
        if not extraction:
            extraction = self.heuristic_classify(user_text, assistant_text)

        if not extraction or not extraction.get("should_store"):
            return None

        # Check confidence threshold
        confidence = float(extraction.get("confidence", 0.7))
        if confidence < Config.MEMORY_AUTO_STORE_CONFIDENCE:
            return None

        # Skip sensitive content — requires manual confirmation
        if extraction.get("sensitive", False):
            return None

        category = extraction.get("category", "general")
        title = extraction.get("title", "User Information")
        content = extraction.get("content", user_text).strip()
        tags = extraction.get("tags", [category])

        # Final guard: never store content that looks like a question with no personal info
        if content.endswith("?") and not self.is_first_person_disclosure(content):
            return None

        memory_item, status = self.memory_manager.add_or_update_memory(
            category=category,
            title=title,
            content=content,
            tags=tags,
            source_conversation=source_conversation,
            confidence=confidence,
            sensitive=False
        )
        return memory_item
