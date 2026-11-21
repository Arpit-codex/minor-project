"""
Intent Detector module.
Analyzes user questions to determine intent:
- 'new_question': Independent, self-contained question
- 'follow_up': Follow-up referencing current conversation turn
- 'continuation_older': Continuation of an older discussion
- 'memory_reference': Explicit inquiry about past personal or stored information
- 'ambiguous': Genuine ambiguity requiring clarification

Resolves pronouns and references ('it', 'that', 'the second option', 'earlier').
"""

import re
from typing import Dict, Any, List, Optional
from config import Config


PRONOUN_PATTERNS = [
    r"\b(it|this|that|these|those|they|them|he|she)\b",
    r"\b(the second|the first|the last|the other|the former|the latter)\b",
    r"\b(what about|how about|explain that|elaborate on that|tell me more about that)\b",
    r"\b(as mentioned|earlier|previously|before|as i said)\b"
]

MEMORY_QUERY_PATTERNS = [
    r"\bwhat did i (tell|say|mention)\b",
    r"\bdo you remember\b",
    r"\bwhat is my\b",
    r"\bwhere do i\b",
    r"\bwho is my\b",
    r"\bwhat were my\b",
    r"\brecall (what|my)\b",
    r"\bwhat do you know about me\b",
    r"\bwhat have you (learned|stored|saved) about me\b",
    r"\btell me (what you know|about me|my info|my details)\b",
    r"\bwhat (information|details|data) do you have (on|about) me\b",
    r"\bmy (profile|information|details|data|preferences|settings)\b",
    r"\bwhat do you (know|have|remember) about me\b",
    r"\bdo you know (anything|something) about me\b",
]


class IntentDetector:
    """Analyzes conversation context to infer query intent and resolve references."""

    def __init__(self, ai_client=None):
        self.ai_client = ai_client

    def has_anaphoric_reference(self, text: str) -> bool:
        """Check if query contains pronouns or referential phrases pointing to prior context."""
        lower = text.lower()
        return any(re.search(pat, lower) for pat in PRONOUN_PATTERNS)

    def is_memory_query(self, text: str) -> bool:
        """Check if user is explicitly querying stored memory."""
        lower = text.lower()
        return any(re.search(pat, lower) for pat in MEMORY_QUERY_PATTERNS)

    def resolve_coreference(self, query: str, recent_messages: List[Dict[str, Any]]) -> str:
        """
        Attempt to resolve referring pronouns ('it', 'that', 'the second option')
        using the subject or entities from the most recent assistant/user messages.
        """
        if not recent_messages or not self.has_anaphoric_reference(query):
            return query

        last_assistant_msg = next((m["content"] for m in reversed(recent_messages) if m.get("role") == "assistant"), "")
        last_user_msg = next((m["content"] for m in reversed(recent_messages) if m.get("role") == "user"), "")

        # Look for explicit topics or noun phrases in recent messages
        context_source = last_assistant_msg or last_user_msg
        if not context_source:
            return query

        # Example: if query is "Can I make it run with less RAM?" and context talked about "language model"
        resolved = query

        # Common pronoun replacements
        lower_q = query.lower()
        if re.search(r"\b(it|that)\b", lower_q):
            # Extract main candidate subject from last user question or assistant heading
            subject_match = re.search(r"(?:about|optimize|run|install|using|create|build)\s+([a-zA-Z0-9_\-\s]{3,30})", last_user_msg, re.IGNORECASE)
            if subject_match:
                candidate = subject_match.group(1).strip()
                # Do not replace if candidate is too generic
                if candidate and len(candidate.split()) <= 5:
                    resolved = re.sub(r"\b(it|that)\b", f"it [{candidate}]", query, count=1, flags=re.IGNORECASE)

        return resolved

    def detect_intent(
        self,
        query: str,
        recent_messages: List[Dict[str, Any]],
        similar_past_questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Classify intent into:
        - intent_type: 'new_question' | 'follow_up' | 'continuation_older' | 'memory_reference'
        - is_ambiguous: bool
        - clarifying_question: Optional[str]
        - resolved_query: str
        - reason: str
        """
        clean_query = query.strip()
        lower_query = clean_query.lower()

        # 1. Check for explicit memory inquiries
        if self.is_memory_query(clean_query):
            return {
                "intent_type": "memory_reference",
                "is_ambiguous": False,
                "clarifying_question": None,
                "resolved_query": clean_query,
                "reason": "User is explicitly asking about stored preferences or personal context."
            }

        # 2. Check for follow-up to the current ongoing conversation
        has_refs = self.has_anaphoric_reference(clean_query)
        is_short_followup = len(clean_query.split()) <= 6 and has_refs

        if recent_messages:
            last_msg = recent_messages[-1]
            last_content = last_msg.get("content", "")

            # Check if reference is ambiguous (e.g. "What about the second one?" but no list was provided)
            if re.search(r"\b(the second|second option|the third|option 2)\b", lower_query):
                # Check if numbers or bullet options existed in last message
                has_options = bool(re.search(r"(1\.|2\.|- Option|Option 1|Option 2|\bfirst\b.*\bsecond\b)", last_content, re.IGNORECASE))
                if not has_options:
                    return {
                        "intent_type": "follow_up",
                        "is_ambiguous": True,
                        "clarifying_question": "You mentioned the second option, but I couldn't find a numbered list in our recent discussion. Which specific option or topic would you like to explore?",
                        "resolved_query": clean_query,
                        "reason": "Numbered option referenced without clear candidate in previous turn."
                    }

            if has_refs or is_short_followup:
                resolved = self.resolve_coreference(clean_query, recent_messages)
                return {
                    "intent_type": "follow_up",
                    "is_ambiguous": False,
                    "clarifying_question": None,
                    "resolved_query": resolved,
                    "reason": "Follow-up question directly building on immediate conversation turn."
                }

        # 3. If no recent messages but contains anaphoric pronouns and similar past questions exist
        if not recent_messages and has_refs and similar_past_questions:
            best_past = similar_past_questions[0]
            if best_past.get("similarity_score", 0.0) >= Config.SIMILARITY_THRESHOLD:
                return {
                    "intent_type": "continuation_older",
                    "is_ambiguous": False,
                    "clarifying_question": None,
                    "resolved_query": clean_query,
                    "reason": f"Continuation of older discussion: '{best_past.get('question')}'"
                }

        # 4. If query has naked pronouns like 'Can it run faster?' with zero recent messages and zero past matches
        if not recent_messages and has_refs and not similar_past_questions:
            # Genuinely ambiguous without prior context
            pronoun_found = re.search(r"\b(it|that|this|they|them)\b", lower_query)
            pronoun_word = pronoun_found.group(1) if pronoun_found else "that"
            return {
                "intent_type": "new_question",
                "is_ambiguous": True,
                "clarifying_question": f"Could you clarify what '{pronoun_word}' refers to so I can give you an accurate answer?",
                "resolved_query": clean_query,
                "reason": f"Anaphoric pronoun '{pronoun_word}' used with no prior conversation context."
            }

        # 5. Check if query is semantically similar to an older question from past sessions
        if similar_past_questions:
            top_match = similar_past_questions[0]
            if top_match.get("similarity_score", 0.0) >= 0.70:
                return {
                    "intent_type": "continuation_older",
                    "is_ambiguous": False,
                    "clarifying_question": None,
                    "resolved_query": clean_query,
                    "reason": f"Strong semantic similarity to earlier question: '{top_match.get('question')}'"
                }

        # Default: New independent question
        return {
            "intent_type": "new_question",
            "is_ambiguous": False,
            "clarifying_question": None,
            "resolved_query": clean_query,
            "reason": "Independent query with self-contained context."
        }
