"""
Prompt Builder module.
Assembles compact, structured prompts incorporating:
- Retrieved local long-term memories (verified facts ONLY)
- Recent conversation turns
- Relevant past questions & answers
- Inferred intent and pronoun resolutions
- Strict instructions against hallucinating memories
"""

from typing import List, Dict, Any, Optional
from config import Config


SYSTEM_INSTRUCTION = """You are an intelligent, privacy-first personal AI assistant with access to the user's verified local memory storage.

STRICT RULES — you MUST follow these exactly:

1. ANSWER ONLY FROM VERIFIED SOURCES:
   - Facts about the user (name, age, college, goals, preferences, etc.) MUST come from the "CONFIRMED LOCAL MEMORIES" section below.
   - If that section is empty or does not contain the answer, say exactly: "I don't have that stored in my memory yet."
   - NEVER guess, infer, or invent personal details about the user.

2. MEMORY PRIORITY:
   - If recent conversation history contradicts an older memory, prefer the RECENT CONVERSATION.
   - If memory contradicts the current question's intent, clarify the contradiction politely.

3. GENERAL KNOWLEDGE vs PERSONAL FACTS:
   - General knowledge questions (history, science, coding, etc.) → answer freely from your training.
   - Personal/contextual questions (my name, my college, my project) → ONLY use confirmed local memory.

4. FOLLOW-UP RESOLUTION:
   - If the current question uses pronouns like 'it', 'that', 'the second option', use the "INTENT & RESOLUTION" section and the recent conversation history to resolve them accurately.

5. CONCISENESS:
   - Be direct. Don't repeat back the user's full memory verbatim unless asked.
   - Use natural language. Don't say "According to your local memory, you are..."
"""

NORMAL_SYSTEM_INSTRUCTION = """You are an intelligent, helpful personal AI assistant.
Answer the user's questions clearly, accurately, and naturally. Maintain context from earlier messages in this conversation.
Be concise, helpful, and direct."""


class PromptBuilder:
    """Builds structured contextual prompts for the AI model."""

    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens or Config.MAX_CONTEXT_TOKENS

    def build_prompt(
        self,
        query: str,
        recent_messages: List[Dict[str, Any]],
        retrieved_memories: List[Dict[str, Any]],
        similar_past_questions: List[Dict[str, Any]],
        intent_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Construct system instruction and user prompt payload.
        Returns {'system_instruction': str, 'prompt': str}
        """
        sections: List[str] = []

        # 1. Retrieved Long-Term Memories — clearly labelled as VERIFIED FACTS
        if retrieved_memories:
            mem_lines = []
            for m in retrieved_memories:
                cat = m.get("category", "general")
                title = m.get("title", "")
                content = m.get("content", "").strip()
                conf = m.get("confidence", 0.9)
                # Show confidence visually only when lower than high-confidence threshold
                conf_note = f" [confidence: {conf:.0%}]" if conf < 0.80 else ""
                mem_lines.append(f"• [{cat.upper()}] {title}: {content}{conf_note}")
            section_text = "\n".join(mem_lines)
            sections.append(
                f"### CONFIRMED LOCAL MEMORIES (Use ONLY these verified facts about the user):\n{section_text}"
            )
        else:
            # Explicitly tell the model: NO personal facts available
            sections.append(
                "### CONFIRMED LOCAL MEMORIES:\n"
                "[No relevant personal memories found for this query. "
                "Do NOT invent or assume any personal details about the user.]"
            )

        # 2. Inferred Intent & Coreference Resolution
        intent_type = intent_data.get("intent_type", "new_question")
        resolved_q = intent_data.get("resolved_query", query)
        reason = intent_data.get("reason", "")
        intent_text = f"• Query Type: {intent_type}\n• Resolved Focus: {resolved_q}"
        if reason:
            intent_text += f"\n• Context Note: {reason}"
        sections.append(f"### INTENT & RESOLUTION:\n{intent_text}")

        # 3. Recent Conversation History (most recent N turns)
        if recent_messages:
            conv_lines = []
            for msg in recent_messages[-Config.MAX_RECENT_MESSAGES:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "").strip()
                # Truncate very long messages to avoid bloating the prompt
                if len(content) > 400:
                    content = content[:400] + "..."
                conv_lines.append(f"{role}: {content}")
            sections.append("### RECENT CONVERSATION HISTORY:\n" + "\n".join(conv_lines))

        # 4. Relevant Similar Past Q&A (from other sessions)
        if similar_past_questions:
            past_lines = []
            for item in similar_past_questions[:2]:
                q = item.get("question", "").strip()
                a = item.get("answer", "").strip()
                if a and q:
                    # Truncate long past answers
                    if len(a) > 300:
                        a = a[:300] + "..."
                    past_lines.append(f"Past Q: {q}\nPast A: {a}")
            if past_lines:
                sections.append("### RELEVANT PAST DISCUSSIONS (for context only):\n" + "\n\n".join(past_lines))

        # 5. Current User Question — always last
        sections.append(f"### CURRENT USER QUESTION:\n{query}")

        full_prompt = "\n\n".join(sections)

        return {
            "system_instruction": SYSTEM_INSTRUCTION,
            "prompt": full_prompt
        }

    def build_direct_prompt(
        self,
        query: str,
        recent_messages: List[Dict[str, Any]],
        system_instruction: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build a lightweight, direct conversation prompt without any memory sections.
        Used for Normal and Memory modes where memory retrieval is disabled.
        """
        sections: List[str] = []
        if recent_messages:
            conv_lines = []
            for msg in recent_messages[-Config.MAX_RECENT_MESSAGES:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "").strip()
                if len(content) > 500:
                    content = content[:500] + "..."
                conv_lines.append(f"{role}: {content}")
            sections.append("### RECENT CONVERSATION HISTORY:\n" + "\n".join(conv_lines))

        sections.append(f"### CURRENT USER QUESTION:\n{query}")
        return {
            "system_instruction": system_instruction or NORMAL_SYSTEM_INSTRUCTION,
            "prompt": "\n\n".join(sections)
        }
