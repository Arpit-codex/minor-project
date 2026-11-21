"""
Assistant Orchestrator module.
Coordinates the end-to-end processing pipeline:
1. Normalization & intent detection
2. Context retrieval (recent turns + past similar questions)
3. Categorized memory retrieval & ranking
4. Ambiguity checking
5. Prompt construction
6. AI API generation
7. Local conversation persistence
8. Long-term memory classification & updating
"""

import time
import threading
from typing import Dict, Any, Optional, List
from config import Config
from core.conversation_manager import ConversationManager
from core.similarity_engine import SimilarityEngine
from core.memory_manager import MemoryManager
from core.memory_classifier import MemoryClassifier
from core.memory_retriever import MemoryRetriever
from core.intent_detector import IntentDetector
from core.ai_client import AIClient, AIResponse
from core.prompt_builder import PromptBuilder


class ContextualAssistant:
    """End-to-end contextual AI assistant engine."""

    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        similarity_engine: Optional[SimilarityEngine] = None,
        memory_manager: Optional[MemoryManager] = None,
        ai_client: Optional[AIClient] = None
    ):
        self.conversation_manager = conversation_manager or ConversationManager()
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.memory_manager = memory_manager or MemoryManager(similarity_engine=self.similarity_engine)
        self.ai_client = ai_client or AIClient()
        self.memory_classifier = MemoryClassifier(self.memory_manager, ai_client=self.ai_client)
        self.retriever = MemoryRetriever(self.memory_manager, self.similarity_engine, self.conversation_manager)
        self.intent_detector = IntentDetector(ai_client=self.ai_client)
        self.prompt_builder = PromptBuilder()

    def _handle_memory_reference_query(self, query: str, conv_id: str) -> Dict[str, Any]:
        """
        Fast local-first handler for 'what do you know about me' type queries.
        Reads all stored memories, builds a compact summary, and sends ONLY that
        compact summary to the AI — avoiding the large-prompt timeout.
        """
        all_memories = self.memory_manager.list_memories()

        if not all_memories:
            answer = (
                "I currently don't have any stored information about you. \n\n"
                "You can add personal context using:\n"
                "  • /add_memory — to manually save a memory\n"
                "  • Just chat with me — I will automatically learn useful context from our conversations"
            )
            self.conversation_manager.add_message(conv_id, "user", query, intent="memory_reference")
            self.conversation_manager.add_message(conv_id, "assistant", answer)
            return {
                "answer": answer,
                "is_error": False,
                "conversation_id": conv_id,
                "intent": {"intent_type": "memory_reference"},
                "retrieved_memories": [],
                "extracted_memory": None
            }

        # Group memories by category for a readable summary
        by_category: Dict[str, List] = {}
        for mem in all_memories:
            by_category.setdefault(mem.category, []).append(mem)

        summary_lines = [f"Here is everything I have stored about you ({len(all_memories)} item(s)):\n"]
        compact_for_ai = []
        for cat, items in sorted(by_category.items()):
            summary_lines.append(f"**{cat.upper()}** ({len(items)} item(s)):")
            compact_for_ai.append(f"[{cat.upper()}]")
            for item in items:
                summary_lines.append(f"  • {item.title}: {item.content[:120]}{'...' if len(item.content) > 120 else ''}")
                compact_for_ai.append(f"  {item.title}: {item.content[:200]}")

        local_summary = "\n".join(summary_lines)

        # Send a compact prompt to AI for natural language formatting
        compact_context = "\n".join(compact_for_ai)
        compact_prompt = (
            f"The user asked: '{query}'\n\n"
            f"Here is all stored personal context about this user:\n{compact_context}\n\n"
            f"Please present this information in a natural, friendly way. "
            f"Don't invent any additional facts. If a category is empty, skip it."
        )

        ai_resp: AIResponse = self.ai_client.generate(
            prompt=compact_prompt,
            system_instruction="You are a helpful personal AI assistant. Present the user's stored memory data clearly and naturally. Do not hallucinate any facts not listed."
        )

        # If AI fails (timeout/error), fall back to the locally built summary
        if ai_resp.is_error:
            answer = local_summary
        else:
            answer = ai_resp.content

        self.conversation_manager.add_message(conv_id, "user", query, intent="memory_reference")
        self.conversation_manager.add_message(conv_id, "assistant", answer)

        return {
            "answer": answer,
            "is_error": False,
            "conversation_id": conv_id,
            "intent": {"intent_type": "memory_reference"},
            "retrieved_memories": [{"item": m, "category": m.category, "title": m.title, "content": m.content} for m in all_memories],
            "extracted_memory": None
        }

    def ask(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        memory_write: Optional[bool] = None,
        memory_retrieval: Optional[bool] = None,
        mode: Optional[str] = None,
        ai_provider: Optional[str] = None,
        model: Optional[str] = None,
        sync_memory_write: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user question through the contextual pipeline configured by two
        independent capabilities: memory_write and memory_retrieval.

        Supported Mode Presets:
        - 'normal':    memory_write=False, memory_retrieval=False
        - 'memory':    memory_write=True,  memory_retrieval=False
        - 'retrieval': memory_write=False, memory_retrieval=True
        - 'both':      memory_write=True,  memory_retrieval=True
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "answer": "Please enter a valid question.",
                "is_error": True,
                "error_type": "empty_input",
                "conversation_id": conversation_id,
                "retrieved_memories": [],
                "intent": {}
            }

        # 1. Resolve active conversation ID and stored mode
        conv_id = conversation_id or self.conversation_manager.get_or_create_active_conversation()
        conv_record = self.conversation_manager.get_conversation(conv_id)
        saved_mode = conv_record.get("mode", "both") if conv_record else "both"

        # 2. Resolve independent capabilities: memory_write and memory_retrieval
        # Determine base mode
        active_mode = (mode or saved_mode or "both").lower().strip()
        if active_mode not in ("normal", "memory", "retrieval", "both"):
            active_mode = "both"

        # Map base mode to boolean flags
        mode_write_map = {
            "normal": False,
            "memory": True,
            "retrieval": False,
            "both": True,
        }
        mode_retrieval_map = {
            "normal": False,
            "memory": False,
            "retrieval": True,
            "both": True,
        }

        # Allow explicit boolean capability overrides if provided
        effective_write = mode_write_map[active_mode] if memory_write is None else bool(memory_write)
        effective_retrieval = mode_retrieval_map[active_mode] if memory_retrieval is None else bool(memory_retrieval)

        # Compute canonical mode tag
        if effective_write and effective_retrieval:
            canonical_mode = "both"
        elif effective_write and not effective_retrieval:
            canonical_mode = "memory"
        elif not effective_write and effective_retrieval:
            canonical_mode = "retrieval"
        else:
            canonical_mode = "normal"

        # Update saved mode if it changed
        if conv_record and conv_record.get("mode") != canonical_mode:
            self.conversation_manager.update_conversation_mode(conv_id, canonical_mode)

        # 3. Retrieve recent conversation history (always maintained for current conversation)
        recent_messages = self.conversation_manager.get_recent_messages(conv_id)

        # 4. Memory Retrieval Pipeline (ONLY executed when memory_retrieval is True)
        retrieved_memories = []
        similar_past = []
        intent_info = {"intent_type": "standard_chat", "resolved_query": clean_q}

        if effective_retrieval:
            # Cross-session past questions search
            past_qa = self.conversation_manager.get_all_user_questions(exclude_conversation_id=conv_id)
            if past_qa:
                similar_past = self.similarity_engine.find_similar_questions(
                    clean_q, past_qa, threshold=Config.SIMILARITY_THRESHOLD, top_k=3
                )

            # Intent detection & pronoun resolution
            intent_info = self.intent_detector.detect_intent(clean_q, recent_messages, similar_past)

            # Check for fast local-first memory reference queries
            if intent_info.get("intent_type") == "memory_reference":
                return self._handle_memory_reference_query(clean_q, conv_id)

            # Hybrid memory retrieval
            retrieval_data = self.retriever.retrieve_full_context(clean_q, conv_id)
            retrieved_memories = retrieval_data.get("retrieved_memories", [])

            # Check ambiguity clarification
            if intent_info.get("is_ambiguous") and intent_info.get("clarifying_question"):
                clarifying_answer = intent_info["clarifying_question"]
                self.conversation_manager.add_message(
                    conv_id, "user", clean_q, intent=intent_info.get("intent_type")
                )
                self.conversation_manager.add_message(
                    conv_id, "assistant", clarifying_answer, topic="Clarification Needed"
                )
                return {
                    "answer": clarifying_answer,
                    "is_clarification": True,
                    "is_error": False,
                    "conversation_id": conv_id,
                    "mode": canonical_mode,
                    "memory_write": effective_write,
                    "memory_retrieval": effective_retrieval,
                    "intent": intent_info,
                    "retrieved_memories": retrieved_memories,
                    "provider": "assistant_clarifier"
                }

            # Build contextual prompt with memory sections
            prompt_payload = self.prompt_builder.build_prompt(
                query=clean_q,
                recent_messages=recent_messages,
                retrieved_memories=retrieved_memories,
                similar_past_questions=similar_past,
                intent_data=intent_info
            )
        else:
            # Fast, lightweight path: direct conversation prompt with current session context only
            prompt_payload = self.prompt_builder.build_direct_prompt(
                query=clean_q,
                recent_messages=recent_messages
            )

        # 5. Route request to configured AI API
        target_client = self.ai_client
        if ai_provider:
            target_client = AIClient(provider_type=ai_provider)

        ai_resp: AIResponse = target_client.generate(
            prompt=prompt_payload["prompt"],
            system_instruction=prompt_payload["system_instruction"],
            model=model
        )

        answer_text = ai_resp.content

        # 6. Store conversation interaction locally in SQLite
        memory_refs = [m.get("item").id for m in retrieved_memories if m.get("item")] if effective_retrieval else []
        self.conversation_manager.add_message(
            conversation_id=conv_id,
            role="user",
            content=clean_q,
            intent=intent_info.get("intent_type"),
            memory_references=memory_refs
        )
        self.conversation_manager.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=answer_text,
            memory_references=memory_refs
        )

        # 7. Memory Writing Pipeline (ONLY executed when memory_write is True)
        extracted_memory = None
        if effective_write and not ai_resp.is_error:
            def _extract_worker():
                try:
                    self.memory_classifier.classify_and_extract(
                        user_text=clean_q,
                        assistant_text=answer_text,
                        source_conversation=conv_id
                    )
                except Exception:
                    pass

            if sync_memory_write:
                # Synchronous execution (useful for deterministic unit tests)
                extracted_memory = self.memory_classifier.classify_and_extract(
                    user_text=clean_q,
                    assistant_text=answer_text,
                    source_conversation=conv_id
                )
            else:
                # Non-blocking background execution so user response is immediate
                thread = threading.Thread(target=_extract_worker, daemon=True)
                thread.start()

        return {
            "answer": answer_text,
            "is_error": ai_resp.is_error,
            "error_type": ai_resp.error_type,
            "provider": ai_resp.provider,
            "model": ai_resp.model,
            "conversation_id": conv_id,
            "mode": canonical_mode,
            "memory_write": effective_write,
            "memory_retrieval": effective_retrieval,
            "intent": intent_info,
            "retrieved_memories": retrieved_memories,
            "similar_past_questions": similar_past,
            "extracted_memory": extracted_memory
        }
