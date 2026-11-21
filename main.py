"""
Main Command-Line Interface for the Local Memory-Based Contextual AI Assistant.
Provides an interactive conversational REPL with rich commands to view history,
manage categorized memories, switch sessions, and inspect system status.
"""

import sys
import os
import shutil
from typing import Optional

from config import Config, DB_PATH, MEMORY_DIR
from core.assistant import ContextualAssistant
from core.memory_manager import MemoryItem


BANNER = r"""
========================================================================
     LOCAL MEMORY-BASED CONTEXTUAL AI ASSISTANT
     Privacy-First * Modular Architecture * Categorized Memory
========================================================================
"""


class AssistantCLI:
    """Command-line interface handler."""

    def __init__(self):
        self.assistant = ContextualAssistant()
        self.active_conv_id = self.assistant.conversation_manager.get_or_create_active_conversation()

    def print_status(self):
        """Print current system status banner."""
        active_provider = self.assistant.ai_client.get_active_provider().provider_name
        model_name = self.assistant.ai_client.get_active_provider().model_name
        mem_count = len(self.assistant.memory_manager.list_memories())
        convs = self.assistant.conversation_manager.list_conversations(limit=100)

        print(BANNER)
        print(f"  [Status] AI Provider: {active_provider.upper()} ({model_name})")
        print(f"  [Status] Active Conversation ID: {self.active_conv_id[:8]}...")
        print(f"  [Status] Total Stored Memories: {mem_count} items")
        print(f"  [Status] Total Conversation Sessions: {len(convs)}")
        print("  Type your question or '/help' for command list. Type '/exit' to quit.\n" + "-" * 72)

    def print_help(self):
        """Display help menu."""
        help_text = """
Available Commands:
  /help                  - Show this help menu
  /new [title]           - Start a new conversation session
  /switch <id>           - Switch to a previous conversation by ID
  /history [limit]       - View list of past conversations
  /view [id]             - View message transcript of current or specified conversation
  /search <term>         - Search conversation messages by keyword
  /memories [category]   - List long-term memories (optional: filter by category)
  /view_memory <id>      - View detailed memory content
  /add_memory            - Interactively add a new memory
  /edit_memory <id>      - Edit an existing memory's content
  /delete_memory <id>    - Delete a memory by ID
  /categories            - List all memory categories
  /add_category <name>   - Add a new memory category
  /provider <name>       - Switch provider ('freellmapi', 'gemini', or 'mock')
  /stats                 - Show system metrics and storage breakdown
  /clear                 - Clear terminal screen
  /exit or /quit         - Exit the assistant
"""
        print(help_text)

    def handle_command(self, cmd_line: str) -> bool:
        """Process slash commands. Returns False if assistant should exit."""
        parts = cmd_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit", "/q"):
            print("\nGoodbye! Your local conversation history and memories are saved safely.")
            return False

        elif cmd == "/help":
            self.print_help()

        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            self.print_status()

        elif cmd == "/new":
            title = arg or "New Conversation"
            self.active_conv_id = self.assistant.conversation_manager.create_conversation(title=title)
            print(f"\n[+] Started new conversation session (ID: {self.active_conv_id[:8]}).")

        elif cmd == "/history":
            limit = int(arg) if arg.isdigit() else 10
            convs = self.assistant.conversation_manager.list_conversations(limit=limit)
            if not convs:
                print("\n[!] No past conversations found.")
            else:
                print(f"\n--- Past Conversations (Showing up to {limit}) ---")
                for c in convs:
                    current_marker = " * " if c["id"] == self.active_conv_id else "   "
                    print(f"{current_marker}[{c['id'][:8]}] {c['title']} ({c.get('message_count', 0)} msgs, updated: {c['updated_at'][:19]})")
                print("Use '/switch <id>' to switch to an earlier conversation.\n")

        elif cmd == "/switch":
            if not arg:
                print("\n[!] Usage: /switch <conversation_id>")
                return True
            convs = self.assistant.conversation_manager.list_conversations(limit=100)
            target = next((c for c in convs if c["id"].startswith(arg)), None)
            if target:
                self.active_conv_id = target["id"]
                print(f"\n[+] Switched to conversation [{target['id'][:8]}] '{target['title']}'")
            else:
                print(f"\n[!] Conversation starting with '{arg}' not found.")

        elif cmd == "/view":
            target_id = arg if arg else self.active_conv_id
            msgs = self.assistant.conversation_manager.get_recent_messages(target_id, limit=50)
            if not msgs:
                print(f"\n[!] No messages in conversation [{target_id[:8]}].")
            else:
                print(f"\n--- Transcript for Conversation [{target_id[:8]}] ---")
                for m in msgs:
                    role_tag = "USER" if m["role"] == "user" else "ASSISTANT"
                    print(f"\n[{role_tag}] ({m['timestamp'][:19]}):")
                    print(f"  {m['content']}")
                print("\n" + "-" * 50)

        elif cmd == "/search":
            if not arg:
                print("\n[!] Usage: /search <keyword>")
                return True
            hits = self.assistant.conversation_manager.search_messages(arg, limit=10)
            if not hits:
                print(f"\n[!] No messages matched '{arg}'.")
            else:
                print(f"\n--- Search results for '{arg}' ({len(hits)} matches) ---")
                for h in hits:
                    print(f"[{h['conversation_id'][:8]}] {h['role'].upper()}: {h['content'][:100]}...")
                print()

        elif cmd == "/memories":
            category = arg if arg else None
            memories = self.assistant.memory_manager.list_memories(category=category)
            if not memories:
                filter_str = f" in category '{category}'" if category else ""
                print(f"\n[!] No memories found{filter_str}.")
            else:
                filter_str = f" in category '{category}'" if category else ""
                print(f"\n--- Stored Memories{filter_str} ({len(memories)} items) ---")
                for m in memories:
                    print(f"[{m.id}] [{m.category.upper()}] {m.title}: {m.content[:80]}... (Conf: {m.confidence:.2f})")
                print("Use '/view_memory <id>' to view complete details.\n")

        elif cmd == "/view_memory":
            if not arg:
                print("\n[!] Usage: /view_memory <memory_id>")
                return True
            item = self.assistant.memory_manager.get_memory(arg)
            if not item:
                print(f"\n[!] Memory '{arg}' not found.")
            else:
                print(f"\n--- Memory [{item.id}] ---")
                print(f"Category: {item.category}")
                print(f"Title: {item.title}")
                print(f"Tags: {', '.join(item.tags)}")
                print(f"Confidence: {item.confidence:.2f}")
                print(f"Created: {item.created_at}")
                print(f"Updated: {item.updated_at}")
                print(f"Content:\n{item.content}\n" + "-" * 50)

        elif cmd == "/add_memory":
            self._interactive_add_memory()

        elif cmd == "/edit_memory":
            if not arg:
                print("\n[!] Usage: /edit_memory <memory_id>")
                return True
            self._interactive_edit_memory(arg)

        elif cmd == "/delete_memory":
            if not arg:
                print("\n[!] Usage: /delete_memory <memory_id>")
                return True
            success = self.assistant.memory_manager.delete_memory(arg)
            if success:
                print(f"\n[+] Memory '{arg}' deleted successfully.")
            else:
                print(f"\n[!] Memory '{arg}' could not be deleted or was not found.")

        elif cmd == "/categories":
            print("\n--- Available Memory Categories ---")
            for cat in Config.CATEGORIES:
                cat_count = len(self.assistant.memory_manager.list_memories(category=cat))
                print(f"  - {cat} ({cat_count} items)")
            print()

        elif cmd == "/add_category":
            if not arg:
                print("\n[!] Usage: /add_category <category_name>")
                return True
            success = self.assistant.memory_manager.add_category(arg)
            if success:
                print(f"\n[+] Category '{arg}' added successfully.")
            else:
                print("\n[!] Invalid category name.")

        elif cmd == "/provider":
            if arg not in ("freellmapi", "gemini", "mock", "auto"):
                print("\n[!] Usage: /provider <freellmapi|gemini|mock|auto>")
                return True
            self.assistant.ai_client.provider_type = arg
            active = self.assistant.ai_client.get_active_provider()
            print(f"\n[+] Switched active AI provider to: {active.provider_name.upper()} ({active.model_name})")

        elif cmd == "/stats":
            self._print_stats()

        else:
            print(f"\n[!] Unknown command '{cmd}'. Type '/help' for available commands.")

        return True

    def _interactive_add_memory(self):
        """Prompt user interactively to create a memory."""
        print("\n--- Add New Memory ---")
        print("Categories: " + ", ".join(Config.CATEGORIES))
        category = input("Category [general]: ").strip().lower() or "general"
        if category not in Config.CATEGORIES:
            self.assistant.memory_manager.add_category(category)

        title = input("Title: ").strip() or "User Note"
        content = input("Content: ").strip()
        if not content:
            print("[!] Content cannot be empty. Cancelled.")
            return

        tags_raw = input("Tags (comma separated): ").strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        item, status = self.assistant.memory_manager.add_or_update_memory(
            category=category,
            title=title,
            content=content,
            tags=tags,
            source_conversation=self.active_conv_id,
            confidence=1.0
        )
        print(f"[+] Memory saved [{item.id}] in '{category}' (Status: {status}).")

    def _interactive_edit_memory(self, memory_id: str):
        """Prompt user interactively to edit a memory."""
        item = self.assistant.memory_manager.get_memory(memory_id)
        if not item:
            print(f"[!] Memory '{memory_id}' not found.")
            return

        print(f"\nEditing Memory [{item.id}] ({item.title}):")
        print(f"Current content:\n{item.content}")
        new_content = input("New content (Enter to keep current): ").strip()
        new_title = input(f"New title (Enter to keep '{item.title}'): ").strip()

        updated_content = new_content if new_content else item.content
        updated_title = new_title if new_title else item.title

        res = self.assistant.memory_manager.update_memory_content(item.id, updated_content, updated_title)
        if res:
            print(f"[+] Memory [{item.id}] updated successfully.")
        else:
            print(f"[!] Failed to update memory [{item.id}].")

    def _print_stats(self):
        """Print system metrics and storage breakdown."""
        convs = self.assistant.conversation_manager.list_conversations(limit=500)
        total_msgs = sum(c.get("message_count", 0) for c in convs)
        memories = self.assistant.memory_manager.list_memories()

        print("\n================ SYSTEM STATISTICS ================")
        print(f"Database Path: {DB_PATH}")
        print(f"Memory Directory: {MEMORY_DIR}")
        print(f"Total Conversations: {len(convs)}")
        print(f"Total Messages Logged: {total_msgs}")
        print(f"Total Stored Memories: {len(memories)}")
        print("\nMemories by Category:")
        for cat in Config.CATEGORIES:
            cat_mems = [m for m in memories if m.category == cat]
            print(f"  - {cat:20s}: {len(cat_mems)} items")
        print("===================================================\n")

    def run(self):
        """Start the interactive REPL."""
        self.print_status()

        while True:
            try:
                short_id = self.active_conv_id[:8]
                user_input = input(f"\n[{short_id}] You > ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process question through the assistant pipeline
                print("\n[Thinking & Retrieving context...]")
                result = self.assistant.ask(user_input, conversation_id=self.active_conv_id)

                # Show contextual hints
                intent_type = result.get("intent", {}).get("intent_type", "unknown")
                resolved = result.get("intent", {}).get("resolved_query", user_input)
                mems = result.get("retrieved_memories", [])

                if intent_type == "follow_up":
                    print(f"  [Intent: Follow-up | Focus: '{resolved}']")
                elif intent_type == "memory_reference":
                    print(f"  [Intent: Memory Inquiry]")

                if mems:
                    mem_titles = [f"{m.get('category')}:{m.get('title')}" for m in mems]
                    print(f"  [Retrieved Local Memory: {', '.join(mem_titles)}]")

                if result.get("extracted_memory"):
                    ex = result["extracted_memory"]
                    print(f"  [+] Automatically learned new context into '{ex.category}': {ex.title}")

                # Display answer
                print(f"\nAssistant:\n{result['answer']}\n")

            except (KeyboardInterrupt, EOFError):
                print("\n\nSession terminated. Goodbye!")
                break
            except Exception as e:
                print(f"\n[!] Unexpected error: {e}")


def main():
    """Application entry point."""
    cli = AssistantCLI()
    cli.run()


if __name__ == "__main__":
    main()
