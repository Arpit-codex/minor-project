"""
Conversation Management & Chat Isolation Engine.
Guarantees strict isolation across conversations with unique UUIDs,
independent message histories, token optimization metadata, and local persistence.
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"


@dataclass
class ChatMessage:
    id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        return cls(
            id=data["id"],
            title=data.get("title", "New Conversation"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            messages=[ChatMessage.from_dict(m) for m in data.get("messages", [])],
        )


class ChatManager:
    """
    Manages isolated multi-chat conversations with persistence.
    """

    def __init__(self, storage_file: Union[str, Path] = CONVERSATIONS_FILE):
        self.storage_file = Path(storage_file)
        self._conversations: Dict[str, Conversation] = {}
        self._load()

    def _load(self) -> None:
        """Loads all conversations from local JSON file."""
        self._conversations = {}
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        conv = Conversation.from_dict(item)
                        self._conversations[conv.id] = conv
            except Exception:
                self._conversations = {}

    def _save(self) -> None:
        """Saves all conversations to local JSON file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = [conv.to_dict() for conv in self._conversations.values()]
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversations: {e}")

    def create_conversation(self, title: Optional[str] = None) -> Conversation:
        """Creates and stores a new isolated conversation with a unique ID."""
        conv_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        conv = Conversation(
            id=conv_id,
            title=title or "New Chat",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        self._conversations[conv_id] = conv
        self._save()
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Retrieves a conversation by its unique ID."""
        return self._conversations.get(conv_id)

    def list_conversations(self) -> List[Conversation]:
        """Returns all conversations sorted by latest update first."""
        convs = list(self._conversations.values())
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs

    def delete_conversation(self, conv_id: str) -> bool:
        """Deletes a conversation by its unique ID."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
            self._save()
            return True
        return False

    def clear_conversation(self, conv_id: str) -> bool:
        """Clears all messages inside a specific conversation without deleting the chat."""
        conv = self._conversations.get(conv_id)
        if conv:
            conv.messages = []
            conv.updated_at = datetime.now().isoformat()
            self._save()
            return True
        return False

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        Adds a message to a specific conversation, maintaining complete chat isolation.
        Auto-generates title from first user message if title is still 'New Chat'.
        """
        conv = self._conversations.get(conv_id)
        if not conv:
            conv = self.create_conversation()
            conv_id = conv.id

        now = datetime.now().isoformat()
        msg_id = str(uuid.uuid4())
        msg = ChatMessage(
            id=msg_id,
            role=role,
            content=content,
            timestamp=now,
            metadata=metadata or {},
        )

        conv.messages.append(msg)
        conv.updated_at = now

        # Auto-update title if it's the first user message
        if role == "user" and conv.title in ["New Chat", "New Conversation"]:
            clean_title = content.strip().split("\n")[0][:36]
            if len(content.strip().split("\n")[0]) > 36:
                clean_title += "..."
            conv.title = clean_title if clean_title else "Chat"

        self._save()
        return msg

    def get_messages_for_llm(
        self,
        conv_id: str,
        system_prompt: Optional[str] = None,
        max_messages: int = 20,
    ) -> List[Dict[str, str]]:
        """
        Retrieves ONLY the conversation history belonging to this specific chat ID,
        formatted for OpenAI-compatible chat completions APIs.
        """
        conv = self._conversations.get(conv_id)
        if not conv:
            return []

        formatted: List[Dict[str, str]] = []

        if system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        # Take the most recent messages up to max_messages
        recent = conv.messages[-max_messages:] if len(conv.messages) > max_messages else conv.messages

        for m in recent:
            formatted.append({
                "role": m.role,
                "content": m.content,
            })

        return formatted
