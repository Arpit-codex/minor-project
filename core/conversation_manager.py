"""
Conversation manager module.
Handles persistent conversation sessions, message history, topic/intent metadata,
and database lifecycle using SQLite.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import DB_PATH, Config


class ConversationManager:
    """Manages conversations and message history stored in SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high performance and integrity
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self):
        """Initialize database tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    mode TEXT NOT NULL DEFAULT 'both'
                );
            """)

            # Safe migration for mode column if upgrading an existing database
            cursor.execute("PRAGMA table_info(conversations);")
            columns = [row[1] for row in cursor.fetchall()]
            if "mode" not in columns:
                cursor.execute("ALTER TABLE conversations ADD COLUMN mode TEXT NOT NULL DEFAULT 'both';")

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    topic TEXT,
                    intent TEXT,
                    memory_references TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)

            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_role
                ON messages(role);
            """)
            conn.commit()

    def create_conversation(self, title: str = "New Conversation", mode: str = "both") -> str:
        """Create a new conversation session and return its ID."""
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        clean_mode = mode.lower().strip() if mode else "both"
        if clean_mode not in ("normal", "memory", "retrieval", "both"):
            clean_mode = "both"
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at, status, mode) VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, title, now, now, "active", clean_mode)
            )
            conn.commit()
        return conv_id

    def get_or_create_active_conversation(self, default_mode: str = "both") -> str:
        """Return the most recent active conversation ID or create a new one."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE status = 'active' ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return row["id"]
        return self.create_conversation(mode=default_mode)

    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, conversation_id)
            )
            conn.commit()

    def update_conversation_mode(self, conversation_id: str, mode: str) -> bool:
        """Update conversation mode ('normal', 'memory', 'retrieval', 'both')."""
        valid_modes = {"normal", "memory", "retrieval", "both"}
        clean_mode = mode.lower().strip() if mode else "both"
        if clean_mode not in valid_modes:
            clean_mode = "both"
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET mode = ?, updated_at = ? WHERE id = ?",
                (clean_mode, now, conversation_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        topic: Optional[str] = None,
        intent: Optional[str] = None,
        memory_references: Optional[List[str]] = None,
    ) -> int:
        """Add a message to a conversation."""
        now = datetime.now(timezone.utc).isoformat()
        refs_json = json.dumps(memory_references or [])

        with self._get_connection() as conn:
            # Check conversation exists
            conv = conn.execute("SELECT id, title, mode FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not conv:
                # Create if it didn't exist
                title = content[:40] + ("..." if len(content) > 40 else "") if role == "user" else "Conversation"
                conn.execute(
                    "INSERT INTO conversations (id, title, created_at, updated_at, status, mode) VALUES (?, ?, ?, ?, ?, 'both')",
                    (conversation_id, title, now, now, "active")
                )
            else:
                # Update title on first user message if default
                if role == "user" and conv["title"] in ("New Conversation", ""):
                    new_title = content[:45] + ("..." if len(content) > 45 else "")
                    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))

            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, timestamp, topic, intent, memory_references)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, now, topic, intent, refs_json)
            )
            msg_id = cursor.lastrowid
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id)
            )
            conn.commit()
            return msg_id

    def get_recent_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent messages for a conversation in chronological order."""
        max_limit = limit or Config.MAX_RECENT_MESSAGES
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, role, content, timestamp, topic, intent, memory_references
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, max_limit)
            ).fetchall()

            # Reverse to return chronological order
            messages = [dict(row) for row in reversed(rows)]
            for msg in messages:
                if msg.get("memory_references"):
                    try:
                        msg["memory_references"] = json.loads(msg["memory_references"])
                    except (json.JSONDecodeError, TypeError):
                        msg["memory_references"] = []
                else:
                    msg["memory_references"] = []
            return messages

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation metadata."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at, status, mode FROM conversations WHERE id = ?",
                (conversation_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List past conversations ordered by most recently updated."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, c.status, c.mode,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_user_questions(self, exclude_conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve all past user questions with their corresponding assistant answers.
        Useful for historical question similarity search.
        """
        with self._get_connection() as conn:
            query = """
                SELECT u.id as user_msg_id,
                       u.conversation_id,
                       u.content as question,
                       u.timestamp,
                       u.topic,
                       u.intent,
                       (
                           SELECT a.content
                           FROM messages a
                           WHERE a.conversation_id = u.conversation_id
                             AND a.id > u.id
                             AND a.role = 'assistant'
                           ORDER BY a.id ASC
                           LIMIT 1
                       ) as answer
                FROM messages u
                WHERE u.role = 'user'
            """
            params = []
            if exclude_conversation_id:
                query += " AND u.conversation_id != ?"
                params.append(exclude_conversation_id)

            query += " ORDER BY u.id DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def search_messages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search message history by keyword."""
        with self._get_connection() as conn:
            like_query = f"%{query}%"
            rows = conn.execute(
                """
                SELECT m.id, m.conversation_id, m.role, m.content, m.timestamp, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.content LIKE ?
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (like_query, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its associated messages."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all_history(self) -> bool:
        """Clear all conversation history."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM conversations")
            conn.commit()
            return True
