"""
Memory Manager module.
Handles categorized local storage in Markdown files with YAML-style metadata headers.
Supports listing, creating, reading, updating, deleting, and index rebuilding.
Detects duplicate memories and potential contradictions.
"""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from config import Config, MEMORY_DIR, MEMORY_INDEX_DIR
from core.similarity_engine import SimilarityEngine


@dataclass
class MemoryItem:
    """Represents a categorized memory item."""
    id: str
    category: str
    title: str
    content: str
    tags: List[str]
    created_at: str
    updated_at: str
    source_conversation: str = "direct"
    confidence: float = 0.9
    sensitive: bool = False
    status: str = "active"

    def to_markdown(self) -> str:
        """Serialize memory to Markdown with YAML-style metadata header."""
        tags_str = ", ".join(self.tags)
        return (
            f"---\n"
            f"id: {self.id}\n"
            f"category: {self.category}\n"
            f"title: {self.title}\n"
            f"tags: [{tags_str}]\n"
            f"created_at: {self.created_at}\n"
            f"updated_at: {self.updated_at}\n"
            f"source_conversation: {self.source_conversation}\n"
            f"confidence: {self.confidence:.2f}\n"
            f"sensitive: {str(self.sensitive).lower()}\n"
            f"status: {self.status}\n"
            f"---\n\n"
            f"{self.content.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, md_text: str, default_category: str = "general") -> Optional["MemoryItem"]:
        """Parse a MemoryItem from Markdown content."""
        pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.search(pattern, md_text, re.DOTALL)
        if not match:
            # Fallback if no frontmatter
            now = datetime.now(timezone.utc).isoformat()
            return cls(
                id=str(uuid.uuid4())[:8],
                category=default_category,
                title="Untitled Note",
                content=md_text.strip(),
                tags=[],
                created_at=now,
                updated_at=now
            )

        header_lines = match.group(1).splitlines()
        content = match.group(2).strip()

        meta: Dict[str, Any] = {}
        for line in header_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k == "tags":
                    # Parse [tag1, tag2]
                    v_clean = v.strip("[]")
                    meta[k] = [t.strip() for t in v_clean.split(",") if t.strip()]
                elif k in ("confidence",):
                    try:
                        meta[k] = float(v)
                    except ValueError:
                        meta[k] = 0.8
                elif k == "sensitive":
                    meta[k] = v.lower() == "true"
                else:
                    meta[k] = v

        return cls(
            id=meta.get("id", str(uuid.uuid4())[:8]),
            category=meta.get("category", default_category),
            title=meta.get("title", "Untitled Note"),
            content=content,
            tags=meta.get("tags", []),
            created_at=meta.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=meta.get("updated_at", datetime.now(timezone.utc).isoformat()),
            source_conversation=meta.get("source_conversation", "direct"),
            confidence=meta.get("confidence", 0.9),
            sensitive=meta.get("sensitive", False),
            status=meta.get("status", "active")
        )


class MemoryManager:
    """Manages categorized local memory storage."""

    def __init__(self, memory_dir: Optional[Path] = None, similarity_engine: Optional[SimilarityEngine] = None):
        self.memory_dir = memory_dir or MEMORY_DIR
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.ensure_categories()

    def ensure_categories(self):
        """Ensure all category directories exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for cat in Config.CATEGORIES:
            (self.memory_dir / cat).mkdir(parents=True, exist_ok=True)

    def add_category(self, category_name: str) -> bool:
        """Dynamically add a new memory category."""
        clean_name = re.sub(r"[^\w\-]", "_", category_name.strip().lower())
        if not clean_name:
            return False
        cat_path = self.memory_dir / clean_name
        cat_path.mkdir(parents=True, exist_ok=True)
        if clean_name not in Config.CATEGORIES:
            Config.CATEGORIES.append(clean_name)
        return True

    def _sanitize_filename(self, text: str) -> str:
        """Create a safe filename from title or text."""
        clean = re.sub(r"[^\w\s-]", "", text.lower()).strip()
        clean = re.sub(r"[-\s]+", "_", clean)
        return clean[:40] or "memory"

    def _get_file_path(self, memory_id: str, category: str, title: str) -> Path:
        """Return the target file path for a memory."""
        safe_title = self._sanitize_filename(title)
        return self.memory_dir / category / f"{safe_title}_{memory_id}.md"

    def save_memory(self, memory: MemoryItem) -> Path:
        """Save a MemoryItem to disk in its designated category directory."""
        self.ensure_categories()
        # Ensure category folder exists
        cat_dir = self.memory_dir / memory.category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Check if already exists under any file in this category
        existing_path = self.find_file_by_id(memory.id)
        if existing_path and existing_path.exists():
            file_path = existing_path
        else:
            file_path = self._get_file_path(memory.id, memory.category, memory.title)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(memory.to_markdown())

        return file_path

    def find_file_by_id(self, memory_id: str) -> Optional[Path]:
        """Find the file corresponding to a given memory ID across all categories."""
        for md_file in self.memory_dir.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read(500)
                    if f"id: {memory_id}" in content:
                        return md_file
            except Exception:
                continue
        return None

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a memory item by ID."""
        file_path = self.find_file_by_id(memory_id)
        if not file_path:
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return MemoryItem.from_markdown(f.read(), default_category=file_path.parent.name)
        except Exception:
            return None

    def list_memories(self, category: Optional[str] = None, include_superseded: bool = False) -> List[MemoryItem]:
        """List all memories, optionally filtered by category."""
        results = []
        search_dirs = [self.memory_dir / category] if category and (self.memory_dir / category).exists() else [self.memory_dir]

        for base_path in search_dirs:
            for md_file in base_path.rglob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        item = MemoryItem.from_markdown(f.read(), default_category=md_file.parent.name)
                        if item:
                            if not include_superseded and item.status != "active":
                                continue
                            results.append(item)
                except Exception:
                    continue

        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory file by its ID."""
        file_path = self.find_file_by_id(memory_id)
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
        return False

    def update_memory_content(self, memory_id: str, new_content: str, new_title: Optional[str] = None) -> Optional[MemoryItem]:
        """Update content of an existing memory."""
        item = self.get_memory(memory_id)
        if not item:
            return None

        item.content = new_content.strip()
        if new_title:
            item.title = new_title.strip()
        item.updated_at = datetime.now(timezone.utc).isoformat()
        self.save_memory(item)
        return item

    def check_duplicate_or_conflict(
        self,
        new_content: str,
        category: str,
        similarity_threshold: float = 0.70
    ) -> Tuple[str, Optional[MemoryItem]]:
        """
        Check if a candidate memory is an exact duplicate, semantically very similar,
        or a potential contradiction/update to existing memories in that category.
        Returns: ('duplicate' | 'update' | 'unique', existing_item)
        """
        existing_memories = self.list_memories(category=category)
        if not existing_memories:
            return ("unique", None)

        # 1. Exact match
        clean_new = new_content.strip().lower()
        for item in existing_memories:
            if clean_new == item.content.strip().lower():
                return ("duplicate", item)

        # 2. Semantic comparison
        candidate_contents = [m.content for m in existing_memories]
        scores = self.similarity_engine.compute_similarity(new_content, candidate_contents)

        for item, score in zip(existing_memories, scores):
            if score >= 0.90:
                # Highly overlapping content -> Duplicate
                return ("duplicate", item)
            elif score >= similarity_threshold:
                # Related content about same subject -> Possible update or contradiction
                return ("update", item)

        return ("unique", None)

    def add_or_update_memory(
        self,
        category: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
        source_conversation: str = "direct",
        confidence: float = 0.9,
        sensitive: bool = False
    ) -> Tuple[MemoryItem, str]:
        """
        Add a new memory or update an existing one if a duplicate/update is detected.
        Returns (MemoryItem, status) where status is 'created', 'updated', or 'skipped_duplicate'.
        """
        self.ensure_categories()
        status_check, existing = self.check_duplicate_or_conflict(content, category)

        now = datetime.now(timezone.utc).isoformat()

        if status_check == "duplicate" and existing:
            # Refresh confidence or source if appropriate, but avoid duplicate file
            existing.updated_at = now
            if confidence > existing.confidence:
                existing.confidence = confidence
            self.save_memory(existing)
            return (existing, "skipped_duplicate")

        if status_check == "update" and existing:
            # Update the existing memory with newer information
            existing.content = content.strip()
            existing.title = title.strip()
            existing.updated_at = now
            existing.confidence = confidence
            if tags:
                existing.tags = list(set(existing.tags + tags))
            self.save_memory(existing)
            return (existing, "updated")

        # Create brand new memory item
        mem_id = str(uuid.uuid4())[:8]
        new_item = MemoryItem(
            id=mem_id,
            category=category,
            title=title.strip(),
            content=content.strip(),
            tags=tags or [],
            created_at=now,
            updated_at=now,
            source_conversation=source_conversation,
            confidence=confidence,
            sensitive=sensitive,
            status="active"
        )
        self.save_memory(new_item)
        return (new_item, "created")
