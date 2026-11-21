"""
Core Editor Module: Token-Efficient Direct Code Editing System.
Implements the 3-Tier Architecture:
  Tier 1: Direct Zero-Token Edit (AST & Regex) - 0 LLM calls
  Tier 2: Targeted Retrieval (Micro-context snippet) - Minimal tokens
  Tier 3: Deep Reasoning (Multi-file / complex logic) - Escalated context
"""

from ai_efficiency.core.editor.models import EditTier, EditOperation, ParsedInstruction, CodeEditResult
from ai_efficiency.core.editor.intent_classifier import EditIntentClassifier
from ai_efficiency.core.editor.direct_engine import DirectEditEngine
from ai_efficiency.core.editor.targeted_retrieval import TargetedRetrievalEngine
from ai_efficiency.core.editor.patch_engine import MinimalPatchEngine
from ai_efficiency.core.editor.orchestrator import TokenEfficientCodeEditor

__all__ = [
    "EditTier",
    "EditOperation",
    "ParsedInstruction",
    "CodeEditResult",
    "EditIntentClassifier",
    "DirectEditEngine",
    "TargetedRetrievalEngine",
    "MinimalPatchEngine",
    "TokenEfficientCodeEditor",
]
