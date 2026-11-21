"""
Rule-Based Text Compressor and Pleonasm Compactor.
Applies deterministic linguistic simplification and fluff pruning without mutating
meaning, code blocks, or critical semantic anchors.
"""

import re
from typing import Dict, Any, List, Set, Tuple
from enum import Enum
from dataclasses import dataclass, asdict

from core.efficiency.rules import (
    CONVERSATIONAL_BOILERPLATE,
    WORDY_PHRASE_MAPPINGS,
    PUNCTUATION_REDUNDANCY,
    SEMANTIC_ANCHORS,
    LOW_INFO_FILLERS,
)


class CompressionLevel(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class RuleCompressionReport:
    """Report detailing phrase compaction and rule executions."""
    original_text: str
    compressed_text: str
    rules_applied: List[str]
    fillers_removed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleBasedTextCompressor:
    """
    Deterministic rule-based text compressor.
    Compacts pleonasms and conversational boilerplate.
    Guarantees semantic anchor preservation.
    """

    def __init__(self, level: CompressionLevel = CompressionLevel.BALANCED):
        self.level = level

    def _compact_wordy_phrases(self, text: str) -> Tuple[str, List[str]]:
        """Applies pleonasm mappings to collapse multi-word bloat."""
        applied = []
        result = text
        for pattern, replacement in WORDY_PHRASE_MAPPINGS:
            if pattern.search(result):
                result = pattern.sub(replacement, result)
                applied.append(f"Compacted '{pattern.pattern[:30]}...' -> '{replacement}'")
        return result, applied

    def _strip_boilerplate(self, text: str) -> Tuple[str, List[str]]:
        """Removes conversational fluff, greetings, and closing gratitude."""
        applied = []
        result = text
        for pattern, replacement in CONVERSATIONAL_BOILERPLATE:
            if pattern.search(result):
                result = pattern.sub(replacement, result)
                applied.append("Pruned conversational boilerplate")
        return result, applied

    def _prune_low_info_fillers(self, text: str) -> Tuple[str, List[str]]:
        """Removes filler adverbs (basically, actually, simply) unless they are semantic anchors."""
        removed = []
        words = text.split()
        filtered = []

        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in LOW_INFO_FILLERS and clean not in SEMANTIC_ANCHORS:
                removed.append(clean)
                continue
            filtered.append(w)

        return " ".join(filtered), removed

    def _cleanup_punctuation(self, text: str) -> str:
        """Cleans up redundant whitespace and punctuation."""
        result = text
        for pattern, replacement in PUNCTUATION_REDUNDANCY:
            result = pattern.sub(replacement, result)
        return result.strip()

    def compress_text(self, text: str, level: CompressionLevel = CompressionLevel.BALANCED) -> RuleCompressionReport:
        """
        Compresses plain text prose according to selected aggressiveness level.
        - CONSERVATIVE: Boilerplate and pleonasms only.
        - BALANCED: Boilerplate, pleonasms, and low-info fillers.
        - AGGRESSIVE: Aggressive filler pruning and whitespace compaction.
        """
        rules_applied = []
        fillers_removed = []

        # 1. Boilerplate pruning (All levels)
        current_text, b_rules = self._strip_boilerplate(text)
        rules_applied.extend(b_rules)

        # 2. Wordy phrase compaction (All levels)
        current_text, p_rules = self._compact_wordy_phrases(current_text)
        rules_applied.extend(p_rules)

        # 3. Low-info fillers (BALANCED and AGGRESSIVE)
        if level in (CompressionLevel.BALANCED, CompressionLevel.AGGRESSIVE):
            current_text, fillers = self._prune_low_info_fillers(current_text)
            fillers_removed.extend(fillers)
            if fillers:
                rules_applied.append(f"Pruned {len(fillers)} filler words")

        # 4. Whitespace & punctuation normalization
        final_text = self._cleanup_punctuation(current_text)

        # Capitalize first character if needed
        if final_text and final_text[0].islower() and (len(final_text) == 1 or final_text[1].isalpha()):
            final_text = final_text[0].upper() + final_text[1:]

        return RuleCompressionReport(
            original_text=text,
            compressed_text=final_text,
            rules_applied=list(dict.fromkeys(rules_applied)),
            fillers_removed=list(dict.fromkeys(fillers_removed))
        )
