"""
Metrics & Information Density Engine.
Calculates Subword Token Estimations, Shannon Entropy H(X), Perplexity,
Lexical Density %, and Financial Cost Savings across LLM API providers.
"""

import math
import re
from typing import Dict, Any, List, Set, Tuple
from collections import Counter


FUNCTION_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "out", "off", "over", "under", "again", "of",
    "i", "me", "my", "myself", "we", "our", "ours", "you", "your", "he", "him",
    "she", "her", "it", "its", "they", "them", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "would", "should",
    "could", "might", "may", "can", "will", "if", "because", "as", "while", "then",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "some", "such", "than", "too", "very", "just", "now",
    "hello", "hi", "hey", "please", "kindly", "thanks", "thank"
}

# Standard LLM Pricing per 1 Million Input Tokens (USD)
PROVIDER_PRICING: Dict[str, float] = {
    "gemini_3_1_flash_lite": 0.075,   # Google Gemini 3.1 Flash-Lite ($0.075 / 1M)
    "gemini_3_5_flash": 0.150,        # Google Gemini 3.5 Flash ($0.150 / 1M)
    "gpt_4o": 2.500,                  # OpenAI GPT-4o ($2.50 / 1M)
    "claude_3_5_sonnet": 3.000,       # Anthropic Claude 3.5 Sonnet ($3.00 / 1M)
    "local_edge_slm": 0.000           # Local Edge Model ($0.00 / 1M)
}


def estimate_tokens(text: str) -> int:
    """
    Subword token estimation matching BPE (Byte-Pair Encoding) behavior.
    Handles word boundaries, code syntax, and multi-byte fragments.
    """
    if not text or not text.strip():
        return 0

    # Split into words and code tokens
    words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    token_count = 0
    for w in words:
        length = len(w)
        if length <= 4:
            token_count += 1
        else:
            token_count += math.ceil(length / 3.7)

    return max(1, token_count)


def compute_information_density(text: str) -> Dict[str, float]:
    """
    Calculates Shannon Entropy H(X) in bits/token, empirical Perplexity 2^H(X),
    and Lexical Density % (Ratio of content words to total words).
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return {
            "shannon_entropy_bits": 0.0,
            "perplexity": 1.0,
            "lexical_density_pct": 0.0,
            "content_word_count": 0,
            "total_words": 0
        }

    total_words = len(words)
    word_counts = Counter(words)

    # Shannon Entropy: H(X) = -Σ P(w) * log2(P(w))
    entropy = 0.0
    for count in word_counts.values():
        p = count / total_words
        entropy -= p * math.log2(p)

    perplexity = 2 ** entropy

    # Lexical Density: Content words (non-grammatical) / Total words
    content_words = [w for w in words if w not in FUNCTION_WORDS]
    lexical_density = round((len(content_words) / total_words) * 100, 1)

    return {
        "shannon_entropy_bits": round(entropy, 3),
        "perplexity": round(perplexity, 2),
        "lexical_density_pct": lexical_density,
        "content_word_count": len(content_words),
        "total_words": total_words
    }


def calculate_cost_savings(raw_tokens: int, compressed_tokens: int) -> Dict[str, Any]:
    """
    Calculates immediate and projected economic cost savings across multiple LLM models.
    """
    tokens_saved = max(0, raw_tokens - compressed_tokens)
    savings_pct = round((tokens_saved / raw_tokens) * 100, 1) if raw_tokens > 0 else 0.0

    breakdown = {}
    for model_key, price_per_1m in PROVIDER_PRICING.items():
        price_per_token = price_per_1m / 1_000_000.0
        raw_cost = raw_tokens * price_per_token
        comp_cost = compressed_tokens * price_per_token
        saved_usd = raw_cost - comp_cost
        savings_10k = saved_usd * 10_000

        breakdown[model_key] = {
            "price_per_1m": price_per_1m,
            "raw_cost_usd": round(raw_cost, 6),
            "compressed_cost_usd": round(comp_cost, 6),
            "saved_per_query_usd": round(saved_usd, 6),
            "projected_10k_savings_usd": round(savings_10k, 2),
        }

    # Primary savings based on GPT-4o frontier API comparison
    gpt4o = breakdown["gpt_4o"]
    gemini_lite = breakdown["gemini_3_1_flash_lite"]

    return {
        "tokens_saved": tokens_saved,
        "savings_pct": savings_pct,
        "immediate_savings_usd": gpt4o["saved_per_query_usd"],
        "projected_10k_savings_usd": gpt4o["projected_10k_savings_usd"],
        "gemini_savings_10k_usd": gemini_lite["projected_10k_savings_usd"],
        "model_breakdown": breakdown
    }
