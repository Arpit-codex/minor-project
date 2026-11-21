"""
High-Level Orchestration Engine for Consumer-Side Prompt Optimization.
Coordinates AST Code Compression, Rule-Based Text Compaction,
Information Density Analysis, and Financial Savings Estimation.
"""

import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from core.efficiency.text_compressor import RuleBasedTextCompressor, CompressionLevel
from core.efficiency.ast_compressor import ASTCodeCompressor
from core.efficiency.metrics import estimate_tokens, compute_information_density, calculate_cost_savings


BENCHMARK_PROMPTS = [
    {
        "id": "python_algorithm",
        "title": "Python Algorithm with Docstrings & Fluff",
        "category": "Code & AST",
        "prompt": (
            "Hello there! Could you please write a python script to implement binary search?\n\n"
            "```python\n"
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    \"\"\"\n"
            "    Performs binary search on a sorted integer list.\n"
            "    Returns index of target if found, otherwise returns -1.\n"
            "    Time complexity must be O(log n).\n"
            "    \"\"\"\n"
            "    # Initialize lower and upper search boundaries\n"
            "    low = 0\n"
            "    high = len(arr) - 1\n"
            "    \n"
            "    # Iterate while search interval is valid\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "            \n"
            "    return -1\n"
            "```\n\n"
            "Can you explain how this works in detail and make sure to highlight any edge cases? Thank you so much!"
        )
    },
    {
        "id": "verbose_business_query",
        "title": "Wordy Business & Technical Request",
        "category": "Prose & Boilerplate",
        "prompt": (
            "Good afternoon! Can you please assist me and help me to write a comprehensive strategy in order to migrate "
            "our monolithic backend architecture to a microservices architecture? Due to the reason that our application is "
            "experiencing high traffic at this point in time, it is crucial that the system must maintain 99.99% uptime. "
            "In addition to that, please kindly provide a step-by-step roadmap and also outline the common pitfalls. "
            "Thank you very much for your help and support!"
        )
    },
    {
        "id": "sql_optimization",
        "title": "SQL Query with Comments & Redundancy",
        "category": "Database & SQL",
        "prompt": (
            "Could you please review and optimize the following SQL query for me?\n\n"
            "```sql\n"
            "-- Query to retrieve active customer orders with transaction totals\n"
            "-- Author: Engineering Team\n"
            "-- Date: 2026-11-21\n"
            "SELECT \n"
            "    c.customer_id, \n"
            "    c.customer_name, \n"
            "    COUNT(o.order_id) AS total_orders, \n"
            "    SUM(o.order_amount) AS total_spent\n"
            "FROM customers c\n"
            "JOIN orders o ON c.customer_id = o.customer_id\n"
            "WHERE o.order_status = 'COMPLETED'\n"
            "-- Group by customer identifiers\n"
            "GROUP BY c.customer_id, c.customer_name\n"
            "-- Sort by highest spenders first\n"
            "ORDER BY total_spent DESC;\n"
            "```\n\n"
            "I want to know how we can make this query run faster on a PostgreSQL database with 10 million rows."
        )
    },
    {
        "id": "javascript_api_service",
        "title": "JavaScript / TypeScript API Client",
        "category": "Frontend / Fullstack",
        "prompt": (
            "Hi! Would you mind writing a JavaScript fetch wrapper with automatic retries?\n\n"
            "```javascript\n"
            "/**\n"
            " * Fetch wrapper with exponential backoff\n"
            " * @param {string} url - API target endpoint\n"
            " * @param {number} retries - Number of retry attempts\n"
            " */\n"
            "async function fetchWithRetry(url, retries = 3) {\n"
            "    // Loop through attempts\n"
            "    for (let i = 0; i < retries; i++) {\n"
            "        try {\n"
            "            const response = await fetch(url);\n"
            "            if (!response.ok) throw new Error(`HTTP status ${response.status}`);\n"
            "            return await response.json();\n"
            "        } catch (err) {\n"
            "            // Delay before next attempt\n"
            "            if (i === retries - 1) throw err;\n"
            "            await new Promise(r => setTimeout(r, 1000 * Math.pow(2, i)));\n"
            "        }\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "Please explain how this handles network disconnection errors. Thanks!"
        )
    }
]


class PromptOptimizer:
    """
    Main entry point for local prompt pre-processing, compression, and metric evaluation.
    """

    def __init__(self):
        self.ast_compressor = ASTCodeCompressor()
        self.text_compressor = RuleBasedTextCompressor()

    def optimize(self, prompt: str, level: str = "balanced") -> Dict[str, Any]:
        """
        Executes end-to-end prompt optimization:
        1. Compresses code blocks (Python AST docstring removal, comment stripping).
        2. Compresses surrounding text prose (boilerplate, pleonasms, fillers).
        3. Generates token, entropy, and financial savings metrics.
        """
        t0 = time.perf_counter()
        clean_prompt = prompt.strip()

        if not clean_prompt:
            return {
                "original_prompt": "",
                "compressed_prompt": "",
                "raw_tokens": 0,
                "compressed_tokens": 0,
                "tokens_saved": 0,
                "reduction_pct": 0.0,
                "ast_code_summaries": [],
                "ast_verified": True,
                "rules_applied": [],
                "information_density": {},
                "cost_savings": {},
                "latency_ms": 0.0
            }

        # Resolve compression level
        level_enum = CompressionLevel.BALANCED
        if level.lower() == "conservative":
            level_enum = CompressionLevel.CONSERVATIVE
        elif level.lower() == "aggressive":
            level_enum = CompressionLevel.AGGRESSIVE

        # 1. Process code blocks first (preserves code from prose regex rules)
        code_compressed_prompt, ast_summaries = self.ast_compressor.process_prompt_code_blocks(clean_prompt)

        # 2. Process non-code prose segments
        # Extract markdown code blocks to keep them safe
        code_pattern = re.compile(r"```[a-zA-Z0-9_-]*\n[\s\S]*?```")
        code_matches = list(code_pattern.finditer(code_compressed_prompt))

        rules_applied = []
        if code_matches:
            # Split and process only text between code blocks
            last_idx = 0
            compressed_parts = []
            for m in code_matches:
                prose_part = code_compressed_prompt[last_idx:m.start()]
                if prose_part.strip():
                    report = self.text_compressor.compress_text(prose_part, level=level_enum)
                    compressed_parts.append(report.compressed_text)
                    rules_applied.extend(report.rules_applied)
                else:
                    compressed_parts.append(prose_part)

                # Append the safe compressed code block
                compressed_parts.append(m.group(0))
                last_idx = m.end()

            # Tail prose
            tail_prose = code_compressed_prompt[last_idx:]
            if tail_prose.strip():
                report = self.text_compressor.compress_text(tail_prose, level=level_enum)
                compressed_parts.append(report.compressed_text)
                rules_applied.extend(report.rules_applied)
            else:
                compressed_parts.append(tail_prose)

            final_compressed = "".join(compressed_parts)
        else:
            # Entire prompt is prose
            report = self.text_compressor.compress_text(code_compressed_prompt, level=level_enum)
            final_compressed = report.compressed_text
            rules_applied.extend(report.rules_applied)

        # 3. Calculate Tokens and Information Density
        raw_tokens = estimate_tokens(clean_prompt)
        compressed_tokens = estimate_tokens(final_compressed)
        tokens_saved = max(0, raw_tokens - compressed_tokens)
        reduction_pct = round((tokens_saved / raw_tokens) * 100, 1) if raw_tokens > 0 else 0.0

        raw_density = compute_information_density(clean_prompt)
        comp_density = compute_information_density(final_compressed)

        density_comparison = {
            "raw_entropy": raw_density["shannon_entropy_bits"],
            "compressed_entropy": comp_density["shannon_entropy_bits"],
            "entropy_change": round(comp_density["shannon_entropy_bits"] - raw_density["shannon_entropy_bits"], 3),
            "raw_lexical_density_pct": raw_density["lexical_density_pct"],
            "compressed_lexical_density_pct": comp_density["lexical_density_pct"],
            "density_gain_pct": round(comp_density["lexical_density_pct"] - raw_density["lexical_density_pct"], 1)
        }

        # 4. Financial Cost Savings
        cost_savings = calculate_cost_savings(raw_tokens, compressed_tokens)

        # 5. AST Semantic Equivalence Flag
        ast_verified = all(s.semantic_equivalence_verified for s in ast_summaries) if ast_summaries else True

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "original_prompt": clean_prompt,
            "compressed_prompt": final_compressed,
            "raw_tokens": raw_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": tokens_saved,
            "reduction_pct": reduction_pct,
            "characters_saved": len(clean_prompt) - len(final_compressed),
            "ast_code_summaries": [s.to_dict() for s in ast_summaries],
            "ast_verified": ast_verified,
            "has_code": len(ast_summaries) > 0,
            "rules_applied": list(dict.fromkeys(rules_applied)),
            "information_density": density_comparison,
            "cost_savings": cost_savings,
            "latency_ms": latency_ms,
            "compression_level": level
        }
