"""
Tests for Token & Efficiency Optimizer:
- Rule-based text compaction & boilerplate pruning
- AST Python code docstring stripping & mathematical equivalence
- Semantic anchor preservation (NOT, MUST, O(1))
- Information density & Shannon entropy
- Multi-model financial cost savings calculator
- Web endpoints (/efficiency, /api/optimize, /api/optimize/benchmarks)
"""

import pytest
from core.efficiency.text_compressor import RuleBasedTextCompressor, CompressionLevel
from core.efficiency.ast_compressor import ASTCodeCompressor
from core.efficiency.metrics import estimate_tokens, compute_information_density, calculate_cost_savings
from core.efficiency.optimizer import PromptOptimizer, BENCHMARK_PROMPTS
from gui_server import app


def test_rule_based_text_compression_and_semantic_anchors():
    """
    Test pleonasm compaction, boilerplate removal, and strict preservation
    of semantic anchors (not, must, o(1)).
    """
    compressor = RuleBasedTextCompressor()
    raw = (
        "Hello! Can you please write a function in order to calculate primes? "
        "It is crucial that the algorithm must not use external libraries, "
        "and time complexity should be O(n). Thank you very much!"
    )
    report = compressor.compress_text(raw, level=CompressionLevel.BALANCED)
    compressed = report.compressed_text.lower()

    # Pleonasms compacted / fluff removed
    assert "hello" not in compressed
    assert "thank you" not in compressed
    assert "in order to" not in compressed

    # Crucial semantic anchors MUST be preserved
    assert "must" in compressed
    assert "not" in compressed
    assert "o(n)" in compressed


def test_ast_python_code_compression_and_equivalence():
    """
    Test Python AST docstring stripping and zero-mutation semantic equivalence.
    """
    ast_comp = ASTCodeCompressor()
    py_code = (
        "def fibonacci(n: int) -> int:\n"
        "    \"\"\"Compute the nth Fibonacci number recursively.\"\"\"\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    )

    res, is_parsed, is_equiv = ast_comp.compress_python_ast(py_code)
    assert is_parsed is True
    assert is_equiv is True
    # Docstring is stripped
    assert "Compute the nth Fibonacci number" not in res
    # Essential logic is completely intact
    assert "fibonacci(n - 1)" in res
    assert "return n" in res


def test_generic_code_comment_stripping():
    """
    Test non-Python comment stripping (SQL and JS).
    """
    ast_comp = ASTCodeCompressor()
    sql_code = (
        "-- Fetch completed user transactions\n"
        "SELECT id, amount FROM transactions\n"
        "WHERE status = 'PAID'; -- Filter paid only\n"
    )
    cleaned = ast_comp.compress_generic_code(sql_code, "sql")
    assert "-- Fetch completed" not in cleaned
    assert "SELECT id, amount" in cleaned


def test_information_density_and_shannon_entropy():
    """
    Test token estimation, Shannon entropy, and lexical density.
    """
    text = "The quick brown fox jumps over the lazy dog."
    tokens = estimate_tokens(text)
    assert tokens >= 8

    density = compute_information_density(text)
    assert density["shannon_entropy_bits"] > 0
    assert density["perplexity"] > 1.0
    assert density["lexical_density_pct"] > 0


def test_cost_savings_calculation():
    """
    Test financial cost savings calculator across models.
    """
    savings = calculate_cost_savings(raw_tokens=500, compressed_tokens=250)
    assert savings["tokens_saved"] == 250
    assert savings["savings_pct"] == 50.0
    assert savings["projected_10k_savings_usd"] > 0
    assert "gemini_3_1_flash_lite" in savings["model_breakdown"]
    assert "gpt_4o" in savings["model_breakdown"]


def test_prompt_optimizer_full_pipeline():
    """
    Test high-level PromptOptimizer on sample benchmark.
    """
    optimizer = PromptOptimizer()
    sample = BENCHMARK_PROMPTS[0]["prompt"]
    result = optimizer.optimize(sample, level="balanced")

    assert result["raw_tokens"] > result["compressed_tokens"]
    assert result["tokens_saved"] > 0
    assert result["reduction_pct"] > 20.0
    assert result["ast_verified"] is True
    assert result["has_code"] is True
    assert result["latency_ms"] < 100.0  # Must be fast


def test_efficiency_web_endpoints():
    """
    Test Flask API endpoints for the Token & Efficiency Optimizer.
    """
    client = app.test_client()

    # 1. View route
    res_view = client.get("/efficiency")
    assert res_view.status_code == 200

    # 2. Benchmarks endpoint
    res_bench = client.get("/api/optimize/benchmarks")
    assert res_bench.status_code == 200
    benchmarks = res_bench.get_json()
    assert len(benchmarks) >= 3

    # 3. Optimize endpoint
    res_opt = client.post("/api/optimize", json={
        "prompt": "Hello! Can you please write a quick python script to sort numbers?",
        "level": "balanced"
    })
    assert res_opt.status_code == 200
    data = res_opt.get_json()
    assert data["tokens_saved"] > 0
    assert len(data["compressed_prompt"]) < len(data["original_prompt"])
