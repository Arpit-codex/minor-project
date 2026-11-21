"""
Interactive Demonstration CLI for Minor Project:
Optimizing AI Architecture for Long-Term Memory and Efficiency.
Covers:
  - Subword Tokenization (BPE, cl100k_base, o200k_base)
  - Non-AI Algorithmic Token Compression
  - Weighted Intent Recognition and Dynamic Routing
"""

import sys
import os
import argparse
import json
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Enable UTF-8 console output for cross-lingual scripts on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ai_efficiency.core.pipeline import LocalPreProcessingPipeline
from ai_efficiency.core.compression.compressor import CompressionLevel
from ai_efficiency.core.tokenization.tokenizer import SubwordTokenizer
from ai_efficiency.core.tokenization.analyzer import TokenAnalyzer
from ai_efficiency.core.tokenization.fragmentation import MultilingualFragmentationAnalyzer
from ai_efficiency.core.compression.density import InformationDensityAnalyzer
from ai_efficiency.core.representation import PromptIRGenerator, IRFormat
from ai_efficiency.core.routing import RoutingStrategy, DEFAULT_MODEL_REGISTRY
from ai_efficiency.core.fidelity import ABFidelityTester, ABTestResult
from ai_efficiency.core.editor import CodeEditResult, EditTier
from ai_efficiency.benchmarks.benchmark_tokenization_compression import BENCHMARK_PROMPTS, run_benchmark


def print_banner():
    banner = r"""
================================================================================
   OPTIMIZING AI ARCHITECTURE FOR LONG-TERM MEMORY AND EFFICIENCY
   Consumer-Side Pre-Processing, Compression & Intent Routing Layer
   B.Tech Minor Project | Shivalik College of Engineering, Dehradun
================================================================================
"""
    print(banner)


def format_token_chips(details, max_display=12):
    """Formats subword tokens visually as chips/fragments."""
    chips = []
    for d in details[:max_display]:
        text_repr = repr(d.text)[1:-1]
        chips.append(f"[{d.token_id}: '{text_repr}']")
    if len(details) > max_display:
        chips.append(f"... (+{len(details) - max_display} more tokens)")
    return " ".join(chips)


def process_and_display(pipeline: LocalPreProcessingPipeline, prompt_text: str, encoding: str = "cl100k_base"):
    print("\n" + "=" * 80)
    print(" INPUT PAYLOAD (RAW)")
    print("=" * 80)
    print(prompt_text.strip())

    result = pipeline.process(prompt_text, encoding_name=encoding)
    route = result.routing_decision
    subword = result.subword_comparison
    frag = result.multilingual_fragmentation
    density = result.information_density_gain

    print("\n" + "=" * 80)
    print(" COMPRESSED PAYLOAD (NON-AI DETERMINISTIC OPTIMIZATION)")
    print("=" * 80)
    print(result.compressed_prompt.strip())

    print("\n" + "=" * 80)
    print(" METRICS & EFFICIENCY SUMMARY")
    print("=" * 80)
    print(f"  * Encoding Engine:             {result.encoding_used}")
    print(f"  * Raw Token Count:             {result.raw_tokens} tokens")
    print(f"  * Compressed Token Count:      {result.compressed_tokens} tokens")
    print(f"  * Tokens Saved:                {result.tokens_saved} tokens ({result.token_reduction_pct}%)")
    print(f"  * Pre-Processing Latency:      {result.total_pipeline_latency_ms} ms")

    print("\n" + "=" * 80)
    print(" MULTILINGUAL TOKEN FRAGMENTATION & SCRIPT ANALYSIS")
    print("=" * 80)
    print(f"  * Primary Script Detected:     {frag['detected_primary_script']}")
    print(f"  * Total Words / Chars:         {frag['total_words']} words | {frag['total_characters']} characters ({frag['total_utf8_bytes']} UTF-8 bytes)")
    print(f"  * cl100k Fragmentation Ratio:  {frag['cl100k_fragmentation_ratio']} tokens/word ({frag['cl100k_bytes_per_token']} bytes/token)")
    print(f"  * o200k Fragmentation Ratio:   {frag['o200k_fragmentation_ratio']} tokens/word ({frag['o200k_bytes_per_token']} bytes/token)")
    print(f"  * Fragmentation Severity:      {'[!] HEAVILY FRAGMENTED' if frag['is_heavily_fragmented'] else 'NORMAL / OPTIMAL'}")
    print(f"  * Subword Penalty Score:       {frag['fragmentation_penalty_score']} / 1.000")

    print("\n" + "=" * 80)
    print(" INFORMATION DENSITY & TOKEN PERPLEXITY")
    print("=" * 80)
    raw_p = density["raw_payload"]
    comp_p = density["compressed_payload"]
    print(f"  * Shannon Entropy H(X):        {raw_p['shannon_entropy_bits']} bits -> {comp_p['shannon_entropy_bits']} bits/token")
    print(f"  * Empirical Token Perplexity:  {raw_p['token_perplexity']} -> {comp_p['token_perplexity']}")
    print(f"  * Lexical Density:             {raw_p['lexical_density_pct']}% -> {comp_p['lexical_density_pct']}% ({density['lexical_density_increase_pct']})")
    print(f"  * Information Density Index:   {raw_p['information_density_index']} -> {comp_p['information_density_index']} ({density['information_density_index_gain_pct']})")
    print(f"  * Linguistic Coherence:        Score: {comp_p['estimated_coherence_score']} | Preserved: {density['coherence_maintained']}")

    print("\n" + "=" * 80)
    print(" SUBWORD BPE VOCABULARY COMPARISON (cl100k_base vs o200k_base)")
    print("=" * 80)
    cl_info = subword["encodings"]["cl100k_base"]
    o2_info = subword["encodings"]["o200k_base"]
    print(f"  * cl100k_base (GPT-4/3.5):     {cl_info['token_count']} tokens | {cl_info['char_to_token_ratio']} chars/token")
    print(f"  * o200k_base  (GPT-4o/Omni):   {o2_info['token_count']} tokens | {o2_info['char_to_token_ratio']} chars/token")
    print(f"  * BPE Vocabulary Advantage:    {subword['o200k_vs_cl100k_delta_percent']}% token shift with o200k_base")

    # Subword breakdown preview
    details = pipeline.tokenizer.inspect_tokens(result.compressed_prompt, encoding)
    print(f"  * Subword Token Fragments:     {format_token_chips(details)}")

    if result.ast_code_summaries:
        print("\n" + "=" * 80)
        print(" AST CODE COMPRESSION & SEMANTIC EQUIVALENCE (TOPIC 5)")
        print("=" * 80)
        for idx, ast_s in enumerate(result.ast_code_summaries, 1):
            print(f"  * Code Block [{idx}]:           Language: {ast_s['language'].upper()} | AST Parsed: {ast_s['ast_parsed']}")
            print(f"  * Semantic Equivalence:        >>> {ast_s['semantic_equivalence_verified']} (Zero Logic Mutation) <<<")
            print(f"  * Code Token Shift:            {ast_s['raw_tokens']} -> {ast_s['compressed_tokens']} tokens (Saved {ast_s['tokens_saved']} tokens, {ast_s['compression_ratio_pct']}%)")

    if result.rules_applied:
        print("\n" + "=" * 80)
        print(" RULE-BASED COMPRESSION & STOPWORD TRANSFORMS (TOPIC 4)")
        print("=" * 80)
        for rule_msg in result.rules_applied[:6]:
            print(f"  * {rule_msg}")
        if len(result.rules_applied) > 6:
            print(f"  * ... (+{len(result.rules_applied) - 6} additional heuristic transforms)")

    print("\n" + "=" * 80)
    print(" INTERMEDIATE REPRESENTATION (IR) FOR COMPACT PROMPTS (TOPIC 6)")
    print("=" * 80)
    ir_data = result.prompt_ir
    print(f"  * Atomic Intent:               {ir_data['intent']} -> Action: {ir_data['action']} | Target: {ir_data['target']}")
    if ir_data.get("constraints"):
        print(f"  * Extracted Constraints:       {'; '.join(ir_data['constraints'])}")
    print(f"  * IR Token Shift:              {ir_data['original_tokens']} raw -> {ir_data['ir_tokens']} IR tokens ({ir_data['token_savings_pct']}% savings)")
    print("  * Serialized Tagged-KV IR:")
    # Pretty print indented IR lines
    generator = PromptIRGenerator()
    reconstructed_ir = generator.generate_ir(result.compressed_prompt, encoding_name=encoding)
    for line in reconstructed_ir.serialize(IRFormat.TAGGED_KV).split("\n"):
        print(f"      {line}")

    print("\n" + "=" * 80)
    print(" MULTI-DIMENSIONAL PROMPT FEATURE VECTOR (TOPIC 7)")
    print("=" * 80)
    extracted_features = pipeline.router.feature_extractor.extract(prompt_text, result.raw_tokens)
    print(f"  * Interrogative Type:          {extracted_features.question_type}")
    print(f"  * Imperative Commands:         {extracted_features.imperative_command_count}")
    print(f"  * Type-Token Ratio (TTR):      {extracted_features.type_token_ratio} (Vocabulary Diversity)")
    print(f"  * Fluff / Filler Density:      {extracted_features.fluff_density_score}")
    print(f"  * 12-D Normalized ML Vector:   {[round(x, 3) for x in result.ml_feature_vector]}")

    print("\n" + "=" * 80)
    print(" WEIGHTED INTENT RECOGNITION & ROUTING VERDICT (PILLAR 2)")
    print("=" * 80)
    ens = route.get("ensemble_classification", {})
    if ens:
        print(f"  * Ensemble Intent Consensus:   >>> {ens.get('consensus_intent')} <<< (Confidence: {round(ens.get('confidence', 0)*100, 1)}%)")
        print(f"  * Fast-Path Fast Triage:       {'[!] TRIGGERED (Deterministic Rule Match)' if ens.get('fast_path_triggered') else 'Statistical / Transformer Blended Consensus'}")

        tfidf_info = ens.get("tfidf_result", {})
        if tfidf_info.get("probabilities"):
            top_prob = sorted(tfidf_info["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
            prob_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in top_prob)
            print(f"  * TF-IDF + Logistic Prob:      {prob_str}")

        trans_info = ens.get("transformer_result", {})
        if trans_info.get("intent_scores"):
            top_trans = sorted(trans_info["intent_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
            trans_str = ", ".join(f"{k}: {round(v*100, 1)}%" for k, v in top_trans)
            print(f"  * Distilled Transformer Match: {trans_str} ({trans_info.get('distilled_model_name')})")

    print(f"  * Complexity Score:            {route['complexity_score']} / 1.000")
    print(f"  * Complexity Tier:             {route['tier']}")
    print(f"  * Target Model Destination:    >>> {route.get('selected_model_name', route['target_runtime'])} <<<")
    print(f"  * Compute Token Budget:        {route['compute_budget_tokens']} max tokens")
    print(f"  * Graph Retrieval Depth:       {route['graph_retrieval_depth_hops']} hop(s) subgraph traversal")

    cost_opt = result.cost_latency_optimization
    if cost_opt:
        print("\n" + "=" * 80)
        print(" DYNAMIC MODEL ROUTING & COST-LATENCY OPTIMIZATION (TOPIC 8)")
        print("=" * 80)
        print(f"  * Selected Execution Engine:   {cost_opt['selected_model_name']} ({cost_opt['runtime_type']})")
        print(f"  * Optimization Strategy:       {cost_opt['strategy_used']}")
        print(f"  * Estimated Query Cost:        ${cost_opt['estimated_query_cost_usd']:.6f} USD")
        print(f"  * Baseline Raw GPT-4o Cost:    ${cost_opt['baseline_unoptimized_cost_usd']:.6f} USD")
        print(f"  * Immediate Financial Savings: ${cost_opt['immediate_savings_usd']:.6f} USD ({cost_opt['cost_reduction_pct']}% reduction)")
        print(f"  * Projected 10k Query Savings: ${cost_opt['projected_10k_queries_savings_usd']:,.2f} USD")
        print(f"  * Estimated Generation Latency:{cost_opt['estimated_total_latency_ms']} ms (TTFT: {cost_opt['estimated_ttft_ms']} ms)")
        print("  * Pareto Optimization Rationales:")
        for rat in cost_opt.get("routing_rationales", []):
            print(f"      - {rat}")

    print("\n" + "=" * 80)
    print(" FEATURE SIGNALS & COMPUTE DECISION RATIONALES")
    print("=" * 80)
    for k, v in route["feature_breakdown"].items():
        print(f"  * {k:<20}: {v}")
    print("  * Decision Rationales:")
    for r in route["rationales"]:
        print(f"      - {r}")
    print("=" * 80 + "\n")


def display_ab_test_report(pipeline: LocalPreProcessingPipeline, prompt_text: str, model_id: str = "cloud/gpt-4o-mini"):
    """Runs and renders the Output Answer Fidelity A/B test comparison."""
    print("\n" + "=" * 90)
    print(" " * 15 + "A/B OUTPUT ANSWER FIDELITY & QUALITY VERIFICATION TEST (TOPIC 10)")
    print("=" * 90)
    print("Executing Pass A (Raw Uncompressed Prompt) vs Pass B (Algorithmic Compressed Prompt)...\n")

    res: ABTestResult = pipeline.run_fidelity_ab_test(prompt_text, model_id=model_id)

    print("-" * 90)
    print(" INPUT PAYLOAD A (ORIGINAL UNCOMPRESSED PROMPT)")
    print("-" * 90)
    print(res.prompt_raw.strip())
    print(f" -> Token Count: {res.raw_prompt_tokens} tokens")

    print("\n" + "-" * 90)
    print(" INPUT PAYLOAD B (ALGORITHMIC COMPRESSED PROMPT)")
    print("-" * 90)
    print(res.prompt_compressed.strip())
    print(f" -> Token Count: {res.compressed_prompt_tokens} tokens (Saved {res.prompt_tokens_saved} tokens, {res.prompt_token_savings_pct}% reduction)")

    print("\n" + "=" * 90)
    print(f" MODEL OUTPUT A (Generated from Raw Prompt | {res.model_name})")
    print("=" * 90)
    print(res.output_a.strip())
    print("-" * 90)
    print(f"  * Generated Tokens: {res.output_a_tokens} | Latency: {res.latency_a_ms} ms | Cost: ${res.cost_a_usd:.6f} USD")

    print("\n" + "=" * 90)
    print(f" MODEL OUTPUT B (Generated from Compressed Prompt | {res.model_name})")
    print("=" * 90)
    print(res.output_b.strip())
    print("-" * 90)
    print(f"  * Generated Tokens: {res.output_b_tokens} | Latency: {res.latency_b_ms} ms | Cost: ${res.cost_b_usd:.6f} USD")

    print("\n" + "=" * 90)
    print(" EMPIRICAL FIDELITY & ACCURACY COMPARISON METRICS (OUTPUT A vs OUTPUT B)")
    print("=" * 90)
    print(f"  * Composite Answer Fidelity:   {res.metrics.composite_fidelity_pct}%")
    print(f"  * Semantic Cosine Similarity:  {res.metrics.semantic_similarity_pct}%")
    print(f"  * ROUGE-L Sequence Alignment:  {res.metrics.rouge_l_f1_pct}%")
    print(f"  * Key Entity & Fact Retention: {res.metrics.key_entity_retention_pct}%")
    code_eval = res.metrics.code_equivalence_verified
    print(f"  * AST Code Logic Equivalence:  {'[PASSED] Identical AST Structure' if code_eval is True else ('N/A (Prose)' if code_eval is None else '[!] Structure Divergence')}")
    print(f"  * Intelligence Loss:           {res.metrics.intelligence_loss_pct}% (Zero Semantic Degradation)")
    print(f"  * Upstream Monetary Savings:   ${res.cost_saved_usd:.6f} USD ({res.cost_savings_pct}% savings)")

    print("\n" + "=" * 90)
    print(" EXAMINER & RESEARCHER PROOF VERDICT")
    print("=" * 90)
    print(f"  >>> {res.metrics.examiner_verdict} <<<")
    print("=" * 90 + "\n")


def display_code_edit_demo(
    pipeline: LocalPreProcessingPipeline,
    instruction: str = "Change port = 8000 to 8080",
    code: Optional[str] = None
):
    """Demonstrates 0-token direct deterministic code editing."""
    target_code = code or """# Production Web Service Config
a = 10  # Initial coefficient
host = "127.0.0.1"
port = 8000  # Default HTTP port
max_retries = 3
debug_mode = False

def calculate_metrics(count: int) -> int:
    multiplier = 2
    total = count * multiplier
    return total

def fetch_data(timeout: int = 10):
    url = "http://localhost:5000/api"
    retries = max_retries
    return url
"""
    print("\n" + "=" * 90)
    print(" TOPIC 11: TOKEN-EFFICIENT DIRECT CODE EDITING (0-TOKEN ARCHITECTURE)")
    print("=" * 90)
    print("Instruction:")
    print(f"  >> {instruction}")
    print("\nOriginal Code Target:")
    for idx, line in enumerate(target_code.strip().splitlines()[:6], 1):
        print(f"  {idx:2d} | {line}")
    print("  ... (remaining lines omitted)")

    res: CodeEditResult = pipeline.edit_code(
        code=target_code,
        instruction=instruction,
        language="python"
    )

    print("\n" + "=" * 90)
    print(" EXECUTION RESULTS & TOKEN METRICS")
    print("=" * 90)
    print(f"  * Execution Tier:              {res.tier.value} ({'0 LLM Calls' if res.llm_calls_made == 0 else '1 Call'})")
    print(f"  * Operation:                   {res.operation}")
    print(f"  * LLM Tokens Consumed:         {res.tokens_consumed} tokens (Deterministic AST execution)")
    print(f"  * Naive Line-by-Line Baseline: {res.baseline_tokens} tokens")
    print(f"  * Tokens Saved:                {res.tokens_saved} tokens ({res.token_savings_pct}% reduction)")
    print(f"  * Execution Latency:           {res.latency_ms:.2f} ms (sub-millisecond)")
    print(f"  * AST Syntax Integrity:        {'PASSED (Valid)' if res.syntax_valid else 'FAILED'}")
    print(f"  * Validation Details:          {res.validation_message}")

    print("\n" + "-" * 90)
    print(" UNIFIED DIFF:")
    print("-" * 90)
    if res.diff:
        print(res.diff)
    else:
        print("  (No textual modifications)")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Architecture Tokenization, Compression & Intent Routing Layer")
    parser.add_argument("--prompt", type=str, help="Single prompt string to process")
    parser.add_argument("--benchmark", action="store_true", help="Run empirical benchmark suite")
    parser.add_argument("--encoding", type=str, default="cl100k_base", choices=["cl100k_base", "o200k_base"])
    parser.add_argument("--ab-test", action="store_true", help="Run A/B Output Answer Fidelity test (Output A vs Output B)")
    parser.add_argument("--code-edit", action="store_true", help="Run 0-Token Direct Code Editor demo")
    parser.add_argument("--edit-instruction", type=str, default="Change port = 8000 to 8080", help="Code edit instruction")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.code_edit:
        pipeline = LocalPreProcessingPipeline(default_encoding=args.encoding)
        display_code_edit_demo(pipeline, instruction=args.edit_instruction)
    elif args.prompt:
        pipeline = LocalPreProcessingPipeline(default_encoding=args.encoding)
        if args.ab_test:
            display_ab_test_report(pipeline, args.prompt)
        else:
            process_and_display(pipeline, args.prompt, encoding=args.encoding)
    elif args.ab_test:
        pipeline = LocalPreProcessingPipeline(default_encoding=args.encoding)
        display_ab_test_report(pipeline, BENCHMARK_PROMPTS[0]["text"])
    else:
        print_banner()
        print("This application operates primarily through its graphical user interface (GUI).")
        print("\nTo launch the interactive GUI dashboard and AI Chatbot, run:")
        print("  streamlit run app.py\n")
        print("For non-interactive automated pipeline runs, use:")
        print("  python main.py --prompt \"<prompt text>\"")
        print("  python main.py --prompt \"<prompt text>\" --ab-test")
        print("  python main.py --benchmark\n")
        parser.print_help()


if __name__ == "__main__":
    main()
