"""
GUI Server for Memory Graph View.
Serves a beautiful Obsidian-style force-directed graph visualization
of the local memory system with full CRUD capabilities.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

from flask import Flask, render_template, jsonify, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import Config, MEMORY_DIR
from core.memory_manager import MemoryManager, MemoryItem
from core.similarity_engine import SimilarityEngine
from core.conversation_manager import ConversationManager
from core.assistant import ContextualAssistant
from core.ai_client import AIClient
from core.efficiency import PromptOptimizer, BENCHMARK_PROMPTS

# Initialize AI Efficiency Subsystem from internal subfolder
AI_EFF_PATH = str(Path(__file__).parent / "ai_efficiency")
if AI_EFF_PATH not in sys.path:
    sys.path.insert(0, AI_EFF_PATH)

try:
    from ai_efficiency.core.pipeline import LocalPreProcessingPipeline
    from ai_efficiency.core.compression.compressor import CompressionLevel
    from ai_efficiency.core.representation import PromptIRGenerator, IRFormat
    from ai_efficiency.core.fidelity.ab_tester import ABFidelityTester, ABTestResult
    from ai_efficiency.core.fidelity.llm_client import LLMClient
    from ai_efficiency.core.editor.orchestrator import TokenEfficientCodeEditor
    from ai_efficiency.core.editor.models import EditTier, CodeEditResult
    from ai_efficiency.core.config import load_settings as load_ai_settings
    from ai_efficiency.benchmarks.benchmark_tokenization_compression import BENCHMARK_PROMPTS as AI_EFF_BENCHMARKS
    ai_eff_pipeline = LocalPreProcessingPipeline()
    ai_eff_code_editor = TokenEfficientCodeEditor()
    ai_eff_ir_gen = PromptIRGenerator()
    ai_eff_available = True
except Exception as e:
    print(f"[Warning] Failed to load ai_efficiency: {e}")
    ai_eff_available = False

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize components
similarity_engine = SimilarityEngine()
memory_manager = MemoryManager(similarity_engine=similarity_engine)
conversation_manager = ConversationManager()
assistant = ContextualAssistant(
    conversation_manager=conversation_manager,
    similarity_engine=similarity_engine,
    memory_manager=memory_manager
)
prompt_optimizer = PromptOptimizer()

# Category color palette (Obsidian-inspired vibrant muted tones)
CATEGORY_COLORS = {
    "personal":             "#7C6FE0",  # Soft purple
    "college":              "#4ECDC4",  # Teal
    "career":               "#FF6B6B",  # Coral
    "exams":                "#FFE66D",  # Warm yellow
    "family":               "#FF8A5C",  # Peach orange
    "projects":             "#45B7D1",  # Sky blue
    "preferences":          "#96CEB4",  # Sage green
    "technical_knowledge":  "#F093FB",  # Pink lavender
    "general":              "#A0A4B8",  # Cool grey
}


def _build_graph_data():
    """Build nodes and edges for the graph visualization."""
    memories = memory_manager.list_memories()
    nodes = []
    edges = []
    tag_to_ids = {}  # tag -> list of memory ids

    for mem in memories:
        node = {
            "id": mem.id,
            "label": mem.title[:40] if mem.title else "Untitled",
            "category": mem.category,
            "content": mem.content,
            "tags": mem.tags,
            "confidence": mem.confidence,
            "created_at": mem.created_at,
            "updated_at": mem.updated_at,
            "source": mem.source_conversation,
            "color": CATEGORY_COLORS.get(mem.category, "#A0A4B8"),
            "size": max(12, min(30, 12 + len(mem.content) // 40)),
        }
        nodes.append(node)

        # Index tags for edge building
        for tag in mem.tags:
            tag_to_ids.setdefault(tag.lower(), []).append(mem.id)

    # Create category hub nodes (larger, central)
    categories_with_memories = set(m.category for m in memories)
    for cat in categories_with_memories:
        hub_id = f"cat_{cat}"
        nodes.append({
            "id": hub_id,
            "label": cat.upper().replace("_", " "),
            "category": cat,
            "content": "",
            "tags": [],
            "confidence": 1.0,
            "created_at": "",
            "updated_at": "",
            "source": "",
            "color": CATEGORY_COLORS.get(cat, "#A0A4B8"),
            "size": 36,
            "is_hub": True,
        })
        # Connect each memory to its category hub
        for mem in memories:
            if mem.category == cat:
                edges.append({
                    "source": hub_id,
                    "target": mem.id,
                    "type": "category",
                    "strength": 0.6,
                })

    # Connect memories sharing tags
    seen_edges = set()
    for tag, ids in tag_to_ids.items():
        if tag in CATEGORY_COLORS:  # skip pure category tags
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edge_key = tuple(sorted([ids[i], ids[j]]))
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "source": ids[i],
                        "target": ids[j],
                        "type": "tag",
                        "strength": 0.3,
                        "label": tag,
                    })

    # Semantic similarity edges (between memories with similarity > 0.4)
    if len(memories) > 1:
        texts = [f"{m.title}. {m.content}" for m in memories]
        for i in range(len(memories)):
            scores = similarity_engine.compute_similarity(texts[i], texts[i+1:])
            for j_offset, score in enumerate(scores):
                j = i + 1 + j_offset
                if score >= 0.35:
                    edge_key = tuple(sorted([memories[i].id, memories[j].id]))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            "source": memories[i].id,
                            "target": memories[j].id,
                            "type": "semantic",
                            "strength": round(score, 3),
                        })

    return {"nodes": nodes, "edges": edges, "categories": list(CATEGORY_COLORS.keys())}


# ── Routes ──

@app.route("/")
@app.route("/memory-graph")
def index():
    return render_template("index.html", categories=list(CATEGORY_COLORS.items()))


@app.route("/chat")
def chat_view():
    return render_template("chat.html", categories=list(CATEGORY_COLORS.items()))


@app.route("/efficiency")
def efficiency_view():
    return render_template("efficiency.html", categories=list(CATEGORY_COLORS.items()))


@app.route("/api/graph")
def api_graph():
    return jsonify(_build_graph_data())


@app.route("/api/memories")
def api_memories():
    category = request.args.get("category")
    memories = memory_manager.list_memories(category=category if category else None)
    return jsonify([{
        "id": m.id,
        "category": m.category,
        "title": m.title,
        "content": m.content,
        "tags": m.tags,
        "confidence": m.confidence,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "color": CATEGORY_COLORS.get(m.category, "#A0A4B8"),
    } for m in memories])


@app.route("/api/memory/<memory_id>")
def api_get_memory(memory_id):
    mem = memory_manager.get_memory(memory_id)
    if not mem:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "id": mem.id,
        "category": mem.category,
        "title": mem.title,
        "content": mem.content,
        "tags": mem.tags,
        "confidence": mem.confidence,
        "created_at": mem.created_at,
        "updated_at": mem.updated_at,
        "source_conversation": mem.source_conversation,
        "color": CATEGORY_COLORS.get(mem.category, "#A0A4B8"),
    })


@app.route("/api/memory", methods=["POST"])
def api_create_memory():
    data = request.json
    if not data or not data.get("content"):
        return jsonify({"error": "Content required"}), 400
    category = data.get("category", "general")
    if category not in Config.CATEGORIES:
        memory_manager.add_category(category)
    item, status = memory_manager.add_or_update_memory(
        category=category,
        title=data.get("title", "User Note"),
        content=data["content"],
        tags=data.get("tags", [category]),
        confidence=float(data.get("confidence", 0.9)),
    )
    return jsonify({"id": item.id, "status": status, "category": item.category})


@app.route("/api/memory/<memory_id>", methods=["PUT"])
def api_update_memory(memory_id):
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400
    item = memory_manager.update_memory_content(
        memory_id,
        new_content=data.get("content", ""),
        new_title=data.get("title"),
    )
    if not item:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"id": item.id, "status": "updated"})


@app.route("/api/memory/<memory_id>", methods=["DELETE"])
def api_delete_memory(memory_id):
    success = memory_manager.delete_memory(memory_id)
    return jsonify({"deleted": success})


@app.route("/api/categories")
def api_categories():
    result = []
    for cat in Config.CATEGORIES:
        count = len(memory_manager.list_memories(category=cat))
        result.append({
            "name": cat,
            "count": count,
            "color": CATEGORY_COLORS.get(cat, "#A0A4B8"),
        })
    return jsonify(result)


# ── Chat & Conversation Endpoints ──

@app.route("/api/conversations")
def api_list_conversations():
    search = request.args.get("search", "").strip().lower()
    limit = int(request.args.get("limit", 50))
    convs = conversation_manager.list_conversations(limit=limit)
    if search:
        convs = [c for c in convs if search in c.get("title", "").lower()]
    return jsonify(convs)


@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    data = request.json or {}
    title = data.get("title", "New Conversation")
    mode = data.get("mode", "both")
    conv_id = conversation_manager.create_conversation(title=title, mode=mode)
    return jsonify({"id": conv_id, "title": title, "mode": mode, "status": "active"})


@app.route("/api/conversations/<conv_id>")
def api_get_conversation(conv_id):
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    messages = conversation_manager.get_recent_messages(conv_id, limit=100)
    return jsonify({
        "conversation": conv,
        "messages": messages
    })


@app.route("/api/conversations/<conv_id>/mode", methods=["PUT"])
def api_update_conversation_mode(conv_id):
    data = request.json or {}
    mode = data.get("mode", "both")
    success = conversation_manager.update_conversation_mode(conv_id, mode)
    if not success:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"id": conv_id, "mode": mode, "status": "updated"})


@app.route("/api/conversations/<conv_id>/title", methods=["PUT"])
def api_update_conversation_title(conv_id):
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    conversation_manager.update_conversation_title(conv_id, title)
    return jsonify({"id": conv_id, "title": title, "status": "updated"})


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    success = conversation_manager.delete_conversation(conv_id)
    return jsonify({"deleted": success})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message content is required"}), 400

    conversation_id = data.get("conversation_id")
    mode = data.get("mode")
    memory_write = data.get("memory_write")
    memory_retrieval = data.get("memory_retrieval")
    ai_provider = data.get("ai_provider")
    model = data.get("model")

    result = assistant.ask(
        question=message,
        conversation_id=conversation_id,
        memory_write=memory_write,
        memory_retrieval=memory_retrieval,
        mode=mode,
        ai_provider=ai_provider,
        model=model
    )

    retrieved_clean = []
    for m in result.get("retrieved_memories", []):
        item = m.get("item")
        cat = getattr(item, "category", m.get("category", "general"))
        retrieved_clean.append({
            "id": getattr(item, "id", m.get("id", "")),
            "title": getattr(item, "title", m.get("title", "Note")),
            "category": cat,
            "content": getattr(item, "content", m.get("content", "")),
            "confidence": getattr(item, "confidence", m.get("confidence", 0.9)),
            "score": m.get("score", 0.0),
            "color": CATEGORY_COLORS.get(cat, "#A0A4B8")
        })

    return jsonify({
        "answer": result.get("answer", ""),
        "is_error": result.get("is_error", False),
        "error_type": result.get("error_type"),
        "conversation_id": result.get("conversation_id"),
        "mode": result.get("mode"),
        "memory_write": result.get("memory_write"),
        "memory_retrieval": result.get("memory_retrieval"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "retrieved_memories": retrieved_clean,
        "extracted_memory": {
            "id": result["extracted_memory"].id,
            "title": result["extracted_memory"].title,
            "category": result["extracted_memory"].category,
            "content": result["extracted_memory"].content,
        } if result.get("extracted_memory") else None
    })


@app.route("/api/config")
def api_config():
    active_p = assistant.ai_client.get_active_provider()
    return jsonify({
        "active_provider": active_p.provider_name,
        "active_model": getattr(active_p, "model_name", "default"),
        "providers": ["freellmapi", "gemini", "mock"],
        "categories": list(CATEGORY_COLORS.keys()),
        "embedding_provider": Config.EMBEDDING_PROVIDER
    })


# ── Token & Efficiency Optimization Endpoints ──

@app.route("/api/optimize", methods=["POST"])
def api_optimize_prompt():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt text is required"}), 400

    level_str = data.get("level", "balanced").upper()
    encoding_name = data.get("encoding", "cl100k_base")

    if ai_eff_available:
        try:
            level_map = {
                "SAFE": CompressionLevel.SAFE,
                "CONSERVATIVE": CompressionLevel.SAFE,
                "BALANCED": CompressionLevel.BALANCED,
                "AGGRESSIVE": CompressionLevel.AGGRESSIVE
            }
            comp_level = level_map.get(level_str, CompressionLevel.BALANCED)
            processed = ai_eff_pipeline.process(
                prompt=prompt,
                encoding_name=encoding_name,
                compression_level=comp_level
            )

            # Generate IR serializations
            ir_obj = ai_eff_ir_gen.generate_ir(processed.compressed_prompt)
            ir_formats = {
                "tagged_kv": ir_obj.serialize(IRFormat.TAGGED_KV),
                "minimal_json": ir_obj.serialize(IRFormat.MINIMAL_JSON),
                "compact_dsl": ir_obj.serialize(IRFormat.COMPACT_DSL),
            }

            # Industry Models Cost Comparison (monthly at 100k requests)
            calls = 100_000
            cost_models = [
                {"name": "GPT-4o (Omni Frontier)", "input_per_m": 2.50, "output_per_m": 10.00, "provider": "OpenAI"},
                {"name": "Claude 3.5 Sonnet", "input_per_m": 3.00, "output_per_m": 15.00, "provider": "Anthropic"},
                {"name": "Gemini 3.1 Flash Lite", "input_per_m": 0.075, "output_per_m": 0.30, "provider": "Google (FreeLLMAPI)"},
                {"name": "Llama 3.2 3B Instruct (Edge SLM)", "input_per_m": 0.0, "output_per_m": 0.0, "provider": "Local Edge Device"}
            ]
            cost_table = []
            for m in cost_models:
                raw_c = (processed.raw_tokens / 1_000_000.0) * m["input_per_m"] * calls
                comp_c = (processed.compressed_tokens / 1_000_000.0) * m["input_per_m"] * calls
                saved = max(0.0, raw_c - comp_c)
                pct = round((saved / raw_c * 100.0), 1) if raw_c > 0 else 0.0
                cost_table.append({
                    "model": m["name"],
                    "provider": m["provider"],
                    "raw_cost_monthly": round(raw_c, 2),
                    "optimized_cost_monthly": round(comp_c, 2),
                    "saved_monthly": round(saved, 2),
                    "savings_pct": pct,
                    "saved_annual": round(saved * 12, 2)
                })

            return jsonify({
                "original_prompt": processed.raw_prompt,
                "raw_prompt": processed.raw_prompt,
                "compressed_prompt": processed.compressed_prompt,
                "encoding_used": processed.encoding_used,
                "raw_tokens": processed.raw_tokens,
                "compressed_tokens": processed.compressed_tokens,
                "tokens_saved": processed.tokens_saved,
                "token_reduction_pct": processed.token_reduction_pct,
                "total_pipeline_latency_ms": processed.total_pipeline_latency_ms,
                "subword_comparison": processed.subword_comparison,
                "multilingual_fragmentation": processed.multilingual_fragmentation,
                "information_density_gain": processed.information_density_gain,
                "ast_code_summaries": processed.ast_code_summaries,
                "rules_applied": processed.rules_applied,
                "routing_decision": processed.routing_decision,
                "prompt_ir": {
                    "serialized": ir_formats,
                    "intent": ir_obj.intent.value if hasattr(ir_obj.intent, 'value') else str(ir_obj.intent),
                    "action": ir_obj.action,
                    "target": ir_obj.target,
                    "constraints": ir_obj.constraints,
                    "code_snippets": ir_obj.code_snippets
                },
                "cost_table": cost_table
            })
        except Exception as e:
            print(f"[Warning] ai_efficiency pipeline error: {e}, falling back to PromptOptimizer")

    # Fallback optimizer
    result = prompt_optimizer.optimize(prompt, level=level_str.lower())
    return jsonify(result)


@app.route("/api/optimize/ab-test", methods=["POST"])
def api_optimize_ab_test():
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt text is required"}), 400

    model = data.get("model", "gemini-3.1-flash-lite")
    force_sim = data.get("force_simulation", False)

    if not ai_eff_available:
        return jsonify({"error": "ai_efficiency module not available"}), 503

    try:
        settings = load_ai_settings()
        llm_client = LLMClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=model,
            provider="freellmapi"
        )
        tester = ABFidelityTester(
            tokenizer=ai_eff_pipeline.tokenizer,
            compressor=ai_eff_pipeline.compressor,
            llm_client=llm_client
        )
        res: ABTestResult = tester.run_ab_test(
            raw_prompt=prompt,
            model_id=model,
            force_simulation=force_sim
        )
        return jsonify(res.to_dict())
    except Exception as e:
        return jsonify({"error": str(e), "is_error": True}), 500


@app.route("/api/optimize/code-edit", methods=["POST"])
def api_optimize_code_edit():
    data = request.json or {}
    code = data.get("code", "").strip()
    instruction = data.get("instruction", "").strip()
    tier_name = data.get("tier")

    if not code or not instruction:
        return jsonify({"error": "Both code and instruction are required"}), 400

    if not ai_eff_available:
        return jsonify({"error": "ai_efficiency module not available"}), 503

    tier = None
    if tier_name:
        try:
            tier = EditTier(tier_name.upper())
        except Exception:
            pass

    try:
        edit_result: CodeEditResult = ai_eff_code_editor.edit_code(
            code=code,
            instruction=instruction,
            force_tier=tier
        )
        return jsonify(edit_result.to_dict())
    except Exception as e:
        return jsonify({"error": str(e), "is_error": True}), 500


@app.route("/api/optimize/benchmarks", methods=["GET"])
def api_get_benchmarks():
    if ai_eff_available:
        return jsonify(AI_EFF_BENCHMARKS)
    return jsonify(BENCHMARK_PROMPTS)


if __name__ == "__main__":
    print("\n  [*] Memory Graph View starting...")
    print(f"  [*] Memory Directory: {MEMORY_DIR}")
    print(f"  [*] Open in browser: http://localhost:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
