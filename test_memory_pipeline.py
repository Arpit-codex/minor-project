"""Full validation test for memory pipeline fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory_classifier import MemoryClassifier
from core.memory_manager import MemoryManager
from core.similarity_engine import SimilarityEngine
from core.memory_retriever import MemoryRetriever

mm = MemoryManager()
se = SimilarityEngine()
mc = MemoryClassifier(mm)
mr = MemoryRetriever(mm, se)

# Test 1: Should NOT store — these are questions or noise
questions = [
    "How can I optimize a small language model for my laptop?",
    "What courses should I focus on for my goal?",
    "What is the capital of France?",
    "hello",
    "ok thanks",
    "tell me about machine learning",
    "how do I install Python?",
]
print("=== CLASSIFIER: Should NOT store ===")
all_ok = True
for q in questions:
    result = mc.heuristic_classify(q)
    status = "STORED (BUG!)" if result else "SKIPPED (correct)"
    if result:
        all_ok = False
    print(f"  [{status}] {q[:65]}")

# Test 2: Should STORE personal facts
facts = [
    "I am studying Computer Science at NIT",
    "My name is Rahul",
    "I am 21 years old",
    "I prefer dark mode responses",
    "I want to become an AI engineer",
    "Remember: I use Python 3.11 for all my projects",
]
print()
print("=== CLASSIFIER: Should store ===")
for f in facts:
    result = mc.heuristic_classify(f)
    if result:
        print(f"  [STORED -> {result['category']}] {f[:65]}")
    else:
        print(f"  [SKIPPED (BUG!)] {f[:65]}")
        all_ok = False

# Test 3: Retrieval for personal queries
print()
print("=== RETRIEVER: Personal fact retrieval ===")
queries = [
    "what college do I attend",
    "how old am I",
    "what is my goal",
    "what do you know about me",
    "where am I studying",
]
for query in queries:
    results = mr.search_memories(query, top_k=3)
    if results:
        print(f"  Query: \"{query}\"")
        for r in results:
            print(f"    -> [{r['category']}] {r['content'][:65]} (score={r['score']:.3f})")
    else:
        print(f"  Query: \"{query}\" -> NO RESULTS (check retrieval)")

# Test 4: Make sure "what do you know about me" intent is detected
print()
print("=== INTENT DETECTOR: Memory reference patterns ===")
from core.intent_detector import IntentDetector
det = IntentDetector()
mem_queries = [
    "what do you know about me",
    "what have you learned about me",
    "tell me about me",
    "do you know anything about me",
    "my preferences",
]
non_mem = ["hello", "how are you", "what is machine learning"]
for q in mem_queries:
    print(f"  [{'MEM OK' if det.is_memory_query(q) else 'MISS BUG'}] {q}")
for q in non_mem:
    print(f"  [{'BUG-WRONG' if det.is_memory_query(q) else 'NOT MEM OK'}] {q}")

print()
if all_ok:
    print("All classifier tests PASSED.")
else:
    print("Some tests FAILED - check output above.")
