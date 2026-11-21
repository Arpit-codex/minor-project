"""
Similarity Engine module.
Provides hybrid semantic and lexical similarity search.
Uses online embedding API (Gemini text-embedding-004) when API key is available,
with a zero-download n-gram TF-IDF + BM25-style lexical fallback.
Includes persistent caching to avoid redundant API requests.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import Config, MEMORY_INDEX_DIR

CACHE_FILE = MEMORY_INDEX_DIR / "embeddings_cache.json"


class SimilarityEngine:
    """Hybrid semantic and lexical similarity engine."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.cache_file = CACHE_FILE
        self.embedding_cache: Dict[str, List[float]] = self._load_cache()
        self._genai_client = None

        if self.api_key and Config.EMBEDDING_PROVIDER in ("auto", "gemini"):
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
            except Exception:
                self._genai_client = None

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cached embeddings from local storage."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Save embedding cache to local file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.embedding_cache, f)
        except Exception:
            pass

    def _get_online_embedding(self, text: str) -> Optional[List[float]]:
        """Fetch dense embedding from FreeLLMAPI or Gemini API."""
        # 1. Try FreeLLMAPI if configured
        if Config.FREELLMAPI_KEY and Config.EMBEDDING_PROVIDER in ("auto", "freellmapi"):
            try:
                import requests
                url = f"{Config.FREELLMAPI_BASE_URL}/embeddings"
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {Config.FREELLMAPI_KEY}", "Content-Type": "application/json"},
                    json={"model": Config.FREELLMAPI_EMBEDDING_MODEL, "input": text},
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    emb_list = data.get("data", [])
                    if emb_list and "embedding" in emb_list[0]:
                        return list(emb_list[0]["embedding"])
            except Exception:
                pass

        # 2. Try Gemini API if configured
        if self._genai_client and Config.EMBEDDING_PROVIDER in ("auto", "gemini"):
            try:
                response = self._genai_client.models.embed_content(
                    model=Config.GEMINI_EMBEDDING_MODEL,
                    contents=text,
                )
                if response and hasattr(response, "embeddings") and response.embeddings:
                    values = response.embeddings[0].values
                    return list(values)
            except Exception:
                pass

        return None

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with caching (online model)."""
        clean_text = text.strip()
        if not clean_text:
            return None

        if clean_text in self.embedding_cache:
            return self.embedding_cache[clean_text]

        embedding = self._get_online_embedding(clean_text)
        if embedding:
            self.embedding_cache[clean_text] = embedding
            self._save_cache()
            return embedding
        return None

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization for lexical overlap."""
        return re.findall(r"\b\w+\b", text.lower())

    def _lexical_similarity(self, query: str, target: str) -> float:
        """Compute Jaccard token overlap between two strings."""
        q_tokens = set(self._tokenize(query))
        t_tokens = set(self._tokenize(target))
        if not q_tokens or not t_tokens:
            return 0.0
        intersection = q_tokens.intersection(t_tokens)
        union = q_tokens.union(t_tokens)
        return len(intersection) / len(union)

    COMMON_SYNONYMS = {
        # Tech terms
        "llm": "language model ai model large language model",
        "notebook": "laptop",
        "notebook computer": "laptop pc",
        "laptop": "notebook computer pc",
        "pc": "computer laptop",
        "efficiently": "optimize performance faster less memory",
        "optimize": "improve efficiency faster less ram performance",
        "ml": "machine learning ai",
        "db": "database storage sql",
        "cs": "computer science",
        "ai": "artificial intelligence machine learning",
        "nlp": "natural language processing text",
        # Personal fact terms — so queries find matching memories
        "college": "university campus degree institute",
        "university": "college campus degree institute",
        "nit": "national institute of technology college university",
        "iit": "indian institute of technology college university",
        "job": "career employment role work profession",
        "exam": "test finals midterm revision",
        "age": "years old born old how old",
        "years old": "age born",
        "name": "called known as",
        "goal": "aspiration dream want to become future plan",
        "engineer": "developer programmer software developer",
        "ai engineer": "artificial intelligence engineer ml engineer machine learning engineer",
        "studying": "enrolled student learning taking classes",
        "interest": "hobby enjoy like passionate about",
    }

    def _normalize_and_expand(self, text: str) -> str:
        """Expand common acronyms and technical synonyms for robust local similarity."""
        lower = text.lower()
        for term, expansion in self.COMMON_SYNONYMS.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, lower):
                lower = re.sub(pattern, f"{term} {expansion}", lower)
        return lower

    def _local_tfidf_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """
        Fast zero-download local semantic similarity using subword character
        and word n-grams. Captures root stems, synonyms, and variations.
        """
        if not candidates:
            return []

        expanded_query = self._normalize_and_expand(query)
        expanded_candidates = [self._normalize_and_expand(c) for c in candidates]

        corpus = [expanded_query] + expanded_candidates
        try:
            # Word vectorizer with unigrams & bigrams
            word_vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
            word_mat = word_vec.fit_transform(corpus)
            word_scores = cosine_similarity(word_mat[0:1], word_mat[1:]).flatten()

            # Character n-gram vectorizer
            char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
            char_mat = char_vec.fit_transform(corpus)
            char_scores = cosine_similarity(char_mat[0:1], char_mat[1:]).flatten()

            # Blend word n-gram and char n-gram scores
            blended = (0.60 * word_scores) + (0.40 * char_scores)
            return [float(s) for s in blended]
        except Exception:
            return [self._lexical_similarity(query, c) for c in candidates]

    def compute_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """
        Compute similarity scores between a query and a list of candidates.
        Uses online Gemini embeddings when available, blended with keyword overlap.
        Falls back to local n-gram TF-IDF + lexical matching seamlessly.
        """
        if not candidates or not query.strip():
            return [0.0] * len(candidates)

        query_emb = self.get_embedding(query)
        if query_emb is not None:
            # We have online embeddings!
            q_vec = np.array(query_emb).reshape(1, -1)
            scores = []
            for candidate in candidates:
                c_emb = self.get_embedding(candidate)
                if c_emb is not None:
                    c_vec = np.array(c_emb).reshape(1, -1)
                    raw_cos = float(cosine_similarity(q_vec, c_vec)[0][0])
                    # Rescale: dense embeddings have ~0.40 baseline for unrelated texts.
                    # Map [0.40, 1.0] -> [0.0, 1.0] so unrelated content maps to ~0.0
                    baseline = 0.40
                    sim = max(0.0, min(1.0, (raw_cos - baseline) / (1.0 - baseline))) if raw_cos > baseline else 0.0
                else:
                    # Fallback for this single candidate
                    sim = self._lexical_similarity(query, candidate)

                # Boost slightly if exact keyword overlap is high
                lex = self._lexical_similarity(query, candidate)
                blended = (0.75 * sim) + (0.25 * lex)
                scores.append(round(blended, 4))
            return scores

        # Local fallback mode (zero download, fast & lightweight)
        tfidf_scores = self._local_tfidf_similarity(query, candidates)
        blended_scores = []
        for i, candidate in enumerate(candidates):
            lex = self._lexical_similarity(query, candidate)
            tfidf = tfidf_scores[i] if i < len(tfidf_scores) else 0.0
            # Blend subword character n-gram TF-IDF with exact word lexical match
            blended = (0.65 * tfidf) + (0.35 * lex)
            blended_scores.append(round(blended, 4))

        return blended_scores

    def find_similar_questions(
        self,
        query: str,
        past_questions: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search past questions for semantic similarity.
        Returns matching records above threshold with added 'similarity_score'.
        """
        if not past_questions or not query.strip():
            return []

        min_thresh = threshold if threshold is not None else Config.SIMILARITY_THRESHOLD
        candidate_texts = [q.get("question", "") for q in past_questions]
        scores = self.compute_similarity(query, candidate_texts)

        results = []
        for q_dict, score in zip(past_questions, scores):
            if score >= min_thresh:
                item = dict(q_dict)
                item["similarity_score"] = score
                results.append(item)

        # Sort descending by similarity score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
