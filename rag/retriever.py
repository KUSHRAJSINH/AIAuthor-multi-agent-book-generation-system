"""
Hybrid retriever — combines FAISS dense search with BM25 sparse retrieval,
then reranks results by score fusion.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Tuple

from rag.embedder import Embedder
from rag.faiss_store import FAISSStore


class BM25:
    """Lightweight BM25 implementation (no external dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[str] = []
        self._tf: List[dict] = []
        self._idf: dict = {}
        self._avgdl: float = 0.0

    def fit(self, corpus: List[str]) -> None:
        self._corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self._avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
        df: dict = defaultdict(int)
        self._tf = []
        for tokens in tokenized:
            tf: dict = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            self._tf.append(dict(tf))
            for tok in set(tokens):
                df[tok] += 1
        N = len(corpus)
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        tokens = query.lower().split()
        tf_doc = self._tf[doc_idx] if doc_idx < len(self._tf) else {}
        dl = sum(tf_doc.values())
        s = 0.0
        for tok in tokens:
            if tok not in tf_doc:
                continue
            idf = self._idf.get(tok, 0.0)
            tf = tf_doc[tok]
            s += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
            )
        return s

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self._corpus:
            return []
        scores = [(i, self.score(query, i)) for i in range(len(self._corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._corpus[i], sc) for i, sc in scores[:top_k]]


class HybridRetriever:
    """Fuses FAISS + BM25 results using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, faiss_store: FAISSStore, embedder: Embedder, rrf_k: int = 60):
        self.faiss = faiss_store
        self.embedder = embedder
        self.bm25 = BM25()
        self.rrf_k = rrf_k
        self._indexed_texts: List[str] = []

    def index(self, texts: List[str]) -> None:
        """Add texts to both FAISS and BM25."""
        if not texts:
            return
        embeddings = self.embedder.embed(texts)
        self.faiss.add(texts, embeddings)
        self._indexed_texts.extend(texts)
        self.bm25.fit(self._indexed_texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Hybrid retrieval with RRF score fusion."""
        query_emb = self.embedder.embed_one(query)

        dense_results = self.faiss.search(query_emb, top_k=top_k * 2)
        sparse_results = self.bm25.retrieve(query, top_k=top_k * 2)

        # RRF fusion
        rrf_scores: dict = defaultdict(float)

        for rank, (text, _) in enumerate(dense_results, start=1):
            rrf_scores[text] += 1.0 / (self.rrf_k + rank)

        for rank, (text, _) in enumerate(sparse_results, start=1):
            rrf_scores[text] += 1.0 / (self.rrf_k + rank)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def retrieve_texts(self, query: str, top_k: int = 5) -> List[str]:
        results = self.retrieve(query, top_k=top_k)
        return [text for text, _ in results]
