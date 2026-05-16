"""
FAISS vector store — manages index creation, persistence, and search.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

INDEX_DIR = Path("rag/indices")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class FAISSStore:
    """
    Wraps a FAISS flat L2 index with a parallel text corpus for retrieval.
    Uses cosine similarity via pre-normalised embeddings.
    """

    def __init__(self, session_id: str, dim: int = 384):
        self.session_id = session_id
        self.dim = dim
        self._index = None
        self._texts: List[str] = []
        self._index_path = INDEX_DIR / f"{session_id}.index"
        self._texts_path = INDEX_DIR / f"{session_id}.texts.pkl"

    def _build_index(self):
        import faiss
        self._index = faiss.IndexFlatIP(self.dim)  # inner product = cosine for normalised vecs

    def add(self, texts: List[str], embeddings: np.ndarray) -> None:
        if self._index is None:
            self._build_index()
        self._index.add(embeddings)
        self._texts.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        q = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._texts):
                results.append((self._texts[idx], float(score)))
        return results

    def save(self) -> None:
        if self._index is None:
            return
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        with open(self._texts_path, "wb") as f:
            pickle.dump(self._texts, f)

    def load(self) -> bool:
        if not self._index_path.exists() or not self._texts_path.exists():
            return False
        import faiss
        self._index = faiss.read_index(str(self._index_path))
        with open(self._texts_path, "rb") as f:
            self._texts = pickle.load(f)
        return True

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0
