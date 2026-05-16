"""
Embedder — wraps sentence-transformers for local dense embeddings.
"""
from __future__ import annotations

from typing import List
import numpy as np
import hashlib


class DeterministicEmbedder:
    """Fallback hash-based embedder."""
    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            # Generate deterministic vector from hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            rng = np.random.default_rng(seed=int.from_bytes(hash_bytes[:8], "big"))
            vec = rng.standard_normal(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)


class Embedder:
    """Lazy-loads sentence-transformers model with fallback."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._is_fallback = False

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._is_fallback = False
        except Exception:
            # Catches OSError [WinError 206] filename too long (torch/Anaconda
            # deep path on Windows), ImportError, and any other load failure.
            self._model = DeterministicEmbedder()
            self._is_fallback = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a (N, D) float32 embedding matrix."""
        self._load()
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        
        if self._is_fallback:
            return self._model.embed(texts)
            
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            import sys
            print(f"[Embedder] PyTorch execution failed ({e}). Switching to fallback.", file=sys.stderr)
            self._model = DeterministicEmbedder()
            self._is_fallback = True
            return self._model.embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        if self._is_fallback:
            return 384
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception:
            self._model = DeterministicEmbedder()
            self._is_fallback = True
            return 384
