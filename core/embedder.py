"""
Embedding model wrapper.
Thin abstraction over sentence-transformers so we can swap models easily
and reuse the same interface for both indexing and query-time encoding.
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from snn_rag.config import EmbeddingConfig


class Embedder:
    """Encode text to dense vectors using a SentenceTransformer model."""

    def __init__(self, cfg: EmbeddingConfig | None = None) -> None:
        self.cfg = cfg or EmbeddingConfig()
        self.model = SentenceTransformer(
            self.cfg.model_name, device=self.cfg.device
        )

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return (N, D) float32 array, optionally L2-normalized."""
        embs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        return embs.astype(np.float32)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
