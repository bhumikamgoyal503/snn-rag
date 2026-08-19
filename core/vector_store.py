"""
Vector store backed by FAISS.

Stores document texts alongside their vectors so retrieval returns
full content, not just IDs.  The interface is deliberately minimal —
index, search, reset — so it maps to Chroma/Qdrant if we swap later.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from snn_rag.core.embedder import Embedder


@dataclass
class RetrievedDoc:
    """A single retrieval result."""
    doc_id: int
    text: str
    score: float           # similarity score (higher = more relevant)
    embedding: np.ndarray  # kept around for Phase 3 (redundancy matrix)


class FAISSVectorStore:
    """In-memory FAISS index with cosine similarity (via IP on L2-normed vecs)."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.dim = embedder.dim
        # Inner-product index on L2-normalized vectors ≡ cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_documents(self, texts: list[str], show_progress: bool = False) -> None:
        """Embed and index a batch of documents."""
        embs = self.embedder.encode(texts, show_progress=show_progress)
        self.index.add(embs)
        self.documents.extend(texts)
        # Cache embeddings for Phase 3 redundancy matrix
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

    def add_precomputed(self, texts: list[str], embs: np.ndarray) -> None:
        """Index already-encoded vectors (avoids double encoding)."""
        self.index.add(embs)
        self.documents.extend(texts)
        if self.embeddings is None:
            self.embeddings = embs
        else:
            self.embeddings = np.vstack([self.embeddings, embs])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> list[RetrievedDoc]:
        """Return top-k docs for a single query string."""
        q_emb = self.embedder.encode([query])
        scores, ids = self.index.search(q_emb, top_k)
        results: list[RetrievedDoc] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < score_threshold:
                continue
            results.append(
                RetrievedDoc(
                    doc_id=int(idx),
                    text=self.documents[idx],
                    score=float(score),
                    embedding=self.embeddings[idx],
                )
            )
        return results

    def search_by_vector(
        self,
        query_emb: np.ndarray,
        top_k: int = 10,
    ) -> list[RetrievedDoc]:
        """Search with a pre-computed query vector."""
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        scores, ids = self.index.search(query_emb, top_k)
        results: list[RetrievedDoc] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append(
                RetrievedDoc(
                    doc_id=int(idx),
                    text=self.documents[idx],
                    score=float(score),
                    embedding=self.embeddings[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.index.reset()
        self.documents.clear()
        self.embeddings = None

    def __len__(self) -> int:
        return self.index.ntotal
