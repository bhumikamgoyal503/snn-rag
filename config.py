"""
Central configuration for the SNN-RAG project.
All tunables live here so experiments are reproducible.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"          # "cuda" if available
    batch_size: int = 64
    normalize: bool = True       # L2 normalize for cosine sim


@dataclass
class RetrieverConfig:
    top_k: int = 10              # docs retrieved per hop
    score_threshold: float = 0.0 # minimum similarity to keep


@dataclass
class RerankerConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5              # docs kept after reranking


@dataclass
class GeneratorConfig:
    # Supports "huggingface" or "api" backends
    backend: str = "huggingface"
    model_name: str = "google/flan-t5-base"   # small enough for CPU dev
    max_new_tokens: int = 256
    temperature: float = 0.1
    # For API backend (e.g., local Ollama or remote)
    api_url: str = "http://localhost:11434/api/generate"
    api_model: str = "mistral"


@dataclass
class QPConfig:
    """Phase 3-4 QP formulation config.

    Both Q and c are pre-normalized onto comparable scales in
    solvers/qp_utils.build_qp_matrices before these weights are applied
    (see that module's docstring), so redundancy_weight and
    relevance_weight are clean, comparable knobs rather than fighting
    uncontrolled matrix magnitudes.
    """
    redundancy_weight: float = 1.0   # scaling factor on the normalized Q
    relevance_weight: float = 1.0    # scaling factor on normalized relevance in c
    retain_threshold: float = 0.5    # x_i > threshold => keep doc
    # Cosine similarity below this is treated as "different topic" and pays
    # no redundancy penalty at all. Only pairs at or above it (near-duplicate
    # paragraphs) are penalized. Without this, topically-adjacent-but-distinct
    # docs (exactly what a bridge question's two gold docs look like) get
    # penalized as if they were duplicates. See solvers/qp_utils.py.
    #
    # Calibrated empirically against all-MiniLM-L6-v2 cosine similarities on
    # real HotpotQA distractor paragraphs: median pairwise sim ~0.23, p90
    # ~0.53, and genuine near-duplicates (e.g. "Ed Wood" vs "Ed Wood (film)")
    # land around 0.65-0.70. 0.8 (a common default elsewhere) turned out to
    # be far too high for this model/domain and left the redundancy penalty
    # never firing at all. Re-calibrate if the embedding model changes.
    redundancy_similarity_threshold: float = 0.6


@dataclass
class PipelineConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    qp: QPConfig = field(default_factory=QPConfig)

    data_dir: Path = Path("data/hotpotqa")
    results_dir: Path = Path("results")
    max_context_docs: int = 5       # docs fed to the LLM
    max_hops: int = 1               # Phase 1 = single-shot (1 hop)
