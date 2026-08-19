"""
Phase 1 Evaluation: Single-shot RAG baseline on HotpotQA (distractor).

Usage:
    python -m snn_rag.run_phase1_eval --max_examples 50

This script:
  1. Loads HotpotQA validation examples
  2. For each example, indexes the 10 context paragraphs into FAISS
  3. Retrieves top-k with optional cross-encoder reranking
  4. Generates an answer with the LLM
  5. Computes EM, F1, retrieval precision/recall, and latency
  6. Dumps per-example and aggregate results to JSON

The results become the baseline that Phases 2-5 improve upon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from snn_rag.config import PipelineConfig
from snn_rag.core.embedder import Embedder
from snn_rag.core.vector_store import FAISSVectorStore
from snn_rag.core.retriever import SingleShotRetriever
from snn_rag.core.generator import Generator
from snn_rag.core.pipeline import RAGPipeline
from snn_rag.data.hotpotqa_loader import load_hotpotqa
from snn_rag.evaluation.metrics import (
    exact_match,
    token_f1,
    retrieval_precision,
    retrieval_recall,
)


def parse_title_from_doc(doc_text: str) -> str:
    """Extract title from our 'Title: body...' format."""
    if ":" in doc_text:
        return doc_text.split(":", 1)[0].strip()
    return doc_text[:50]


def run_eval(cfg: PipelineConfig, max_examples: int = 50) -> dict:
    print(f"[Phase 1] Loading HotpotQA validation (max {max_examples})...")
    examples = load_hotpotqa(split="validation", max_examples=max_examples)
    print(f"  Loaded {len(examples)} examples.")

    # Shared embedder (loaded once)
    embedder = Embedder(cfg.embedding)
    generator = Generator(cfg.generator)

    results = []
    t_total = perf_counter()

    for ex in tqdm(examples, desc="Phase 1 eval"):
        # -- Per-example FAISS index (10 context docs per question) --
        store = FAISSVectorStore(embedder)
        store.add_documents(ex.context_docs)

        retriever = SingleShotRetriever(
            store=store,
            cfg=cfg.retriever,
            reranker_cfg=cfg.reranker,
        )
        pipeline = RAGPipeline(retriever, generator, cfg)

        # -- Run --
        output = pipeline.run(ex.question)

        # -- Extract titles for retrieval metrics --
        retrieved_titles = [
            parse_title_from_doc(d.text) for d in output.retrieval.docs
        ]

        # -- Score --
        em = exact_match(output.answer, ex.answer)
        f1 = token_f1(output.answer, ex.answer)
        r_prec = retrieval_precision(retrieved_titles, ex.gold_titles)
        r_rec = retrieval_recall(retrieved_titles, ex.gold_titles)

        results.append({
            "qid": ex.qid,
            "question": ex.question,
            "gold_answer": ex.answer,
            "predicted_answer": output.answer,
            "exact_match": em,
            "token_f1": f1,
            "retrieval_precision": r_prec,
            "retrieval_recall": r_rec,
            "num_hops": output.retrieval.num_hops,
            "num_docs_retrieved": len(output.retrieval.docs),
            "latency_s": output.latency_s,
            "level": ex.level,
            "type": ex.question_type,
        })

    elapsed = perf_counter() - t_total

    # -- Aggregate --
    n = len(results)
    agg = {
        "phase": 1,
        "description": "Single-shot RAG baseline",
        "n_examples": n,
        "mean_exact_match": sum(r["exact_match"] for r in results) / n,
        "mean_token_f1": sum(r["token_f1"] for r in results) / n,
        "mean_retrieval_precision": sum(r["retrieval_precision"] for r in results) / n,
        "mean_retrieval_recall": sum(r["retrieval_recall"] for r in results) / n,
        "mean_latency_s": sum(r["latency_s"] for r in results) / n,
        "total_time_s": elapsed,
        "config": {
            "embedding_model": cfg.embedding.model_name,
            "generator_model": cfg.generator.model_name,
            "retriever_top_k": cfg.retriever.top_k,
            "reranker_enabled": cfg.reranker.enabled,
            "reranker_top_k": cfg.reranker.top_k,
            "max_context_docs": cfg.max_context_docs,
        },
    }

    # -- Save --
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_results.json"
    with open(out_path, "w") as f:
        json.dump({"aggregate": agg, "per_example": results}, f, indent=2)
    print(f"\n[Phase 1] Results saved to {out_path}")
    print(f"  EM:  {agg['mean_exact_match']:.3f}")
    print(f"  F1:  {agg['mean_token_f1']:.3f}")
    print(f"  Retrieval Recall: {agg['mean_retrieval_recall']:.3f}")
    print(f"  Avg Latency: {agg['mean_latency_s']:.2f}s")

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: single-shot RAG eval")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument("--generator_model", type=str, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.no_reranker:
        cfg.reranker.enabled = False
    if args.generator_model:
        cfg.generator.model_name = args.generator_model
    run_eval(cfg, max_examples=args.max_examples)
