"""
Phase 5 Evaluation: ablation across retrieval strategies on HotpotQA (distractor).

Ablation arms, all run on the SAME per-example FAISS index so the
comparison is apples-to-apples:
  1. single_shot          - Phase 1 baseline (one retrieval pass)
  2. multi_hop             - Phase 2 planner, no QP doc-selection
  3. multi_hop_qp_classical - Phase 2 planner + Phase 3 cvxpy/OSQP solver
  4. multi_hop_qp_snn       - Phase 2 planner + Phase 4 SNN solver
                              (always also runs the classical solver
                              internally and reports the gap - see
                              solvers/snn_solver.py)

Metrics per arm: EM, F1, retrieval precision/recall, mean retrieval
calls per question (= mean hops), mean latency. QP arms additionally
report mean docs retained and solver timing; the SNN arm additionally
reports the SNN-vs-classical solution gap (objective, ||x diff||,
selection agreement) as a first-class result, not an afterthought.

Published ReAct / IRCoT numbers are included in the output for context
ONLY, cited from their papers - see CITED_BASELINES below. They were
NOT reproduced on this machine, use much larger backbone LLMs (PaLM-540B,
GPT-3, Flan-T5-XXL) than this project's flan-t5-base, and are not a
like-for-like comparison. Treat them as orientation, not a leaderboard.

Usage:
    python -m snn_rag.run_phase5_eval --max_examples 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from snn_rag.config import PipelineConfig
from snn_rag.core.embedder import Embedder
from snn_rag.core.generator import Generator
from snn_rag.core.pipeline import RAGPipeline
from snn_rag.core.retriever import BaseRetriever, SingleShotRetriever
from snn_rag.core.vector_store import FAISSVectorStore
from snn_rag.data.hotpotqa_loader import load_hotpotqa
from snn_rag.data.musique_loader import load_musique
from snn_rag.evaluation.metrics import (
    exact_match,
    retrieval_precision_by_index,
    retrieval_recall_by_index,
    token_f1,
)
from snn_rag.evaluation.stats import bootstrap_ci, format_ci, paired_bootstrap_ci
from snn_rag.planning.multi_hop_planner import MultiHopRetriever
from snn_rag.solvers.classical import ClassicalQPSolver
from snn_rag.solvers.snn_solver import SNNQPSolver

ARMS = ["single_shot", "multi_hop", "multi_hop_qp_classical", "multi_hop_qp_snn"]

# Metrics that get a bootstrap CI per arm, and a paired CI against single-shot.
CI_METRICS = ["retrieval_precision", "exact_match", "token_f1", "retrieval_recall"]

DATASETS = {
    "hotpotqa": load_hotpotqa,
    "musique": load_musique,
}

# Cited from the published papers - never measured on this project's
# hardware/model. Kept separate from our own results in the output.
CITED_BASELINES = {
    "react_palm540b_hotpotqa_em": {
        "source": "Yao et al. 2023, ReAct: Synergizing Reasoning and Acting "
                   "in Language Models (ICLR 2023), Table 1. arXiv:2210.03629",
        "model": "PaLM-540B, HotpotQA validation set",
        "scores_em": {
            "Standard": 28.7, "CoT": 29.4, "CoT-SC": 33.4, "Act-only": 25.7,
            "ReAct": 27.4, "ReAct->CoT-SC": 35.1, "CoT-SC->ReAct": 34.2,
        },
    },
    "react_gpt3_hotpotqa_em": {
        "source": "Yao et al. 2023, ReAct (ICLR 2023), Appendix Table 5. "
                   "arXiv:2210.03629",
        "model": "GPT-3, 500 randomly sampled HotpotQA validation questions",
        "scores_em": {"ReAct": 30.8},
    },
    "ircot_hotpotqa_f1": {
        "source": "Trivedi et al. 2023, Interleaving Retrieval with "
                   "Chain-of-Thought Reasoning (ACL 2023), Table 4. "
                   "arXiv:2212.10509. Retriever: BM25/Elasticsearch, "
                   "500 sampled HotpotQA dev questions.",
        "scores_f1": {
            "GPT-3 OneR (one-step retrieval)": 53.6,
            "GPT-3 + IRCoT": 60.7,
            "Flan-T5-XXL OneR (one-step retrieval)": 43.1,
            "Flan-T5-XXL + IRCoT": 59.1,
        },
    },
    "caveat": (
        "These are orientation points, not a controlled comparison: they use "
        "PaLM-540B / GPT-3 / Flan-T5-XXL (11B+ params) as the reader, vs this "
        "project's flan-t5-base (~250M params), and different retrievers "
        "(BM25/Elasticsearch vs FAISS+MiniLM). A lower EM/F1 here reflects "
        "reader model scale at least as much as the retrieval strategy."
    ),
}


def parse_title_from_doc(doc_text: str) -> str:
    if ":" in doc_text:
        return doc_text.split(":", 1)[0].strip()
    return doc_text[:50]


def build_retriever(
    arm: str,
    store: FAISSVectorStore,
    generator: Generator,
    cfg: PipelineConfig,
    max_hops: int,
) -> BaseRetriever:
    if arm == "single_shot":
        return SingleShotRetriever(store, cfg.retriever, cfg.reranker)
    if arm == "multi_hop":
        return MultiHopRetriever(
            store, generator, cfg.retriever, cfg.reranker, max_hops=max_hops,
        )
    if arm == "multi_hop_qp_classical":
        return MultiHopRetriever(
            store, generator, cfg.retriever, cfg.reranker, max_hops=max_hops,
            doc_selector=ClassicalQPSolver(), qp_cfg=cfg.qp,
        )
    if arm == "multi_hop_qp_snn":
        return MultiHopRetriever(
            store, generator, cfg.retriever, cfg.reranker, max_hops=max_hops,
            doc_selector=SNNQPSolver(), qp_cfg=cfg.qp,
        )
    raise ValueError(f"Unknown arm: {arm}")


def extract_qp_stats(retrieval_metadata: dict) -> list[dict]:
    """Pull the per-hop QP solver record (if any) out of RetrievalResult.metadata."""
    hops = retrieval_metadata.get("hops", [])
    return [hop.qp for hop in hops if hop.qp is not None]


def infer_hop_bucket(example) -> str | None:
    """MuSiQue encodes hop count in the question ID and level field."""
    level = str(getattr(example, "level", "")).lower()
    if level.startswith("2hop"):
        return "2"
    if level.startswith("3hop"):
        return "3"
    if level.startswith("4hop"):
        return "4"
    return None


def filter_genuinely_multihop(
    candidates: list,
    embedder: Embedder,
    cfg: PipelineConfig,
    target_n: int,
) -> list:
    """Keep only examples where a single retrieval pass can't reach every
    gold supporting doc (single-shot recall < 1.0).

    On the full HotpotQA distractor split, single-shot top-k often already
    covers both gold paragraphs, so multi-hop has nothing to add and the
    ablation can't show a gain even if the mechanism works. No generator
    calls here - pure retrieval + FAISS - so filtering a large candidate
    pool is cheap relative to the generation-heavy ablation itself.
    """
    kept = []
    for ex in tqdm(candidates, desc="Filtering for genuinely multi-hop"):
        store = FAISSVectorStore(embedder)
        store.add_documents(ex.context_docs)
        docs = SingleShotRetriever(store, cfg.retriever, cfg.reranker).retrieve(ex.question).docs
        retrieved_doc_ids = [d.doc_id for d in docs]
        if retrieval_recall_by_index(retrieved_doc_ids, ex.gold_doc_ids) < 1.0:
            kept.append(ex)
        if len(kept) >= target_n:
            break
    return kept


def run_eval(
    cfg: PipelineConfig,
    max_examples: int,
    max_hops: int,
    genuinely_multihop: bool = False,
    candidate_pool: int = 300,
    dataset: str = "hotpotqa",
    results_name: str | None = None,
) -> dict:
    embedder = Embedder(cfg.embedding)
    loader = DATASETS[dataset]
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = results_name or f"phase5_{dataset}"
    checkpoint_path = out_dir / f"{output_stem}.partial.json"

    if genuinely_multihop:
        print(f"[Phase 5] Loading a candidate pool of {candidate_pool} {dataset} "
              f"validation examples to filter for genuinely multi-hop cases...")
        candidates = loader(split="validation", max_examples=candidate_pool)
        examples = filter_genuinely_multihop(candidates, embedder, cfg, max_examples)
        print(f"  Kept {len(examples)}/{len(candidates)} examples where single-shot "
              f"recall < 1.0 (single retrieval alone can't reach every gold doc).")
    else:
        print(f"[Phase 5] Loading {dataset} validation (max {max_examples})...")
        examples = loader(split="validation", max_examples=max_examples)
        print(f"  Loaded {len(examples)} examples.")

    generator = Generator(cfg.generator)

    per_arm_results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        if checkpoint.get("config", {}).get("dataset") != dataset:
            raise ValueError(f"Checkpoint dataset mismatch: {checkpoint_path}")
        per_arm_results = checkpoint["per_example"]
        completed_qids = {
            row["qid"] for row in per_arm_results["single_shot"]
        }
        examples = [ex for ex in examples if ex.qid not in completed_qids]
        print(
            f"  Resuming from {checkpoint_path}: "
            f"{len(completed_qids)} examples already complete."
        )
    t_total = perf_counter()

    for ex in tqdm(examples, desc="Phase 5 ablation"):
        store = FAISSVectorStore(embedder)
        store.add_documents(ex.context_docs)

        for arm in ARMS:
            retriever = build_retriever(arm, store, generator, cfg, max_hops)
            pipeline = RAGPipeline(retriever, generator, cfg)
            output = pipeline.run(ex.question)

            retrieved_doc_ids = [d.doc_id for d in output.retrieval.docs]
            qp_records = extract_qp_stats(output.retrieval.metadata)

            per_arm_results[arm].append({
                "qid": ex.qid,
                "exact_match": exact_match(output.answer, ex.answer),
                "token_f1": token_f1(output.answer, ex.answer),
                "retrieval_precision": retrieval_precision_by_index(retrieved_doc_ids, ex.gold_doc_ids),
                "retrieval_recall": retrieval_recall_by_index(retrieved_doc_ids, ex.gold_doc_ids),
                "num_hops": output.retrieval.num_hops,
                "num_docs_final": len(output.retrieval.docs),
                "latency_s": output.latency_s,
                "hop_bucket": infer_hop_bucket(ex),
                "qp_records": qp_records,
            })

        with open(checkpoint_path, "w") as f:
            json.dump({
                "phase": 5,
                "config": {
                    "dataset": dataset,
                    "embedding_model": cfg.embedding.model_name,
                    "generator_model": cfg.generator.model_name,
                    "retriever_top_k": cfg.retriever.top_k,
                    "reranker_enabled": cfg.reranker.enabled,
                    "max_hops": max_hops,
                    "genuinely_multihop_filter": genuinely_multihop,
                    "candidate_pool": candidate_pool if genuinely_multihop else None,
                },
                "per_example": per_arm_results,
            }, f)

    elapsed = perf_counter() - t_total

    aggregate = {}
    for arm, rows in per_arm_results.items():
        n = len(rows)
        agg = {
            "n_examples": n,
            "mean_exact_match": sum(r["exact_match"] for r in rows) / n,
            "mean_token_f1": sum(r["token_f1"] for r in rows) / n,
            "mean_retrieval_precision": sum(r["retrieval_precision"] for r in rows) / n,
            "mean_retrieval_recall": sum(r["retrieval_recall"] for r in rows) / n,
            "mean_retrieval_calls": sum(r["num_hops"] for r in rows) / n,
            "mean_latency_s": sum(r["latency_s"] for r in rows) / n,
        }

        by_hop = {}
        for bucket in ["2", "3", "4"]:
            bucket_rows = [r for r in rows if r.get("hop_bucket") == bucket]
            if not bucket_rows:
                continue
            bucket_precision = sum(r["retrieval_precision"] for r in bucket_rows) / len(bucket_rows)
            by_hop[bucket] = {
                "n_examples": len(bucket_rows),
                "mean_retrieval_precision": bucket_precision,
            }
        if by_hop:
            agg["precision_by_hop"] = by_hop
        all_qp = [rec for r in rows for rec in r["qp_records"]]
        if all_qp:
            agg["qp"] = {
                "mean_docs_in": sum(q["num_docs_in"] for q in all_qp) / len(all_qp),
                "mean_docs_retained": sum(q["num_docs_retained"] for q in all_qp) / len(all_qp),
                "mean_solve_time_s": sum(q["solve_time_s"] for q in all_qp) / len(all_qp),
            }
            comparisons = [q["comparison"] for q in all_qp if q.get("comparison")]
            if comparisons:
                agg["qp"]["snn_vs_classical"] = {
                    "mean_objective_gap": sum(c["objective_gap"] for c in comparisons) / len(comparisons),
                    "mean_x_l2_gap": sum(c["x_l2_gap"] for c in comparisons) / len(comparisons),
                    "mean_selected_agreement": sum(c["selected_agreement"] for c in comparisons) / len(comparisons),
                    "mean_classical_solve_time_s": sum(c["classical_solve_time_s"] for c in comparisons) / len(comparisons),
                    "mean_snn_solve_time_s": sum(c["snn_solve_time_s"] for c in comparisons) / len(comparisons),
                }

        # Marginal 95% bootstrap CIs for each headline metric.
        agg["ci"] = {
            m: bootstrap_ci([r[m] for r in rows]) for m in CI_METRICS
        }
        aggregate[arm] = agg

    # Paired CIs on the difference against the single-shot baseline. Every arm
    # ran on the identical example set, so the paired interval is the one that
    # supports (or refuses) a claim of improvement.
    baseline_rows = per_arm_results["single_shot"]
    for arm, rows in per_arm_results.items():
        if arm == "single_shot":
            continue
        aggregate[arm]["ci_vs_single_shot"] = {
            m: paired_bootstrap_ci([r[m] for r in rows], [b[m] for b in baseline_rows])
            for m in CI_METRICS
        }
        if "precision_by_hop" in aggregate[arm]:
            baseline_by_hop = {}
            for bucket in ["2", "3", "4"]:
                bucket_rows = [r for r in baseline_rows if r.get("hop_bucket") == bucket]
                if bucket_rows:
                    baseline_by_hop[bucket] = sum(r["retrieval_precision"] for r in bucket_rows) / len(bucket_rows)
            aggregate[arm]["precision_gain_vs_single_shot_by_hop"] = {}
            for bucket, info in aggregate[arm]["precision_by_hop"].items():
                base = baseline_by_hop.get(bucket)
                if base is None:
                    continue
                info["gain_vs_single_shot"] = info["mean_retrieval_precision"] - base
                aggregate[arm]["precision_gain_vs_single_shot_by_hop"][bucket] = info["gain_vs_single_shot"]

    result = {
        "phase": 5,
        "description": "Ablation: single-shot vs multi-hop vs multi-hop+classical-QP vs multi-hop+SNN",
        "total_time_s": elapsed,
        "config": {
            "dataset": dataset,
            "embedding_model": cfg.embedding.model_name,
            "generator_model": cfg.generator.model_name,
            "retriever_top_k": cfg.retriever.top_k,
            "reranker_enabled": cfg.reranker.enabled,
            "max_hops": max_hops,
            "genuinely_multihop_filter": genuinely_multihop,
            "genuinely_multihop_rule": "keep only examples with single-shot retrieval recall < 1.0; applied equally to all datasets",
            "candidate_pool": candidate_pool if genuinely_multihop else None,
        },
        "aggregate": aggregate,
        "per_example": per_arm_results,
        "cited_baselines": CITED_BASELINES,
    }

    # Per-dataset filename so a second dataset's run can't clobber the first's.
    # results/phase5_results.json stays as the original frozen HotpotQA spine.
    out_path = out_dir / f"{output_stem}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    checkpoint_path.unlink(missing_ok=True)

    print(f"\n[Phase 5] Results saved to {out_path}\n")
    print(
        "Retrieval recall (does the final selected set still contain every "
        "gold supporting doc?) is the primary evidence here - EM/F1 also "
        "depend on flan-t5-base's answer quality, which is a separate, weak "
        "link in the chain.\n"
    )
    print(f"{'Arm':<26}{'EM':>7}{'F1':>7}{'Prec.':>8}{'R.Recall':>10}{'Calls/Q':>9}{'Latency':>10}")
    for arm in ARMS:
        a = aggregate[arm]
        print(
            f"{arm:<26}{a['mean_exact_match']:>7.3f}{a['mean_token_f1']:>7.3f}"
            f"{a['mean_retrieval_precision']:>8.3f}"
            f"{a['mean_retrieval_recall']:>10.3f}{a['mean_retrieval_calls']:>9.2f}"
            f"{a['mean_latency_s']:>9.2f}s"
        )

    print(f"\n95% bootstrap CIs (n={aggregate['single_shot']['n_examples']}):")
    for arm in ARMS:
        print(f"  {arm}")
        for m in CI_METRICS:
            print(f"    {m:<22}{format_ci(aggregate[arm]['ci'][m])}")

    print("\nPaired difference vs single-shot (95% CI; * = excludes zero):")
    for arm in ARMS:
        if arm == "single_shot":
            continue
        print(f"  {arm}")
        for m in CI_METRICS:
            ci = aggregate[arm]["ci_vs_single_shot"][m]
            star = " *" if ci["significant"] else ""
            print(f"    {m:<22}{format_ci(ci)}{star}")
    snn_qp = aggregate.get("multi_hop_qp_snn", {}).get("qp", {}).get("snn_vs_classical")
    if snn_qp:
        # Lead with solution equivalence, not wall-clock: a spiking solver
        # simulated on CPU will always lose to a compiled OSQP call, and
        # that comparison isn't the meaningful one. The result that matters
        # is how close the SNN's solution lands to the classical optimum.
        print("\nSNN solution-equivalence vs classical QP (multi_hop_qp_snn arm):")
        print(f"  selection agreement:     {snn_qp['mean_selected_agreement']:.1%}")
        print(f"  mean objective gap:      {snn_qp['mean_objective_gap']:.4f}")
        print(f"  mean ||x_snn - x_cls||:  {snn_qp['mean_x_l2_gap']:.4f}")
        print(
            "  (CPU wall-clock solve time recorded in the JSON but not "
            "reported here as a result - a simulated spiking network will "
            "always lose to compiled OSQP on CPU; the efficiency story "
            "belongs on neuromorphic/FPGA hardware, tracked separately.)"
        )

    print(
        "\nCited baselines (ReAct/IRCoT, NOT measured here - see "
        "cited_baselines in the JSON for full citations and the model-scale caveat)."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: retrieval-strategy ablation")
    parser.add_argument("--max_examples", type=int, default=25)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--no_reranker", action="store_true")
    parser.add_argument(
        "--genuinely_multihop", action="store_true",
        help="Filter the candidate pool to examples where single-shot recall "
             "< 1.0 before running the ablation, so multi-hop has room to help.",
    )
    parser.add_argument(
        "--candidate_pool", type=int, default=300,
        help="Pool size to draw genuinely-multi-hop examples from (only used "
             "with --genuinely_multihop).",
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Override retriever top_k. IMPORTANT for --genuinely_multihop: "
             "HotpotQA distractor context has exactly 10 paragraphs, so "
             "top_k=10 (the config default) retrieves the entire corpus and "
             "single-shot recall is always 1.0 - the filter can never find "
             "anything. Use a real subset (e.g. 5) so single-shot retrieval "
             "can genuinely miss a gold doc.",
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="hotpotqa",
        help="Which multi-hop benchmark to evaluate on.",
    )
    parser.add_argument(
        "--results_name", type=str, default=None,
        help="Output filename stem under results/ (default: phase5_<dataset>).",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.no_reranker:
        cfg.reranker.enabled = False
    if args.top_k is not None:
        cfg.retriever.top_k = args.top_k
    run_eval(
        cfg, max_examples=args.max_examples, max_hops=args.max_hops,
        genuinely_multihop=args.genuinely_multihop, candidate_pool=args.candidate_pool,
        dataset=args.dataset, results_name=args.results_name,
    )
