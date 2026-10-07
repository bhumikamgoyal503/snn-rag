"""
MuSiQue dataset loader.

Second multi-hop benchmark alongside HotpotQA, added so the QP/SNN selector's
precision result isn't demonstrated on a single dataset. MuSiQue is the harder
of the two by design:

  * 20 candidate paragraphs per question, vs HotpotQA distractor's 10 - twice
    the distractors for the selector to prune.
  * 2-4 gold supporting paragraphs, vs HotpotQA's fixed 2, including genuine
    3-hop and 4-hop questions.
  * Built specifically to resist the single-hop shortcuts that make parts of
    HotpotQA answerable without real multi-hop reasoning.

Emits the same MultiHopExample records as the HotpotQA loader, so the
retrievers, solvers, and ablation script consume both datasets unchanged.
"""
from __future__ import annotations

from datasets import load_dataset

from snn_rag.data.hotpotqa_loader import HotpotQAExample

HF_REPO = "dgslibisey/MuSiQue"


def _hop_count(qid: str) -> str:
    """MuSiQue encodes hop count in the id, e.g. '3hop1__12345_678' -> '3hop'."""
    prefix = qid.split("__", 1)[0]
    return prefix[:4] if prefix[:1].isdigit() else "unknown"


def load_musique(
    split: str = "validation",
    max_examples: int | None = None,
) -> list[HotpotQAExample]:
    """
    Load MuSiQue as MultiHopExample records.

    Parameters
    ----------
    split : "train" or "validation" (test answers are not public)
    max_examples : cap for quick dev iteration (None = full split)
    """
    ds = load_dataset(HF_REPO, split=split)

    examples: list[HotpotQAExample] = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        paragraphs = row["paragraphs"]
        titles = [p["title"] for p in paragraphs]
        context_docs = [f"{p['title']}: {p['paragraph_text']}" for p in paragraphs]
        gold_titles = {p["title"] for p in paragraphs if p["is_supporting"]}
        gold_doc_ids = {idx for idx, p in enumerate(paragraphs) if p.get("is_supporting", False)}

        examples.append(
            HotpotQAExample(
                qid=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_titles=gold_titles,
                gold_doc_ids=gold_doc_ids,
                context_docs=context_docs,
                context_titles=titles,
                # MuSiQue ships no easy/medium/hard label; hop count is the
                # closest available difficulty signal.
                level=_hop_count(row["id"]),
                question_type=_hop_count(row["id"]),
            )
        )
    return examples
