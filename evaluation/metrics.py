"""
Evaluation metrics for the SNN-RAG project.

Answer quality:
  - Exact Match (EM): binary, after normalization
  - Token F1: precision/recall over answer tokens

Retrieval quality:
  - Retrieval Precision/Recall: did we retrieve the gold paragraphs?

All normalization follows the HotpotQA official eval script conventions.
"""
from __future__ import annotations

import re
import string
from collections import Counter


# ── Text normalization (HotpotQA-standard) ─────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace."""
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> list[str]:
    return _normalize(text).split()


# ── Answer-level metrics ───────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized prediction == normalized gold, else 0.0."""
    return float(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Retrieval-level metrics ────────────────────────────────────────

def retrieval_precision(
    retrieved_titles: list[str], gold_titles: set[str]
) -> float:
    """Fraction of retrieved docs that are gold."""
    if not retrieved_titles:
        return 0.0
    hits = sum(1 for t in retrieved_titles if t in gold_titles)
    return hits / len(retrieved_titles)


def retrieval_recall(
    retrieved_titles: list[str], gold_titles: set[str]
) -> float:
    """Fraction of gold docs that were retrieved."""
    if not gold_titles:
        return 1.0
    hits = sum(1 for t in retrieved_titles if t in gold_titles)
    return hits / len(gold_titles)


# ── Batch evaluation ───────────────────────────────────────────────

def evaluate_batch(
    predictions: list[str],
    golds: list[str],
) -> dict[str, float]:
    """Compute mean EM and F1 over a batch."""
    n = len(predictions)
    assert n == len(golds), "predictions and golds must have the same length"
    em_total = sum(exact_match(p, g) for p, g in zip(predictions, golds))
    f1_total = sum(token_f1(p, g) for p, g in zip(predictions, golds))
    return {
        "exact_match": em_total / n,
        "token_f1": f1_total / n,
        "n": n,
    }
