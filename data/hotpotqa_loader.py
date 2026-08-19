"""
HotpotQA dataset loader for the distractor setting.

Each HotpotQA example in the distractor setting has:
  - question: str
  - answer: str
  - supporting_facts: titles + sentence indices of gold evidence
  - context: list of (title, sentences) — 10 paragraphs total
    (2 gold + 8 distractors)

We flatten each paragraph's sentences into a single document string
(one doc per Wikipedia paragraph title) so the retriever has 10
candidate docs per question — matching the standard evaluation setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from datasets import load_dataset


@dataclass
class HotpotQAExample:
    """One evaluation example."""
    qid: str
    question: str
    answer: str
    gold_titles: set[str]          # titles of the 2 supporting paragraphs
    context_docs: list[str]         # 10 flattened paragraph texts
    context_titles: list[str]       # corresponding titles
    level: str                     # "easy", "medium", "hard"
    question_type: str             # "comparison" or "bridge"


def _flatten_paragraph(title: str, sentences: list[str]) -> str:
    """Join a paragraph's sentences into one retrieval-unit string."""
    body = " ".join(sentences)
    return f"{title}: {body}"


def load_hotpotqa(
    split: str = "validation",
    max_examples: int | None = None,
) -> list[HotpotQAExample]:
    """
    Load HotpotQA distractor split from HuggingFace.

    Parameters
    ----------
    split : "train" or "validation" (no public test labels)
    max_examples : cap for quick dev iteration (None = full split)
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    examples: list[HotpotQAExample] = []
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break

        titles: list[str] = row["context"]["title"]
        sentences_list: list[list[str]] = row["context"]["sentences"]

        context_docs = [
            _flatten_paragraph(t, sents)
            for t, sents in zip(titles, sentences_list)
        ]

        gold_titles = set(row["supporting_facts"]["title"])

        examples.append(
            HotpotQAExample(
                qid=row["id"],
                question=row["question"],
                answer=row["answer"],
                gold_titles=gold_titles,
                context_docs=context_docs,
                context_titles=titles,
                level=row["level"],
                question_type=row["type"],
            )
        )
    return examples


def iter_examples(
    split: str = "validation",
    max_examples: int | None = None,
) -> Generator[HotpotQAExample, None, None]:
    """Memory-friendly streaming variant."""
    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split=split,
        streaming=True,
    )
    for i, row in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        titles = row["context"]["title"]
        sentences_list = row["context"]["sentences"]
        context_docs = [
            _flatten_paragraph(t, sents)
            for t, sents in zip(titles, sentences_list)
        ]
        gold_titles = set(row["supporting_facts"]["title"])
        yield HotpotQAExample(
            qid=row["id"],
            question=row["question"],
            answer=row["answer"],
            gold_titles=gold_titles,
            context_docs=context_docs,
            context_titles=titles,
            level=row["level"],
            question_type=row["type"],
        )
