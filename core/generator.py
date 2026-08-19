"""
LLM answer generator.

Two backends:
  - "huggingface": local model via transformers (default: flan-t5-base)
  - "api": any OpenAI-compatible or Ollama endpoint

The prompt template is explicit and swappable — important for
Phase 2 where we'll need hop-aware prompts (gap detection, sub-query
generation).
"""
from __future__ import annotations

from dataclasses import dataclass

from snn_rag.config import GeneratorConfig
from snn_rag.core.vector_store import RetrievedDoc

# ── Prompt templates ────────────────────────────────────────────────

SINGLE_SHOT_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


GAP_DETECTION_PROMPT = """You are checking whether the context below is enough \
to fully answer the question.

Context:
{context}

Question: {question}

Is any information still missing to answer the question? Reply with exactly \
one line starting with YES or NO. If YES, follow it with a short phrase \
naming the missing fact or entity. If NO, just reply "NO".

Reply:"""


SUBQUERY_PROMPT = """We are answering a multi-hop question and already have \
some context, but it is missing this information: {missing_info}

Original question: {question}

Existing context:
{context}

Write ONE focused search query (not a full sentence) to find the missing \
information.

Search query:"""


def build_context(docs: list[RetrievedDoc], max_docs: int = 5) -> str:
    """Concatenate top-k doc texts into a single context string."""
    snippets = [
        f"[Doc {i+1}] {d.text}" for i, d in enumerate(docs[:max_docs])
    ]
    return "\n\n".join(snippets)


@dataclass
class GapCheck:
    """Result of asking the LLM whether the context has a knowledge gap."""
    has_gap: bool
    missing_info: str


# ── Generator ───────────────────────────────────────────────────────

class Generator:
    """Generate answers from retrieved context + question."""

    def __init__(self, cfg: GeneratorConfig | None = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self._model = None
        self._tokenizer = None

    # -- lazy init so import is fast --------------------------------
    def _load_hf_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name)
        self._model.eval()

    # ---------------------------------------------------------------
    def generate(
        self,
        question: str,
        docs: list[RetrievedDoc],
        max_context_docs: int = 5,
        prompt_template: str = SINGLE_SHOT_PROMPT,
    ) -> str:
        """Build prompt from docs + question, then run the LLM."""
        context = build_context(docs, max_docs=max_context_docs)
        prompt = prompt_template.format(context=context, question=question)
        return self._run(prompt)

    def detect_gap(
        self,
        question: str,
        docs: list[RetrievedDoc],
        max_context_docs: int = 5,
    ) -> GapCheck:
        """Ask the LLM whether the current context is missing information."""
        context = build_context(docs, max_docs=max_context_docs)
        prompt = GAP_DETECTION_PROMPT.format(context=context, question=question)
        reply = self._run(prompt).strip()

        has_gap = reply.upper().startswith("YES")
        missing_info = ""
        if has_gap:
            missing_info = reply[3:].strip(" :.-\n")
        return GapCheck(has_gap=has_gap, missing_info=missing_info or question)

    def generate_subquery(
        self,
        question: str,
        docs: list[RetrievedDoc],
        missing_info: str,
        max_context_docs: int = 5,
    ) -> str:
        """Ask the LLM to formulate a follow-up search query for a hop."""
        context = build_context(docs, max_docs=max_context_docs)
        prompt = SUBQUERY_PROMPT.format(
            missing_info=missing_info, question=question, context=context
        )
        subquery = self._run(prompt).strip()
        return subquery or missing_info

    # ── backend dispatch ────────────────────────────────────────────
    def _run(self, prompt: str) -> str:
        if self.cfg.backend == "huggingface":
            return self._generate_hf(prompt)
        elif self.cfg.backend == "api":
            return self._generate_api(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

    # ── HuggingFace local ──────────────────────────────────────────
    def _generate_hf(self, prompt: str) -> str:
        if self._model is None:
            self._load_hf_model()
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature if self.cfg.temperature > 0 else None,
            do_sample=self.cfg.temperature > 0,
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── API backend (Ollama / OpenAI-compat) ───────────────────────
    def _generate_api(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            self.cfg.api_url,
            json={
                "model": self.cfg.api_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_new_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
