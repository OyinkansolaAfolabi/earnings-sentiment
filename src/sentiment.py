"""Sentence-level sentiment scoring with FinBERT (plus an offline test scorer)."""
from __future__ import annotations

import re

import numpy as np

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c(\[])')
MIN_WORDS = 5  # drop greetings / fragments such as "Thank you." or "Next question."


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text.replace("\n", " "))]
    return [s for s in sents if len(s.split()) >= MIN_WORDS]


class FinBertScorer:
    """ProsusAI/finbert. Returns an (n, 3) array of probabilities ordered [pos, neg, neu]."""

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32,
                 max_length: int = 128, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        # Read label order from the model config instead of hard-coding it.
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        inv = {v: k for k, v in id2label.items()}
        self._cols = [inv["positive"], inv["negative"], inv["neutral"]]

    def score_sentences(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 3))
        out = []
        with self.torch.no_grad():
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
                out.append(probs[:, self._cols])
        return np.vstack(out)


class LexiconScorer:
    """Tiny word-list scorer for OFFLINE SMOKE TESTS ONLY. Not a research-grade model."""

    POS = {"strong", "growth", "record", "improved", "exceeded", "confident", "momentum",
           "increase", "increased", "gains", "robust", "outperformed", "success", "profitable"}
    NEG = {"weak", "decline", "declined", "challenging", "headwinds", "loss", "losses",
           "decrease", "decreased", "pressure", "missed", "uncertain", "softness", "impairment"}

    def score_sentences(self, sentences: list[str]) -> np.ndarray:
        rows = []
        for s in sentences:
            words = set(re.findall(r"[a-z]+", s.lower()))
            p, n = len(words & self.POS), len(words & self.NEG)
            if p == n:
                rows.append([0.05, 0.05, 0.90])
            elif p > n:
                rows.append([0.80, 0.05, 0.15])
            else:
                rows.append([0.05, 0.80, 0.15])
        return np.array(rows).reshape(-1, 3)


def score_text(text: str, scorer) -> dict:
    """Aggregate sentence probabilities into one section-level summary.

    score    = mean(P(pos) - P(neg))  in [-1, 1]
    pos_frac = share of sentences whose top label is positive (likewise neg_frac)
    """
    sents = split_sentences(text)
    if not sents:
        return {"score": np.nan, "pos_frac": np.nan, "neg_frac": np.nan, "n_sent": 0}
    probs = scorer.score_sentences(sents)
    top = probs.argmax(axis=1)
    return {
        "score": float((probs[:, 0] - probs[:, 1]).mean()),
        "pos_frac": float((top == 0).mean()),
        "neg_frac": float((top == 1).mean()),
        "n_sent": int(len(sents)),
    }
