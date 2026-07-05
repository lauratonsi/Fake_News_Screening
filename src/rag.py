"""Reference-corpus retrieval for the screening demo.

This is a retrieval-first support layer: it does not generate answers.
It finds the closest known real/fake snippets for an input text and returns
the best matching evidence together with a light verdict heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


@dataclass(frozen=True)
class RetrievalHit:
    label: str
    score: float
    text: str


class ReferenceRAG:
    """TF-IDF retrieval over the committed reference corpus."""

    def __init__(self, real_file: Path | None = None, fake_file: Path | None = None):
        self.real_file = real_file or config.REF_REAL_FILE
        self.fake_file = fake_file or config.REF_FAKE_FILE
        self.enabled = self.real_file.exists() and self.fake_file.exists()
        self.vectorizer = None
        self.real_matrix = None
        self.fake_matrix = None
        self.real_texts = []
        self.fake_texts = []

        if not self.enabled:
            return

        real = pd.read_csv(self.real_file)["text"].fillna("")
        fake = pd.read_csv(self.fake_file)["text"].fillna("")
        self.real_texts = real.tolist()
        self.fake_texts = fake.tolist()

        self.vectorizer = TfidfVectorizer(max_features=config.REF_TFIDF_FEATURES)
        self.vectorizer.fit(pd.concat([real, fake], ignore_index=True))
        self.real_matrix = self.vectorizer.transform(real)
        self.fake_matrix = self.vectorizer.transform(fake)

    def _top_hits(self, label: str, scores, texts, top_k: int) -> list[RetrievalHit]:
        if len(scores) == 0:
            return []
        top_idx = scores.argsort()[::-1][:top_k]
        return [
            RetrievalHit(label=label, score=float(scores[i]), text=str(texts[i]))
            for i in top_idx
        ]

    def query(self, text: str, top_k: int = 3) -> dict:
        result = {
            "verdict": None,
            "score": 0.0,
            "message": "reference corpus disabled",
            "evidence": [],
        }
        if not self.enabled:
            return result

        snippet = text[: config.REF_SNIPPET_CHARS].lower().strip()
        vec = self.vectorizer.transform([snippet])

        real_scores = cosine_similarity(vec, self.real_matrix).ravel()
        fake_scores = cosine_similarity(vec, self.fake_matrix).ravel()

        best_real = float(real_scores.max()) if len(real_scores) else 0.0
        best_fake = float(fake_scores.max()) if len(fake_scores) else 0.0

        evidence = self._top_hits("REAL", real_scores, self.real_texts, top_k)
        evidence += self._top_hits("FAKE", fake_scores, self.fake_texts, top_k)
        evidence = sorted(evidence, key=lambda hit: hit.score, reverse=True)[: 2 * top_k]

        if best_fake > config.REF_MATCH_THRESHOLD and best_fake > best_real + config.REF_MARGIN:
            result.update(
                {
                    "verdict": "FAKE",
                    "score": best_fake,
                    "message": f"closest match is a known fake snippet ({best_fake:.0%})",
                }
            )
        elif best_real > config.REF_MATCH_THRESHOLD and best_real > best_fake + config.REF_MARGIN:
            result.update(
                {
                    "verdict": "REAL",
                    "score": best_real,
                    "message": f"closest match is a known real snippet ({best_real:.0%})",
                }
            )
        else:
            result["message"] = "no strong match in the reference corpus"

        result["evidence"] = [
            {"label": hit.label, "score": round(hit.score, 4), "text": hit.text}
            for hit in evidence
        ]
        return result