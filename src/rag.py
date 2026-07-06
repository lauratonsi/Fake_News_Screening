"""Reference-corpus retrieval for the screening demo.

Semantic retrieval-first support layer: it does not generate answers. It
finds the closest known real/fake snippets for an input text using sentence
embeddings — semantic similarity, not literal word overlap — and returns the
best matching evidence together with a light verdict heuristic.

Classification stays classical (TF-IDF + SVM/RNNs) by tested choice: on this
dataset the fake/real signal is mostly surface style and source markers,
which literal term-matching exploits and semantic embeddings are built to
ignore (see experiments/embeddings_baseline.py and the README). Retrieval is
a different task — finding *paraphrases* of a known claim regardless of
wording — which is exactly what semantic embeddings are good at.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class RetrievalHit:
    label: str
    score: float
    text: str


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def _embedding_model_source() -> str:
    """Prefer the committed local copy; fall back to the Hub name so a
    fresh clone (before models/embedding_model exists) can still run
    src.train to regenerate everything."""
    local = config.EMBEDDING_MODEL_PATH
    if local.exists() and (local / "config.json").exists():
        return str(local)
    return config.EMBEDDING_MODEL_NAME


def build_and_save_embeddings(
    real_file: Path | None = None,
    fake_file: Path | None = None,
    out_file: Path | None = None,
) -> None:
    """Encode the committed reference corpus once and cache it to disk.

    Encoding ~68k snippets takes tens of minutes on CPU — far too slow to
    redo on every app start — so this runs once during training/data prep
    and the result (float16, well under GitHub's 100 MB file limit) is
    committed alongside the existing real.csv.gz / fake.csv.gz snippets.
    """
    from sentence_transformers import SentenceTransformer

    real_file = real_file or config.REF_REAL_FILE
    fake_file = fake_file or config.REF_FAKE_FILE
    out_file = out_file or config.REF_EMBEDDINGS_FILE

    real = pd.read_csv(real_file)["text"].fillna("").tolist()
    fake = pd.read_csv(fake_file)["text"].fillna("").tolist()

    model = SentenceTransformer(_embedding_model_source())
    print(f">>> Encoding reference corpus ({len(real)} real + {len(fake)} fake snippets)...")
    real_emb = model.encode(real, batch_size=64, show_progress_bar=True).astype(np.float16)
    fake_emb = model.encode(fake, batch_size=64, show_progress_bar=True).astype(np.float16)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, real=real_emb, fake=fake_emb)
    print(f">>> Reference embeddings saved -> {out_file}")


class ReferenceRAG:
    """Sentence-embedding retrieval over the committed reference corpus."""

    def __init__(
        self,
        real_file: Path | None = None,
        fake_file: Path | None = None,
        embeddings_file: Path | None = None,
    ):
        self.real_file = real_file or config.REF_REAL_FILE
        self.fake_file = fake_file or config.REF_FAKE_FILE
        self.embeddings_file = embeddings_file or config.REF_EMBEDDINGS_FILE
        self.enabled = (
            self.real_file.exists() and self.fake_file.exists() and self.embeddings_file.exists()
        )
        self.model = None
        self.real_matrix = None
        self.fake_matrix = None
        self.real_texts: list[str] = []
        self.fake_texts: list[str] = []

        if not self.enabled:
            return

        self.real_texts = pd.read_csv(self.real_file)["text"].fillna("").tolist()
        self.fake_texts = pd.read_csv(self.fake_file)["text"].fillna("").tolist()

        cached = np.load(self.embeddings_file)
        self.real_matrix = _normalize(cached["real"].astype(np.float32))
        self.fake_matrix = _normalize(cached["fake"].astype(np.float32))

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(_embedding_model_source())

    def _top_hits(self, label: str, scores: np.ndarray, texts: list[str], top_k: int) -> list[RetrievalHit]:
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

        snippet = text[: config.REF_SNIPPET_CHARS].strip()
        vec = self.model.encode([snippet], normalize_embeddings=True)[0].astype(np.float32)

        real_scores = self.real_matrix @ vec
        fake_scores = self.fake_matrix @ vec

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
                    "message": f"closest match is a known fake snippet ({best_fake:.0%} semantic similarity)",
                }
            )
        elif best_real > config.REF_MATCH_THRESHOLD and best_real > best_fake + config.REF_MARGIN:
            result.update(
                {
                    "verdict": "REAL",
                    "score": best_real,
                    "message": f"closest match is a known real snippet ({best_real:.0%} semantic similarity)",
                }
            )
        else:
            result["message"] = "no strong match in the reference corpus"

        result["evidence"] = [
            {"label": hit.label, "score": round(hit.score, 4), "text": hit.text}
            for hit in evidence
        ]
        return result
