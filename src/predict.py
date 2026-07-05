"""Inference: the screening system used by the Streamlit demo and benchmarks.

The system combines three signals:

1. An ensemble (simple average) of the calibrated SVM, the Bi-GRU and the
   Bi-LSTM fake-probability scores.
2. A retrieval-based reference layer: cosine similarity against snippets of
    articles already known to be real or fake. This is honest retrieval over
    the committed corpora — NOT fact-checking. A strong match can override the
    ensemble; a weaker one only shifts the score.
3. An agreement check: when the individual models disagree strongly, the
   verdict is flagged for human review instead of being reported as confident.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
from . import config
from .claim_rag import analyze_claims
from .rag import ReferenceRAG


class ScreeningSystem:
    """Loads all artifacts once; ``predict`` scores a single text."""

    def __init__(self, with_reference: bool = True):
        artifact = joblib.load(config.SVM_FILE)
        self.svm = artifact["model"]
        self.vectorizer = artifact["vectorizer"]
        self.tokenizer = joblib.load(config.TOKENIZER_FILE)

        from tf_keras.models import load_model

        self.gru = load_model(config.GRU_FILE)
        self.lstm = load_model(config.LSTM_FILE)

        self.reference = ReferenceRAG() if with_reference else None

    # ------------------------------------------------------------------ signals
    def model_scores(self, text: str) -> dict:
        from tf_keras.preprocessing.sequence import pad_sequences

        scores = {}
        scores["svm"] = float(
            self.svm.predict_proba(self.vectorizer.transform([text]))[0, 1]
        )
        seq = pad_sequences(
            self.tokenizer.texts_to_sequences([text]),
            maxlen=config.MAX_LEN,
            padding="post",
        )
        scores["gru"] = float(self.gru(seq, training=False).numpy()[0, 0])
        scores["lstm"] = float(self.lstm(seq, training=False).numpy()[0, 0])
        return scores

    def reference_check(self, text: str) -> dict:
        """Retrieve the closest known real/fake snippets and score them."""
        result = {"verdict": None, "score": 0.0, "message": "reference corpus disabled", "evidence": []}
        if self.reference is None:
            return result
        return self.reference.query(text)

    # ------------------------------------------------------------------ verdict
    def predict(self, text: str) -> dict:
        scores = self.model_scores(text)
        reference = self.reference_check(text)
        claim_analysis = analyze_claims(text, self.reference) if self.reference is not None else {
            "verdict": None,
            "message": "reference corpus disabled",
            "claims": [],
            "summary": {"claims_total": 0, "supported": 0, "refuted": 0, "unsupported": 0},
        }

        score = float(np.mean(list(scores.values())))
        spread = max(scores.values()) - min(scores.values())

        if reference["verdict"] and reference["score"] > config.REF_OVERRIDE_THRESHOLD:
            verdict = reference["verdict"]
            score = 1.0 if verdict == "FAKE" else 0.0
            reason = "reference-corpus override (strong match)"
        elif reference["verdict"] == "FAKE":
            score = min(1.0, score + config.REF_BOOST)
            verdict = "FAKE" if score > 0.5 else "REAL"
            reason = "ensemble + reference-corpus boost"
        elif reference["verdict"] == "REAL":
            score = max(0.0, score - config.REF_BOOST)
            verdict = "FAKE" if score > 0.5 else "REAL"
            reason = "ensemble + reference-corpus boost"
        else:
            verdict = "FAKE" if score > 0.5 else "REAL"
            reason = "ensemble consensus"

        return {
            "verdict": verdict,
            "fake_probability": score,
            "model_scores": scores,
            "reference": reference,
            "claim_analysis": claim_analysis,
            "needs_review": spread > config.DISAGREEMENT_SPREAD,
            "reason": reason,
        }
