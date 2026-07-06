"""Inference: the screening system used by the Streamlit demo and benchmarks.

The system combines three signals:

1. An ensemble (simple average) of the calibrated SVM, the Bi-GRU and the
   Bi-LSTM fake-probability scores. The RNNs run as TFLite models via the
   lightweight ``ai-edge-litert`` interpreter (~10 MB) rather than the full
   TensorFlow runtime (~500+ MB) — see experiments/ for why that headroom
   matters: it is what makes the embeddings-based reference layer below
   affordable on a free, memory-capped deployment.
2. A retrieval-based reference layer: sentence-embedding similarity against
   snippets of articles already known to be real or fake. This is honest
   retrieval over the committed corpora — NOT fact-checking. A strong match
   can override the ensemble; a weaker one only shifts the score.
3. An agreement check: when the individual models disagree strongly, the
   verdict is flagged for human review instead of being reported as confident.
"""
from __future__ import annotations

import joblib
import numpy as np
from . import config
from .claim_rag import analyze_claims
from .external_retrieval import ExternalEvidenceRetriever
from .rag import ReferenceRAG
from .tokenizer import pad_sequences


class ScreeningSystem:
    """Loads all artifacts once; ``predict`` scores a single text."""

    def __init__(self, with_reference: bool = True, with_live: bool = True):
        artifact = joblib.load(config.SVM_FILE)
        self.svm = artifact["model"]
        self.vectorizer = artifact["vectorizer"]
        self.tokenizer = joblib.load(config.TOKENIZER_FILE)

        from ai_edge_litert.interpreter import Interpreter

        self.gru = Interpreter(model_path=str(config.GRU_TFLITE_FILE))
        self.gru.allocate_tensors()
        self.lstm = Interpreter(model_path=str(config.LSTM_TFLITE_FILE))
        self.lstm.allocate_tensors()

        self.reference = ReferenceRAG() if with_reference else None
        self.live_retriever = ExternalEvidenceRetriever() if with_live else None

    # ------------------------------------------------------------------ signals
    def _rnn_score(self, interpreter, seq: np.ndarray) -> float:
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        interpreter.set_tensor(inp["index"], seq.astype(inp["dtype"]))
        interpreter.invoke()
        return float(interpreter.get_tensor(out["index"])[0, 0])

    def model_scores(self, text: str) -> dict:
        scores = {}
        scores["svm"] = float(
            self.svm.predict_proba(self.vectorizer.transform([text]))[0, 1]
        )
        seq = pad_sequences(self.tokenizer.texts_to_sequences([text]), maxlen=config.MAX_LEN)
        scores["gru"] = self._rnn_score(self.gru, seq)
        scores["lstm"] = self._rnn_score(self.lstm, seq)
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
        claim_analysis = analyze_claims(text, self.reference, self.live_retriever) if self.reference is not None else {
            "message": "reference corpus disabled",
            "source": None,
            "claims": [],
            "summary": {"claims_total": 0, "matches_fake": 0, "matches_real": 0, "no_match": 0},
        }

        verdict_info = combine_verdict(scores, reference)
        return {**verdict_info, "model_scores": scores, "reference": reference, "claim_analysis": claim_analysis}


def combine_verdict(scores: dict, reference: dict) -> dict:
    """Pure decision logic: ensemble mean + reference-corpus signal -> verdict.

    Kept separate from ScreeningSystem so it can be unit-tested with
    synthetic scores, without loading the SVM/RNN/embedding artifacts.
    """
    score = float(np.mean(list(scores.values())))
    spread = max(scores.values()) - min(scores.values())

    # Asymmetric by design (see src/rag.py): a fake-side match is evidence of
    # fakeness and can override or boost the ensemble; a real-side match only
    # reaches this function when it is near-verbatim (it never fires on mere
    # topical similarity), and even then it only overrides — a weak real match
    # must not quietly pull a text toward "true" just for sharing a topic with
    # genuine reporting.
    if reference["verdict"] and reference["score"] > config.REF_OVERRIDE_THRESHOLD:
        verdict = reference["verdict"]
        score = 1.0 if verdict == "FAKE" else 0.0
        kind = "known fake" if verdict == "FAKE" else "known real"
        reason = f"reference-corpus override (near-verbatim {kind} match)"
    elif reference["verdict"] == "FAKE":
        score = min(1.0, score + config.REF_BOOST)
        verdict = "FAKE" if score > 0.5 else "REAL"
        reason = "ensemble + reference match to a known fake claim"
    else:
        verdict = "FAKE" if score > 0.5 else "REAL"
        reason = "ensemble consensus"

    return {
        "verdict": verdict,
        "fake_probability": score,
        "needs_review": spread > config.DISAGREEMENT_SPREAD,
        "reason": reason,
    }
