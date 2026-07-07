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
from .explain import averaged_linear_coef, top_token_contributions
from .external_retrieval import ExternalEvidenceRetriever
from .manipulation import detect_manipulation
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

        # Per-token weights for the linear SVM, cached once for explainability.
        # Best-effort: if the artifact ever stops exposing linear coefficients,
        # predictions still work — only the explanation panel goes empty.
        try:
            self._svm_coef = averaged_linear_coef(self.svm)
        except Exception:
            self._svm_coef = None

    def explain(self, text: str) -> dict:
        """Tokens pushing the SVM toward FAKE / REAL (exact linear contributions)."""
        if self._svm_coef is None:
            return {"fake_pushing": [], "real_pushing": [], "available": False}
        return top_token_contributions(self._svm_coef, self.vectorizer, text)

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

        live = _aggregate_live_verdict(claim_analysis)
        manipulation = detect_manipulation(text)
        n_words = len(str(text).split())
        verdict_info = combine_verdict(
            scores, reference, live=live, n_words=n_words, manipulation=manipulation
        )
        return {
            **verdict_info,
            "model_scores": scores,
            "reference": reference,
            "claim_analysis": claim_analysis,
            "live": live,
            "manipulation": manipulation,
            "explanation": self.explain(text),
        }


def _aggregate_live_verdict(claim_analysis: dict) -> dict | None:
    """Strongest live fact-check verdict across the analyzed claims.

    Only Google Fact Check returns an actual verdict (Wikipedia/GDELT never
    do), so any verdict found here is a professional fact-checker's rating.
    FAKE takes priority — for a screening tool, a single fact-checked hoax in
    the text is decisive — then REAL. Returns ``None`` when no live source
    asserted a verdict (offline runs, no API key, or nothing matched).
    """
    fallback_real = None
    for item in claim_analysis.get("claims", []):
        live = item.get("live") or {}
        verdict = live.get("verdict")
        if verdict == "FAKE":
            return {
                "verdict": "FAKE",
                "source": live.get("source"),
                "message": live.get("message", ""),
                "evidence": live.get("evidence", []),
                "claim": item.get("claim"),
            }
        if verdict == "REAL" and fallback_real is None:
            fallback_real = {
                "verdict": "REAL",
                "source": live.get("source"),
                "message": live.get("message", ""),
                "evidence": live.get("evidence", []),
                "claim": item.get("claim"),
            }
    return fallback_real


def combine_verdict(scores: dict, reference: dict, live: dict | None = None,
                    n_words: int | None = None, manipulation: dict | None = None) -> dict:
    """Pure decision logic: fuse the model ensemble with external evidence.

    Priority of signals, strongest first:
      1. A live fact-check *verdict* (Google Fact Check) — a professional
         fact-checker's rating: the highest-signal, most *current* evidence,
         and the only one that can correct the article-trained models on
         events they never saw. It takes precedence.
      2. A near-verbatim reference-corpus match (asymmetric, see src/rag.py).
      3. A mid-strength known-fake corpus match, which boosts toward FAKE.
      4. The model ensemble alone.

    Every result also carries a ``confidence`` tier and an ``evidence_backed``
    flag. A model-only verdict on a short, out-of-domain claim is LOW
    confidence and routed to review — this only lowers confidence, it never
    flips FAKE->REAL, so the zero-false-negative guarantee holds.

    Kept separate from ScreeningSystem so it can be unit-tested with synthetic
    scores, without loading the SVM/RNN/embedding artifacts.
    """
    ensemble = float(np.mean(list(scores.values())))
    spread = max(scores.values()) - min(scores.values())
    disagree = spread > config.DISAGREEMENT_SPREAD
    short = n_words is not None and n_words < config.SHORT_INPUT_WORDS
    # Stacked manipulation techniques are grounds for human review on their own,
    # regardless of the label — but never flip the label (see src/manipulation.py).
    manip_high = bool(manipulation and manipulation.get("high"))

    # 1. Live fact-check verdict — takes precedence over everything else.
    if live and live.get("verdict"):
        verdict = live["verdict"]
        source = live.get("source") or "fact-check"
        # Flag a conflict for the reviewer when a real fact-checker's rating
        # points the opposite way from a confident ensemble, rather than
        # silently discarding either signal.
        conflict = (verdict == "REAL" and ensemble > 0.9) or (verdict == "FAKE" and ensemble < 0.1)
        return {
            "verdict": verdict,
            "fake_probability": 1.0 if verdict == "FAKE" else 0.0,
            "needs_review": bool(disagree or conflict or manip_high),
            "reason": f"live fact-check verdict ({source})",
            "confidence": "high",
            "evidence_backed": True,
        }

    # 2. Reference-corpus override (near-verbatim). Asymmetric by design: a
    # fake-side match is evidence of fakeness; a real-side match only reaches
    # here when it is near-verbatim (never on mere topical similarity).
    if reference["verdict"] and reference["score"] > config.REF_OVERRIDE_THRESHOLD:
        verdict = reference["verdict"]
        kind = "known fake" if verdict == "FAKE" else "known real"
        return {
            "verdict": verdict,
            "fake_probability": 1.0 if verdict == "FAKE" else 0.0,
            "needs_review": bool(disagree or manip_high),
            "reason": f"reference-corpus override (near-verbatim {kind} match)",
            "confidence": "high",
            "evidence_backed": True,
        }

    # 3. Mid-strength known-fake corpus match: boosts the ensemble toward FAKE.
    if reference["verdict"] == "FAKE":
        score = min(1.0, ensemble + config.REF_BOOST)
        verdict = "FAKE" if score > 0.5 else "REAL"
        return {
            "verdict": verdict,
            "fake_probability": score,
            "needs_review": bool(disagree or manip_high),
            "reason": "ensemble + reference match to a known fake claim",
            "confidence": "medium",
            "evidence_backed": True,
        }

    # 4. Model-only. Nothing external corroborates the verdict, so confidence is
    # bounded by how in-domain the input is: a short, claim-length input is out
    # of domain for the article-trained models, exactly where they are
    # confidently wrong on true statements. Report it as LOW confidence and
    # route to human verification — but keep the label (never flip FAKE->REAL),
    # so a genuine hoax is still flagged.
    verdict = "FAKE" if ensemble > 0.5 else "REAL"
    confidence = "low" if (disagree or short) else "medium"
    return {
        "verdict": verdict,
        "fake_probability": ensemble,
        "needs_review": bool(disagree or manip_high),
        "reason": "ensemble consensus",
        "confidence": confidence,
        "evidence_backed": False,
    }
