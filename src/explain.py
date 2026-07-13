"""Explainability across the ensemble.

Three complementary "why" views, each matched to how that part of the system
actually decides:

* **SVM (linear, exact).** The SVM is a linear model over TF-IDF features, so
  each token's contribution is exactly its TF-IDF weight times the model
  coefficient — the model's own arithmetic, not a LIME/SHAP approximation.
  ``top_token_contributions`` surfaces the tokens pushing toward FAKE / REAL.
* **RNNs (occlusion).** The Bi-GRU/Bi-LSTM are not linear and, served as TFLite,
  expose no gradients — but they can be re-run cheaply. ``occlusion_importance``
  turns "score with token i removed" into a per-token attribution: how much each
  word moved the neural sub-ensemble's fake-probability. Model-agnostic and
  honest about being an ablation, not the model's internals.
* **Retrieval (evidence).** The reference layer already *is* its own
  explanation — the snippets it matched. ``top_evidence`` names the retrieved
  real/fake article that most drove the verdict, so the reviewer can read the
  actual evidence rather than trust a score.

The scoring math lives here as pure functions; ``src.predict`` supplies the
model calls (SVM coefficients, the occluded RNN scores, the retrieval result).
"""
from __future__ import annotations

import numpy as np


def averaged_linear_coef(calibrated_model) -> np.ndarray:
    """Recover per-feature weights from a ``CalibratedClassifierCV``.

    Calibration wraps one fitted linear SVM per CV fold and hides ``coef_`` on
    the outer object. We average the folds' coefficients — the same ensemble
    the calibrated probabilities average over — to get a single weight vector.
    """
    coefs = []
    for cc in calibrated_model.calibrated_classifiers_:
        inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if inner is None or not hasattr(inner, "coef_"):
            raise AttributeError("calibrated sub-classifier exposes no linear coef_")
        coefs.append(np.asarray(inner.coef_[0]))
    return np.mean(coefs, axis=0)


def top_token_contributions(coef: np.ndarray, vectorizer, text: str, top_k: int = 8) -> dict:
    """Tokens in ``text`` that most push the SVM toward FAKE / toward REAL.

    Contribution of a token = its TF-IDF value in this text × its model weight.
    Positive pushes toward FAKE (class 1), negative toward REAL (class 0).
    """
    row = vectorizer.transform([text]).tocoo()
    feats = vectorizer.get_feature_names_out()
    contribs = [
        {"token": str(feats[j]), "weight": float(row.data[k] * coef[j])}
        for k, j in enumerate(row.col)
    ]
    contribs.sort(key=lambda c: c["weight"], reverse=True)

    fake_pushing = [c for c in contribs if c["weight"] > 0][:top_k]
    real_pushing = [c for c in contribs if c["weight"] < 0]
    real_pushing = sorted(real_pushing, key=lambda c: c["weight"])[:top_k]

    return {
        "fake_pushing": fake_pushing,
        "real_pushing": real_pushing,
        "available": bool(contribs),
    }


def occlusion_importance(base_score: float, occluded_scores, tokens, top_k: int = 8) -> dict:
    """Per-token importance for a black-box model via leave-one-out occlusion.

    ``base_score`` is the model's fake-probability on the full text;
    ``occluded_scores[i]`` is the fake-probability with ``tokens[i]`` removed.
    Importance = ``base - occluded``: a positive value means removing the token
    *lowered* the fake-probability, i.e. the token pushed the verdict toward
    FAKE; negative pushes toward REAL. Repeated tokens are aggregated (summed),
    so a word's reported weight is its total influence across occurrences.
    """
    agg: dict[str, float] = {}
    for tok, occ in zip(tokens, occluded_scores):
        agg[str(tok)] = agg.get(str(tok), 0.0) + (float(base_score) - float(occ))
    contribs = [{"token": t, "weight": w} for t, w in agg.items()]

    fake_pushing = sorted((c for c in contribs if c["weight"] > 0),
                          key=lambda c: c["weight"], reverse=True)[:top_k]
    real_pushing = sorted((c for c in contribs if c["weight"] < 0),
                          key=lambda c: c["weight"])[:top_k]
    return {
        "fake_pushing": fake_pushing,
        "real_pushing": real_pushing,
        "available": any(c["weight"] != 0 for c in contribs),
    }


def top_evidence(reference: dict, top_k: int = 2, snippet_chars: int = 200) -> dict:
    """The retrieved snippet(s) that most drove the reference-layer verdict.

    Reads the ``evidence`` list of a ``ReferenceRAG.query`` result (each hit
    carries ``label``/``score``/``text``, already sorted by similarity) and
    returns the strongest matches, truncated for display. Pure — no model call.
    """
    hits = (reference or {}).get("evidence") or []
    driving = [
        {
            "label": h.get("label"),
            "score": round(float(h.get("score", 0.0)), 3),
            "snippet": str(h.get("text", ""))[:snippet_chars].strip(),
        }
        for h in hits[:top_k]
    ]
    return {
        "available": bool(driving),
        "driving": driving,
        "best_label": driving[0]["label"] if driving else None,
    }
