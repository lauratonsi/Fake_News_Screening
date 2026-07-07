"""Explainability for the SVM half of the ensemble.

The SVM is a *linear* model over TF-IDF features, which makes it fully
transparent: the contribution of each token to the score is simply its TF-IDF
weight times the model coefficient. We surface the tokens pushing the verdict
toward FAKE and toward REAL so the human reviewer can see *why* the model
scored the text the way it did — not a post-hoc approximation (LIME/SHAP), but
the model's own arithmetic.

Only the SVM is explained this way: it is the transparent, full-training-set
member of the ensemble, and (unlike the RNNs) its decision decomposes exactly
into per-token contributions. The RNNs remain black boxes by nature; the
manipulation layer (src/manipulation.py) and the retrieved evidence cover the
"why" for the rest of the system.
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
