import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.explain import (averaged_linear_coef, occlusion_importance,
                         top_evidence, top_token_contributions)


def test_top_contributions_split_fake_and_real_by_sign():
    vec = TfidfVectorizer().fit(["leaked secret memo report", "senate confirmed official report"])
    feats = list(vec.get_feature_names_out())
    coef = np.zeros(len(feats))
    coef[feats.index("leaked")] = 3.0        # pushes toward FAKE (class 1)
    coef[feats.index("confirmed")] = -3.0    # pushes toward REAL (class 0)

    out = top_token_contributions(coef, vec, "leaked report confirmed")
    assert out["available"] is True
    assert "leaked" in [c["token"] for c in out["fake_pushing"]]
    assert "confirmed" in [c["token"] for c in out["real_pushing"]]
    # Contributions carry the right sign.
    assert all(c["weight"] > 0 for c in out["fake_pushing"])
    assert all(c["weight"] < 0 for c in out["real_pushing"])


def test_empty_or_unseen_text_has_no_contributions():
    vec = TfidfVectorizer().fit(["alpha beta gamma", "delta epsilon zeta"])
    coef = np.ones(len(vec.get_feature_names_out()))
    out = top_token_contributions(coef, vec, "nonexistenttoken")
    assert out["available"] is False
    assert out["fake_pushing"] == [] and out["real_pushing"] == []


def test_averaged_linear_coef_recovers_a_weight_vector():
    texts = ["fake hoax lie scam"] * 4 + ["true real fact news"] * 4
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    model = CalibratedClassifierCV(LinearSVC(), cv=2).fit(X, labels)

    coef = averaged_linear_coef(model)
    assert coef.shape == (len(vec.get_feature_names_out()),)
    # "hoax" should weigh toward FAKE more than "fact".
    feats = list(vec.get_feature_names_out())
    assert coef[feats.index("hoax")] > coef[feats.index("fact")]


# --- RNN occlusion attribution ----------------------------------------------

def test_occlusion_splits_tokens_by_effect_on_fake_probability():
    # base fake-prob 0.80. Removing "leaked" drops it to 0.50 (that word pushed
    # FAKE); removing "confirmed" raises it to 0.90 (that word pushed REAL).
    out = occlusion_importance(
        base_score=0.80,
        occluded_scores=[0.50, 0.78, 0.90],
        tokens=["leaked", "report", "confirmed"],
    )
    assert out["available"] is True
    assert "leaked" in [c["token"] for c in out["fake_pushing"]]
    assert "confirmed" in [c["token"] for c in out["real_pushing"]]
    assert all(c["weight"] > 0 for c in out["fake_pushing"])
    assert all(c["weight"] < 0 for c in out["real_pushing"])


def test_occlusion_aggregates_repeated_tokens():
    out = occlusion_importance(0.9, [0.7, 0.7], ["hoax", "hoax"])
    weights = {c["token"]: c["weight"] for c in out["fake_pushing"]}
    assert abs(weights["hoax"] - 0.4) < 1e-9  # 0.2 + 0.2 summed across occurrences


def test_occlusion_with_no_effect_is_unavailable():
    out = occlusion_importance(0.5, [0.5, 0.5], ["a", "b"])
    assert out["available"] is False
    assert out["fake_pushing"] == [] and out["real_pushing"] == []


# --- retrieval evidence -----------------------------------------------------

def test_top_evidence_returns_the_strongest_matched_snippet():
    reference = {"verdict": "FAKE", "evidence": [
        {"label": "FAKE", "score": 0.91, "text": "A known fabricated claim about X " * 20},
        {"label": "REAL", "score": 0.40, "text": "A true report about Y"},
    ]}
    out = top_evidence(reference, top_k=2, snippet_chars=50)
    assert out["available"] is True
    assert out["best_label"] == "FAKE"
    assert len(out["driving"]) == 2
    assert len(out["driving"][0]["snippet"]) <= 50


def test_top_evidence_without_hits_is_unavailable():
    out = top_evidence({"verdict": None, "evidence": []})
    assert out["available"] is False
    assert out["best_label"] is None
