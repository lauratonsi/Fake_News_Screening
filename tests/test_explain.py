import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.explain import averaged_linear_coef, top_token_contributions


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
