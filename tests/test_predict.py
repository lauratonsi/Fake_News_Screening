from src import config
from src.predict import combine_verdict

NO_REFERENCE = {"verdict": None, "score": 0.0, "message": "no strong match", "evidence": []}


def test_ensemble_consensus_when_no_reference_match():
    scores = {"svm": 0.9, "gru": 0.85, "lstm": 0.88}
    result = combine_verdict(scores, NO_REFERENCE)
    assert result["verdict"] == "FAKE"
    assert result["reason"] == "ensemble consensus"
    assert result["fake_probability"] == sum(scores.values()) / 3


def test_needs_review_flag_on_model_disagreement():
    scores = {"svm": 0.95, "gru": 0.10, "lstm": 0.50}  # spread 0.85 > 0.40
    result = combine_verdict(scores, NO_REFERENCE)
    assert result["needs_review"] is True


def test_no_review_flag_when_models_agree():
    scores = {"svm": 0.90, "gru": 0.85, "lstm": 0.88}  # spread 0.05 < 0.40
    result = combine_verdict(scores, NO_REFERENCE)
    assert result["needs_review"] is False


def test_reference_override_forces_extreme_probability():
    scores = {"svm": 0.4, "gru": 0.45, "lstm": 0.5}  # ensemble alone -> REAL
    reference = {"verdict": "FAKE", "score": config.REF_OVERRIDE_THRESHOLD + 0.1, "evidence": []}
    result = combine_verdict(scores, reference)
    assert result["verdict"] == "FAKE"
    assert result["fake_probability"] == 1.0
    assert "override" in result["reason"]


def test_reference_boost_can_flip_a_borderline_verdict():
    scores = {"svm": 0.55, "gru": 0.55, "lstm": 0.55}  # ensemble alone -> FAKE (barely)
    reference = {"verdict": "REAL", "score": config.REF_MATCH_THRESHOLD + 0.02, "evidence": []}
    result = combine_verdict(scores, reference)
    assert result["fake_probability"] == 0.55 - config.REF_BOOST
    assert result["verdict"] == "REAL"
    assert "boost" in result["reason"]


def test_reference_boost_is_capped_at_zero_and_one():
    high = combine_verdict({"svm": 0.95, "gru": 0.95, "lstm": 0.95},
                            {"verdict": "FAKE", "score": config.REF_MATCH_THRESHOLD, "evidence": []})
    assert high["fake_probability"] == 1.0

    low = combine_verdict({"svm": 0.05, "gru": 0.05, "lstm": 0.05},
                           {"verdict": "REAL", "score": config.REF_MATCH_THRESHOLD, "evidence": []})
    assert low["fake_probability"] == 0.0
