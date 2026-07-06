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


def test_midstrength_fake_match_boosts_toward_fake():
    # Ensemble alone is borderline REAL; a fake-side match below the override
    # threshold boosts the score over 0.5 and flips the verdict to FAKE.
    scores = {"svm": 0.45, "gru": 0.45, "lstm": 0.45}  # mean 0.45 -> REAL alone
    reference = {"verdict": "FAKE", "score": config.REF_MATCH_THRESHOLD + 0.02, "evidence": []}
    result = combine_verdict(scores, reference)
    assert result["fake_probability"] == 0.45 + config.REF_BOOST
    assert result["verdict"] == "FAKE"
    assert "known fake" in result["reason"]


def test_topical_real_match_below_override_does_not_change_score():
    # A real-side match below the override threshold is topical similarity, not
    # evidence of truth: it must NOT quietly pull the score toward REAL. (This
    # is the bug behind "Trump won 2016" showing FAKE with a green REAL panel.)
    scores = {"svm": 0.6, "gru": 0.6, "lstm": 0.6}  # mean 0.6 -> FAKE
    reference = {"verdict": "REAL", "score": 0.80, "evidence": []}  # strong-ish, still < 0.90
    result = combine_verdict(scores, reference)
    assert result["fake_probability"] == 0.6  # unchanged by the topical real match
    assert result["verdict"] == "FAKE"
    assert result["reason"] == "ensemble consensus"


def test_fake_boost_and_real_override_pin_to_the_extremes():
    # A mid-strength fake match boosts and is capped at 1.0 ...
    high = combine_verdict({"svm": 0.95, "gru": 0.95, "lstm": 0.95},
                            {"verdict": "FAKE", "score": config.REF_MATCH_THRESHOLD, "evidence": []})
    assert high["fake_probability"] == 1.0
    # ... and a near-verbatim real match overrides straight to 0.0.
    low = combine_verdict({"svm": 0.05, "gru": 0.05, "lstm": 0.05},
                           {"verdict": "REAL", "score": config.REF_OVERRIDE_THRESHOLD + 0.05, "evidence": []})
    assert low["fake_probability"] == 0.0
    assert "override" in low["reason"]
