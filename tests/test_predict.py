from src import config
from src.predict import _aggregate_live_verdict, combine_verdict

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


# --- A1: live fact-check verdict is a first-class, top-priority signal -------

def test_live_fake_verdict_overrides_the_ensemble():
    # Models are confident REAL, but a professional fact-checker rated it FAKE.
    scores = {"svm": 0.05, "gru": 0.02, "lstm": 0.03}  # ensemble -> REAL
    live = {"verdict": "FAKE", "source": "google_fact_check", "evidence": []}
    result = combine_verdict(scores, NO_REFERENCE, live=live)
    assert result["verdict"] == "FAKE"
    assert result["fake_probability"] == 1.0
    assert result["confidence"] == "high"
    assert result["evidence_backed"] is True
    assert "fact-check" in result["reason"]


def test_live_real_verdict_rescues_a_true_claim_the_models_flag():
    # The dominant failure mode: models scream FAKE on a true short claim.
    # A live REAL fact-check corrects the headline verdict and marks the
    # model/fact-check conflict for a reviewer.
    scores = {"svm": 0.99, "gru": 0.99, "lstm": 0.99}  # ensemble -> confident FAKE
    live = {"verdict": "REAL", "source": "google_fact_check", "evidence": []}
    result = combine_verdict(scores, NO_REFERENCE, live=live, n_words=8)
    assert result["verdict"] == "REAL"
    assert result["fake_probability"] == 0.0
    assert result["needs_review"] is True  # conflicts with a confident ensemble
    assert result["evidence_backed"] is True


def test_live_verdict_takes_precedence_over_reference_override():
    scores = {"svm": 0.5, "gru": 0.5, "lstm": 0.5}
    reference = {"verdict": "FAKE", "score": config.REF_OVERRIDE_THRESHOLD + 0.05, "evidence": []}
    live = {"verdict": "REAL", "source": "google_fact_check", "evidence": []}
    result = combine_verdict(scores, reference, live=live)
    assert result["verdict"] == "REAL"


# --- A2/A3: confidence tier without ever flipping FAKE->REAL -----------------

def test_short_model_only_verdict_is_low_confidence_but_still_flags_fake():
    # A short claim the models call FAKE with no external evidence: keep the
    # FAKE label (zero-false-negative guarantee) but mark it low confidence.
    scores = {"svm": 0.97, "gru": 0.98, "lstm": 0.96}  # agree, spread tiny
    result = combine_verdict(scores, NO_REFERENCE, n_words=9)
    assert result["verdict"] == "FAKE"          # label preserved, not abstained
    assert result["confidence"] == "low"        # ... but not asserted as certain
    assert result["evidence_backed"] is False


def test_long_in_domain_agreement_is_medium_confidence():
    scores = {"svm": 0.9, "gru": 0.88, "lstm": 0.91}
    result = combine_verdict(scores, NO_REFERENCE, n_words=200)
    assert result["confidence"] == "medium"


def test_model_disagreement_is_low_confidence_regardless_of_length():
    scores = {"svm": 0.95, "gru": 0.10, "lstm": 0.50}  # spread 0.85
    result = combine_verdict(scores, NO_REFERENCE, n_words=200)
    assert result["confidence"] == "low"
    assert result["needs_review"] is True


# --- fluent fabricated-authority (ai_fluent) review signal ------------------

HIGH_AI_STYLE = {"high": True, "count": 3, "score": 1.0}


def test_ai_fluent_fabrication_routed_to_review_without_flipping_label():
    # The ai_fluent failure mode: the ensemble waves a fluent fabrication
    # through as REAL. The fabricated-authority signal must route it to a human
    # (verify the cited source) — but never flip the label or change the score.
    scores = {"svm": 0.20, "gru": 0.15, "lstm": 0.25}  # agree -> REAL
    baseline = combine_verdict(scores, NO_REFERENCE, n_words=60)
    flagged = combine_verdict(scores, NO_REFERENCE, n_words=60, ai_style=HIGH_AI_STYLE)
    assert baseline["needs_review"] is False          # ensemble alone: silent miss
    assert flagged["needs_review"] is True            # now surfaced for review
    assert flagged["verdict"] == "REAL"               # label untouched (no censorship risk)
    assert flagged["fake_probability"] == baseline["fake_probability"]  # score untouched
    assert flagged["confidence"] == "low"             # not presented as settled


def test_ai_style_high_never_flips_a_fake_verdict_off():
    # Zero-false-negative guarantee: a real hoax stays FAKE regardless.
    scores = {"svm": 0.9, "gru": 0.88, "lstm": 0.91}
    result = combine_verdict(scores, NO_REFERENCE, n_words=200, ai_style=HIGH_AI_STYLE)
    assert result["verdict"] == "FAKE"
    assert result["needs_review"] is True


def test_low_ai_style_does_not_trigger_review():
    scores = {"svm": 0.20, "gru": 0.15, "lstm": 0.25}
    result = combine_verdict(scores, NO_REFERENCE, n_words=60,
                             ai_style={"high": False, "count": 1})
    assert result["needs_review"] is False


# --- live-verdict aggregation from claim analysis ---------------------------

def test_aggregate_live_verdict_prioritises_fake():
    claim_analysis = {"claims": [
        {"claim": "a", "live": {"verdict": "REAL", "source": "google_fact_check"}},
        {"claim": "b", "live": {"verdict": "FAKE", "source": "google_fact_check"}},
    ]}
    assert _aggregate_live_verdict(claim_analysis)["verdict"] == "FAKE"


def test_aggregate_live_verdict_is_none_without_a_verdict():
    claim_analysis = {"claims": [
        {"claim": "a", "live": {"verdict": None, "source": "wikipedia"}},
        {"claim": "b", "live": None},
    ]}
    assert _aggregate_live_verdict(claim_analysis) is None
