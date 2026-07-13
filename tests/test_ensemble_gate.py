"""Unit tests for the pure ensemble promotion gate."""
from src.ensemble_gate import EnsembleMetrics, should_promote

# A realistic incumbent: strong in-domain, no censorship, zero classic FN, and a
# weak ai_fluent recall (the documented gap a candidate would try to help).
CURRENT = EnsembleMetrics(
    indomain_accuracy=0.9458,
    adversarial_false_positives=2,
    classic_false_negatives=0,
    ai_fluent_recall=0.60,
)


def test_promotes_when_it_helps_ai_fluent_without_regressing():
    better = EnsembleMetrics(0.9460, 2, 0, 0.68)  # same acc/FP, higher ai_fluent recall
    decision = should_promote(CURRENT, better)
    assert decision["promote"] is True
    assert decision["benefit"]["ai_fluent_recall_delta"] > 0


def test_rejects_in_domain_regression():
    # The rejected embeddings ablation shape: markedly weaker in-domain.
    worse = EnsembleMetrics(0.8854, 2, 0, 0.70)
    decision = should_promote(CURRENT, worse)
    assert decision["promote"] is False
    assert any("in-domain accuracy regressed" in r for r in decision["reasons"])


def test_rejects_new_false_positives():
    censoring = EnsembleMetrics(0.95, 5, 0, 0.75)  # gains acc but adds censorship
    decision = should_promote(CURRENT, censoring)
    assert decision["promote"] is False
    assert any("false positives increased" in r for r in decision["reasons"])


def test_rejects_broken_zero_false_negative():
    leaky = EnsembleMetrics(0.96, 2, 1, 0.80)  # misses a classic hoax
    decision = should_promote(CURRENT, leaky)
    assert decision["promote"] is False
    assert any("zero-false-negative" in r for r in decision["reasons"])


def test_rejects_harmless_but_useless_candidate():
    flat = EnsembleMetrics(0.9458, 2, 0, 0.60)  # identical everywhere
    decision = should_promote(CURRENT, flat)
    assert decision["promote"] is False
    assert any("no measurable benefit" in r for r in decision["reasons"])


def test_small_accuracy_dip_within_tolerance_is_allowed_if_ai_fluent_helps():
    # Accuracy dips 0.3pp (< default 0.5pp tolerance) but ai_fluent recall jumps.
    tradeoff = EnsembleMetrics(0.9428, 2, 0, 0.78)
    decision = should_promote(CURRENT, tradeoff)
    assert decision["promote"] is True


def test_tolerance_boundary_is_respected():
    # Exactly at the tolerance edge: a 0.6pp drop with a tight 0.5pp tolerance fails.
    dip = EnsembleMetrics(CURRENT.indomain_accuracy - 0.006, 2, 0, 0.90)
    assert should_promote(CURRENT, dip, accuracy_tolerance=0.005)["promote"] is False
