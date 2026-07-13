"""Promotion gate: may a candidate model join the production ensemble?

The embeddings ablation in ``experiments/`` was *rejected* on honest numbers —
it scored 88.5% in-domain (vs the 95.3% SVM) and produced 10 false positives on
the 30-scenario adversarial set. Adding a weaker, more censorship-prone model to
the ensemble would have degraded it. So a candidate 4th member is not adopted
because it exists; it is adopted only if adding it measurably helps *without*
breaking the system's guarantees.

This module encodes that decision as a pure function so it is testable and the
rule is explicit rather than a judgement call. The comparison is always between
the **current** ensemble and the **augmented** ensemble (current + candidate),
measured on the same untouched in-domain test set and the same adversarial
benchmark — never the candidate in isolation.

The gates mirror the invariants the project already commits to
(``tests/test_benchmark_invariants.py``):

1. **No in-domain regression.** Augmented accuracy must not drop more than a
   small tolerance below the current ensemble.
2. **No new censorship risk.** Augmented adversarial false positives (a REAL
   statement called FAKE) must not exceed the current count.
3. **Zero-false-negative preserved.** The augmented ensemble must still catch
   every classic ("human_typical") disinformation scenario.
4. **Actual benefit.** It must *help* somewhere that matters — in-domain
   accuracy or ai_fluent recall — or it is not worth its memory budget on a
   1 GB-capped deployment. A model that is merely harmless does not get added.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleMetrics:
    """The handful of numbers the promotion decision depends on.

    ``indomain_accuracy`` — accuracy on the shared held-out test split.
    ``adversarial_false_positives`` — REAL scenarios called FAKE (censorship).
    ``classic_false_negatives`` — missed FAKE among ``human_typical`` scenarios
        (must stay 0 — the project's hard invariant).
    ``ai_fluent_recall`` — catch rate on the fluent-fabrication FAKE bucket, the
        documented weak spot a new model would be expected to help.
    """
    indomain_accuracy: float
    adversarial_false_positives: int
    classic_false_negatives: int
    ai_fluent_recall: float


def should_promote(
    current: EnsembleMetrics,
    augmented: EnsembleMetrics,
    *,
    accuracy_tolerance: float = 0.005,
) -> dict:
    """Decide whether ``augmented`` (current ensemble + candidate) should replace
    ``current``. Returns ``{"promote": bool, "reasons": [...], "benefit": {...}}``.

    ``reasons`` lists every gate that FAILED (empty-safe: a passing decision
    still returns a one-line rationale). Deterministic and side-effect free.
    """
    failures: list[str] = []

    # 1. No in-domain regression.
    if augmented.indomain_accuracy < current.indomain_accuracy - accuracy_tolerance:
        failures.append(
            f"in-domain accuracy regressed "
            f"{current.indomain_accuracy:.4f} -> {augmented.indomain_accuracy:.4f} "
            f"(tolerance {accuracy_tolerance})"
        )

    # 2. No new censorship risk.
    if augmented.adversarial_false_positives > current.adversarial_false_positives:
        failures.append(
            f"adversarial false positives increased "
            f"{current.adversarial_false_positives} -> {augmented.adversarial_false_positives} "
            f"(censorship risk)"
        )

    # 3. Zero-false-negative on classic disinformation preserved.
    if augmented.classic_false_negatives > 0:
        failures.append(
            f"classic-disinfo false negatives introduced "
            f"({augmented.classic_false_negatives}) — breaks the zero-false-negative guarantee"
        )

    # 4. Must actually help somewhere that matters.
    helps_indomain = augmented.indomain_accuracy > current.indomain_accuracy + 1e-9
    helps_ai_fluent = augmented.ai_fluent_recall > current.ai_fluent_recall + 1e-9
    if not failures and not (helps_indomain or helps_ai_fluent):
        failures.append(
            "no measurable benefit (in-domain accuracy flat and ai_fluent recall "
            "not improved) — not worth the added memory budget"
        )

    benefit = {
        "indomain_accuracy_delta": round(augmented.indomain_accuracy - current.indomain_accuracy, 4),
        "ai_fluent_recall_delta": round(augmented.ai_fluent_recall - current.ai_fluent_recall, 4),
        "false_positive_delta": augmented.adversarial_false_positives - current.adversarial_false_positives,
    }
    return {
        "promote": not failures,
        "reasons": failures or ["passes all gates: helps without regressing any guarantee"],
        "benefit": benefit,
    }
