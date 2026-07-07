"""Regression gate on the committed adversarial benchmark.

The 64-scenario benchmark (``benchmarks/adversarial_scenarios.json``) tags
every FAKE scenario with a ``style`` (``human_typical`` vs ``ai_fluent``) and,
for ``ai_fluent`` items, a ``provenance``:

* **``external_dataset``** (6 scenarios) — ChatGPT-3.5 paraphrases of real
  human-written misinformation, drawn (fixed random seed, unbiased selection)
  from Chen & Shu's LLMFake dataset (ICLR 2024,
  github.com/llm-misinformation/llm-misinformation), PolitiFact/GossipCop
  subset. Nobody on this project wrote these, so they are not at risk of
  being tuned — consciously or not — against this system's own detection
  logic. **This is the citable, non-circular comparison.**
* **``hand_authored``** (8 scenarios) — written for this benchmark to avoid
  overt disinformation tropes. Still useful as a data point, but cannot rule
  out the circularity risk above, so it is disclosed as exploratory rather
  than headlined.

Measured recall (the hoax "catch rate"):

* ``human_typical`` (23 scenarios, all hand-authored but modeled on
  well-documented real hoaxes): **100%**, zero false negatives. The original,
  non-negotiable guarantee — still holds.
* ``ai_fluent`` / ``external_dataset``: **83.3%** (5/6) — a real but modest
  gap versus classic-style disinformation in the same domains (also 100%).
* ``ai_fluent`` / ``hand_authored``: **50.0%** (4/8) — lower, but carries the
  circularity caveat above.

An earlier version of this benchmark used only hand-authored ai_fluent
scenarios and measured a much starker gap (28.6% recall, and a reversed
length effect where long articles scored worse). Replacing the long-form
half with real, independently-sourced text roughly halved the measured gap
and reversed the length finding — direct evidence that the hand-authored
cohort had been inadvertently (or not) tuned against this system. This is
exactly why the ``external_dataset`` cohort, not the blended number, is the
one to trust and cite.

Re-run ``python -m src.evaluate --adversarial`` to refresh the results file.
"""
import json

from src import config


def _load():
    data = json.loads(config.ADVERSARIAL_RESULTS_FILE.read_text())
    return (
        data["summary"],
        data.get("by_style_fake", {}),
        data.get("by_provenance_ai_fluent", {}),
        data.get("by_length", {}),
        data["results"],
    )


def test_zero_false_negatives_on_classic_disinformation():
    # The original, non-negotiable guarantee: no hoax written with classic
    # disinformation tropes is ever waved through as REAL.
    _, by_style_fake, _, _, _ = _load()
    human_typical = by_style_fake.get("human_typical")
    assert human_typical is not None, "expected a 'human_typical' style bucket"
    assert human_typical["false_negatives"] == 0
    assert human_typical["recall_fake"] == 1.0


def test_every_human_typical_fake_scenario_is_caught():
    # Row-wise dual of the above.
    _, _, _, _, results = _load()
    missed = [
        r["text"] for r in results
        if r["label"] == "FAKE" and r.get("style") == "human_typical" and r["predicted"] != "FAKE"
    ]
    assert not missed, f"missed classic-style hoaxes: {missed}"


def test_ai_fluent_external_dataset_recall_does_not_regress():
    # The citable, non-circular metric: recall on real, independently-sourced
    # LLM-paraphrased misinformation (not written by this project).
    _, _, by_provenance, _, _ = _load()
    external = by_provenance.get("external_dataset")
    assert external is not None, "expected an 'external_dataset' provenance bucket"
    assert external["recall_fake"] >= config.AI_FLUENT_RECALL_FLOOR_EXTERNAL


def test_ai_fluent_hand_authored_recall_does_not_regress_further():
    # Exploratory metric, disclosed with a circularity caveat — tracked so it
    # cannot silently get worse, not presented as a target.
    _, _, by_provenance, _, _ = _load()
    hand_authored = by_provenance.get("hand_authored")
    assert hand_authored is not None, "expected a 'hand_authored' provenance bucket"
    assert hand_authored["recall_fake"] >= config.AI_FLUENT_RECALL_FLOOR_HAND_AUTHORED


def test_overall_accuracy_above_floor():
    summary, _, _, _, _ = _load()
    assert summary["overall"]["accuracy"] >= config.ADVERSARIAL_ACCURACY_FLOOR


def test_length_and_style_breakdowns_are_present():
    # Guards against evaluate.py silently dropping the breakdown fields.
    summary, by_style_fake, by_provenance, by_length, _ = _load()
    assert {"short", "long"} <= set(by_length)
    assert {"human_typical", "ai_fluent"} <= set(by_style_fake)
    assert {"external_dataset", "hand_authored"} <= set(by_provenance)
    assert sum(s["n"] for s in by_length.values()) == summary["overall"]["n"]
