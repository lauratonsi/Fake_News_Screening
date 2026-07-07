"""Regression gate on the committed adversarial benchmark.

The 101-scenario benchmark (``benchmarks/adversarial_scenarios.json``) tags
every FAKE scenario with a ``style`` (``human_typical`` vs ``ai_fluent``) and,
for ``ai_fluent`` items, a ``provenance``:

* **``external_dataset``** (43 scenarios) — ChatGPT-3.5 outputs (paraphrase,
  rewrite, open-ended, "information manipulation" of real true articles, and
  a small fully-fabricated set) on real human-written articles, drawn
  (fixed random seed, stratified by domain and generation method, no
  cherry-picking) from Chen & Shu's LLMFake dataset (ICLR 2024,
  github.com/llm-misinformation/llm-misinformation), PolitiFact/GossipCop
  subset. Every candidate was individually read before inclusion: roughly
  half of the raw, automatically-filtered draws turned out to be real
  transcripts, true statements that survived the "text changed" filter,
  debunking/correction text, or model refusals, and were discarded on
  manual review rather than counted. Nobody on this project *wrote* the
  ones that remain, so they are not at risk of being tuned — consciously or
  not — against this system's own detection logic. **This is the citable,
  non-circular comparison.**
* **``hand_authored``** (8 scenarios) — written for this benchmark to avoid
  overt disinformation tropes. Still useful as a data point, but cannot rule
  out the circularity risk above, so it is disclosed as exploratory rather
  than headlined.

Measured recall (the hoax "catch rate"):

* ``human_typical`` (23 scenarios, all hand-authored but modeled on
  well-documented real hoaxes): **100%**, zero false negatives. The original,
  non-negotiable guarantee — still holds.
* ``ai_fluent`` / ``external_dataset``: **74.4%** (32/43). History of this
  number as the sample grew: n=6 -> 83.3%, n=18 -> 61.1%, n=43 -> 74.4%. The
  gap first widened, then narrowed — evidence that recall on any single
  small subgroup is noisy, not that either earlier reading was wrong. The
  two mature, unchanged-since-n=18 buckets remain the most trustworthy
  individually: **75.0%** for paraphrase-generated text vs. **33.3%** for
  rewrite-generated text. Newer buckets added at n=43
  (open_ended_generation, six information_manipulation sub-strategies,
  hallucination/partially_arbitrary_generation) are each n<=10 and tracked
  in ``by_generation_method``, but not yet individually trustworthy.
* ``ai_fluent`` / ``hand_authored``: **50.0%** (4/8) — lower, but carries the
  circularity caveat above.
* The ``information_manipulation`` sub-strategies also fixed a real gap in
  the benchmark's design: every ``external_dataset`` item before n=43 was
  "long" (article-length), while every short, single-claim item was
  ``hand_authored`` — meaning length and provenance were fully confounded.
  ``information_manipulation`` (which distorts real TRUE articles into
  short, single-claim misinformation) supplied the first genuinely short
  ``external_dataset`` items, decoupling the two.

An earlier version of this benchmark used only hand-authored ai_fluent
scenarios and measured a much starker gap (28.6% recall, and a reversed
length effect where long articles scored worse). Replacing the long-form
half with real, independently-sourced text (n=6, then 18, then 43) first
narrowed that gap, then widened it, then narrowed it again as the sample
grew — evidence that circularity was real, but so is sampling variance at
small n. This is exactly why the ``external_dataset`` cohort, not the
blended number, is the one to trust and cite.

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
        data.get("by_generation_method", {}),
        data["results"],
    )


def test_zero_false_negatives_on_classic_disinformation():
    # The original, non-negotiable guarantee: no hoax written with classic
    # disinformation tropes is ever waved through as REAL.
    _, by_style_fake, _, _, _, _ = _load()
    human_typical = by_style_fake.get("human_typical")
    assert human_typical is not None, "expected a 'human_typical' style bucket"
    assert human_typical["false_negatives"] == 0
    assert human_typical["recall_fake"] == 1.0


def test_every_human_typical_fake_scenario_is_caught():
    # Row-wise dual of the above.
    _, _, _, _, _, results = _load()
    missed = [
        r["text"] for r in results
        if r["label"] == "FAKE" and r.get("style") == "human_typical" and r["predicted"] != "FAKE"
    ]
    assert not missed, f"missed classic-style hoaxes: {missed}"


def test_ai_fluent_external_dataset_recall_does_not_regress():
    # The citable, non-circular metric: recall on real, independently-sourced
    # LLM-paraphrased/rewritten misinformation (not written by this project).
    _, _, by_provenance, _, _, _ = _load()
    external = by_provenance.get("external_dataset")
    assert external is not None, "expected an 'external_dataset' provenance bucket"
    assert external["recall_fake"] >= config.AI_FLUENT_RECALL_FLOOR_EXTERNAL


def test_ai_fluent_hand_authored_recall_does_not_regress_further():
    # Exploratory metric, disclosed with a circularity caveat — tracked so it
    # cannot silently get worse, not presented as a target.
    _, _, by_provenance, _, _, _ = _load()
    hand_authored = by_provenance.get("hand_authored")
    assert hand_authored is not None, "expected a 'hand_authored' provenance bucket"
    assert hand_authored["recall_fake"] >= config.AI_FLUENT_RECALL_FLOOR_HAND_AUTHORED


def test_overall_accuracy_above_floor():
    summary, _, _, _, _, _ = _load()
    assert summary["overall"]["accuracy"] >= config.ADVERSARIAL_ACCURACY_FLOOR


def test_length_and_style_breakdowns_are_present():
    # Guards against evaluate.py silently dropping the breakdown fields.
    summary, by_style_fake, by_provenance, by_length, by_generation_method, _ = _load()
    assert {"short", "long"} <= set(by_length)
    assert {"human_typical", "ai_fluent"} <= set(by_style_fake)
    assert {"external_dataset", "hand_authored"} <= set(by_provenance)
    assert {"paraphrase_generation", "rewrite_generation"} <= set(by_generation_method)
    assert sum(s["n"] for s in by_length.values()) == summary["overall"]["n"]


def test_generation_method_recall_does_not_regress():
    # Robustness check on the external_dataset headline: is the effect
    # specific to one prompting method? Most buckets are too small (n<=10)
    # for their own dedicated floor, so every bucket present is held to the
    # shared hand_authored floor as a minimum — loose enough not to be
    # brittle at small n, still enough to catch a real collapse.
    _, _, _, _, by_generation_method, _ = _load()
    for method in ("paraphrase_generation", "rewrite_generation"):
        assert method in by_generation_method, f"expected a '{method}' bucket"
    for method, bucket in by_generation_method.items():
        assert bucket["recall_fake"] >= config.AI_FLUENT_RECALL_FLOOR_HAND_AUTHORED, (
            f"{method} recall {bucket['recall_fake']:.1%} fell below the shared floor"
        )
