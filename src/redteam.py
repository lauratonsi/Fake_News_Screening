"""Red-team stress generator for the screening system.

Usage:
    python -m src.redteam --generate 60        # grow the red-team stress set
    python -m src.redteam --evaluate           # score the stress set (needs models)

Purpose and the honesty guardrails that make it worth having:

The hand-authored and external_dataset benchmark buckets are the *citable*
numbers. A generator that writes into them would be circular — worse, a
generator that simply emits what this project's own ai_style / manipulation
layers already match would inflate the score by construction, exactly the trap
the README warns about. So this module is built to avoid that:

* **Separate bucket.** Generated scenarios go ONLY to
  ``config.REDTEAM_SCENARIOS_FILE``; :func:`write_redteam` refuses to write the
  citable benchmark. The stress recall is reported on its own, tagged synthetic.
* **Adversaries that EVADE, not flatter.** The most important category is
  ``plain_fluent`` — fluent false claims carrying *none* of the manipulation
  tropes and *none* of the fabricated-authority markers, i.e. the genuine blind
  spot. ``authority_fluent`` and ``trope_laden`` are included too, tagged, so an
  evaluation shows *which* category slips through the models rather than
  confirming the regex catches what the regex was written to catch.
* **False-positive control.** A curated set of plainly-true statements is mixed
  in so the stress set measures censorship risk, not just catch rate.
* **Deterministic.** Seeded, so a run is reproducible and diffable.

Generation is a pure function (no LLM, no network); only ``--evaluate`` loads
the models.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from . import config

# --- fillers -----------------------------------------------------------------
_ROLES = ["The mayor of", "The transport authority of", "The health department of",
          "A city council in", "The education ministry of", "The water utility of"]
_PLACES = ["Turin", "Lyon", "Valencia", "Bristol", "Graz", "Malmö", "Porto", "Utrecht"]
_ACTIONS = ["announced a permanent ban on", "made mandatory the use of",
            "quietly rolled back", "began taxing", "outlawed the sale of",
            "started requiring a licence for"]
_OBJECTS = ["home gardening", "second-hand bicycles", "wood-burning stoves",
            "private rainwater collection", "gas cookers", "diesel scooters",
            "backyard chickens", "single-use umbrellas"]
_WHEN = ["starting next month", "effective immediately", "from January",
         "within ninety days", "beginning this autumn"]

# Fabricated-authority scaffolding (this category SHOULD trip ai_style — it is
# tagged so an evaluation can separate it from the plain_fluent blind spot).
_JOURNALS = ["the Journal of Applied Metabolism", "the European Review of Public Health",
             "the Institute for Urban Policy"]
_METRICS = ["cut commuting time by 38 percent", "reduced household energy use by 47 percent",
            "lowered infection rates by an average of 22 percent"]

# Classic-trope scaffolding (this category SHOULD trip manipulation.py).
_TROPES = [
    "SHOCKING: a leaked memo reveals the truth about {place}'s new ban on {object} — they don't want you to know, so share before it's deleted!",
    "A whistleblower exposed the real reason {place} secretly outlawed {object} — wake up and do your own research!",
    "They don't want you to know: {place}'s officials quietly banned {object} behind closed doors. Spread the word before it's removed!",
]

_REAL_CONTROLS = [
    "Water boils at about 100 degrees Celsius at sea-level atmospheric pressure.",
    "The Earth completes one orbit of the Sun in roughly 365 days.",
    "Paris is the capital of France.",
    "The human heart pumps blood through the circulatory system.",
    "A leap year has 366 days instead of the usual 365.",
    "Sound travels more slowly than light through air.",
    "The Pacific is the largest of Earth's oceans.",
    "Photosynthesis lets plants convert sunlight into chemical energy.",
]

_DOMAINS = ["politics", "science", "economy", "technology", "mixed"]


def _plain_fluent(rng: random.Random) -> str:
    return (f"{rng.choice(_ROLES)} {rng.choice(_PLACES)} {rng.choice(_ACTIONS)} "
            f"{rng.choice(_OBJECTS)} {rng.choice(_WHEN)}.")


def _authority_fluent(rng: random.Random) -> str:
    return (f"A peer-reviewed study in {rng.choice(_JOURNALS)} reported that "
            f"banning {rng.choice(_OBJECTS)} {rng.choice(_METRICS)}.")


def _trope_laden(rng: random.Random) -> str:
    return rng.choice(_TROPES).format(place=rng.choice(_PLACES), object=rng.choice(_OBJECTS))

_GENERATORS = {
    "plain_fluent": (_plain_fluent, "ai_fluent"),
    "authority_fluent": (_authority_fluent, "ai_fluent"),
    "trope_laden": (_trope_laden, "human_typical"),
}


def generate_scenarios(n_fake: int = 60, *, seed: int = config.REDTEAM_SEED,
                       include_controls: bool = True) -> list[dict]:
    """Deterministically build a red-team stress set.

    ``n_fake`` fabricated FAKE scenarios are spread evenly across the three
    categories (plain_fluent, authority_fluent, trope_laden), plus the curated
    REAL controls when ``include_controls``. Every scenario is tagged
    ``provenance="redteam_synthetic"`` and a ``category``.
    """
    rng = random.Random(seed)
    categories = list(_GENERATORS)
    scenarios: list[dict] = []
    for i in range(n_fake):
        cat = categories[i % len(categories)]
        make, style = _GENERATORS[cat]
        scenarios.append({
            "text": make(rng),
            "label": "FAKE",
            "style": style,
            "category": cat,
            "domain": rng.choice(_DOMAINS),
            "provenance": "redteam_synthetic",
        })
    if include_controls:
        for text in _REAL_CONTROLS:
            scenarios.append({
                "text": text,
                "label": "REAL",
                "style": "reporting",
                "category": "real_control",
                "domain": "mixed",
                "provenance": "redteam_synthetic",
            })
    return scenarios


_REQUIRED = ("text", "label", "style", "category", "domain", "provenance")


def validate_scenario(s: dict) -> list[str]:
    """Return a list of schema problems (empty = valid)."""
    problems = []
    for key in _REQUIRED:
        if not s.get(key):
            problems.append(f"missing '{key}'")
    if s.get("label") not in ("FAKE", "REAL"):
        problems.append("label must be FAKE or REAL")
    if s.get("provenance") != "redteam_synthetic":
        problems.append("provenance must be 'redteam_synthetic'")
    return problems


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def dedupe(new: list[dict], existing_texts: set[str]) -> list[dict]:
    """Drop scenarios whose normalized text is already present or repeated."""
    seen = set(existing_texts)
    out = []
    for s in new:
        key = _norm(s.get("text"))
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def write_redteam(scenarios: list[dict], path: Path | None = None) -> dict:
    """Append validated, deduped scenarios to the red-team file — never the
    citable benchmark. Refuses if pointed at ``config.SCENARIOS_FILE``.
    """
    path = path or config.REDTEAM_SCENARIOS_FILE
    if Path(path).resolve() == config.SCENARIOS_FILE.resolve():
        raise ValueError(
            "refusing to write generated scenarios into the citable benchmark "
            "(adversarial_scenarios.json) — synthetic data stays in the red-team bucket."
        )
    invalid = [(i, validate_scenario(s)) for i, s in enumerate(scenarios)]
    invalid = [(i, p) for i, p in invalid if p]
    if invalid:
        raise ValueError(f"invalid scenarios: {invalid[:3]}")

    existing = []
    if Path(path).exists():
        existing = json.loads(Path(path).read_text(encoding="utf-8")).get("scenarios", [])
    fresh = dedupe(scenarios, {_norm(s["text"]) for s in existing})
    merged = existing + fresh
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"description": "Synthetic red-team stress set — exploratory, NOT the "
                        "citable benchmark. See src/redteam.py.",
         "scenarios": merged}, indent=2), encoding="utf-8")
    return {"added": len(fresh), "total": len(merged), "path": str(path)}


def evaluate(path: Path | None = None) -> dict:
    """Score the red-team set with the full system, per category. Loads models."""
    path = path or config.REDTEAM_SCENARIOS_FILE
    scenarios = json.loads(Path(path).read_text(encoding="utf-8"))["scenarios"]
    from .predict import ScreeningSystem

    system = ScreeningSystem(with_live=False)
    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "caught": 0, "fp": 0, "review": 0})
    for c in scenarios:
        pred = system.predict(c["text"], explain=False)
        b = buckets[c["category"]]
        b["n"] += 1
        b["review"] += int(pred["needs_review"])
        if c["label"] == "FAKE":
            b["caught"] += int(pred["verdict"] == "FAKE")
        elif pred["verdict"] == "FAKE":
            b["fp"] += 1
    report = {cat: {**v,
                    "recall": round(v["caught"] / v["n"], 3) if v["n"] and cat != "real_control" else None,
                    "review_rate": round(v["review"] / v["n"], 3) if v["n"] else None}
              for cat, v in buckets.items()}
    config.REDTEAM_RESULTS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", type=int, metavar="N", default=0,
                        help="generate N fabricated scenarios and append to the red-team file")
    parser.add_argument("--evaluate", action="store_true",
                        help="score the red-team set with the full system (loads models)")
    args = parser.parse_args()

    if args.generate:
        result = write_redteam(generate_scenarios(args.generate))
        print(f"Red-team set: +{result['added']} new (total {result['total']}) -> {result['path']}")
    if args.evaluate:
        report = evaluate()
        print(json.dumps(report, indent=2))
    if not args.generate and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    main()
