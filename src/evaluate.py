"""Evaluation entry point.

Usage:
    python -m src.evaluate                # print the in-domain metrics report
    python -m src.evaluate --adversarial  # run the out-of-domain scenario benchmark

The in-domain metrics are produced by ``src.train`` on the shared held-out
test split and stored in ``models/metrics.json``. The adversarial benchmark
re-runs the full screening system (ensemble + reference heuristic) on the
versioned scenarios in ``benchmarks/adversarial_scenarios.json`` and writes
the measured results next to them, so every number in the README can be
traced back to a script.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from . import config


def print_in_domain() -> None:
    if not config.METRICS_FILE.exists():
        raise SystemExit("models/metrics.json not found — run `python -m src.train` first.")
    report = json.loads(config.METRICS_FILE.read_text())
    print(f"Protocol: {report['protocol']}")
    print(f"Test rows: {report['dataset']['test_rows']} "
          f"({report['dataset']['test_sources']})\n")
    header = f"{'model':<10} {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>8}"
    print(header)
    print("-" * len(header))
    for name, m in report["models"].items():
        print(f"{name:<10} {m['accuracy']:>9.2%} {m['precision_fake']:>10.2%} "
              f"{m['recall_fake']:>8.2%} {m['f1_fake']:>8.2%}")
    print("\nPer-source accuracy:")
    for name, sources in report["per_source_accuracy"].items():
        line = "  ".join(f"{s}={a:.2%}" for s, a in sources.items())
        print(f"  {name:<10} {line}")


def run_adversarial() -> None:
    from .predict import ScreeningSystem

    scenarios = json.loads(config.SCENARIOS_FILE.read_text())["scenarios"]
    # No live retrieval: the benchmark must stay offline-reproducible, and the
    # measured numbers must not depend on what external APIs return today.
    system = ScreeningSystem(with_live=False)

    results = []
    for case in scenarios:
        pred = system.predict(case["text"])
        results.append(
            {
                **case,
                "predicted": pred["verdict"],
                "fake_probability": round(pred["fake_probability"], 3),
                "model_scores": {k: round(v, 3) for k, v in pred["model_scores"].items()},
                "reference_message": pred["reference"]["message"],
                "needs_review": pred["needs_review"],
                "correct": pred["verdict"] == case["label"],
            }
        )

    def _summarize(subset: list[dict]) -> dict:
        tp = sum(r["label"] == "FAKE" and r["predicted"] == "FAKE" for r in subset)
        tn = sum(r["label"] == "REAL" and r["predicted"] == "REAL" for r in subset)
        fp = sum(r["label"] == "REAL" and r["predicted"] == "FAKE" for r in subset)
        fn = sum(r["label"] == "FAKE" and r["predicted"] == "REAL" for r in subset)
        return {
            "n": len(subset),
            "accuracy": round((tp + tn) / len(subset), 4) if subset else None,
            "precision_fake": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall_fake": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positives": fp,
            "false_negatives": fn,
            "flagged_for_review": sum(r["needs_review"] for r in subset),
        }

    domains = sorted({c["domain"] for c in scenarios})
    summary = {
        domain: _summarize([r for r in results if domain == "overall" or r["domain"] == domain])
        for domain in domains + ["overall"]
    }

    # Length breakdown: are short claims and long articles caught equally well?
    lengths = sorted({c.get("length", "short") for c in scenarios})
    by_length = {
        length: _summarize([r for r in results if r.get("length", "short") == length])
        for length in lengths
    }

    # Style breakdown, FAKE scenarios only: recall_fake (the hoax "catch rate")
    # for classic human-style disinformation (secret/leaked/whistleblower
    # tropes) vs fluent, source-attributed prose with none of those tropes.
    # NOTE: this blends two provenances within `ai_fluent` (see below) and is
    # kept for transparency, but by_provenance_ai_fluent is the citable,
    # non-circular number. See README "AI-generated disinformation is harder
    # to detect".
    fake_results = [r for r in results if r["label"] == "FAKE"]
    styles = sorted({c.get("style", "human_typical") for c in scenarios if c["label"] == "FAKE"})
    by_style_fake = {
        style: _summarize([r for r in fake_results if r.get("style", "human_typical") == style])
        for style in styles
    }

    # Provenance breakdown, ai_fluent FAKE scenarios only: `external_dataset`
    # items are ChatGPT-3.5 paraphrases of real human-written misinformation
    # drawn from Chen & Shu's LLMFake dataset (ICLR 2024) — nobody on this
    # project wrote them, so they carry no risk of being (consciously or not)
    # adversarially tuned against this specific system. `hand_authored` items
    # were written for this benchmark and, while designed only to avoid overt
    # tropes, cannot rule out that risk. This split is what makes the headline
    # ai_fluent-vs-human_typical comparison citable rather than circular.
    ai_fluent_results = [r for r in fake_results if r.get("style") == "ai_fluent"]
    provenances = sorted({r.get("provenance", "hand_authored") for r in ai_fluent_results})
    by_provenance_ai_fluent = {
        prov: _summarize([r for r in ai_fluent_results if r.get("provenance", "hand_authored") == prov])
        for prov in provenances
    }

    # Generation-method breakdown, external_dataset items only: are ChatGPT-3.5
    # paraphrases and rewrites equally hard to catch, or is the effect an
    # artifact of one specific prompting method? Robustness check for the
    # external_dataset headline number.
    external_results = [r for r in ai_fluent_results if r.get("provenance") == "external_dataset"]
    methods = sorted({r.get("generation_method", "unknown") for r in external_results})
    by_generation_method = {
        method: _summarize([r for r in external_results if r.get("generation_method", "unknown") == method])
        for method in methods
    }

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": "SVM + Bi-GRU + Bi-LSTM ensemble with reference-corpus heuristic",
        "summary": summary,
        "by_length": by_length,
        "by_style_fake": by_style_fake,
        "by_provenance_ai_fluent": by_provenance_ai_fluent,
        "by_generation_method": by_generation_method,
        "results": results,
    }
    config.ADVERSARIAL_RESULTS_FILE.write_text(json.dumps(payload, indent=2))

    print(f"{'domain':<10} {'n':>3} {'accuracy':>9} {'FP':>4} {'FN':>4} {'review':>7}")
    print("-" * 42)
    for domain, s in summary.items():
        print(f"{domain:<10} {s['n']:>3} {s['accuracy']:>9.1%} "
              f"{s['false_positives']:>4} {s['false_negatives']:>4} "
              f"{s['flagged_for_review']:>7}")

    print(f"\n{'length':<10} {'n':>3} {'accuracy':>9}")
    print("-" * 26)
    for length, s in by_length.items():
        print(f"{length:<10} {s['n']:>3} {s['accuracy']:>9.1%}")

    print(f"\n{'FAKE style':<16} {'n':>3} {'recall (catch rate)':>20}")
    print("-" * 42)
    for style, s in by_style_fake.items():
        recall = f"{s['recall_fake']:.1%}" if s["recall_fake"] is not None else "n/a"
        print(f"{style:<16} {s['n']:>3} {recall:>20}")

    print(f"\n{'ai_fluent provenance':<20} {'n':>3} {'recall (catch rate)':>20}")
    print("-" * 46)
    for prov, s in by_provenance_ai_fluent.items():
        recall = f"{s['recall_fake']:.1%}" if s["recall_fake"] is not None else "n/a"
        print(f"{prov:<20} {s['n']:>3} {recall:>20}")

    print(f"\n{'generation method (external)':<30} {'n':>3} {'recall':>10}")
    print("-" * 46)
    for method, s in by_generation_method.items():
        recall = f"{s['recall_fake']:.1%}" if s["recall_fake"] is not None else "n/a"
        print(f"{method:<30} {s['n']:>3} {recall:>10}")

    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\nErrors:")
        for r in wrong:
            kind = "FP (censorship risk)" if r["label"] == "REAL" else "FN (missed hoax)"
            style = r.get("style", "")
            print(f"  [{r['domain']}/{style}] {kind}: {r['text'][:70]}...")
    print(f"\nResults written to {config.ADVERSARIAL_RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adversarial", action="store_true",
                        help="run the out-of-domain scenario benchmark")
    args = parser.parse_args()
    if args.adversarial:
        run_adversarial()
    else:
        print_in_domain()


if __name__ == "__main__":
    main()
