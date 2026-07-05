"""Evaluation entry point.

Usage:
    python -m src.evaluate                # print the in-domain metrics report
    python -m src.evaluate --adversarial  # run the 30 out-of-domain scenarios

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

    domains = sorted({c["domain"] for c in scenarios})
    summary = {}
    for domain in domains + ["overall"]:
        subset = [r for r in results if domain == "overall" or r["domain"] == domain]
        tp = sum(r["label"] == "FAKE" and r["predicted"] == "FAKE" for r in subset)
        tn = sum(r["label"] == "REAL" and r["predicted"] == "REAL" for r in subset)
        fp = sum(r["label"] == "REAL" and r["predicted"] == "FAKE" for r in subset)
        fn = sum(r["label"] == "FAKE" and r["predicted"] == "REAL" for r in subset)
        summary[domain] = {
            "n": len(subset),
            "accuracy": round((tp + tn) / len(subset), 4),
            "precision_fake": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall_fake": round(tp / (tp + fn), 4) if tp + fn else None,
            "false_positives": fp,
            "false_negatives": fn,
            "flagged_for_review": sum(r["needs_review"] for r in subset),
        }

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": "SVM + Bi-GRU + Bi-LSTM ensemble with reference-corpus heuristic",
        "summary": summary,
        "results": results,
    }
    config.ADVERSARIAL_RESULTS_FILE.write_text(json.dumps(payload, indent=2))

    print(f"{'domain':<10} {'n':>3} {'accuracy':>9} {'FP':>4} {'FN':>4} {'review':>7}")
    print("-" * 42)
    for domain, s in summary.items():
        print(f"{domain:<10} {s['n']:>3} {s['accuracy']:>9.1%} "
              f"{s['false_positives']:>4} {s['false_negatives']:>4} "
              f"{s['flagged_for_review']:>7}")
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\nErrors:")
        for r in wrong:
            kind = "FP (censorship risk)" if r["label"] == "REAL" else "FN (missed hoax)"
            print(f"  [{r['domain']}] {kind}: {r['text'][:70]}...")
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
