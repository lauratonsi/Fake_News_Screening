"""Run the same 30 adversarial scenarios through the experimental embeddings
classifier, for direct comparison with benchmarks/adversarial_results.json.

Usage:
    python -m experiments.embeddings_adversarial
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from experiments.embeddings_baseline import EMBEDDING_MODEL_NAME  # noqa: E402

RESULTS_FILE = ROOT / "experiments" / "embeddings_adversarial_results.json"


def main():
    from sentence_transformers import SentenceTransformer

    scenarios = json.loads(config.SCENARIOS_FILE.read_text())["scenarios"]
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    clf = joblib.load(ROOT / "experiments" / "embeddings_classifier.joblib")["model"]

    texts = [s["text"] for s in scenarios]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    probs = clf.predict_proba(embeddings)[:, 1]

    results = []
    for case, prob in zip(scenarios, probs):
        predicted = "FAKE" if prob > 0.5 else "REAL"
        results.append({**case, "predicted": predicted, "fake_probability": round(float(prob), 3),
                         "correct": predicted == case["label"]})

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
            "false_positives": fp,
            "false_negatives": fn,
        }

    payload = {"system": "embeddings_svm (experimental)", "summary": summary, "results": results}
    RESULTS_FILE.write_text(json.dumps(payload, indent=2))

    print(f"{'domain':<10} {'n':>3} {'accuracy':>9} {'FP':>4} {'FN':>4}")
    print("-" * 34)
    for domain, s in summary.items():
        print(f"{domain:<10} {s['n']:>3} {s['accuracy']:>9.1%} {s['false_positives']:>4} {s['false_negatives']:>4}")
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\nErrors:")
        for r in wrong:
            kind = "FP" if r["label"] == "REAL" else "FN"
            print(f"  [{r['domain']}] {kind}: {r['text'][:70]}...")
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
