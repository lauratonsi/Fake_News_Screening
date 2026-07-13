"""Active learning: fold verified user corrections into the *models*.

Usage:
    python -m src.retrain_from_feedback [--dry-run] [--no-rnns]

``incorporate_feedback.py`` closes the retrieval half of the loop — a
correction becomes a retrievable snippet. It deliberately stops there, noting
that touching the SVM/RNN weights "would need far more care (adversarial
submissions, label noise, class balance) than a single append-only log can
justify". This module is that care, made explicit and testable, so a batch of
*verified* corrections can safely reach the classifier weights:

* **Verified only.** A record counts when the user marked the result wrong
  (``agrees is False``) AND supplied the correct label — the same bar
  ``incorporate_feedback`` uses. Agreements and unlabeled thumbs-down are
  ignored.
* **Idempotent.** Folded records are stamped ``retrained`` in the log, so
  re-running never double-counts them.
* **Minimum batch.** Fewer than ``RETRAIN_MIN_CORRECTIONS`` verified, novel
  corrections and it refuses — a handful of clicks must not reshape the model.
* **Capped and balanced.** Corrections may become at most
  ``RETRAIN_MAX_FEEDBACK_FRACTION`` of the training rows, split across the two
  classes, so feedback nudges the boundary but can never dominate it.
* **No leakage, no contradictions.** Corrections join the TRAINING split only
  (the held-out test set stays clean, so the after-metrics are honest), and any
  correction whose text already sits in training under the OPPOSITE label — the
  classic poisoning shape — is dropped.
* **Regression-aware.** After retraining it re-scores the *same untouched* test
  set and flags any drop beyond ``RETRAIN_REGRESSION_TOLERANCE`` loudly, so a
  bad batch is caught (revert with git) rather than silently deployed.

The planning logic (which corrections survive the guards) is a pure function,
:func:`plan_corrections`, unit-tested without any ML dependency. Only
:func:`retrain_from_feedback` touches models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .feedback import load_feedback

_LABEL_TO_TARGET = {"REAL": 0, "FAKE": 1}


def plan_corrections(
    records: list[dict],
    train_targets_by_text: dict[str, set[int]],
    test_texts: set[str],
    n_train: int,
    *,
    min_chars: int = config.RETRAIN_MIN_CORRECTION_CHARS,
    max_fraction: float = config.RETRAIN_MAX_FEEDBACK_FRACTION,
    min_corrections: int = config.RETRAIN_MIN_CORRECTIONS,
) -> dict:
    """Decide which verified corrections may be folded into training.

    Pure and dependency-free so every guard is unit-testable. ``records`` is the
    raw feedback log; ``train_targets_by_text`` maps an existing training
    ``full_text`` to the set of labels it already carries (for dedup + the
    contradiction guard); ``test_texts`` is the held-out test ``full_text`` set
    (leakage guard); ``n_train`` is the current training row count (for the cap).

    Returns a dict with the accepted rows (``full_text``/``target``/``source``),
    the indices of the log records they came from (to stamp ``retrained``), a
    per-reason tally of what was skipped, and whether the minimum-batch gate
    passed.
    """
    accepted: list[dict] = []
    accepted_idx: list[int] = []
    seen_in_batch: set[str] = set()
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for i, r in enumerate(records):
        if r.get("agrees") is not False or r.get("correct_label") not in _LABEL_TO_TARGET:
            continue  # not an actionable correction at all (don't count as "skipped")
        if r.get("retrained"):
            skip("already_retrained")
            continue
        text = str(r.get("text") or "").strip()
        target = _LABEL_TO_TARGET[r["correct_label"]]
        if len(text) < min_chars:
            skip("too_short")
            continue
        if text in seen_in_batch:
            skip("duplicate_in_batch")
            continue
        if text in test_texts:
            skip("would_leak_into_test")  # never train on a held-out eval example
            continue
        existing = train_targets_by_text.get(text)
        if existing is not None:
            if target in existing:
                skip("already_in_training")  # same text+label already learned
            else:
                skip("contradicts_training_label")  # poisoning shape: opposite label
            continue
        seen_in_batch.add(text)
        accepted.append({"full_text": text, "target": target, "source": "feedback"})
        accepted_idx.append(i)

    # Balanced cap: at most `budget` rows total, split evenly across the classes
    # so one label cannot swamp the other. Deterministic (keep earliest records).
    budget = int(n_train * max_fraction)
    capped = _balance_and_cap(accepted, accepted_idx, budget)
    n_capped_out = len(accepted) - len(capped["rows"])
    if n_capped_out:
        skip_key = "over_cap"
        skipped[skip_key] = skipped.get(skip_key, 0) + n_capped_out

    return {
        "rows": capped["rows"],
        "record_indices": capped["indices"],
        "skipped": skipped,
        "n_accepted": len(capped["rows"]),
        "gate_passed": len(capped["rows"]) >= min_corrections,
        "min_corrections": min_corrections,
        "budget": budget,
    }


def _balance_and_cap(rows: list[dict], indices: list[int], budget: int) -> dict:
    """Keep at most ``budget`` rows, balanced across the two classes.

    A zero/negative budget keeps nothing. Earliest records win (deterministic).
    """
    if budget <= 0:
        return {"rows": [], "indices": []}
    per_class = max(1, budget // 2)
    kept_rows: list[dict] = []
    kept_idx: list[int] = []
    counts = {0: 0, 1: 0}
    for row, idx in zip(rows, indices):
        t = row["target"]
        if counts[t] < per_class and len(kept_rows) < budget:
            kept_rows.append(row)
            kept_idx.append(idx)
            counts[t] += 1
    return {"rows": kept_rows, "indices": kept_idx}


def _format_skips(skipped: dict[str, int]) -> str:
    if not skipped:
        return "none"
    return ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))


def retrain_from_feedback(
    feedback_path: Path | None = None,
    *,
    dry_run: bool = False,
    retrain_rnns: bool = True,
) -> dict:
    """Retrain the classifiers on the base dataset augmented with verified,
    guard-passing corrections. Returns a summary dict.

    Imports the heavy ML/data modules lazily so the pure planner (and its tests)
    never pay for TensorFlow/sklearn.
    """
    import numpy as np
    import pandas as pd

    from . import data
    feedback_path = feedback_path or config.FEEDBACK_LOG

    records = load_feedback(feedback_path)
    df = data.build_dataset()
    train_df, test_df = data.train_test_frames(df)

    train_targets_by_text: dict[str, set[int]] = {}
    for txt, tgt in zip(train_df["full_text"], train_df["target"]):
        train_targets_by_text.setdefault(str(txt), set()).add(int(tgt))
    test_texts = set(map(str, test_df["full_text"]))

    plan = plan_corrections(records, train_targets_by_text, test_texts, len(train_df))
    summary = {
        "n_accepted": plan["n_accepted"],
        "skipped": plan["skipped"],
        "gate_passed": plan["gate_passed"],
        "budget": plan["budget"],
        "dry_run": dry_run,
    }

    if not plan["gate_passed"]:
        summary["status"] = "refused_below_minimum"
        summary["message"] = (
            f"{plan['n_accepted']} novel verified correction(s) after guards — "
            f"need {plan['min_corrections']} to retrain. Skipped: "
            f"{_format_skips(plan['skipped'])}."
        )
        return summary

    if dry_run:
        by_class = {"real": sum(r["target"] == 0 for r in plan["rows"]),
                    "fake": sum(r["target"] == 1 for r in plan["rows"])}
        summary["status"] = "dry_run"
        summary["would_add"] = by_class
        return summary

    from . import train

    aug = pd.DataFrame(plan["rows"], columns=["full_text", "target", "source"])
    augmented_train = pd.concat([train_df, aug], ignore_index=True)
    augmented_train = augmented_train.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)

    y_test = test_df["target"].to_numpy()
    before = _load_ensemble_accuracy()

    probas = {"svm": train.train_svm(augmented_train, test_df)}
    if retrain_rnns:
        probas.update(train.train_rnns(augmented_train, test_df))
        probas["ensemble"] = np.mean([probas["svm"], probas["gru"], probas["lstm"]], axis=0)
        headline = probas["ensemble"]
    else:
        headline = probas["svm"]

    after_acc = float((( headline > 0.5).astype(int) == y_test).mean())
    summary["test_accuracy_after"] = round(after_acc, 4)
    summary["test_accuracy_before"] = before
    summary["regressed"] = (
        before is not None and after_acc < before - config.RETRAIN_REGRESSION_TOLERANCE
    )

    # Refresh the metrics report for whatever models were retrained.
    _write_metrics(probas, test_df, augmented_train, df, retrain_rnns)

    # Stamp the folded records so re-running is idempotent.
    for i in plan["record_indices"]:
        records[i]["retrained"] = True
    feedback_path.write_text(
        ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n") if records else "",
        encoding="utf-8",
    )

    summary["status"] = "retrained"
    return summary


def _load_ensemble_accuracy() -> float | None:
    """Best-effort read of the current ensemble test accuracy (pre-retrain)."""
    if not config.METRICS_FILE.exists():
        return None
    try:
        report = json.loads(config.METRICS_FILE.read_text())
        models = report.get("models", {})
        node = models.get("ensemble") or models.get("svm")
        return float(node["accuracy"]) if node else None
    except (ValueError, KeyError, TypeError):
        return None


def _write_metrics(probas, test_df, augmented_train, base_df, retrain_rnns: bool) -> None:
    from datetime import datetime, timezone

    from . import train

    y_test = test_df["target"].to_numpy()
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "feedback-augmented retrain: base dataset + guard-passing "
        "verified corrections on the training side only (test set untouched)",
        "dataset": {
            "base_unique_articles": int(len(base_df)),
            "train_rows": int(len(augmented_train)),
            "feedback_rows": int((augmented_train["source"] == "feedback").sum()),
            "test_rows": int(len(test_df)),
        },
        "models": {},
        "per_source_accuracy": {},
    }
    for name, proba in probas.items():
        y_pred = (proba > 0.5).astype(int)
        report["models"][name] = train._metrics(y_test, y_pred)
        report["per_source_accuracy"][name] = train._per_source_accuracy(test_df, y_pred)
    config.METRICS_FILE.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be folded in, without training")
    parser.add_argument("--no-rnns", action="store_true",
                        help="retrain only the SVM baseline (skip the slower RNNs)")
    args = parser.parse_args()

    result = retrain_from_feedback(dry_run=args.dry_run, retrain_rnns=not args.no_rnns)
    status = result["status"]

    if status == "refused_below_minimum":
        print(result["message"])
        return
    if status == "dry_run":
        add = result["would_add"]
        print(f"Would retrain on {result['n_accepted']} verified correction(s): "
              f"+{add['real']} real, +{add['fake']} fake "
              f"(cap {result['budget']} rows). Skipped: {_format_skips(result['skipped'])}.")
        return

    print(f"Retrained on {result['n_accepted']} verified correction(s). "
          f"Test accuracy {result['test_accuracy_before']} -> {result['test_accuracy_after']}.")
    if result["regressed"]:
        print("  ⚠️  REGRESSION: test accuracy dropped beyond tolerance. "
              "Inspect the feedback batch and consider reverting the models (git checkout).")


if __name__ == "__main__":
    main()
