"""Experiment (not yet part of the production pipeline): does a modern
sentence-embedding classifier beat the current TF-IDF+SVM / Bi-GRU / Bi-LSTM
models?

This reuses the exact same fused dataset, split protocol and shared test set
as ``src.train`` (see ``src/data.py``), so the numbers are directly
comparable to ``models/metrics.json`` and ``benchmarks/adversarial_results.json``.

Nothing here touches the production pipeline, ``requirements.txt``, or the
deployed app — it only writes ``experiments/embeddings_results.json``. The
decision to adopt (and what to retire) is made after looking at these
numbers, not before.

Usage:
    python -m experiments.embeddings_baseline
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, data  # noqa: E402

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_FILE = ROOT / "experiments" / "embeddings_results.json"
EMBEDDINGS_CACHE = ROOT / "experiments" / "_embeddings_cache.npz"


def _metrics(y_true, y_pred) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
    }


def _per_source_accuracy(test_df, y_pred) -> dict:
    out = {}
    for source in sorted(test_df["source"].unique()):
        mask = (test_df["source"] == source).to_numpy()
        out[source] = round(float(accuracy_score(test_df["target"].to_numpy()[mask], y_pred[mask])), 4)
    return out


def encode_all(model, train_df, test_df):
    """Encode train+test once and cache to disk (encoding is the slow part)."""
    if EMBEDDINGS_CACHE.exists():
        print(f">>> Loading cached embeddings from {EMBEDDINGS_CACHE}")
        cached = np.load(EMBEDDINGS_CACHE)
        return cached["X_train"], cached["X_test"]

    print(f">>> Encoding {len(train_df)} train + {len(test_df)} test texts with {EMBEDDING_MODEL_NAME}...")
    t0 = time.time()
    X_train = model.encode(train_df["full_text"].tolist(), batch_size=64, show_progress_bar=True)
    X_test = model.encode(test_df["full_text"].tolist(), batch_size=64, show_progress_bar=True)
    print(f"    encoding took {(time.time() - t0) / 60:.1f} min")

    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBEDDINGS_CACHE, X_train=X_train, X_test=X_test)
    return X_train, X_test


def main():
    from sentence_transformers import SentenceTransformer

    df = data.build_dataset()
    train_df, test_df = data.train_test_frames(df)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    print(
        f">>> Dataset: {len(df)} unique articles | train {len(train_df)} "
        f"(after COVID boost) | test {len(test_df)} — identical split to src.train"
    )

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    X_train, X_test = encode_all(model, train_df, test_df)

    print(">>> Training calibrated LinearSVC on top of the embeddings...")
    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual=False))
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba > 0.5).astype(int)

    joblib.dump({"model": clf}, ROOT / "experiments" / "embeddings_classifier.joblib")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "note": "Same fused dataset/split as src.train — directly comparable to "
                "models/metrics.json. Experimental: not yet wired into "
                "src/predict.py or requirements.txt.",
        "dataset": {
            "unique_articles": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_sources": test_df["source"].value_counts().to_dict(),
        },
        "models": {"embeddings_svm": _metrics(y_test, y_pred)},
        "per_source_accuracy": {"embeddings_svm": _per_source_accuracy(test_df, y_pred)},
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["models"], indent=2))
    print(f">>> Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
