"""Train every model of the system with one shared protocol.

Usage:
    python -m src.train

Protocol (see data.py):
    1. Fuse ISOT + WELFake + COVID, deduplicate, shuffle (seed 42).
    2. Stratified 80/20 split; COVID balancing/boost on the training side only.
    3. Train the TF-IDF + calibrated LinearSVC baseline on the full training set.
    4. Train light bidirectional GRU and LSTM on a 5,000-article subsample.
    5. Evaluate every model AND their average (the ensemble) on the same
       held-out test set; write models/metrics.json.
    6. Export the reference corpus used by the demo's similarity heuristic.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# Keep TensorFlow on CPU and on the legacy Keras 2 API (matches tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from . import config, data


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


def train_svm(train_df, test_df):
    print(">>> Training TF-IDF + calibrated LinearSVC baseline...")
    vectorizer = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES, stop_words="english")
    X_train = vectorizer.fit_transform(train_df["full_text"])
    X_test = vectorizer.transform(test_df["full_text"])

    model = CalibratedClassifierCV(LinearSVC(class_weight="balanced", dual=False))
    model.fit(X_train, train_df["target"].to_numpy())

    proba = model.predict_proba(X_test)[:, 1]
    joblib.dump({"model": model, "vectorizer": vectorizer}, config.SVM_FILE)
    print(f"    saved -> {config.SVM_FILE}")
    return proba


def train_rnns(train_df, test_df):
    import tf_keras  # noqa: F401  (ensures the legacy backend is present)
    from tf_keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU, LSTM
    from tf_keras.models import Sequential
    from tf_keras.preprocessing.sequence import pad_sequences
    from tf_keras.preprocessing.text import Tokenizer

    sample = train_df.sample(
        n=min(config.RNN_TRAIN_SAMPLE, len(train_df)), random_state=config.SEED
    )
    print(f">>> Training RNNs on a {len(sample)}-article subsample (CPU budget)...")

    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, lower=True)
    tokenizer.fit_on_texts(sample["full_text"].values)
    joblib.dump(tokenizer, config.TOKENIZER_FILE)

    def to_seq(texts):
        return pad_sequences(
            tokenizer.texts_to_sequences(texts), maxlen=config.MAX_LEN, padding="post"
        ).astype(np.int32)

    X_train = to_seq(sample["full_text"].values)
    y_train = sample["target"].to_numpy().astype(np.float32)
    X_test = to_seq(test_df["full_text"].values)

    probas = {}
    for arch, layer in (("gru", GRU), ("lstm", LSTM)):
        print(f"    {arch.upper()}...")
        model = Sequential(
            [
                Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM, input_length=config.MAX_LEN),
                Bidirectional(layer(config.RNN_UNITS)),
                Dense(16, activation="relu"),
                Dropout(0.5),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.fit(
            X_train,
            y_train,
            epochs=config.RNN_EPOCHS,
            batch_size=config.RNN_BATCH_SIZE,
            verbose=2,
        )
        path = config.GRU_FILE if arch == "gru" else config.LSTM_FILE
        model.save(path)
        print(f"    saved -> {path}")
        probas[arch] = model.predict(X_test, verbose=0)[:, 0]
    return probas


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = data.build_dataset()
    train_df, test_df = data.train_test_frames(df)
    y_test = test_df["target"].to_numpy()
    print(
        f">>> Dataset: {len(df)} unique articles | train {len(train_df)} "
        f"(after COVID boost) | test {len(test_df)}"
    )

    probas = {"svm": train_svm(train_df, test_df)}
    probas.update(train_rnns(train_df, test_df))
    probas["ensemble"] = np.mean([probas["svm"], probas["gru"], probas["lstm"]], axis=0)

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "shared stratified 80/20 split, dedup before split, "
        "COVID oversampling on training side only",
        "dataset": {
            "unique_articles": int(len(df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_sources": test_df["source"].value_counts().to_dict(),
        },
        "models": {},
        "per_source_accuracy": {},
    }
    for name, proba in probas.items():
        y_pred = (proba > 0.5).astype(int)
        report["models"][name] = _metrics(y_test, y_pred)
        report["per_source_accuracy"][name] = _per_source_accuracy(test_df, y_pred)

    config.METRICS_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["models"], indent=2))
    print(f">>> Metrics written to {config.METRICS_FILE}")

    data.save_reference_corpus()


if __name__ == "__main__":
    main()
