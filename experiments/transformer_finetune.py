"""Experiment: does a *properly fine-tuned* transformer earn a place in the
ensemble — where the frozen-embedding ablation did not?

``experiments/embeddings_baseline.py`` froze all-MiniLM-L6-v2 and trained a
linear head on top; it landed at 88.5% in-domain (vs the 95.3% SVM) with 10
false positives on the adversarial set, and was rightly rejected. That test
measured frozen embeddings + a linear classifier, NOT a fine-tuned transformer.
This script does the honest redo: it fine-tunes a small transformer
end-to-end on the SAME de-biased fused dataset and split as ``src.train``, so
the numbers are directly comparable to ``models/metrics.json``.

Crucially, it does not decide for itself. It measures the transformer alone AND
the *augmented* ensemble (production SVM+GRU+LSTM + transformer), then runs the
promotion gate (``src.ensemble_gate.should_promote``) and prints whether the
candidate should be adopted — enforcing: no in-domain regression, no new
adversarial false positives, zero-false-negative on classic disinfo preserved,
and a real benefit (in-domain or ai_fluent recall) to justify the memory.

Nothing here changes production. It writes ``experiments/transformer_results.json``
and a quantized artifact, and prints a verdict. Wiring it into the deployed app
(behind ``config.TRANSFORMER_ENABLED``) is a separate, deliberate step taken
only if this verdict says "promote".

Usage:
    python -m experiments.transformer_finetune                 # default: distilbert, CPU subsample
    python -m experiments.transformer_finetune --full          # train on the full training set
    python -m experiments.transformer_finetune --model distilbert-base-uncased --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, data  # noqa: E402
from src.ensemble_gate import EnsembleMetrics, should_promote  # noqa: E402

RESULTS_FILE = ROOT / "experiments" / "transformer_results.json"
ARTIFACT_DIR = ROOT / "experiments" / "transformer_model"
QUANTIZED_FILE = ROOT / "experiments" / "transformer_int8.pt"
MAX_TOKENS = 256  # articles are long; 256 tokens is the CPU-affordable window


def _binary_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
    }


def _fine_tune(model_name: str, train_texts, train_labels, epochs: int):
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    class _DS(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(list(texts), truncation=True, padding=True,
                                 max_length=MAX_TOKENS, return_tensors="pt")
            self.labels = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_DS(train_texts, train_labels))
    trainer.train()
    return tokenizer, model


def _predict_proba(tokenizer, model, texts, batch_size: int = 32) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=MAX_TOKENS, return_tensors="pt")
            logits = model(**enc).logits
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--full", action="store_true",
                        help="train on the full training set (default: CPU subsample)")
    args = parser.parse_args()

    df = data.build_dataset()
    train_df, test_df = data.train_test_frames(df)
    if not args.full:
        train_df = train_df.sample(
            n=min(config.RNN_TRAIN_SAMPLE, len(train_df)), random_state=config.SEED
        )
    y_test = test_df["target"].to_numpy()
    print(f">>> {len(df)} unique | train {len(train_df)} | test {len(test_df)} "
          f"— identical split to src.train")

    t0 = time.time()
    tokenizer, model = _fine_tune(args.model, train_df["full_text"].tolist(),
                                  train_df["target"].tolist(), args.epochs)
    print(f">>> fine-tuning took {(time.time() - t0) / 60:.1f} min")

    proba_tf = _predict_proba(tokenizer, model, test_df["full_text"].tolist())
    tf_metrics = _binary_metrics(y_test, (proba_tf > 0.5).astype(int))
    print(">>> transformer (alone) in-domain:", json.dumps(tf_metrics))

    # Augmented ensemble = production SVM+GRU+LSTM scores + the transformer.
    print(">>> scoring the production ensemble for the augmented comparison...")
    from src.predict import ScreeningSystem

    system = ScreeningSystem(with_reference=False, with_live=False)
    base = np.array([list(system.model_scores(t).values()) for t in test_df["full_text"]])
    current_ens = base.mean(axis=1)
    augmented_ens = np.column_stack([base, proba_tf]).mean(axis=1)

    def _acc(p):
        return float(((p > 0.5).astype(int) == y_test).mean())

    # Adversarial: false positives + classic-disinfo false negatives + ai_fluent recall
    scenarios = json.loads(config.SCENARIOS_FILE.read_text())["scenarios"]
    adv_texts = [c["text"] for c in scenarios]
    proba_tf_adv = _predict_proba(tokenizer, model, adv_texts)
    base_adv = np.array([list(system.model_scores(t).values()) for t in adv_texts])
    cur_adv = base_adv.mean(axis=1)
    aug_adv = np.column_stack([base_adv, proba_tf_adv]).mean(axis=1)

    def _adv_stats(p):
        pred = (p > 0.5).astype(int)
        fp = fn_classic = tp_ai = n_ai = 0
        for c, yhat in zip(scenarios, pred):
            is_fake = c["label"] == "FAKE"
            if c["label"] == "REAL" and yhat == 1:
                fp += 1
            if is_fake and c.get("style") == "human_typical" and yhat == 0:
                fn_classic += 1
            if is_fake and c.get("style") == "ai_fluent":
                n_ai += 1
                tp_ai += int(yhat == 1)
        return fp, fn_classic, (tp_ai / n_ai if n_ai else 0.0)

    cur_fp, cur_fn, cur_air = _adv_stats(cur_adv)
    aug_fp, aug_fn, aug_air = _adv_stats(aug_adv)

    current = EnsembleMetrics(round(_acc(current_ens), 4), cur_fp, cur_fn, round(cur_air, 4))
    augmented = EnsembleMetrics(round(_acc(augmented_ens), 4), aug_fp, aug_fn, round(aug_air, 4))
    decision = should_promote(current, augmented)

    # Save in a runtime-loadable form (production loads via from_pretrained and
    # quantizes on load — see src/predict.py). Promotion = copy this dir to
    # models/transformer_model and set config.TRANSFORMER_ENABLED = True.
    model.save_pretrained(ARTIFACT_DIR)
    tokenizer.save_pretrained(ARTIFACT_DIR)

    # Quantize + size (budget check).
    import torch

    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized.state_dict(), QUANTIZED_FILE)
    size_mb = QUANTIZED_FILE.stat().st_size / 1e6

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "note": "Honest redo of the embeddings ablation with an END-TO-END "
                "fine-tuned transformer. Same fused dataset/split as src.train. "
                "Experimental until the promotion gate says otherwise.",
        "transformer_alone_indomain": tf_metrics,
        "current_ensemble": current.__dict__,
        "augmented_ensemble": augmented.__dict__,
        "promotion": decision,
        "quantized_int8_mb": round(size_mb, 1),
        "memory_budget_mb": 600,
    }
    RESULTS_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps({"current": current.__dict__, "augmented": augmented.__dict__}, indent=2))
    print(f">>> quantized int8 artifact: {size_mb:.1f} MB")
    print(f">>> PROMOTION VERDICT: {'ADOPT' if decision['promote'] else 'REJECT'}")
    for r in decision["reasons"]:
        print(f"    - {r}")
    print(f">>> Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
