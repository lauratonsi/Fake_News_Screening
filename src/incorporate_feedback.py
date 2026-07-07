"""Fold verified user corrections into the reference-retrieval corpus.

Usage:
    python -m src.incorporate_feedback [--dry-run]

This is what actually closes the feedback loop described in
``src/feedback.py``: a 👍/👎 log that nobody reads stays a log, not a loop.
Scope is intentionally narrow. A user correction ("this is wrong, it's
actually REAL/FAKE") is *verified* evidence about one specific text, so it is
safe to fold directly into the retrieval corpus (``reference_corpus/``) the
same way the committed ISOT/WELFake/COVID snippets already are — it only ever
adds retrievable evidence, it does not retrain or reweight the SVM/RNN
classifiers. Retraining on unaudited user submissions would need far more
care (adversarial submissions, label noise, class balance) than a single
append-only log can justify; this module deliberately does not attempt it.

Each incorporated record is marked so re-running this script is idempotent —
it only processes new, unprocessed corrections since the last run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .feedback import load_feedback


def _default_encoder() -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    from .rag import _embedding_model_source

    model = SentenceTransformer(_embedding_model_source())
    # Unnormalized, matching how build_and_save_embeddings() stores the
    # committed corpus — ReferenceRAG normalizes at load time, not storage time.
    return lambda texts: model.encode(texts, batch_size=32).astype(np.float16)


def _append(
    csv_file: Path, embeddings_file: Path, key: str, new_texts: list[str],
    encode: Callable[[list[str]], np.ndarray],
) -> int:
    """Append ``new_texts`` to one side (real/fake) of the corpus + cached embeddings.

    Skips exact-duplicate text already present, so resubmitting feedback on
    the same claim doesn't bloat the corpus. Returns the number of rows added.
    """
    existing = pd.read_csv(csv_file)["text"].fillna("").astype(str)
    existing_set = set(existing)
    fresh = [t for t in new_texts if t not in existing_set]
    if not fresh:
        return 0

    new_embeddings = encode(fresh)
    cached = dict(np.load(embeddings_file))
    cached[key] = np.concatenate([cached[key], new_embeddings], axis=0)
    np.savez_compressed(embeddings_file, **cached)

    updated = pd.concat([pd.DataFrame({"text": existing}), pd.DataFrame({"text": fresh})],
                        ignore_index=True)
    updated.to_csv(csv_file, index=False)
    return len(fresh)


def incorporate_feedback(
    feedback_path: Path | None = None,
    real_file: Path | None = None,
    fake_file: Path | None = None,
    embeddings_file: Path | None = None,
    encoder: Callable[[list[str]], np.ndarray] | None = None,
    dry_run: bool = False,
) -> dict:
    """Fold unprocessed, user-verified corrections into the retrieval corpus.

    A record is actionable when the user marked a result wrong (``agrees is
    False``) AND supplied the correct label. Records already folded in are
    skipped on subsequent runs via the ``incorporated`` flag written back to
    the feedback log.
    """
    feedback_path = feedback_path or config.FEEDBACK_LOG
    real_file = real_file or config.REF_REAL_FILE
    fake_file = fake_file or config.REF_FAKE_FILE
    embeddings_file = embeddings_file or config.REF_EMBEDDINGS_FILE

    records = load_feedback(feedback_path)
    actionable_idx = [
        i for i, r in enumerate(records)
        if r.get("agrees") is False
        and r.get("correct_label") in ("REAL", "FAKE")
        and not r.get("incorporated")
    ]
    if not actionable_idx:
        return {"incorporated": 0, "real_added": 0, "fake_added": 0}

    snippets = {"REAL": [], "FAKE": []}
    for i in actionable_idx:
        snippet = str(records[i]["text"])[: config.REF_SNIPPET_CHARS].strip()
        if snippet:
            snippets[records[i]["correct_label"]].append(snippet)

    if dry_run:
        return {
            "incorporated": len(actionable_idx),
            "real_added": len(snippets["REAL"]),
            "fake_added": len(snippets["FAKE"]),
            "dry_run": True,
        }

    encode = encoder or _default_encoder()
    real_added = _append(real_file, embeddings_file, "real", snippets["REAL"], encode)
    fake_added = _append(fake_file, embeddings_file, "fake", snippets["FAKE"], encode)

    for i in actionable_idx:
        records[i]["incorporated"] = True
    feedback_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
        if records else "",
        encoding="utf-8",
    )

    return {"incorporated": len(actionable_idx), "real_added": real_added, "fake_added": fake_added}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be incorporated without changing anything")
    args = parser.parse_args()

    result = incorporate_feedback(dry_run=args.dry_run)
    if result["incorporated"] == 0:
        print("No new, actionable feedback to incorporate.")
        return
    label = "Would incorporate" if args.dry_run else "Incorporated"
    print(f"{label} {result['incorporated']} correction(s): "
          f"+{result['real_added']} real, +{result['fake_added']} fake snippet(s).")


if __name__ == "__main__":
    main()
