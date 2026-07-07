"""Capture user corrections for later use — the write side of the feedback loop.

A screening aid that never learns from being wrong stays wrong. This module
appends one JSON line per user judgement to a local log. Those lines are the
raw material for three things the tool needs to actually improve over time:

* a real-world evaluation set (how the system does on what people actually
  paste, not just the curated benchmark),
* hard negatives to fold into a future training run (the false positives on
  true statements this system is known to produce), and
* a growing corpus of user-verified claims the retrieval layer can draw on —
  see ``src/incorporate_feedback.py`` for the read side that actually does
  this, folding verified corrections into ``reference_corpus/``.

The log is intentionally simple (append-only JSONL, no database, stdlib only)
and lives under ``data/`` which is git-ignored — user submissions are not
committed. Nothing here blocks or slows a prediction; recording is best-effort.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config


def record_feedback(
    text: str,
    result: dict,
    agrees: bool | None = None,
    correct_label: str | None = None,
    comment: str | None = None,
    path: Path | None = None,
) -> Path:
    """Append one feedback record and return the log path.

    ``result`` is the dict returned by ``ScreeningSystem.predict``; we store the
    signals that make a correction useful later (the verdict, its probability,
    confidence and whether it was evidence-backed) alongside the user's input
    and judgement. Best-effort: any I/O error is swallowed so the UI never
    breaks on a failed write.
    """
    path = path or config.FEEDBACK_LOG
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": str(text),
        "verdict": result.get("verdict"),
        "fake_probability": result.get("fake_probability"),
        "confidence": result.get("confidence"),
        "evidence_backed": result.get("evidence_backed"),
        "needs_review": result.get("needs_review"),
        "live_verdict": (result.get("live") or {}).get("verdict"),
        "manipulation_count": (result.get("manipulation") or {}).get("count"),
        "agrees": agrees,
        "correct_label": correct_label,
        "comment": (comment or "").strip() or None,
        # Set to True by src.incorporate_feedback once folded into the
        # retrieval corpus, so re-running it is idempotent.
        "incorporated": False,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # never let feedback logging break the app
    return path


def load_feedback(path: Path | None = None) -> list[dict]:
    """Read all feedback records (skipping any malformed lines)."""
    path = path or config.FEEDBACK_LOG
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
