"""Temporal refresh: fold RECENT fact-checked claims into the retrieval corpus.

Usage:
    python -m src.refresh_corpus [--dry-run]

The reference corpus is frozen at 2015-2017 (``reports/figures/temporal_window``
shows real coverage only starts in 2016), so semantic retrieval decays on
anything current. The live layer already knows how to fetch fresh evidence per
claim; this reuses its Google Fact Check path — the *only* source that returns
an actual verdict — in the opposite direction: instead of querying per user
input, it periodically pulls recent professionally-fact-checked claims across a
set of seed topics and folds the FAKE/REAL ones into ``reference_corpus/``, the
same append path ``incorporate_feedback`` uses.

Guardrails (fact-checks are third-party verdicts, but the ingest must still be
safe and idempotent):

* **Verdict required.** Only claims Google Fact Check rated clearly FAKE or
  REAL are ingested (``ExternalEvidenceRetriever._rating_to_verdict``); mixed /
  unrated reviews are skipped.
* **Idempotent.** Each fact-check review URL is recorded in a manifest; a claim
  already ingested in a previous run is never re-added.
* **Deduplicated & capped.** Claims already in the corpus are skipped, and one
  run adds at most ``REFRESH_MAX_TOTAL`` claims, balanced across the two
  verdicts so a lopsided news cycle can't skew the corpus.
* **Key required.** Without ``GOOGLE_FACTCHECK_API_KEY`` there is no verdict
  source, so it refuses loudly instead of silently ingesting nothing.

The selection logic (:func:`plan_refresh`) is pure and unit-tested; only
:func:`refresh_corpus` touches the network and the embedding model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .external_retrieval import ExternalEvidenceRetriever

_VERDICTS = ("REAL", "FAKE")


def plan_refresh(
    hits: list[dict],
    existing_texts: set[str],
    seen_ids: set[str],
    *,
    min_chars: int = config.REFRESH_MIN_CLAIM_CHARS,
    max_total: int = config.REFRESH_MAX_TOTAL,
) -> dict:
    """Select which fetched fact-check hits to ingest, applying every guard.

    ``hits`` is a list of ``{"text","verdict","id"}`` (``id`` = the fact-check
    review URL). ``existing_texts`` are the corpus texts already present;
    ``seen_ids`` are review URLs ingested in previous runs. Pure and
    dependency-free. Returns accepted rows (``text``/``verdict``), the ids to
    add to the manifest, and a per-reason skip tally.
    """
    accepted: list[dict] = []
    seen_text: set[str] = set()
    seen_batch_ids: set[str] = set()
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for h in hits:
        text = str(h.get("text") or "").strip()
        verdict = h.get("verdict")
        hid = h.get("id") or ""
        if verdict not in _VERDICTS:
            skip("no_clear_verdict")
            continue
        if hid and (hid in seen_ids):
            skip("already_ingested")
            continue
        if len(text) < min_chars:
            skip("too_short")
            continue
        if text in existing_texts:
            skip("already_in_corpus")
            continue
        if text in seen_text or (hid and hid in seen_batch_ids):
            skip("duplicate_in_batch")
            continue
        seen_text.add(text)
        if hid:
            seen_batch_ids.add(hid)
        accepted.append({"text": text, "verdict": verdict, "id": hid})

    capped = _balance_and_cap(accepted, max_total)
    dropped = len(accepted) - len(capped)
    if dropped:
        skipped["over_cap"] = skipped.get("over_cap", 0) + dropped

    return {
        "rows": capped,
        "ids": [r["id"] for r in capped if r["id"]],
        "skipped": skipped,
        "n_accepted": len(capped),
        "by_verdict": {"REAL": sum(r["verdict"] == "REAL" for r in capped),
                       "FAKE": sum(r["verdict"] == "FAKE" for r in capped)},
    }


def _balance_and_cap(rows: list[dict], max_total: int) -> list[dict]:
    """Keep at most ``max_total`` rows, balanced across REAL/FAKE (earliest win)."""
    if max_total <= 0:
        return []
    per_class = max(1, max_total // 2)
    kept: list[dict] = []
    counts = {"REAL": 0, "FAKE": 0}
    for r in rows:
        v = r["verdict"]
        if counts[v] < per_class and len(kept) < max_total:
            kept.append(r)
            counts[v] += 1
    return kept


def _load_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return set()


def _save_manifest(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), ensure_ascii=False), encoding="utf-8")


def _fetch_factcheck_hits(retriever: ExternalEvidenceRetriever) -> list[dict]:
    """Pull recent fact-checked claims across the seed topics, per language.

    Queries Google Fact Check in each supported language with that language's
    seed topics (English topics against languageCode=en, Italian against
    languageCode=it), so the refresh keeps the corpus current in every language
    the evidence layer serves. Flattens evidence into ``{"text","verdict","id"}``
    rows; only reviews carrying a clear verdict survive plan_refresh, but all are
    passed through so the skip tally is honest.
    """
    by_lang = {"en": config.REFRESH_SEED_QUERIES, "it": config.REFRESH_SEED_QUERIES_IT}
    hits: list[dict] = []
    for lang in config.SUPPORTED_LANGUAGES:
        for q in by_lang.get(lang, []):
            result = retriever._query_google_factcheck(q, lang)
            for ev in result.get("evidence", []):
                hits.append({"text": ev.get("title") or "", "verdict": ev.get("label"),
                             "id": ev.get("url") or ""})
    return hits


def refresh_corpus(
    *,
    dry_run: bool = False,
    retriever: ExternalEvidenceRetriever | None = None,
    encoder=None,
) -> dict:
    """Fetch recent fact-checked claims and fold the verdict-bearing ones into
    the reference corpus. Returns a summary dict. Heavy imports are lazy.
    """
    import pandas as pd

    retriever = retriever or ExternalEvidenceRetriever()
    if not retriever.factcheck_api_key:
        return {"status": "no_api_key",
                "message": "GOOGLE_FACTCHECK_API_KEY not set — no verdict source to ingest from."}

    hits = _fetch_factcheck_hits(retriever)

    existing = set()
    for f in (config.REF_REAL_FILE, config.REF_FAKE_FILE):
        if f.exists():
            existing |= set(pd.read_csv(f)["text"].fillna("").astype(str))
    seen_ids = _load_manifest(config.REFRESH_MANIFEST)

    plan = plan_refresh(hits, existing, seen_ids)
    summary = {"n_fetched": len(hits), "n_accepted": plan["n_accepted"],
               "by_verdict": plan["by_verdict"], "skipped": plan["skipped"], "dry_run": dry_run}

    if plan["n_accepted"] == 0:
        summary["status"] = "nothing_to_ingest"
        return summary
    if dry_run:
        summary["status"] = "dry_run"
        return summary

    from .incorporate_feedback import _append, _default_encoder
    encode = encoder or _default_encoder()
    real_texts = [r["text"] for r in plan["rows"] if r["verdict"] == "REAL"]
    fake_texts = [r["text"] for r in plan["rows"] if r["verdict"] == "FAKE"]
    real_added = _append(config.REF_REAL_FILE, config.REF_EMBEDDINGS_FILE, "real", real_texts, encode)
    fake_added = _append(config.REF_FAKE_FILE, config.REF_EMBEDDINGS_FILE, "fake", fake_texts, encode)

    _save_manifest(config.REFRESH_MANIFEST, seen_ids | set(plan["ids"]))

    summary["status"] = "ingested"
    summary["real_added"] = real_added
    summary["fake_added"] = fake_added
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be ingested without changing the corpus")
    args = parser.parse_args()

    result = refresh_corpus(dry_run=args.dry_run)
    status = result["status"]
    if status == "no_api_key":
        print(result["message"])
        return
    if status == "nothing_to_ingest":
        print(f"Fetched {result['n_fetched']} fact-check(s); nothing new to ingest. "
              f"Skipped: {result['skipped']}.")
        return
    if status == "dry_run":
        bv = result["by_verdict"]
        print(f"Would ingest {result['n_accepted']} claim(s): +{bv['REAL']} real, "
              f"+{bv['FAKE']} fake (of {result['n_fetched']} fetched). Skipped: {result['skipped']}.")
        return
    print(f"Ingested {result['n_accepted']} recent fact-checked claim(s): "
          f"+{result['real_added']} real, +{result['fake_added']} fake snippet(s).")


if __name__ == "__main__":
    main()
