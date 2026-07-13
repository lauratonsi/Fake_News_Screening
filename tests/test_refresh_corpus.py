"""Unit tests for the pure selection logic of the temporal corpus refresh."""
from src.refresh_corpus import _balance_and_cap, plan_refresh


def _hit(text, verdict, id):
    return {"text": text, "verdict": verdict, "id": id}


def test_only_verdict_bearing_claims_are_ingested():
    hits = [
        _hit("a fabricated claim about vaccines " * 2, "FAKE", "u1"),
        _hit("a verified true report on climate " * 2, "REAL", "u2"),
        _hit("a mixed / unrated review of something " * 2, None, "u3"),  # no verdict
    ]
    plan = plan_refresh(hits, set(), set(), min_chars=5)
    assert plan["n_accepted"] == 2
    assert plan["by_verdict"] == {"REAL": 1, "FAKE": 1}
    assert plan["skipped"].get("no_clear_verdict") == 1


def test_already_ingested_ids_are_skipped():
    hits = [_hit("some fact checked claim here now", "FAKE", "seen-url")]
    plan = plan_refresh(hits, set(), {"seen-url"}, min_chars=5)
    assert plan["n_accepted"] == 0
    assert plan["skipped"].get("already_ingested") == 1


def test_corpus_dedup_and_short_and_batch_dupes():
    existing = {"already in the corpus text here"}
    hits = [
        _hit("already in the corpus text here", "FAKE", "u1"),   # dup vs corpus
        _hit("tiny", "FAKE", "u2"),                              # too short
        _hit("a brand new fake claim to ingest", "FAKE", "u3"),  # ok
        _hit("a brand new fake claim to ingest", "FAKE", "u4"),  # dup text in batch
        _hit("distinct text same id repeated now", "REAL", "u3"),# dup id in batch
    ]
    plan = plan_refresh(hits, existing, set(), min_chars=5)
    assert plan["n_accepted"] == 1
    assert plan["skipped"].get("already_in_corpus") == 1
    assert plan["skipped"].get("too_short") == 1
    assert plan["skipped"].get("duplicate_in_batch") == 2


def test_cap_is_balanced_across_verdicts():
    hits = ([_hit(f"fake claim number {i} here now", "FAKE", f"f{i}") for i in range(6)]
            + [_hit(f"real claim number {i} here now", "REAL", f"r{i}") for i in range(6)])
    plan = plan_refresh(hits, set(), set(), min_chars=5, max_total=4)
    assert plan["n_accepted"] == 4
    assert plan["by_verdict"] == {"REAL": 2, "FAKE": 2}
    assert plan["skipped"].get("over_cap") == 8


def test_manifest_ids_returned_for_accepted_only():
    hits = [
        _hit("kept fake claim text here now", "FAKE", "keep1"),
        _hit("unrated skipped claim text now", None, "skip1"),
    ]
    plan = plan_refresh(hits, set(), set(), min_chars=5)
    assert plan["ids"] == ["keep1"]


def test_zero_cap_keeps_nothing():
    assert _balance_and_cap([{"text": "t", "verdict": "FAKE", "id": "x"}], 0) == []
