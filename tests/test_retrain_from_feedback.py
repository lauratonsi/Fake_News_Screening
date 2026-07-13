"""Unit tests for the pure poisoning-guard planner in retrain_from_feedback.

These never touch models/pandas — they exercise the logic that decides which
verified corrections may reach the classifier weights.
"""
from src.retrain_from_feedback import _balance_and_cap, plan_corrections


def _rec(label, text, **kw):
    return {"agrees": False, "correct_label": label, "text": text, **kw}


def test_only_verified_labelled_corrections_are_selected():
    records = [
        _rec("FAKE", "a fabricated claim " * 5),                     # actionable
        {"agrees": True, "correct_label": "FAKE", "text": "b " * 40},  # agreement -> ignored
        {"agrees": False, "correct_label": None, "text": "c " * 40},   # no label -> ignored
        _rec("REAL", "a true statement " * 5),                       # actionable
    ]
    plan = plan_corrections(records, {}, set(), n_train=1000, min_chars=5, min_corrections=1)
    assert plan["n_accepted"] == 2
    assert {r["target"] for r in plan["rows"]} == {0, 1}      # REAL->0, FAKE->1
    assert all(r["source"] == "feedback" for r in plan["rows"])


def test_short_duplicate_and_already_retrained_are_skipped():
    records = [
        _rec("FAKE", "x" * 60),
        _rec("FAKE", "x" * 60),                       # duplicate within batch
        _rec("FAKE", "tiny"),                         # too short
        _rec("REAL", "y" * 60, retrained=True),       # already folded in
    ]
    plan = plan_corrections(records, {}, set(), 1000, min_chars=50, min_corrections=1)
    assert plan["n_accepted"] == 1
    assert plan["skipped"].get("duplicate_in_batch") == 1
    assert plan["skipped"].get("too_short") == 1
    assert plan["skipped"].get("already_retrained") == 1


def test_leakage_training_dedup_and_contradiction_guards():
    train_map = {"known real text here": {0}, "known fake text here": {1}}
    test_texts = {"a held out test example"}
    records = [
        _rec("REAL", "a held out test example"),      # would leak into the eval set
        _rec("REAL", "known real text here"),         # same text+label already trained
        _rec("FAKE", "known real text here"),         # contradiction: train says REAL
        _rec("FAKE", "a brand new fake claim here"),  # actionable
    ]
    plan = plan_corrections(records, train_map, test_texts, 1000, min_chars=5, min_corrections=1)
    assert plan["skipped"].get("would_leak_into_test") == 1
    assert plan["skipped"].get("already_in_training") == 1
    assert plan["skipped"].get("contradicts_training_label") == 1
    assert plan["n_accepted"] == 1


def test_minimum_batch_gate_blocks_small_batches():
    plan = plan_corrections([_rec("FAKE", "x" * 60)], {}, set(), 1000,
                            min_chars=5, min_corrections=20)
    assert plan["n_accepted"] == 1
    assert plan["gate_passed"] is False


def test_cap_is_enforced_and_class_balanced():
    records = ([_rec("FAKE", f"fake claim number {i} here now") for i in range(8)]
               + [_rec("REAL", f"real claim number {i} here now") for i in range(8)])
    # n_train=100, max_fraction=0.06 -> budget=6, split 3 per class
    plan = plan_corrections(records, {}, set(), 100,
                            min_chars=5, max_fraction=0.06, min_corrections=1)
    assert plan["n_accepted"] == 6
    assert sum(r["target"] == 1 for r in plan["rows"]) == 3
    assert sum(r["target"] == 0 for r in plan["rows"]) == 3
    assert plan["skipped"].get("over_cap") == 10


def test_record_indices_point_at_accepted_records():
    records = [
        {"agrees": True, "correct_label": "FAKE", "text": "ignored " * 20},
        _rec("FAKE", "kept one " * 10),
        _rec("REAL", "kept two " * 10),
    ]
    plan = plan_corrections(records, {}, set(), 1000, min_chars=5, min_corrections=1)
    assert plan["record_indices"] == [1, 2]


def test_balance_and_cap_zero_budget_keeps_nothing():
    rows = [{"full_text": "t", "target": 1, "source": "feedback"}]
    out = _balance_and_cap(rows, [0], budget=0)
    assert out["rows"] == []
    assert out["indices"] == []
