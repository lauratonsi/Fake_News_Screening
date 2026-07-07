from src.feedback import load_feedback, record_feedback

_RESULT = {
    "verdict": "FAKE", "fake_probability": 0.91, "confidence": "low",
    "evidence_backed": False, "needs_review": True,
    "live": {"verdict": None}, "manipulation": {"count": 2},
}


def test_record_and_load_roundtrip(tmp_path):
    log = tmp_path / "feedback.jsonl"
    record_feedback("some claim", _RESULT, agrees=False,
                    correct_label="REAL", comment="this is actually true", path=log)
    record_feedback("another claim", _RESULT, agrees=True, path=log)

    records = load_feedback(log)
    assert len(records) == 2
    first = records[0]
    assert first["text"] == "some claim"
    assert first["agrees"] is False
    assert first["correct_label"] == "REAL"
    assert first["verdict"] == "FAKE"
    assert first["manipulation_count"] == 2


def test_load_missing_log_is_empty(tmp_path):
    assert load_feedback(tmp_path / "nope.jsonl") == []


def test_blank_comment_stored_as_none(tmp_path):
    log = tmp_path / "feedback.jsonl"
    record_feedback("c", _RESULT, agrees=True, comment="   ", path=log)
    assert load_feedback(log)[0]["comment"] is None
