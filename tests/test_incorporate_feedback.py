import json

import numpy as np
import pandas as pd
import pytest

from src.incorporate_feedback import incorporate_feedback


def _stub_encoder(dim=4):
    """Deterministic, cheap stand-in for the real sentence-transformer model."""
    def encode(texts):
        return np.array([[len(t) % 7, 1.0, 0.0, 0.0][:dim] for t in texts], dtype=np.float16)
    return encode


@pytest.fixture
def corpus(tmp_path):
    real_file = tmp_path / "real.csv"
    fake_file = tmp_path / "fake.csv"
    emb_file = tmp_path / "embeddings.npz"
    pd.DataFrame({"text": ["an existing real snippet"]}).to_csv(real_file, index=False)
    pd.DataFrame({"text": ["an existing fake snippet"]}).to_csv(fake_file, index=False)
    np.savez_compressed(
        emb_file,
        real=np.array([[1, 0, 0, 0]], dtype=np.float16),
        fake=np.array([[0, 1, 0, 0]], dtype=np.float16),
    )
    return {"real_file": real_file, "fake_file": fake_file, "embeddings_file": emb_file}


def _write_feedback(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_incorporates_disagreed_corrections_into_the_right_side(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "actually this is true", "agrees": False, "correct_label": "REAL"},
        {"text": "actually this is false", "agrees": False, "correct_label": "FAKE"},
    ])

    result = incorporate_feedback(
        feedback_path=feedback_path, encoder=_stub_encoder(), **corpus,
    )

    assert result == {"incorporated": 2, "real_added": 1, "fake_added": 1}
    real_texts = pd.read_csv(corpus["real_file"])["text"].tolist()
    fake_texts = pd.read_csv(corpus["fake_file"])["text"].tolist()
    assert "actually this is true" in real_texts
    assert "actually this is false" in fake_texts
    emb = np.load(corpus["embeddings_file"])
    assert emb["real"].shape[0] == 2
    assert emb["fake"].shape[0] == 2


def test_agreeing_feedback_is_not_incorporated(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "the system got this right", "agrees": True, "correct_label": None},
    ])
    result = incorporate_feedback(feedback_path=feedback_path, encoder=_stub_encoder(), **corpus)
    assert result == {"incorporated": 0, "real_added": 0, "fake_added": 0}


def test_disagreement_without_a_correct_label_is_not_actionable(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "wrong but no correction given", "agrees": False, "correct_label": None},
    ])
    result = incorporate_feedback(feedback_path=feedback_path, encoder=_stub_encoder(), **corpus)
    assert result == {"incorporated": 0, "real_added": 0, "fake_added": 0}


def test_rerunning_is_idempotent(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "a fresh correction", "agrees": False, "correct_label": "REAL"},
    ])
    first = incorporate_feedback(feedback_path=feedback_path, encoder=_stub_encoder(), **corpus)
    assert first["real_added"] == 1

    # The feedback log should now be marked incorporated; a second run must
    # not re-add the same snippet or re-count it.
    records = [json.loads(line) for line in feedback_path.read_text().splitlines()]
    assert records[0]["incorporated"] is True

    second = incorporate_feedback(feedback_path=feedback_path, encoder=_stub_encoder(), **corpus)
    assert second == {"incorporated": 0, "real_added": 0, "fake_added": 0}
    assert len(pd.read_csv(corpus["real_file"])) == 2  # existing + the one addition, not doubled


def test_duplicate_text_already_in_corpus_is_skipped(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "an existing real snippet", "agrees": False, "correct_label": "REAL"},
    ])
    result = incorporate_feedback(feedback_path=feedback_path, encoder=_stub_encoder(), **corpus)
    assert result["real_added"] == 0
    assert len(pd.read_csv(corpus["real_file"])) == 1


def test_dry_run_reports_without_modifying_anything(tmp_path, corpus):
    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(feedback_path, [
        {"text": "would be added", "agrees": False, "correct_label": "FAKE"},
    ])
    result = incorporate_feedback(feedback_path=feedback_path, dry_run=True, **corpus)
    assert result["incorporated"] == 1
    assert result["fake_added"] == 1
    assert result["dry_run"] is True
    assert len(pd.read_csv(corpus["fake_file"])) == 1  # untouched
    records = [json.loads(line) for line in feedback_path.read_text().splitlines()]
    assert records[0].get("incorporated") in (None, False)  # not marked either


def test_no_feedback_file_is_a_no_op(tmp_path, corpus):
    result = incorporate_feedback(feedback_path=tmp_path / "missing.jsonl", encoder=_stub_encoder(), **corpus)
    assert result == {"incorporated": 0, "real_added": 0, "fake_added": 0}
