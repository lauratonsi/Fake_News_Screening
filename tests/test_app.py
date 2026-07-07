"""End-to-end UI tests for app.py via Streamlit's AppTest framework.

AppTest runs the real script (real models, real retrieval corpus — no
mocking) and exposes the resulting widget/output tree for assertions,
without needing a browser. ``st.cache_resource`` persists across AppTest
instances within the same process, so only the first test in the file pays
the full model-loading cost (~10s); the rest reuse the warm cache (~2s each).

A fresh AppTest instance per test gives full isolation (no leaked widget
state between tests) at that modest, one-time cost.
"""
import json

import pytest
from streamlit.testing.v1 import AppTest

from src import config

# Examples defined in app.py, referenced here by their exact dropdown label.
FAKE_HOAX_EXAMPLE = "COVID conspiracy claim"
CALM_HOAX_EXAMPLE = "Hoax (calm, no sensational style)"  # contains "leaked"/"secretly"
CLEAN_TRUE_EXAMPLE = "True political fact (a known hard case)"


def _fresh_app():
    at = AppTest.from_file("app.py", default_timeout=90)
    at.run()
    return at


def _analyze(at, example_label):
    at.selectbox[0].select(example_label).run()
    at.button[0].click().run()
    return at


def _all_text(at):
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def test_app_loads_without_error():
    at = _fresh_app()
    assert not at.exception
    assert at.title[0].value == "🛡️ Fake News Screening"


def test_sidebar_documents_all_seven_layers():
    # Regression guard: this sidebar went stale once already (still describing
    # the pre-Track-A/B/C system) before anyone noticed. Pin the key phrase
    # for each of the 7 layers so a future rewrite can't silently drop one.
    at = _fresh_app()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    for phrase in [
        "Ensemble",
        "Live fact-check overrides the ensemble",
        "Most similar known articles",
        "Manipulation-technique detection",
        "Confidence tier",
        "Explainability",
        "Agreement check + feedback",
    ]:
        assert phrase in sidebar_text, f"sidebar is missing the '{phrase}' layer"


def test_sidebar_limitations_discloses_the_ai_fluent_gap():
    at = _fresh_app()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "AI-generated disinformation" in sidebar_text
    assert "73.7%" in sidebar_text  # current measured out-of-domain accuracy


def test_analyzing_a_hoax_shows_a_fake_verdict_and_model_scores():
    at = _analyze(_fresh_app(), FAKE_HOAX_EXAMPLE)
    assert not at.exception
    text = _all_text(at)
    assert "FAKE" in text
    assert "SVM" in text and "GRU" in text and "LSTM" in text


def test_manipulation_panel_appears_for_trope_laden_text():
    at = _analyze(_fresh_app(), CALM_HOAX_EXAMPLE)
    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Manipulation techniques detected" in s for s in subheaders)


def test_no_manipulation_message_for_clean_text():
    at = _analyze(_fresh_app(), CLEAN_TRUE_EXAMPLE)
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("No manipulation techniques detected" in c for c in captions)


def test_empty_input_does_not_produce_a_result_panel():
    at = _fresh_app()
    at.button[0].click().run()  # no example selected, text area empty
    assert not at.exception
    assert not any("Model scores" in s.value for s in at.subheader)


def test_feedback_submission_keeps_the_result_visible_and_is_recorded(tmp_path, monkeypatch):
    # Regression test for a real bug: the result panel (and the feedback form
    # inside it) lived only under `if st.button("Analyze") and text.strip():`.
    # st.button() is True only in the exact run it was clicked in, so any
    # later widget interaction — including submitting the feedback form
    # itself — made the whole panel vanish before the correction was ever
    # recorded. Fixed by persisting the result in st.session_state.
    monkeypatch.setattr(config, "FEEDBACK_LOG", tmp_path / "feedback.jsonl")

    at = _analyze(_fresh_app(), FAKE_HOAX_EXAMPLE)
    assert any("Model scores" in s.value for s in at.subheader)

    at.radio[0].set_value("👎 No, wrong")
    at.selectbox[1].set_value("REAL (true statement)")
    submit = [b for b in at.button if b.label == "Submit feedback"][0]
    submit.click().run()

    assert not at.exception
    assert any("recorded" in s.value for s in at.success)
    # The result panel must still be there after submitting, not wiped out.
    assert any("Model scores" in s.value for s in at.subheader)

    assert config.FEEDBACK_LOG.exists()
    record = json.loads(config.FEEDBACK_LOG.read_text().splitlines()[0])
    assert record["agrees"] is False
    assert record["correct_label"] == "REAL"
    assert record["verdict"] == "FAKE"


def test_live_fact_check_panel_renders_when_a_verdict_is_present(monkeypatch):
    # Stub the live retriever so this test doesn't depend on network access
    # or a configured API key: it only checks that app.py renders the panel
    # correctly when predict() returns a live verdict, not that the live
    # integration itself works (src/test_external_retrieval.py covers that).
    at = _fresh_app()
    at.selectbox[0].select(FAKE_HOAX_EXAMPLE).run()

    from src import predict as predict_module

    original_predict = predict_module.ScreeningSystem.predict

    def fake_predict(self, text):
        result = original_predict(self, text)
        result["live"] = {"verdict": "FAKE", "source": "google_fact_check", "evidence": []}
        result["evidence_backed"] = True
        result["confidence"] = "high"
        result["reason"] = "live fact-check verdict (google_fact_check)"
        return result

    monkeypatch.setattr(predict_module.ScreeningSystem, "predict", fake_predict)
    at.button[0].click().run()

    assert not at.exception
    assert any("Live fact-check" in e.value and "rated FALSE" in e.value for e in at.error)
