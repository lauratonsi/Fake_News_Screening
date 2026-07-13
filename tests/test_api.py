"""Unit tests for the pure API layer (request validation + response shaping).

No server, no models — build_response takes a synthetic predict() result.
"""
from src import config
from src.api import build_response, validate_request

# A synthetic ScreeningSystem.predict() result, shaped like the real one.
_RESULT = {
    "verdict": "FAKE",
    "fake_probability": 0.82,
    "confidence": "medium",
    "needs_review": True,
    "evidence_backed": False,
    "reason": "ensemble consensus",
    "manipulation": {"count": 2, "high": False,
                     "techniques": [{"label": "Fear language"}, {"label": "Urgency"}]},
    "ai_style": {"count": 2, "high": True},
    "live": {"verdict": "FAKE", "source": "google_fact_check"},
    "explanation": {"fake_pushing": [{"token": "leaked", "weight": 0.4}],
                    "real_pushing": [{"token": "confirmed", "weight": -0.3}]},
    "explanation_neural": {"fake_pushing": [{"token": "secret", "weight": 0.2}],
                           "real_pushing": []},
    "explanation_rag": {"driving": [{"label": "FAKE", "score": 0.91, "snippet": "known hoax"}]},
}


def test_valid_request_passes():
    text, errors = validate_request({"text": "some article text"})
    assert errors == []
    assert text == "some article text"


def test_missing_or_nonstring_text_is_rejected():
    assert validate_request({})[1]
    assert validate_request({"text": 123})[1]
    assert validate_request("not a dict")[1]


def test_empty_and_oversized_text_are_rejected():
    assert any("empty" in e for e in validate_request({"text": "   "})[1])
    big = "x" * (config.API_MAX_TEXT_CHARS + 1)
    assert any("exceeds" in e for e in validate_request({"text": big})[1])


def test_build_response_exposes_the_stable_contract():
    out = build_response(_RESULT)
    assert out["verdict"] == "FAKE"
    assert out["fake_probability"] == 0.82
    assert out["needs_review"] is True
    # signals are summarised, not the raw internal dicts
    assert out["signals"]["manipulation"]["techniques"] == ["Fear language", "Urgency"]
    assert out["signals"]["fabricated_authority"]["high"] is True
    assert out["signals"]["live_factcheck"] == {"verdict": "FAKE", "source": "google_fact_check"}
    # explanations flattened to token lists + evidence
    assert out["explanation"]["svm_top_fake"] == ["leaked"]
    assert out["explanation"]["rnn_top_fake"] == ["secret"]
    assert out["explanation"]["evidence"][0]["label"] == "FAKE"
    assert "disclaimer" in out


def test_build_response_tolerates_missing_optional_blocks():
    minimal = {"verdict": "REAL", "fake_probability": 0.1, "confidence": "low",
               "needs_review": False, "evidence_backed": False, "reason": "ensemble consensus"}
    out = build_response(minimal)
    assert out["verdict"] == "REAL"
    assert out["signals"]["live_factcheck"] is None
    assert out["signals"]["manipulation"]["count"] == 0
    assert out["explanation"]["svm_top_fake"] == []
    assert out["explanation"]["evidence"] == []


def test_live_factcheck_omitted_when_no_verdict():
    result = dict(_RESULT, live={"verdict": None, "source": "wikipedia"})
    assert build_response(result)["signals"]["live_factcheck"] is None
