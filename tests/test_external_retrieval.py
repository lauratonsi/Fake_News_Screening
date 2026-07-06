"""Live-retrieval tests — all offline via a mocked ``urlopen``.

They lock in two real bugs fixed after the demo "never showed a concrete live
response": GDELT's plain-text rate-limit notice used to be swallowed as a
generic failure (indistinguishable from "no matches"), and an over-long
AND-ed query starved every request of hits.
"""
import io
import json
from contextlib import contextmanager
from unittest import mock

from src import config
from src.external_retrieval import ExternalEvidenceRetriever, _extract_keywords


@contextmanager
def _fake_response(body: str):
    yield io.BytesIO(body.encode("utf-8"))


def _retriever():
    r = ExternalEvidenceRetriever()
    r.factcheck_api_key = ""  # force the GDELT path, skip Google Fact Check
    ExternalEvidenceRetriever._last_gdelt_call = 0.0  # bypass the rate limiter
    return r


def test_gdelt_rate_limit_notice_is_reported_not_swallowed():
    notice = "Please limit requests to one every 5 seconds or contact ..."
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response(notice)):
        result = _retriever().query("The Federal Reserve raised interest rates.")
    assert result["evidence"] == []
    assert "rate-limited" in result["message"].lower()


def test_gdelt_returns_evidence_on_valid_json():
    payload = json.dumps(
        {"articles": [
            {"title": "Fed warns inflation remains elevated",
             "url": "http://example.com/a", "domain": "example.com"},
        ]}
    )
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response(payload)):
        result = _retriever().query("The Federal Reserve raised interest rates on inflation.")
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["title"].startswith("Fed warns")


def test_gdelt_null_articles_does_not_crash():
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response('{"articles": null}')):
        result = _retriever().query("Some claim about something.")
    assert result["evidence"] == []  # no crash on a structurally-valid null field


def test_keywords_stay_small_and_drop_weekdays():
    kw = _extract_keywords(
        "The Federal Reserve announced on Wednesday a new set of regulations "
        "to monitor inflation and support the labor market."
    )
    terms = kw.split()
    # multi-word proper nouns are quoted, so count quoted phrases + bare words
    assert kw.count('"') <= 2 * config.LIVE_KEYWORD_TERMS
    assert "Wednesday" not in kw  # weekday names are noise for a news search
    assert '"Federal Reserve"' in kw  # the key named entity is kept
