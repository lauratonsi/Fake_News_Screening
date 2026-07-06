"""Live-retrieval tests — all offline via a mocked ``urlopen``.

Precedence is Google Fact Check (verdict) -> Wikipedia (reliable context) ->
GDELT (last-resort news search). These tests lock in the reliable Wikipedia
default and the real bugs fixed earlier: GDELT's plain-text rate-limit notice
must be reported, not swallowed as "no matches", and an over-long AND-ed query
must not starve every request of hits.
"""
import io
import json
from contextlib import contextmanager
from unittest import mock

from src import config
from src.external_retrieval import (
    ExternalEvidenceRetriever,
    _extract_keywords,
    _strip_html,
)


@contextmanager
def _fake_response(body: str):
    yield io.BytesIO(body.encode("utf-8"))


def _retriever():
    r = ExternalEvidenceRetriever()
    r.factcheck_api_key = ""  # no Google key: exercise Wikipedia/GDELT
    ExternalEvidenceRetriever._last_gdelt_call = 0.0  # bypass the rate limiter
    return r


# --- Wikipedia: the reliable, key-free default ------------------------------

def test_wikipedia_returns_context_evidence_never_a_verdict():
    payload = json.dumps(
        {"query": {"search": [
            {"title": "Donald Trump",
             "snippet": 'Trump <span class="searchmatch">won</span> the 2016 election'},
        ]}}
    )
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response(payload)):
        result = _retriever()._query_wikipedia("Donald Trump won the 2016 election")
    assert result["verdict"] is None  # context, never a verdict
    assert len(result["evidence"]) == 1
    hit = result["evidence"][0]
    assert hit["publisher"] == "Wikipedia"
    assert hit["url"] == "https://en.wikipedia.org/wiki/Donald_Trump"
    assert "won the 2016 election" in hit["title"]  # HTML tags stripped


def test_query_prefers_wikipedia_and_does_not_touch_gdelt_when_wiki_hits():
    r = _retriever()
    wiki_result = {
        "source": "wikipedia", "verdict": None, "score": 0.0, "message": "ok",
        "evidence": [{"title": "x", "url": "u", "publisher": "Wikipedia",
                      "source": "wikipedia", "score": 1.0, "label": None}],
    }
    with mock.patch.object(r, "_query_wikipedia", return_value=wiki_result), \
         mock.patch.object(r, "_query_gdelt",
                           side_effect=AssertionError("GDELT must not run when Wikipedia has hits")):
        result = r.query("some claim")
    assert result["source"] == "wikipedia"


# --- GDELT: last-resort, tested directly ------------------------------------

def test_gdelt_rate_limit_notice_is_reported_not_swallowed():
    notice = "Please limit requests to one every 5 seconds or contact ..."
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response(notice)):
        result = _retriever()._query_gdelt("The Federal Reserve raised interest rates.")
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
        result = _retriever()._query_gdelt("The Federal Reserve raised rates on inflation.")
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["title"].startswith("Fed warns")


def test_gdelt_null_articles_does_not_crash():
    with mock.patch("src.external_retrieval.urlopen", return_value=_fake_response('{"articles": null}')):
        result = _retriever()._query_gdelt("Some claim about something.")
    assert result["evidence"] == []  # no crash on a structurally-valid null field


# --- helpers ----------------------------------------------------------------

def test_keywords_stay_small_and_drop_weekdays():
    kw = _extract_keywords(
        "The Federal Reserve announced on Wednesday a new set of regulations "
        "to monitor inflation and support the labor market.",
        max_terms=config.LIVE_KEYWORD_TERMS,
    )
    assert kw.count('"') <= 2 * config.LIVE_KEYWORD_TERMS
    assert "Wednesday" not in kw  # weekday names are noise for a news search
    assert '"Federal Reserve"' in kw  # the key named entity is kept


def test_strip_html_removes_tags_and_entities():
    assert _strip_html('a <span class="x">b</span> &amp; c') == "a b & c"
