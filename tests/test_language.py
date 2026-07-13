"""Language detection + language-aware live-retrieval routing (offline)."""
import io
import json
from contextlib import contextmanager
from unittest import mock

from src.external_retrieval import ExternalEvidenceRetriever
from src.language import detect_language


# --- pure detector ----------------------------------------------------------

def test_detects_italian():
    text = ("Il governo ha approvato una nuova legge che secondo molti esperti "
            "non è sufficiente per affrontare la crisi economica del paese.")
    assert detect_language(text) == "it"


def test_detects_english():
    text = ("The government approved a new law that according to many experts "
            "is not enough to address the economic crisis of the country.")
    assert detect_language(text) == "en"


def test_empty_or_signal_less_text_falls_back_to_default():
    assert detect_language("") == "en"
    assert detect_language("Trump Biden NASA 2024") == "en"  # proper nouns only, no signal


# --- language-aware routing -------------------------------------------------

_WIKI_PAYLOAD = json.dumps(
    {"query": {"search": [{"title": "Crisi", "snippet": "una <span>crisi</span> economica"}]}}
)


@contextmanager
def _fake(body):
    yield io.BytesIO(body.encode("utf-8"))


def _retriever():
    r = ExternalEvidenceRetriever()
    r.factcheck_api_key = ""  # no Google key -> falls through to Wikipedia
    ExternalEvidenceRetriever._last_gdelt_call = 0.0
    return r


def _capture_url(text):
    captured = {}

    def _cap(req, *a, **k):
        captured["url"] = getattr(req, "full_url", req)
        return _fake(_WIKI_PAYLOAD)

    with mock.patch("src.external_retrieval.urlopen", side_effect=_cap):
        _retriever().query(text)
    return captured["url"]


def test_italian_claim_routes_to_italian_wikipedia():
    url = _capture_url("Il vaccino è stato approvato secondo gli esperti della sanità italiana.")
    assert url.startswith("https://it.wikipedia.org")


def test_english_claim_routes_to_english_wikipedia():
    url = _capture_url("The vaccine was approved according to health experts in the country.")
    assert url.startswith("https://en.wikipedia.org")


def test_forced_language_overrides_detection():
    # English text but language forced to Italian -> request must hit it.wikipedia.
    captured = {}

    def _cap(req, *a, **k):
        captured["url"] = getattr(req, "full_url", req)
        return _fake(_WIKI_PAYLOAD)

    with mock.patch("src.external_retrieval.urlopen", side_effect=_cap):
        _retriever().query("This is clearly English text about a crisis", lang="it")
    assert captured["url"].startswith("https://it.wikipedia.org")
