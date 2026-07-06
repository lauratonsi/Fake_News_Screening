"""Live retrieval against free external sources.

The default path is GDELT, which is free and does not require an API key.
If the optional Google Fact Check API key is present, the retriever also
queries Google Fact Check Tools and uses it as the highest-signal source.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from . import config


@dataclass(frozen=True)
class ExternalHit:
    source: str
    title: str
    url: str
    score: float
    label: str | None = None
    publisher: str | None = None


def _safe_fetch_json(url: str, timeout: int = 6) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


_STOPWORDS = frozenset("""
the a an and or but of to in on for with at by from as is are was were be
been being that this these those it its his her their our your my he she
they we you i not no do does did has have had will would can could should
may might said says according also after before during over into about than
then which who whom what when where how why up down out if so such some
any all most
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _extract_keywords(text: str, max_terms: int = config.LIVE_KEYWORD_TERMS) -> str:
    """Reduce a claim to its most search-relevant terms.

    GDELT and Google Fact Check index real articles and fact-checks, not
    arbitrary paraphrases: a full grammatical sentence rarely matches
    anything verbatim, even when the underlying event is well covered.
    Multi-word proper nouns (e.g. "Federal Reserve") are kept as quoted
    phrases; other stopword-free content words fill the rest of the query.
    """
    words = _WORD_RE.findall(text)

    proper_nouns = []
    i = 0
    while i < len(words):
        if words[i][0].isupper() and words[i].lower() not in _STOPWORDS:
            j = i + 1
            while j < len(words) and words[j][0].isupper():
                j += 1
            proper_nouns.append(" ".join(words[i:j]))
            i = j
        else:
            i += 1

    # Longer content words are, on average, rarer and more discriminative than
    # short common ones ("regulations"/"inflation" beat "announced"/"support"),
    # so rank them by length: with only a handful of AND-ed terms allowed, the
    # few we keep should be the most distinctive ones.
    content_words = sorted(
        (
            w for w in words
            if len(w) > 3 and w.lower() not in _STOPWORDS and not w[0].isupper()
        ),
        key=len,
        reverse=True,
    )

    seen: set[str] = set()
    terms = []
    for term in proper_nouns + content_words:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            terms.append(term)

    terms = terms[:max_terms] or words[:max_terms]
    return " ".join(f'"{t}"' if " " in t else t for t in terms)


class ExternalEvidenceRetriever:
    """Free live retrieval with an optional fact-check API upgrade."""

    # GDELT enforces ~1 request / 5 s; shared across instances so parallel
    # claims don't burn the quota on rate-limit rejections.
    _last_gdelt_call: float = 0.0

    def __init__(self, timeout_seconds: int = config.LIVE_TIMEOUT_SECONDS, max_records: int = 5):
        self.timeout_seconds = timeout_seconds
        self.max_records = max_records
        self.factcheck_api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()

    @staticmethod
    def _rating_to_verdict(rating: str | None) -> str | None:
        if not rating:
            return None
        value = rating.lower()
        if any(token in value for token in ("false", "pants on fire", "fake", "misleading")):
            return "FAKE"
        if any(token in value for token in ("true", "correct", "supported", "accurate")):
            return "REAL"
        return None

    def _query_google_factcheck(self, text: str) -> dict:
        if not self.factcheck_api_key:
            return {
                "source": "google_fact_check",
                "verdict": None,
                "score": 0.0,
                "message": "Google Fact Check API key not configured",
                "evidence": [],
            }

        params = urlencode(
            {
                "query": _extract_keywords(text),
                "languageCode": config.GOOGLE_FACTCHECK_LANGUAGE,
                "pageSize": self.max_records,
                "key": self.factcheck_api_key,
            }
        )
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{params}"

        try:
            payload = _safe_fetch_json(url, timeout=self.timeout_seconds)
        except Exception as exc:
            return {
                "source": "google_fact_check",
                "verdict": None,
                "score": 0.0,
                "message": f"Google Fact Check unavailable: {exc}",
                "evidence": [],
            }

        evidence: list[ExternalHit] = []
        verdict = None
        score = 0.0

        for claim in (payload.get("claims") or [])[: self.max_records]:
            for review in (claim.get("claimReview") or [])[: self.max_records]:
                rating = review.get("textualRating") or review.get("rating")
                review_verdict = self._rating_to_verdict(rating)
                if review_verdict and verdict is None:
                    verdict = review_verdict
                    score = 1.0
                evidence.append(
                    ExternalHit(
                        source="google_fact_check",
                        title=review.get("title") or claim.get("text") or text,
                        url=review.get("url") or "",
                        score=1.0 if review_verdict else 0.5,
                        label=review_verdict,
                        publisher=(review.get("publisher") or {}).get("name") if isinstance(review.get("publisher"), dict) else None,
                    )
                )

        message = "Google Fact Check results found" if evidence else "No Google Fact Check matches"
        return {
            "source": "google_fact_check",
            "verdict": verdict,
            "score": score,
            "message": message,
            "evidence": [hit.__dict__ for hit in evidence],
        }

    def _query_gdelt(self, text: str) -> dict:
        now = time.monotonic()
        if now - ExternalEvidenceRetriever._last_gdelt_call < config.GDELT_MIN_INTERVAL:
            return {
                "source": "gdelt",
                "verdict": None,
                "score": 0.0,
                "message": "GDELT skipped (rate limit: one request every 5 s)",
                "evidence": [],
            }
        ExternalEvidenceRetriever._last_gdelt_call = now

        keywords = _extract_keywords(text)
        query = f"{keywords} sourcelang:{config.GDELT_SOURCE_LANGUAGE}" if keywords else f"sourcelang:{config.GDELT_SOURCE_LANGUAGE}"
        params = urlencode(
            {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": self.max_records,
                "sort": "hybridrel",
                "timespan": config.GDELT_TIMESPAN,
            }
        )
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"

        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return {
                "source": "gdelt",
                "verdict": None,
                "score": 0.0,
                "message": f"GDELT unavailable: {exc}",
                "evidence": [],
            }

        # GDELT's free shared endpoint answers with a plain-text notice (NOT
        # JSON) when it is rate-limited. json.loads() then throws and the old
        # code swallowed it as a generic failure, so the demo just showed
        # nothing — indistinguishable from "no matches". Detect it and say so.
        stripped = raw.lstrip()
        if not stripped.startswith("{"):
            limited = "limit requests" in raw.lower() or "rate" in raw.lower()
            return {
                "source": "gdelt",
                "verdict": None,
                "score": 0.0,
                "message": (
                    "GDELT is rate-limited right now (shared free endpoint, "
                    "~1 request / 5 s) — try again in a few seconds."
                    if limited
                    else "GDELT returned an unexpected (non-JSON) response."
                ),
                "evidence": [],
            }

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "source": "gdelt",
                "verdict": None,
                "score": 0.0,
                "message": "GDELT returned an unexpected response.",
                "evidence": [],
            }

        articles = payload.get("articles") or []
        evidence = []
        for article in articles[: self.max_records]:
            title = article.get("title") or article.get("seendate") or text
            url = article.get("url") or ""
            domain = article.get("domain") or article.get("sourceCountry") or "news"
            evidence.append(
                ExternalHit(
                    source="gdelt",
                    title=title,
                    url=url,
                    score=1.0,
                    label=None,
                    publisher=domain,
                )
            )

        message = "GDELT live news matches found" if evidence else "No GDELT matches"
        return {
            "source": "gdelt",
            "verdict": None,
            "score": 0.0,
            "message": message,
            "evidence": [hit.__dict__ for hit in evidence],
        }

    def query(self, text: str) -> dict:
        google = self._query_google_factcheck(text)
        if google["evidence"]:
            return google

        gdelt = self._query_gdelt(text)
        # Return GDELT's own result when it has evidence OR when it has a
        # meaningful non-"no matches" message (rate-limited / unavailable), so
        # that state reaches the UI instead of a misleading generic fallback.
        if gdelt["evidence"] or any(
            token in gdelt["message"].lower()
            for token in ("rate-limited", "unavailable", "unexpected")
        ):
            return gdelt

        return {
            "source": "live",
            "verdict": None,
            "score": 0.0,
            "message": "No live evidence found in Google Fact Check or GDELT",
            "evidence": [],
        }