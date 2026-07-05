"""Live retrieval against free external sources.

The default path is GDELT, which is free and does not require an API key.
If the optional Google Fact Check API key is present, the retriever also
queries Google Fact Check Tools and uses it as the highest-signal source.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


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


class ExternalEvidenceRetriever:
    """Free live retrieval with an optional fact-check API upgrade."""

    def __init__(self, timeout_seconds: int = 6, max_records: int = 5):
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

        params = urlencode({"query": text[:280], "pageSize": self.max_records, "key": self.factcheck_api_key})
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

        for claim in payload.get("claims", [])[: self.max_records]:
            for review in claim.get("claimReview", [])[: self.max_records]:
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
        params = urlencode(
            {
                "query": text[:280],
                "mode": "ArtList",
                "format": "json",
                "maxrecords": self.max_records,
                "sort": "hybridrel",
                "timespan": "30d",
            }
        )
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"

        try:
            payload = _safe_fetch_json(url, timeout=self.timeout_seconds)
        except Exception as exc:
            return {
                "source": "gdelt",
                "verdict": None,
                "score": 0.0,
                "message": f"GDELT unavailable: {exc}",
                "evidence": [],
            }

        articles = payload.get("articles") or payload.get("articles", [])
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
        if gdelt["evidence"]:
            return gdelt

        return {
            "source": "live",
            "verdict": None,
            "score": 0.0,
            "message": "No live evidence found in Google Fact Check or GDELT",
            "evidence": [],
        }