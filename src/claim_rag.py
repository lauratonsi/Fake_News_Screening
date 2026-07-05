"""Claim-level retrieval analysis built on top of the reference corpus.

This is the first step toward a true RAG-style workflow: the input is split
into claim-like sentences, each claim is retrieved independently, and the demo
can show which statements are supported, refuted, or unsupported.
"""
from __future__ import annotations

import re

from . import config


_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str) -> list[str]:
    """Split a news item into short claim-like segments."""
    parts = [part.strip() for part in _CLAIM_SPLIT_RE.split(str(text).strip())]
    claims = []
    for part in parts:
        cleaned = part.strip(" \n\t-•")
        words = cleaned.split()
        if len(words) < 6:
            continue
        if len(cleaned) < 30:
            continue
        claims.append(cleaned)
    return claims


def analyze_claims(text: str, retriever, live_retriever=None, top_k: int = 2) -> dict:
    """Run retrieval for each claim-like sentence and aggregate the evidence."""
    claims = split_claims(text)
    if not claims:
        claims = [str(text).strip()]

    analyzed = []
    supported = 0
    refuted = 0
    unknown = 0
    live_sources = []

    for index, claim in enumerate(claims):
        hit = retriever.query(claim, top_k=top_k)
        # Live lookups are capped: they cost seconds each and GDELT rate-limits.
        live_hit = (
            live_retriever.query(claim)
            if live_retriever is not None and index < config.LIVE_MAX_CLAIMS
            else None
        )
        if live_hit and live_hit.get("evidence"):
            live_sources.append(live_hit["source"])

        # A live fact-check verdict is the highest-signal source; the committed
        # reference corpus decides only when live evidence is absent or neutral.
        live_verdict = live_hit.get("verdict") if live_hit else None
        if live_verdict == "REAL" or (live_verdict is None and hit["verdict"] == "REAL"):
            status = "SUPPORTED"
            supported += 1
        elif live_verdict == "FAKE" or (live_verdict is None and hit["verdict"] == "FAKE"):
            status = "REFUTED"
            refuted += 1
        else:
            status = "UNSUPPORTED"
            unknown += 1

        analyzed.append(
            {
                "claim": claim,
                "status": status,
                "retrieval_verdict": hit["verdict"],
                "score": round(float(hit["score"]), 4),
                "message": hit["message"],
                "evidence": hit["evidence"],
                "live": live_hit,
            }
        )

    if refuted > supported and refuted > 0:
        verdict = "FAKE"
        message = "At least one claim is strongly refuted by the reference corpus."
    elif supported > refuted and supported > 0:
        verdict = "REAL"
        message = "The claims are closer to known real snippets than to fake ones."
    else:
        verdict = None
        message = "The claims are mostly unsupported by the current reference corpus."

    return {
        "verdict": verdict,
        "message": message,
        "source": live_sources[0] if live_sources else None,
        "claims": analyzed,
        "summary": {
            "claims_total": len(analyzed),
            "supported": supported,
            "refuted": refuted,
            "unsupported": unknown,
        },
    }