"""Claim-level retrieval analysis built on top of the reference corpus.

This is the first step toward a true RAG-style workflow: the input is split
into claim-like sentences and each claim is retrieved independently, so the
demo can show, per claim, the closest known snippets and whether they were
labelled real or fake. These are *evidence* labels ("matches a known false
claim", "matches known reporting", "no close match"), not truth judgements:
only a live fact-check verdict is an actual verdict here — similarity to a
stored snippet is not verification.
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
    matches_fake = 0
    matches_real = 0
    no_match = 0
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

        # These statuses describe the EVIDENCE we retrieved, not a truth
        # judgement of the claim. A live fact-check verdict (from an actual
        # fact-checker) is the only real verdict here and takes precedence;
        # otherwise we only report that the claim is close to a stored snippet
        # that was labelled real or fake, which is similarity, not verification.
        live_verdict = live_hit.get("verdict") if live_hit else None
        if live_verdict == "FAKE" or (live_verdict is None and hit["verdict"] == "FAKE"):
            status = "MATCHES_KNOWN_FALSE"
            matches_fake += 1
        elif live_verdict == "REAL" or (live_verdict is None and hit["verdict"] == "REAL"):
            status = "MATCHES_KNOWN_REAL"
            matches_real += 1
        else:
            status = "NO_CLOSE_MATCH"
            no_match += 1

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

    return {
        "message": "Evidence retrieved per claim — similarity to known snippets, not a truth check.",
        "source": live_sources[0] if live_sources else None,
        "claims": analyzed,
        "summary": {
            "claims_total": len(analyzed),
            "matches_fake": matches_fake,
            "matches_real": matches_real,
            "no_match": no_match,
        },
    }