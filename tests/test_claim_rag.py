from src.claim_rag import analyze_claims, split_claims


class _StubRetriever:
    """Minimal stand-in for ReferenceRAG.query — no models, no I/O."""

    def query(self, text, top_k=2):
        return {"verdict": None, "score": 0.0, "message": "no strong match", "evidence": []}


def test_splits_multiple_long_sentences():
    text = (
        "The Federal Reserve announced new regulations on Wednesday. "
        "Officials said the measures target inflation expectations directly."
    )
    claims = split_claims(text)
    assert len(claims) == 2


def test_drops_short_fragments():
    text = "Yes. No way. This one sentence is long enough to count as a real claim here."
    claims = split_claims(text)
    assert len(claims) == 1
    assert "long enough" in claims[0]


def test_split_claims_on_empty_or_short_text_returns_nothing():
    assert split_claims("") == []
    assert split_claims("a b c") == []  # too short to count as a claim


def test_analyze_claims_falls_back_to_whole_text_when_no_claim_found():
    result = analyze_claims("too short", _StubRetriever())
    assert result["claims"][0]["claim"] == "too short"
    assert result["summary"]["claims_total"] == 1
