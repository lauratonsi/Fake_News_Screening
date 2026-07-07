from src.manipulation import detect_manipulation


def _ids(result):
    return {t["technique"] for t in result["techniques"]}


def test_detects_conspiracy_and_unverifiable_source():
    text = "According to a whistleblower, the WHO secretly had a cure but decided to hide it."
    result = detect_manipulation(text)
    assert "conspiracy" in _ids(result)
    assert "unverifiable_source" in _ids(result)
    assert result["count"] >= 2


def test_stacked_techniques_flag_high_for_review():
    text = ("SHOCKING: a leaked memo reveals the elite secretly agreed to hide "
            "the truth — you won't believe it. Share before it's deleted!")
    result = detect_manipulation(text)
    assert result["high"] is True  # several techniques stack up
    assert result["score"] == 1.0  # >= MANIPULATION_TECHNIQUES_FOR_FULL_SCORE


def test_true_plain_claims_are_not_flagged():
    # The exact statements the classifier over-flags as FAKE carry no
    # manipulation markers — this layer stays silent on them (orthogonality).
    for text in [
        "Donald Trump won the 2016 presidential election defeating Hillary Clinton.",
        "FBI Director James Comey was fired by President Trump in May 2017.",
        "The national football team qualified for the World Cup after winning the last match.",
    ]:
        result = detect_manipulation(text)
        assert result["count"] == 0, f"false-flagged: {text}"
        assert result["high"] is False


def test_each_hit_carries_a_prebunking_explanation():
    result = detect_manipulation("They don't want you to know the real reason.")
    assert result["count"] >= 1
    for tech in result["techniques"]:
        assert tech["explanation"]
        assert tech["matches"]


def test_empty_text_is_safe():
    result = detect_manipulation("")
    assert result["count"] == 0
    assert result["techniques"] == []
    assert result["score"] == 0.0
