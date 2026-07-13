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


# --- Italian prebunking (language-aware) -------------------------------------

def test_detects_italian_techniques_and_flags_high():
    text = ("SCONVOLGENTE: un documento segreto trapelato rivela il vero motivo "
            "che i poteri forti non vogliono che tu sappia. Condividi prima che "
            "lo cancellino!")
    result = detect_manipulation(text)  # language auto-detected as Italian
    ids = _ids(result)
    assert "conspiracy" in ids
    assert "unverifiable_source" in ids
    assert "urgency" in ids
    assert "fear_emotion" in ids
    assert result["high"] is True


def test_italian_plain_true_claim_is_not_flagged():
    text = "Il governo ha approvato la legge di bilancio dopo il voto del parlamento."
    result = detect_manipulation(text)
    assert result["count"] == 0
    assert result["high"] is False


def test_forced_language_selects_the_right_patterns():
    # Italian phrasing but language forced to English -> English patterns miss it.
    text = "gli scienziati confermano che il vaccino è sicuro"
    assert detect_manipulation(text, lang="it")["count"] >= 1   # 'gli scienziati confermano'
    assert detect_manipulation(text, lang="en")["count"] == 0   # no English pattern matches
