from src.ai_style import detect_ai_style


def _ids(result):
    return {m["marker"] for m in result["markers"]}


def test_detects_study_citation_and_precise_stat():
    # The canonical ai_fluent FAKE register: a specific-sounding citation plus a
    # fabricated hard number, with none of the classic disinformation tropes.
    text = ("A peer-reviewed study in the Journal of Clinical Nutrition found that "
            "a daily 500-milligram dose reverses early-stage osteoporosis.")
    result = detect_ai_style(text)
    assert "specific_study_citation" in _ids(result)
    assert "fabricated_precision" in _ids(result)
    assert result["count"] >= 2


def test_stacked_markers_flag_high_for_review():
    text = ("Researchers at a leading university report that in a placebo-controlled "
            "clinical trial of 1200 participants, the treatment cut mortality by 47 percent; "
            "the peer-reviewed study was statistically significant.")
    result = detect_ai_style(text)
    assert result["high"] is True  # several marker categories stack up
    assert result["score"] == 1.0  # >= AI_STYLE_MARKERS_FOR_FULL_SCORE


def test_institutional_fabrication_with_metric_is_flagged():
    # The other ai_fluent register: a named working paper / analysis attributing
    # a suspiciously precise causal metric — caught via two stacked categories.
    for text in [
        "An IMF working paper estimated that a 2% global wealth tax would lower "
        "sovereign debt ratios by an average of 11 percent.",
        "A Congressional Budget Office analysis found that the bill will reduce "
        "interstate freight delays by an average of 38 minutes.",
    ]:
        result = detect_ai_style(text)
        assert result["high"] is True, f"missed: {text}"


def test_classic_human_tropes_are_not_flagged():
    # Orthogonality: the classic-disinformation register (handled by the
    # manipulation layer) carries none of the fabricated-authority markers.
    for text in [
        "SHOCKING: a leaked memo reveals the elite secretly agreed to hide the truth.",
        "They don't want you to know the real reason — wake up and share before it's deleted!",
    ]:
        result = detect_ai_style(text)
        assert result["count"] == 0, f"false-flagged: {text}"
        assert result["high"] is False


def test_plain_true_claims_are_not_flagged():
    # The short true statements the ensemble over-flags carry no authority
    # register either — this layer stays silent on them.
    for text in [
        "Donald Trump won the 2016 presidential election defeating Hillary Clinton.",
        "FBI Director James Comey was fired by President Trump in May 2017.",
    ]:
        result = detect_ai_style(text)
        assert result["count"] == 0, f"false-flagged: {text}"
        assert result["high"] is False


def test_single_marker_is_not_high():
    # One category alone (a real article can cite one study) must not trip review.
    result = detect_ai_style("A recent study found that exercise improves sleep quality.")
    assert result["count"] == 1
    assert result["high"] is False


def test_each_hit_carries_a_prebunking_explanation():
    result = detect_ai_style("A double-blind clinical trial showed a 30 percent reduction.")
    assert result["count"] >= 1
    for marker in result["markers"]:
        assert marker["explanation"]
        assert marker["matches"]


def test_empty_text_is_safe():
    result = detect_ai_style("")
    assert result["count"] == 0
    assert result["markers"] == []
    assert result["score"] == 0.0
