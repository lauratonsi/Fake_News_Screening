"""Manipulation-technique detection (the *prebunking* / inoculation layer).

Where the classifier asks "is this content false?", this layer asks a different,
complementary question: "*how* is this text trying to persuade me?" It flags the
rhetorical manipulation techniques present in the text — appeals to secret
knowledge, unverifiable sources, fabricated authority, fear and absolutist
language, urgency, and us-vs-them framing.

Why it matters here:

* **It is robust to domain and time.** The classifier is trained on 2015-2020
  English articles and is out of domain on anything newer; a manipulation
  technique ("a leaked memo reveals...", "they don't want you to know") works
  the same way regardless of topic or year, so this signal does not decay.
* **It is orthogonal to the content signal.** On the adversarial benchmark the
  true statements the models over-flag ("Trump won the 2016 election") carry
  *no* manipulation markers, while the fabricated claims are saturated with
  them ("secret deals", "a whistleblower", "secretly agreed"). So this layer
  adds information exactly where the classifier is weakest.
* **It is inoculation, not a verdict.** Following Roozenbeek & van der Linden
  (2019), naming the technique to the reader ("this is an appeal to hidden
  knowledge — here is how it works") builds resistance far better than a bare
  true/false stamp. Each hit therefore ships with a short explanation.

By design this NEVER asserts a claim is false on its own — legitimate reporting
can use forceful language too. It is surfaced as independent evidence and can
raise the human-review flag; it does not flip the FAKE/REAL label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class _Technique:
    id: str
    label: str
    explanation: str
    patterns: dict[str, re.Pattern]  # language code -> compiled pattern


def _compile(*phrases: str) -> re.Pattern:
    # Word-boundary, case-insensitive alternation. Phrases may contain \b-safe
    # spaces; short function words inside a phrase are fine because the whole
    # phrase is anchored, not each word.
    body = "|".join(phrases)
    return re.compile(rf"(?<![\w])(?:{body})(?![\w])", re.IGNORECASE)


# The taxonomy mirrors the "prebunking" manipulation techniques from the
# inoculation literature (Roozenbeek & van der Linden, 2019): conspiracy /
# appeal to hidden knowledge, discrediting via unverifiable sources, fake
# authority, emotional (fear) language, false certainty, urgency, and
# polarization (us-vs-them). Patterns are deliberately conservative — they aim
# for the distinctive giveaway phrasing, not every word that could co-occur in
# honest reporting.
_TECHNIQUES: list[_Technique] = [
    _Technique(
        "conspiracy",
        "Appeal to hidden knowledge / cover-up",
        "Frames a claim as a suppressed truth 'they' are hiding, which makes it "
        "unfalsifiable — any absence of evidence is recast as proof of the "
        "cover-up. Genuine reporting cites what is known, not what is secret.",
        {
            "en": _compile(
                r"secret(?:ly|s)?", r"leaked?", r"hidden", r"cover[- ]?up",
                r"suppress(?:ed|ing)?", r"they don'?t want you to know",
                r"what they'?re hiding", r"the truth about", r"exposed?",
                r"behind closed doors", r"kept (?:secret|hidden|quiet)",
                r"real reason", r"deep state",
            ),
            "it": _compile(
                r"segret[oaie]", r"trapelat[oaie]", r"nascost[oaie]",
                r"insabbiamento", r"occultat[oaie]", r"non vogliono che (?:tu )?sappia",
                r"quello che (?:ti )?nascondono", r"la verit[àa] su",
                r"a porte chiuse", r"tenut[oaie] (?:segret|nascost)",
                r"il vero motivo", r"stato profondo", r"poteri forti",
            ),
        },
    ),
    _Technique(
        "unverifiable_source",
        "Anonymous / unverifiable source",
        "Attributes the claim to a source that cannot be checked — a "
        "'whistleblower', 'insiders', 'a leaked memo'. Real evidence can be "
        "traced to a named, accountable origin; an unnameable one shifts the "
        "burden of proof onto you.",
        {
            "en": _compile(
                r"whistle[- ]?blowers?", r"anonymous sources?", r"insiders? (?:say|claim|reveal)",
                r"sources? (?:say|claim|reveal|told)", r"a leaked (?:memo|document|report|study)",
                r"reportedly", r"some (?:say|claim)", r"it is (?:said|claimed|rumou?red)",
                r"an? (?:internal|secret) (?:report|memo|document|study)",
            ),
            "it": _compile(
                r"informator[ei]", r"fonti anonime", r"fonti (?:vicine|interne)",
                r"secondo (?:indiscrezioni|fonti)", r"un memo trapelato",
                r"si dice", r"alcuni (?:dicono|sostengono)", r"pare che",
                r"un documento (?:interno|segreto|riservato)", r"gira voce",
            ),
        },
    ),
    _Technique(
        "fake_authority",
        "Fabricated or vague authority",
        "Borrows the credibility of 'science' or 'experts' without a checkable "
        "reference — 'scientists confirmed', 'a study proves'. The claim of "
        "proof does the persuading; the actual study is never locatable.",
        {
            "en": _compile(
                r"scientists? (?:confirm(?:ed)?|prove[nd]?|agree)",
                r"doctors? (?:agree|confirm|warn)",
                r"experts? (?:confirm(?:ed)?|agree|warn|say)",
                r"studies? (?:prove|show|confirm)", r"a study (?:proves|shows|confirms)",
                r"research proves", r"proven fact", r"scientifically proven",
            ),
            "it": _compile(
                r"gli scienziati (?:confermano|dimostrano|concordano)",
                r"i medici (?:avvertono|confermano|concordano)",
                r"gli esperti (?:confermano|concordano|avvertono|dicono)",
                r"(?:gli )?studi (?:dimostrano|provano|confermano)",
                r"uno studio (?:dimostra|prova|conferma)", r"la ricerca dimostra",
                r"scientificamente provat[oa]", r"fatto (?:provato|scientifico)",
            ),
        },
    ),
    _Technique(
        "fear_emotion",
        "Fear / emotional trigger language",
        "Uses alarming, high-arousal words to push a fast, intuitive reaction "
        "and pre-empt slower, analytical checking. The emotional charge, not "
        "the evidence, is doing the work.",
        {
            "en": _compile(
                r"shocking", r"terrifying", r"horrifying", r"you won'?t believe",
                r"outrageous", r"catastrophic", r"devastating", r"alarming",
                r"dangerous", r"deadly", r"nightmare", r"disaster", r"chilling",
            ),
            "it": _compile(
                r"scioccante", r"terrificante", r"sconvolgente", r"non crederai",
                r"scandalos[oaie]", r"catastrofic[oaie]", r"devastante",
                r"allarmante", r"pericolos[oaie]", r"mortale", r"incubo",
                r"disastro", r"agghiacciante", r"choc",
            ),
        },
    ),
    _Technique(
        "false_certainty",
        "Absolutist / false certainty",
        "Replaces the hedged language of real evidence ('suggests', 'is "
        "associated with') with sweeping absolutes ('always', '100%', "
        "'undeniable'). Reality is rarely that clean; the certainty is a tell.",
        {
            "en": _compile(
                r"always", r"never", r"everyone knows", r"no one", r"nobody",
                r"completely", r"totally", r"100 ?%", r"undeniable", r"irrefutable",
                r"guaranteed", r"definitely", r"without a doubt", r"permanently",
            ),
            "it": _compile(
                r"sempre", r"mai", r"tutti sanno", r"nessuno", r"completamente",
                r"totalmente", r"100 ?%", r"innegabile", r"inconfutabile",
                r"garantito", r"sicuramente", r"senza dubbio", r"assolutamente",
            ),
        },
    ),
    _Technique(
        "urgency",
        "Urgency / call to spread",
        "Pressures you to act or share before verifying — 'must watch', 'share "
        "before it's deleted'. Urgency is engineered to skip the verification "
        "step, which is exactly when false claims travel fastest.",
        {
            "en": _compile(
                r"share (?:this )?before", r"before it'?s (?:deleted|removed|too late|banned)",
                r"act now", r"must[- ]?(?:watch|read|see|share)", r"spread the word",
                r"wake up", r"do your own research", r"they'?re deleting this",
            ),
            "it": _compile(
                r"condividi (?:questo )?prima", r"prima che (?:lo )?(?:cancellino|rimuovano|censurino)",
                r"agisci (?:ora|subito)", r"(?:guarda|leggi|condividi) subito",
                r"diffondi(?:lo)?", r"svegli(?:ati|amoci)", r"fai le tue ricerche",
                r"lo stanno (?:cancellando|censurando)", r"passa parola",
            ),
        },
    ),
    _Technique(
        "polarization",
        "Us-vs-them / polarization",
        "Splits the world into a virtuous in-group and a malicious out-group "
        "('the elite', 'the mainstream media', 'sheeple'). It reframes a factual "
        "question as a loyalty test, so disagreeing feels like betraying your side.",
        {
            "en": _compile(
                r"the elite?s?", r"the establishment", r"mainstream media",
                r"sheeple", r"globalists?", r"the (?:corrupt )?(?:politicians|government) (?:don'?t|won'?t)",
                r"they want (?:you|us)", r"the powers that be", r"puppet masters?",
            ),
            "it": _compile(
                r"le [eé]lite", r"l'establishment", r"i (?:media )?mainstream",
                r"pecoron[ei]", r"globalist[iea]", r"i potenti",
                r"(?:i politici|il governo) non (?:vogliono|vuole)",
                r"vogliono (?:che tu|che voi|farci)", r"il sistema (?:ci|vi|ti)",
            ),
        },
    ),
]


def detect_manipulation(text: str, lang: str | None = None) -> dict:
    """Scan ``text`` for manipulation techniques, in the text's language.

    Returns a structured summary with, per detected technique, the matched
    phrases and a short prebunking explanation. ``score`` is a 0..1 intensity
    normalised by the number of technique *categories* in the taxonomy, so it
    reflects breadth of manipulation, not raw phrase count.

    The taxonomy is shared across languages; only the giveaway *phrasing*
    differs, so an Italian hoax is flagged with the same technique labels as an
    English one. ``lang`` is auto-detected (see src/language.py) unless forced,
    and falls back to English patterns for any language without its own set.
    """
    text = str(text or "")
    if lang is None:
        from .language import detect_language
        lang = detect_language(text)

    detected = []
    total_hits = 0
    for tech in _TECHNIQUES:
        pattern = tech.patterns.get(lang) or tech.patterns[config.DEFAULT_LANGUAGE]
        found = pattern.findall(text)
        if not found:
            continue
        # De-duplicate matches case-insensitively, preserve first casing seen.
        seen: dict[str, str] = {}
        for m in found:
            seen.setdefault(m.lower(), m)
        matches = list(seen.values())
        total_hits += len(matches)
        detected.append(
            {
                "technique": tech.id,
                "label": tech.label,
                "explanation": tech.explanation,
                "matches": matches,
            }
        )

    count = len(detected)
    score = min(1.0, count / config.MANIPULATION_TECHNIQUES_FOR_FULL_SCORE)
    high = count >= config.MANIPULATION_REVIEW_MIN_TECHNIQUES

    if count == 0:
        summary = "No manipulation techniques detected."
    else:
        names = ", ".join(d["label"].split(" / ")[0].lower() for d in detected)
        summary = (
            f"{count} manipulation technique{'s' if count != 1 else ''} detected "
            f"({names})."
        )

    return {
        "count": count,
        "total_hits": total_hits,
        "score": round(score, 3),
        "high": high,
        "summary": summary,
        "techniques": detected,
    }
