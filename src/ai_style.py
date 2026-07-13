"""Fluent fabricated-authority detection (the *ai_fluent* companion layer).

Where :mod:`src.manipulation` catches the classic-disinformation register —
secret leaks, whistleblowers, fear language, us-vs-them — this layer catches a
different and harder one: **fluent, source-attributed prose that borrows the
*register of legitimate authority* without any of the classic tells.** It is
the "ai_fluent" gap made visible: fabrications that cite a specific-sounding
study, name a precise dosage, and read like a science-desk brief.

Why it exists here:

* **It targets the documented blind spot.** On the adversarial benchmark the
  article-trained models and the trope-based manipulation layer both *under-*
  flag ``ai_fluent`` FAKE scenarios — fluent claims such as "a peer-reviewed
  study in the Journal of Clinical Nutrition found that a daily 500-milligram
  dose reverses osteoporosis". These carry no conspiracy/fear markers at all,
  so ``manipulation.py`` is silent on them and the ensemble often calls them
  REAL. This layer fires exactly there (see README, "AI-generated
  disinformation is harder to detect" and ``AI_FLUENT_RECALL_FLOOR_*``).
* **It is orthogonal to content and robust to time/domain.** The *register* of
  fabricated authority — a specific-sounding but uncheckable citation, a
  suspiciously precise statistic, clinical-trial vocabulary — works the same
  way regardless of topic or year, like the manipulation layer.
* **It is inoculation, not a verdict.** Each marker ships with a short
  prebunking note: a *real* study is locatable — name, journal, year, authors,
  a DOI you can open. The persuasive force here is the *sound* of rigour, so
  naming that sound is what builds resistance.

By design this NEVER asserts a claim is false on its own: legitimate science
reporting uses this exact register too, and flipping a true article to FAKE
would be a censorship error. Like the manipulation layer it is surfaced as
independent evidence and can raise the human-review flag — turning a silent
miss into "a human should verify the cited source exists" — but it never flips
the FAKE/REAL label and never changes the probability.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class _Marker:
    id: str
    label: str
    explanation: str
    pattern: re.Pattern


def _compile(*phrases: str) -> re.Pattern:
    # Word-boundary, case-insensitive alternation, mirroring manipulation._compile
    # so the two layers behave identically on tokenisation edge cases.
    body = "|".join(phrases)
    return re.compile(rf"(?<![\w])(?:{body})(?![\w])", re.IGNORECASE)


# The markers are the giveaways of *fabricated rigour*, deliberately distinct
# from manipulation.py's vague-authority patterns ("scientists confirmed"):
# these fire on the SPECIFIC, citation-shaped register — a named study, a
# precise dose, clinical-trial vocabulary — that fluent misinformation uses to
# pass as reporting. Conservative by design: they aim at the distinctive
# scholarly phrasing, accepting that honest science writing will also trip them
# (which is safe here — it only raises review, never flips the label).
_MARKERS: list[_Marker] = [
    _Marker(
        "specific_study_citation",
        "Specific-but-uncheckable study citation",
        "Cites a study with just enough specificity to sound authoritative — "
        "'a peer-reviewed study', 'published in the Journal of…', 'a 2019 "
        "clinical trial' — but never a locatable reference. A real study can be "
        "traced to authors, a journal, a year and a DOI you can open; a citation "
        "that stops at the register of rigour is the tell.",
        _compile(
            r"peer[- ]?reviewed", r"published in the journal", r"in the journal of",
            r"a (?:recent )?study (?:found|shows?|showed|proves?|confirms?|suggests?|revealed)",
            r"according to the study", r"the study (?:found|concluded|showed)",
            r"clinical trials?", r"randomi[sz]ed controlled trial", r"double[- ]?blind",
            r"meta[- ]?analysis", r"systematic review", r"a landmark study",
            r"researchers? (?:found|discovered|concluded|reported)",
            # Citation-shaped attributions common in fluent fabrications: a named
            # working paper / analysis "estimating" or "finding" a result. These
            # are fabrication tells (a real one links the document); measured to
            # add ai_fluent coverage with zero benchmark false-flags on real
            # reporting, which phrases the same facts as record, not as a study.
            r"working papers?", r"analysis (?:found|estimated|projected|concluded|shows?)",
        ),
    ),
    _Marker(
        "named_institution_attribution",
        "Attributed to a research body / authors",
        "Pins the claim on a research team, university lab or 'the authors' to "
        "borrow institutional credibility, while keeping the actual source just "
        "out of reach. Attribution is not evidence: the question is whether the "
        "named body actually published this, and where.",
        _compile(
            r"researchers? at", r"scientists? at", r"a team (?:of researchers|at)",
            r"the authors (?:conclude[d]?|found|note[d]?|report(?:ed)?)",
            r"the paper (?:found|argues?|concludes?)",
            r"the report (?:found|concludes?|states?)",
            r"experts? at", r"professors? (?:at|of)", r"a study (?:led|conducted) by",
        ),
    ),
    _Marker(
        "fabricated_precision",
        "Suspiciously precise statistic",
        "Delivers a hard number — an exact dose, a percentage, a sample size — "
        "that reads as measured fact. Fabrications lean on false precision "
        "because a specific figure feels checked. The number is only meaningful "
        "if you can find the study it came from.",
        _compile(
            r"\d{1,4}(?:\.\d+)?[- ]?(?:mg|milligrams?|micrograms?|mcg|grams?|kilograms?|kg|ml|milliliters?)",
            r"(?:by|of|to|reaching|up to) \d{1,3}(?:\.\d+)? ?(?:percent|%)",
            r"\d{2,5} ?(?:patients|participants|subjects|volunteers|respondents|adults|people were)",
            r"\d+(?:\.\d+)?[- ]?(?:fold|times) (?:more|higher|lower|greater|likely)",
            r"a \d{1,4}(?:\.\d+)?[- ]?(?:percent|%) (?:increase|decrease|reduction|rise|drop)",
            # Quantified causal-improvement claims ("reduced X by an average of
            # 11 percent", "cut ... by 92%") and engineered technical units
            # ("620 watt-hours", "five milliseconds"): the fake-precision register
            # of fluent fabrications. Record-style real reporting states figures
            # flatly (e.g. "prices rose 0.3%"), which these do not match.
            r"(?:reduc\w+|cut|increas\w+|lower\w+|boost\w+|improv\w+|raising|reaching) (?:[\w-]+ ){0,6}?by (?:an average of |roughly |about |nearly |up to )?\d[\d.,]*",
            r"\d+(?:\.\d+)? ?(?:watt-hours?|kilowatt-hours?|milliseconds?|megawatts?)",
        ),
    ),
    _Marker(
        "clinical_register",
        "Clinical / statistical vocabulary",
        "Uses lab-report vocabulary — 'statistically significant', 'placebo-"
        "controlled', 'efficacy', 'cohort' — to set a tone of methodological "
        "rigour. The vocabulary is easy to copy; the underlying method is what "
        "would need checking, and it is never shown.",
        _compile(
            r"statistically significant", r"placebo[- ]?controlled", r"efficacy",
            r"control group", r"sample size", r"longitudinal (?:study|cohort)",
            r"cohort study", r"p[- ]?value", r"confidence interval",
            r"biomarkers?", r"in vitro", r"in vivo",
        ),
    ),
]


def detect_ai_style(text: str) -> dict:
    """Scan ``text`` for the fluent fabricated-authority register.

    Returns a structured summary with, per detected marker, the matched phrases
    and a short prebunking explanation. ``score`` is a 0..1 intensity
    normalised by the number of marker *categories*, so it reflects the breadth
    of the authority register invoked, not the raw phrase count. ``high`` is set
    when several categories stack — the point at which a text is not merely
    citing a source but performing rigour, and a human should confirm the cited
    evidence exists.

    Mirrors :func:`src.manipulation.detect_manipulation` exactly in shape so the
    two orthogonal signals can be surfaced and combined uniformly.
    """
    text = str(text or "")
    detected = []
    total_hits = 0
    for marker in _MARKERS:
        found = marker.pattern.findall(text)
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
                "marker": marker.id,
                "label": marker.label,
                "explanation": marker.explanation,
                "matches": matches,
            }
        )

    count = len(detected)
    score = min(1.0, count / config.AI_STYLE_MARKERS_FOR_FULL_SCORE)
    high = count >= config.AI_STYLE_REVIEW_MIN_MARKERS

    if count == 0:
        summary = "No fabricated-authority register detected."
    else:
        names = ", ".join(d["label"].split(" / ")[0].lower() for d in detected)
        summary = (
            f"{count} fabricated-authority marker{'s' if count != 1 else ''} detected "
            f"({names}). A real source is locatable — verify it exists."
        )

    return {
        "count": count,
        "total_hits": total_hits,
        "score": round(score, 3),
        "high": high,
        "summary": summary,
        "markers": detected,
    }
