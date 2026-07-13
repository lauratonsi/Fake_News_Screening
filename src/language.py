"""Tiny, dependency-free language detector for routing live retrieval.

The classifiers are English-trained, but the *evidence* layer (Wikipedia,
GDELT, Google Fact Check) exists in many languages. To surface Italian evidence
for an Italian claim we first need to know the claim is Italian — cheaply, with
no extra dependency and no network call. A stopword-overlap heuristic is enough
for this job: it only has to pick between the handful of ``SUPPORTED_LANGUAGES``,
and it fails safe to the default language when the signal is weak.

This is deliberately not a general-purpose language identifier — it decides
"which supported language's sources should we query", nothing more.
"""
from __future__ import annotations

import re

from . import config

_WORD_RE = re.compile(r"[a-zàèéìíîòóùú\w']+", re.IGNORECASE)

# Strongly language-specific function words. Chosen to minimise cross-language
# collisions (e.g. Italian "il/che/di/sono", English "the/of/and/is"), so a
# normal sentence lands decisively on one side.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("""
        the of and is are was were be been to that this these those with for
        from as but not have has had will would can could should they them
        their there here what which who because about into over than then
    """.split()),
    "it": frozenset("""
        il lo la i gli le un uno una di che è e per con non si sono anche
        come ma più della delle degli nel nella alla dei questo questa essere
        hanno può perché quando dove molto senza sempre già anche loro
    """.split()),
}


def detect_language(text: str) -> str:
    """Return a code in :data:`config.SUPPORTED_LANGUAGES`.

    Scores the text by how many of its words are distinctive stopwords of each
    supported language and returns the best match, normalised by that
    language's stopword-list size so a longer list doesn't win by default.
    Falls back to :data:`config.DEFAULT_LANGUAGE` when no language shows a clear
    signal (empty text, code, proper nouns only).
    """
    words = [w.lower() for w in _WORD_RE.findall(str(text or ""))]
    if not words:
        return config.DEFAULT_LANGUAGE

    best_lang = config.DEFAULT_LANGUAGE
    best_score = 0.0
    for lang in config.SUPPORTED_LANGUAGES:
        stop = _STOPWORDS.get(lang)
        if not stop:
            continue
        hits = sum(1 for w in words if w in stop)
        score = hits / len(words)
        if score > best_score:
            best_score = score
            best_lang = lang

    # Require a minimal signal; otherwise stay on the default (English).
    return best_lang if best_score >= 0.06 else config.DEFAULT_LANGUAGE
