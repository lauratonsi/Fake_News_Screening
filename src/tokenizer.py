"""A minimal, framework-independent word tokenizer.

Standalone replacement for keras' ``Tokenizer``/``pad_sequences``: the
deployed app should not need to import TensorFlow just to turn text into
padded integer sequences for the (TFLite) RNNs. Behavior only needs to be
internally consistent between training (``src.train``) and serving
(``src.predict``), not identical to the old keras implementation, since the
embedding table is trained from scratch either way.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9']+")


class SimpleTokenizer:
    def __init__(self, num_words: int, lower: bool = True):
        self.num_words = num_words
        self.lower = lower
        self.word_index: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower() if self.lower else text
        return _WORD_RE.findall(text)

    def fit_on_texts(self, texts) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for word in self._tokenize(text):
                counts[word] = counts.get(word, 0) + 1
        # index 1..N ranked by frequency (most frequent word = index 1);
        # index 0 is reserved for padding, matching keras' convention.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.word_index = {word: i + 1 for i, (word, _) in enumerate(ranked)}

    def texts_to_sequences(self, texts) -> list[list[int]]:
        sequences = []
        for text in texts:
            seq = []
            for word in self._tokenize(text):
                idx = self.word_index.get(word)
                if idx is not None and idx < self.num_words:
                    seq.append(idx)
            sequences.append(seq)
        return sequences


def pad_sequences(sequences, maxlen: int):
    """Post-pad/truncate to ``maxlen`` with zeros, like keras' ``padding="post"``."""
    import numpy as np

    out = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        trimmed = seq[:maxlen]
        out[i, : len(trimmed)] = trimmed
    return out
