import numpy as np

from src.tokenizer import SimpleTokenizer, pad_sequences


def test_fit_ranks_indices_by_frequency():
    tok = SimpleTokenizer(num_words=100)
    tok.fit_on_texts(["cat cat cat dog dog bird"])
    assert tok.word_index["cat"] == 1  # most frequent -> lowest index
    assert tok.word_index["dog"] == 2
    assert tok.word_index["bird"] == 3


def test_texts_to_sequences_drops_unknown_and_out_of_vocab_words():
    tok = SimpleTokenizer(num_words=2)  # only index 1 ("cat") fits under num_words
    tok.fit_on_texts(["cat cat dog"])
    seqs = tok.texts_to_sequences(["cat dog unknownword"])
    assert seqs == [[1]]


def test_pad_sequences_pads_and_truncates():
    out = pad_sequences([[1, 2, 3], [1, 2, 3, 4, 5]], maxlen=4)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [1, 2, 3, 0]
    assert out[1].tolist() == [1, 2, 3, 4]  # truncated, keeps the first 4
