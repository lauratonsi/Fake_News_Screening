import pytest

from src.rag import ReferenceRAG
from src import config


def _artifacts_available() -> bool:
    return (
        config.REF_REAL_FILE.exists()
        and config.REF_FAKE_FILE.exists()
        and config.REF_EMBEDDINGS_FILE.exists()
    )


requires_reference_artifacts = pytest.mark.skipif(
    not _artifacts_available(),
    reason="reference_corpus/{real,fake}.csv.gz + embeddings.npz not built yet (python -m src.train)",
)


def test_disabled_when_artifacts_missing(tmp_path):
    rag = ReferenceRAG(
        real_file=tmp_path / "missing_real.csv.gz",
        fake_file=tmp_path / "missing_fake.csv.gz",
        embeddings_file=tmp_path / "missing_embeddings.npz",
    )
    assert rag.enabled is False
    result = rag.query("any text")
    assert result["verdict"] is None
    assert result["evidence"] == []


@requires_reference_artifacts
def test_known_fake_snippet_retrieves_as_fake():
    rag = ReferenceRAG()
    assert rag.enabled is True
    snippet = rag.fake_texts[0]
    result = rag.query(snippet)
    assert result["verdict"] == "FAKE"
    assert result["score"] > 0.9  # near-identical text -> near-1.0 cosine similarity


@requires_reference_artifacts
def test_known_real_snippet_retrieves_as_real():
    rag = ReferenceRAG()
    snippet = rag.real_texts[0]
    result = rag.query(snippet)
    assert result["verdict"] == "REAL"
    assert result["score"] > 0.9


@requires_reference_artifacts
def test_semantic_paraphrase_retrieves_the_matching_fake_claim():
    """The whole point of embeddings over TF-IDF: a reworded version of a
    known fake claim should retrieve it as the closest fake-side match, even
    with almost no literal word overlap ("permanently alters your genetic
    code" vs. "permanently alter our DNA").

    This checks retrieval, not the final verdict: a same-topic real article
    (e.g. about COVID genetics research in general) can legitimately score
    even closer on the real side for an ambiguous paraphrase — top-1
    nearest-neighbor confusing topic with claim is a real, known limitation,
    not something this test should paper over.
    """
    rag = ReferenceRAG()
    fake_examples = [t for t in rag.fake_texts if "vaccine" in t.lower() and "dna" in t.lower()]
    if not fake_examples:
        pytest.skip("no DNA/vaccine fake example in this reference corpus build")
    paraphrase = "Getting the COVID shot permanently alters your genetic code."
    result = rag.query(paraphrase, top_k=5)
    fake_hits = [hit["text"] for hit in result["evidence"] if hit["label"] == "FAKE"]
    assert any(example in fake_hits for example in fake_examples)
