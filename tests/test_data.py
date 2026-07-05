import pytest

from src import config, data
from conftest import requires_datasets


def test_train_test_frames_rejects_duplicated_input(toy_frame_with_duplicate):
    """train_test_frames() must refuse non-deduplicated input rather than
    silently leaking a copy of the same article across the train/test split."""
    with pytest.raises(AssertionError):
        data.train_test_frames(toy_frame_with_duplicate)


def test_train_test_split_has_no_leakage(toy_frame):
    """The central methodological claim of the project: no article sits on
    both sides of the split."""
    train_df, test_df = data.train_test_frames(toy_frame)
    assert set(train_df["full_text"]) & set(test_df["full_text"]) == set()


def test_covid_boost_only_applies_to_train(toy_frame):
    train_df, test_df = data.train_test_frames(toy_frame)

    train_covid = train_df[train_df["source"] == "covid"]
    test_covid = test_df[test_df["source"] == "covid"]

    # oversampling with replace=True must produce duplicate rows in train...
    assert train_covid["full_text"].duplicated().any()
    # ...but the test side keeps the natural, untouched distribution.
    assert not test_covid["full_text"].duplicated().any()


def test_covid_boost_balances_classes_in_train(toy_frame):
    train_df, _ = data.train_test_frames(toy_frame)
    train_covid = train_df[train_df["source"] == "covid"]
    counts = train_covid["target"].value_counts()
    assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0
    assert counts[0] == counts[1]  # real/fake balanced by oversampling
    assert counts[0] % config.COVID_BOOST_FACTOR == 0  # then boosted x3


def test_split_is_deterministic(toy_frame):
    train_a, test_a = data.train_test_frames(toy_frame)
    train_b, test_b = data.train_test_frames(toy_frame)
    assert train_a["full_text"].tolist() == train_b["full_text"].tolist()
    assert test_a["full_text"].tolist() == test_b["full_text"].tolist()


@requires_datasets
def test_build_dataset_has_no_exact_duplicates():
    df = data.build_dataset()
    assert not df["full_text"].duplicated().any()


@requires_datasets
def test_build_dataset_drops_duplicates_present_in_raw_sources():
    isot = data.load_isot()
    welfake = data.load_welfake()
    covid = data.load_covid()
    fused = data.build_dataset()
    assert len(fused) < len(isot) + len(welfake) + len(covid)
