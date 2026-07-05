import pandas as pd
import pytest

from src import config


def _datasets_available() -> bool:
    return (config.DATA_DIR / "Fake.csv").exists() and (config.DATA_DIR / "True.csv").exists()


requires_datasets = pytest.mark.skipif(
    not _datasets_available(),
    reason="raw datasets not present in data/ (see data/README.md) — split-protocol tests need them",
)


@pytest.fixture
def toy_frame_with_duplicate() -> pd.DataFrame:
    """A tiny, hand-built frame with the same columns as data.build_dataset()'s
    output BEFORE deduplication, plus a covid slice, for fast protocol tests
    that don't need the real (multi-GB, gitignored) datasets."""
    rows = []
    for i in range(200):
        rows.append({"full_text": f"real isot article number {i}", "target": 0, "source": "isot"})
        rows.append({"full_text": f"fake isot article number {i}", "target": 1, "source": "isot"})
    for i in range(30):
        rows.append({"full_text": f"real covid claim {i}", "target": 0, "source": "covid"})
    for i in range(10):
        rows.append({"full_text": f"fake covid claim {i}", "target": 1, "source": "covid"})
    # an exact duplicate: build_dataset() must drop this before splitting
    rows.append({"full_text": "real isot article number 0", "target": 0, "source": "isot"})
    return pd.DataFrame(rows)


@pytest.fixture
def toy_frame(toy_frame_with_duplicate) -> pd.DataFrame:
    """The same toy frame after deduplication — what train_test_frames()
    actually expects as input (i.e. build_dataset()'s output)."""
    return toy_frame_with_duplicate.drop_duplicates(subset="full_text").reset_index(drop=True)
