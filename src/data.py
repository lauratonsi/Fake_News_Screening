"""Unified data pipeline: one loading, filtering and splitting protocol.

Three public corpora are fused into a single dataset:

* ISOT       — 2015-2017 US politics articles (Reuters vs. flagged outlets)
* WELFake    — a larger, more heterogeneous fake/real collection
* COVID-19   — claim-level statements about the pandemic

Labels follow one convention everywhere: ``target = 1`` fake, ``0`` real.

The split protocol is deliberately strict: exact duplicates are removed and
the train/test split happens BEFORE any oversampling, so no article (or copy
of it) can sit on both sides of the split.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def _combine(title: pd.Series, text: pd.Series) -> pd.Series:
    return title.fillna("") + " " + text.fillna("")


def load_isot(data_dir=config.DATA_DIR) -> pd.DataFrame:
    df_fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"), usecols=["title", "text"])
    df_real = pd.read_csv(os.path.join(data_dir, "True.csv"), usecols=["title", "text"])
    df_fake["target"], df_real["target"] = 1, 0
    df = pd.concat([df_fake, df_real], ignore_index=True)
    df["full_text"] = _combine(df["title"], df["text"])
    df["source"] = "isot"
    return df[["full_text", "target", "source"]]


def load_welfake(data_dir=config.DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(data_dir, "WELFake_Dataset.csv"),
        usecols=["title", "text", "label"],
    ).rename(columns={"label": "target"})
    df["full_text"] = _combine(df["title"], df["text"])

    # Quality filters (see config for the rationale)
    length = df["full_text"].str.len()
    df = df[(length >= config.WELFAKE_MIN_CHARS) & (length <= config.WELFAKE_MAX_CHARS)]
    caps_ratio = df["full_text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    punct = df["full_text"].str.count(r"[!?]")
    df = df[(caps_ratio <= config.WELFAKE_MAX_CAPS_RATIO) & (punct <= config.WELFAKE_MAX_EXCLAMATIONS)]

    df = df.sample(frac=config.WELFAKE_SUBSAMPLE, random_state=config.SEED)
    df["source"] = "welfake"
    assert set(df["target"].unique()) <= {0, 1}
    return df[["full_text", "target", "source"]]


def load_covid(data_dir=config.DATA_DIR) -> pd.DataFrame:
    """Load the COVID claim files (news only — tweets and replies are skipped)."""
    folder = os.path.join(data_dir, "Covid_Fake_New")
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        fname = os.path.basename(path).lower()
        if "tweet" in fname or "replies" in fname:
            continue
        if "fake" in fname:
            label = 1
        elif "real" in fname:
            label = 0
        else:
            continue
        tmp = pd.read_csv(path)
        parts = [tmp[c].fillna("") for c in ("title", "newstitle", "content", "text") if c in tmp.columns]
        if not parts:
            continue
        full_text = parts[0].astype(str)
        for part in parts[1:]:
            full_text = full_text + " " + part.astype(str)
        frames.append(pd.DataFrame({"full_text": full_text, "target": label, "source": "covid"}))
    df = pd.concat(frames, ignore_index=True)
    assert set(df["target"].unique()) <= {0, 1}
    return df


def build_dataset(data_dir=config.DATA_DIR) -> pd.DataFrame:
    """Fuse the three corpora, drop exact duplicates, shuffle deterministically."""
    df = pd.concat(
        [load_isot(data_dir), load_welfake(data_dir), load_covid(data_dir)],
        ignore_index=True,
    )
    df = df.dropna(subset=["full_text"])
    df = df[df["full_text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="full_text")
    return df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)


def train_test_frames(df: pd.DataFrame):
    """Stratified 80/20 split, then balance + boost the COVID slice on the
    training side only. The test set stays untouched (no duplicates, natural
    class distribution), so every model is measured on the same clean data.

    Requires a deduplicated frame (call ``build_dataset()`` first): splitting
    before deduplication can put copies of the same article on both sides —
    exactly the leakage this protocol exists to prevent — so it fails loudly
    instead of silently."""
    assert not df["full_text"].duplicated().any(), (
        "train_test_frames() expects deduplicated input — call build_dataset() "
        "first, or duplicate articles can leak across the train/test split."
    )
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df["target"],
    )

    covid = train_df[train_df["source"] == "covid"]
    rest = train_df[train_df["source"] != "covid"]

    covid_real = covid[covid["target"] == 0]
    covid_fake = covid[covid["target"] == 1]
    if 0 < len(covid_fake) < len(covid_real):
        covid_fake = covid_fake.sample(n=len(covid_real), replace=True, random_state=config.SEED)
    covid_balanced = pd.concat([covid_real, covid_fake], ignore_index=True)
    covid_boosted = pd.concat([covid_balanced] * config.COVID_BOOST_FACTOR, ignore_index=True)

    train_df = pd.concat([rest, covid_boosted], ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    return train_df, test_df.reset_index(drop=True)


def build_reference_corpus(data_dir=config.DATA_DIR):
    """Snippets (title + first 300 chars) of every known real/fake article.

    These feed the similarity heuristic of the demo — a memory of what the
    system has already seen, not a fact-checking database.
    """
    real_texts, fake_texts = [], []

    df_real = pd.read_csv(os.path.join(data_dir, "True.csv"), usecols=["title", "text"])
    df_fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"), usecols=["title", "text"])
    real_texts += (df_real["title"].fillna("") + ". " + df_real["text"].fillna("").str[: config.REF_SNIPPET_CHARS]).tolist()
    fake_texts += (df_fake["title"].fillna("") + ". " + df_fake["text"].fillna("").str[: config.REF_SNIPPET_CHARS]).tolist()

    df_wel = pd.read_csv(os.path.join(data_dir, "WELFake_Dataset.csv"), usecols=["title", "text", "label"])
    for label, bucket in ((0, real_texts), (1, fake_texts)):
        part = df_wel[df_wel["label"] == label]
        bucket += (part["title"].fillna("") + ". " + part["text"].fillna("").str[: config.REF_SNIPPET_CHARS]).tolist()

    covid = load_covid(data_dir)
    snippets = covid["full_text"].str[: config.REF_SNIPPET_CHARS]
    real_texts += snippets[covid["target"] == 0].tolist()
    fake_texts += snippets[covid["target"] == 1].tolist()

    real = pd.DataFrame({"text": real_texts}).drop_duplicates()
    fake = pd.DataFrame({"text": fake_texts}).drop_duplicates()
    return real, fake


def save_reference_corpus(data_dir=config.DATA_DIR) -> None:
    real, fake = build_reference_corpus(data_dir)
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    real.to_csv(config.REF_REAL_FILE, index=False, compression="gzip")
    fake.to_csv(config.REF_FAKE_FILE, index=False, compression="gzip")
    print(f"Reference corpus saved: {len(real)} real / {len(fake)} fake snippets")
