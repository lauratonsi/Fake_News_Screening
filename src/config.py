"""Central configuration: paths, hyperparameters and decision thresholds.

Every tunable of the project lives here so that the data pipeline, the
training script, the evaluation script and the Streamlit demo all agree
on a single source of truth.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REFERENCE_DIR = ROOT / "reference_corpus"
BENCHMARKS_DIR = ROOT / "benchmarks"
REPORTS_DIR = ROOT / "reports"

SEED = 42

# --- Dataset fusion (ISOT + WELFake + COVID) ---------------------------------
# WELFake quality filters: drop very short/long articles and articles whose
# styling (ALL CAPS, !!!) would let models learn punctuation instead of content.
WELFAKE_MIN_CHARS = 50
WELFAKE_MAX_CHARS = 5_000
WELFAKE_MAX_CAPS_RATIO = 0.30
WELFAKE_MAX_EXCLAMATIONS = 10
WELFAKE_SUBSAMPLE = 0.6  # keep 60% so WELFake does not dominate the fusion

# The COVID slice is small; it is balanced and oversampled (x3) so the models
# actually learn from it. This happens on the TRAINING split only — the test
# split keeps the natural, deduplicated distribution.
COVID_BOOST_FACTOR = 3

TEST_SIZE = 0.2

# --- Models -------------------------------------------------------------------
TFIDF_MAX_FEATURES = 50_000

VOCAB_SIZE = 10_000
MAX_LEN = 100
EMBEDDING_DIM = 32
RNN_UNITS = 16
RNN_TRAIN_SAMPLE = 5_000  # RNNs are trained on a subsample (CPU-friendly)
RNN_EPOCHS = 5
RNN_BATCH_SIZE = 16

# --- Reference-corpus similarity heuristic ------------------------------------
# A lookup against the texts the system already knows to be real or fake.
# This is a heuristic support signal, NOT fact-checking: it can only recognise
# claims *semantically similar* to ones already present in the reference
# corpus (sentence embeddings, not literal word overlap — see src/rag.py).
# Thresholds are calibrated empirically for this embedding model in
# experiments/calibrate_rag_thresholds.py; re-run it if the model changes.
REF_SNIPPET_CHARS = 300
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Loaded from this committed local copy (~88 MB), NOT downloaded from the
# Hugging Face Hub at runtime: Streamlit Cloud containers restart from a
# clean filesystem on every redeploy, so a Hub download at cold start is a
# real availability risk (network hiccup or Hub rate limit = app won't boot)
# for a component that runs on every single prediction, not just live
# retrieval. Falls back to the Hub name if the local copy is ever missing
# (e.g. a fresh clone before running src.train, which regenerates it).
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding_model"
REF_MATCH_THRESHOLD = 0.45   # minimum cosine similarity to count as a match
REF_MARGIN = 0.05            # required gap between the two corpora
# Empirically, semantic similarity does not separate "same claim, reworded"
# from "same topic, different claim" nearly as cleanly as TF-IDF's near-
# literal matches did: on the 30 adversarial scenarios, 3 of 4 wrong
# reference-only calls scored 0.69-0.82 — well above the old 0.65 override
# threshold. Overriding the ensemble is now reserved for near-verbatim
# repeats of a known snippet, not merely topically-similar ones.
REF_OVERRIDE_THRESHOLD = 0.90  # above this the match overrides the ensemble
REF_BOOST = 0.25             # otherwise it only shifts the ensemble score

# --- Live retrieval -------------------------------------------------------------
# Free external evidence: Google Fact Check when an API key is configured,
# GDELT otherwise. GDELT allows roughly one request every 5 seconds, so live
# lookups are capped per input and rate-limited; the committed reference
# corpus is always available as the offline fallback.
LIVE_MAX_CLAIMS = 3          # query live sources for at most this many claims
LIVE_TIMEOUT_SECONDS = 6
GDELT_MIN_INTERVAL = 5.0     # seconds between GDELT requests (API rate limit)

# A full, grammatical sentence almost never appears verbatim in a news
# article or a fact-check, so sending the raw claim as the query starves
# both GDELT and Google Fact Check of hits. Queries are reduced to their
# most search-relevant terms instead (see external_retrieval._extract_keywords).
LIVE_KEYWORD_TERMS = 8
# GDELT's DOC 2.0 API archive starts ~Feb 2017. A 1-year window (the
# previous value) makes it mathematically impossible to find coverage of
# anything older than 1 year ago — which is most of this project's demo
# examples and adversarial scenarios (2016 election, 2017 firings, 2020-21
# COVID claims). Widened so GDELT actually has a chance at retrospective
# mentions of older claims; genuinely current topics (e.g. interest rates)
# still work fine with a wide window. GDELT is still a *news* search engine,
# not a fact-checking archive: a historical claim with no retrospective
# coverage will still correctly show "no live evidence".
GDELT_TIMESPAN = "10y"
GDELT_SOURCE_LANGUAGE = "english"  # the demo is English-only (see README)
GOOGLE_FACTCHECK_LANGUAGE = "en"

# --- Ensemble -----------------------------------------------------------------
# If the individual model scores span more than this, the system flags the
# input for human review instead of pretending to be confident.
DISAGREEMENT_SPREAD = 0.40

# --- Artifact locations --------------------------------------------------------
SVM_FILE = MODELS_DIR / "svm_tfidf.joblib"
GRU_FILE = MODELS_DIR / "gru.keras"
LSTM_FILE = MODELS_DIR / "lstm.keras"
GRU_TFLITE_FILE = MODELS_DIR / "gru.tflite"
LSTM_TFLITE_FILE = MODELS_DIR / "lstm.tflite"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
REF_REAL_FILE = REFERENCE_DIR / "real.csv.gz"
REF_FAKE_FILE = REFERENCE_DIR / "fake.csv.gz"
REF_EMBEDDINGS_FILE = REFERENCE_DIR / "embeddings.npz"
SCENARIOS_FILE = BENCHMARKS_DIR / "adversarial_scenarios.json"
ADVERSARIAL_RESULTS_FILE = BENCHMARKS_DIR / "adversarial_results.json"
