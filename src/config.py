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
# claims similar to ones already present in the reference corpus.
REF_SNIPPET_CHARS = 300
REF_TFIDF_FEATURES = 2_000
REF_MATCH_THRESHOLD = 0.55   # minimum cosine similarity to count as a match
REF_MARGIN = 0.10            # required gap between the two corpora
REF_OVERRIDE_THRESHOLD = 0.75  # above this the match overrides the ensemble
REF_BOOST = 0.25             # otherwise it only shifts the ensemble score

# --- Live retrieval -------------------------------------------------------------
# Free external evidence: Google Fact Check when an API key is configured,
# GDELT otherwise. GDELT allows roughly one request every 5 seconds, so live
# lookups are capped per input and rate-limited; the committed reference
# corpus is always available as the offline fallback.
LIVE_MAX_CLAIMS = 3          # query live sources for at most this many claims
LIVE_TIMEOUT_SECONDS = 6
GDELT_MIN_INTERVAL = 5.0     # seconds between GDELT requests (API rate limit)

# --- Ensemble -----------------------------------------------------------------
# If the individual model scores span more than this, the system flags the
# input for human review instead of pretending to be confident.
DISAGREEMENT_SPREAD = 0.40

# --- Artifact locations --------------------------------------------------------
SVM_FILE = MODELS_DIR / "svm_tfidf.joblib"
GRU_FILE = MODELS_DIR / "gru.keras"
LSTM_FILE = MODELS_DIR / "lstm.keras"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
REF_REAL_FILE = REFERENCE_DIR / "real.csv.gz"
REF_FAKE_FILE = REFERENCE_DIR / "fake.csv.gz"
SCENARIOS_FILE = BENCHMARKS_DIR / "adversarial_scenarios.json"
ADVERSARIAL_RESULTS_FILE = BENCHMARKS_DIR / "adversarial_results.json"
