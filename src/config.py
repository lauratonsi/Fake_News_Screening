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
# Free external evidence, queried per claim in this order of precedence:
#   1. Google Fact Check Tools — only when GOOGLE_FACTCHECK_API_KEY is set;
#      the only source that returns an actual fact-check *verdict*, so it wins.
#   2. Wikipedia (MediaWiki search API) — no key, fast and reliable; returns
#      topic *context*, not a verdict. This is the dependable default so the
#      demo actually shows concrete live evidence.
#   3. GDELT — last-resort news search. Its free shared endpoint is heavily
#      rate-limited (HTTP 429) and flaky, so it only runs when the two above
#      return nothing. Kept for completeness, not relied upon.
# The committed reference corpus is always available as the offline fallback.
LIVE_MAX_CLAIMS = 3          # query live sources for at most this many claims
LIVE_TIMEOUT_SECONDS = 6
GDELT_MIN_INTERVAL = 5.0     # seconds between GDELT requests (API rate limit)
# Wikipedia's API asks clients to send a descriptive User-Agent; a generic
# urllib default can get throttled or blocked.
LIVE_USER_AGENT = "FakeNewsScreening/1.0 (https://github.com/lauratonsi/Fake_News_Screening)"
WIKIPEDIA_MAX_TERMS = 6      # Wikipedia search ranks by relevance (not strict AND)

# A full, grammatical sentence almost never appears verbatim in a news
# article or a fact-check, so sending the raw claim as the query starves
# both GDELT and Google Fact Check of hits. Queries are reduced to their
# most search-relevant terms instead (see external_retrieval._extract_keywords).
#
# GDELT ANDs every term together, so this must stay SMALL: 8 terms meant no
# single article ever contained all of them and *every* live query returned
# nothing, even for current topics (e.g. a Federal Reserve / inflation claim).
# A 2-term query verifiably returns real recent coverage; we keep a few of the
# most distinctive terms (proper-noun phrases first, then the longest content
# words — see external_retrieval._extract_keywords) to stay specific without
# over-constraining the AND.
LIVE_KEYWORD_TERMS = 3
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

# --- Input-type gating / confidence ------------------------------------------
# The classifiers are trained on full news *articles*; a short, claim-length
# input is out of domain for them, and that is exactly where they produce
# confident false positives on true statements (every adversarial-benchmark
# error is a false positive on a short true claim). Below this word count a
# model-only verdict — one with no live fact-check verdict and no near-verbatim
# corpus match to back it — is reported as LOW confidence and routed to human
# verification, instead of being presented as a settled truth judgement. This
# only lowers confidence; it never flips FAKE->REAL, so the zero-false-negative
# guarantee is preserved.
SHORT_INPUT_WORDS = 40

# --- Manipulation-technique layer (prebunking / inoculation) ------------------
# A complementary, domain- and time-robust signal: it flags *how* a text tries
# to persuade (appeal to hidden knowledge, unverifiable source, fake authority,
# fear language, false certainty, urgency, polarization). See src/manipulation.py.
# This never flips the FAKE/REAL label — it is surfaced as evidence and can
# raise the human-review flag when several techniques stack up.
MANIPULATION_TECHNIQUES_FOR_FULL_SCORE = 4  # this many distinct techniques -> score 1.0
MANIPULATION_REVIEW_MIN_TECHNIQUES = 3      # at/above this, flag for review

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

# User feedback log (append-only JSONL). Under data/, which is git-ignored, so
# user submissions are never committed. See src/feedback.py.
FEEDBACK_LOG = DATA_DIR / "feedback.jsonl"

# Minimum overall accuracy the 76-scenario adversarial benchmark must hold
# (measured 73.7%; floor set with a margin below that). The hard invariant is
# zero false negatives on classic ("human_typical") disinformation, not overall
# accuracy — this floor only guards against a silent further drop. See
# tests/test_benchmark_invariants.py.
ADVERSARIAL_ACCURACY_FLOOR = 0.65

# Minimum recall (catch rate) on "ai_fluent"-style FAKE scenarios: fluent,
# source-attributed prose with none of the classic disinformation tropes (see
# README "AI-generated disinformation is harder to detect"). Split by
# provenance because that split is what makes the finding citable rather than
# circular:
#   - `external_dataset` (18 scenarios): ChatGPT-3.5 paraphrases/rewrites of
#     real human-written misinformation from Chen & Shu's LLMFake dataset
#     (ICLR 2024) — nobody on this project wrote them, so this is the
#     trustworthy, non-circular number. Measured recall: 61.1%, vs. 100% on
#     human_typical hoaxes in the same domains — a real, more substantial gap
#     than the first, smaller (n=6) sample suggested (83.3%). Growing the
#     sample from 6 to 18 (adding topical diversity and a second generation
#     method) made the measured gap *wider*, not narrower — the smaller
#     sample was, in hindsight, not just circularity-free but also lucky.
#     A further breakdown by generation method (see by_generation_method in
#     the results file) found paraphrase-generated text far easier to catch
#     (75.0%) than rewrite-generated text (33.3%) — a real difference in how
#     the misinformation was produced, not just whether it was.
#   - `hand_authored` (8 scenarios): written for this benchmark specifically
#     to avoid overt tropes. Measured recall: 50.0% — markedly lower, but
#     cannot rule out (consciously or not) being tuned against this system's
#     own detection logic, so it is disclosed as exploratory, not headlined.
# Both floors are DISCLOSED limitations, not targets: they only stop a future
# change from making either number silently worse.
AI_FLUENT_RECALL_FLOOR_EXTERNAL = 0.45
AI_FLUENT_RECALL_FLOOR_HAND_AUTHORED = 0.30
