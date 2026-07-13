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
GDELT_SOURCE_LANGUAGE = "english"  # default; per-language routing uses GDELT_SOURCELANG
GOOGLE_FACTCHECK_LANGUAGE = "en"   # default; per-language routing uses GOOGLE_FACTCHECK_LANG

# --- Multilingual evidence routing (see src/language.py) ----------------------
# The SVM/RNN classifiers are English-trained, but the *evidence* layer is not
# language-bound: for an Italian claim we can query Italian Wikipedia, Italian
# news (GDELT sourcelang) and Italian fact-checks (Google Fact Check
# languageCode). detect_language() picks the language and these maps route each
# source to it. Full Italian *classification* still needs an Italian-labelled
# training set and a multilingual embedder (EMBEDDING_MODEL_MULTILINGUAL below);
# that is out of scope here — this makes the retrieved evidence multilingual.
SUPPORTED_LANGUAGES = ("en", "it")
DEFAULT_LANGUAGE = "en"
WIKIPEDIA_HOST = {"en": "en.wikipedia.org", "it": "it.wikipedia.org"}
GDELT_SOURCELANG = {"en": "english", "it": "italian"}
GOOGLE_FACTCHECK_LANG = {"en": "en", "it": "it"}
# Swap EMBEDDING_MODEL_NAME to this to make reference retrieval cross-lingual
# (requires re-encoding the corpus: python -m src.train). Not the default so the
# committed English embeddings keep working unchanged.
EMBEDDING_MODEL_MULTILINGUAL = "paraphrase-multilingual-MiniLM-L12-v2"

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

# --- Fluent fabricated-authority layer (the ai_fluent gap) -------------------
# The companion to the manipulation layer for the OTHER hard register: fluent,
# source-attributed prose that mimics legitimate reporting — a specific-sounding
# study citation, a suspiciously precise dose/percentage, clinical-trial
# vocabulary — with none of the classic disinformation tropes. This is exactly
# where the article-trained ensemble and the trope-based manipulation layer are
# both blind (see AI_FLUENT_RECALL_FLOOR_* below and src/ai_style.py). Like the
# manipulation layer it NEVER flips the FAKE/REAL label: legitimate science
# writing uses this register too, so it only surfaces as evidence and can raise
# the human-review flag when several markers stack — turning a silent ai_fluent
# miss into "a human should confirm the cited source exists".
AI_STYLE_MARKERS_FOR_FULL_SCORE = 3  # this many distinct marker categories -> score 1.0
AI_STYLE_REVIEW_MIN_MARKERS = 2      # at/above this, flag for review

# --- Explainability -----------------------------------------------------------
# The RNN half is explained by leave-one-out occlusion (see src/explain.py):
# each token is removed and the neural sub-ensemble re-scored. That is one extra
# forward pass per token, so cap how many leading tokens are probed to keep a
# single prediction interactive in the Streamlit demo.
EXPLAIN_MAX_TOKENS = 40
EXPLAIN_TOP_K = 8

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

# --- Optional transformer ensemble member (experiments/transformer_finetune) --
# OFF by default. A 4th ensemble vote from an end-to-end fine-tuned transformer
# is added ONLY after the promotion gate (src/ensemble_gate.should_promote)
# confirms, on the untouched test set and the adversarial benchmark, that it
# helps without regressing accuracy, adding censorship-side false positives, or
# breaking the zero-false-negative guarantee. When enabled AND the artifact dir
# exists, ScreeningSystem loads it (dynamic-int8-quantized on load) as a 4th
# score; combine_verdict already averages an arbitrary number of members.
# Enabling it at runtime also requires adding `transformers` to requirements.txt.
TRANSFORMER_ENABLED = False
TRANSFORMER_DIR = MODELS_DIR / "transformer_model"
TRANSFORMER_MAX_TOKENS = 256

# User feedback log (append-only JSONL). Under data/, which is git-ignored, so
# user submissions are never committed. See src/feedback.py.
FEEDBACK_LOG = DATA_DIR / "feedback.jsonl"

# --- Active-learning retraining from feedback (see src/retrain_from_feedback) --
# incorporate_feedback.py folds verified corrections into the *retrieval* corpus
# only, and explicitly defers retraining the classifiers because "retraining on
# unaudited user submissions would need far more care (adversarial submissions,
# label noise, class balance)". These guards are that care, so a batch of
# verified corrections can safely reach the SVM/RNN weights without letting a
# flood of adversarial or mislabelled submissions reshape the model:
#   * only VERIFIED corrections (user marked the result wrong + gave the right
#     label) that were not already folded in (idempotent);
#   * a MINIMUM batch size, so a handful of clicks never reshapes the model;
#   * a hard CAP on how much of the training set feedback may become, balanced
#     across classes, so submissions can nudge but never dominate;
#   * corrections are added to the TRAINING split only — never the held-out test
#     set — and any correction whose text already sits in training with the
#     OPPOSITE label (a contradiction, the classic poisoning shape) is dropped.
RETRAIN_MIN_CORRECTIONS = 20        # refuse to retrain on fewer verified corrections
RETRAIN_MAX_FEEDBACK_FRACTION = 0.05  # corrections may be at most 5% of the training rows
RETRAIN_MIN_CORRECTION_CHARS = 50   # drop junk/too-short submissions (cf. WELFAKE_MIN_CHARS)
# A retrain that makes accuracy on the untouched test set worse by more than this
# is surfaced as a loud regression warning (revert with git), not silently kept.
RETRAIN_REGRESSION_TOLERANCE = 0.01

# --- Temporal corpus refresh (see src/refresh_corpus) ------------------------
# The reference corpus is frozen at 2015-2017 (see reports/figures/temporal_window
# — real coverage only starts in 2016). This pulls RECENT, professionally
# fact-checked claims from Google Fact Check — the one live source that returns
# an actual verdict — and folds them into the retrieval corpus the same way
# verified user corrections are, so semantic retrieval keeps seeing current
# claims instead of decaying with time. Meant to run on a schedule (cron / the
# user's scheduler). Requires GOOGLE_FACTCHECK_API_KEY; without it, it refuses
# rather than silently ingesting nothing. Idempotent via a manifest of already-
# ingested fact-check review URLs.
REFRESH_SEED_QUERIES = [
    "vaccine", "election", "climate change", "economy", "public health",
    "immigration", "artificial intelligence", "war", "covid",
]
REFRESH_SEED_QUERIES_IT = [
    "vaccino", "elezioni", "clima", "economia", "salute pubblica",
    "immigrazione", "intelligenza artificiale", "guerra", "covid",
]
REFRESH_MAX_PER_QUERY = 10
REFRESH_MAX_TOTAL = 200            # cap one refresh run, balanced across REAL/FAKE
REFRESH_MIN_CLAIM_CHARS = 40       # drop fragmentary claim texts
REFRESH_MANIFEST = DATA_DIR / "refresh_manifest.json"  # ingested review URLs (idempotency)

# Minimum overall accuracy the 101-scenario adversarial benchmark must hold
# (measured 76.2%; floor set with a margin below that). The hard invariant is
# zero false negatives on classic ("human_typical") disinformation, not overall
# accuracy — this floor only guards against a silent further drop. See
# tests/test_benchmark_invariants.py.
ADVERSARIAL_ACCURACY_FLOOR = 0.70

# Minimum recall (catch rate) on "ai_fluent"-style FAKE scenarios: fluent,
# source-attributed prose with none of the classic disinformation tropes (see
# README "AI-generated disinformation is harder to detect"). Split by
# provenance because that split is what makes the finding citable rather than
# circular:
#   - `external_dataset` (43 scenarios): ChatGPT-3.5 outputs on real
#     human-written articles from Chen & Shu's LLMFake dataset (ICLR 2024) —
#     nobody on this project wrote them, so this is the trustworthy,
#     non-circular number. Measured recall: 74.4%, vs. 100% on human_typical
#     hoaxes in the same domains.
#     History of this number as the sample grew: n=6 -> 83.3%, n=18 -> 61.1%,
#     n=43 -> 74.4%. The gap first widened, then narrowed, as more generation
#     methods and domains were added — the honest reading is that recall on
#     any single small subgroup is noisy, and only the two largest, most
#     mature buckets (paraphrase_generation, rewrite_generation: 75.0% /
#     33.3%, unchanged since n=18) should be treated as remotely stable.
#     Newer, smaller buckets (open_ended_generation, information_manipulation
#     sub-strategies, hallucination/partially_arbitrary_generation) each carry
#     n<=10 and are tracked (see by_generation_method) but not yet trustworthy
#     individually.
#   - `hand_authored` (8 scenarios): written for this benchmark specifically
#     to avoid overt tropes. Measured recall: 50.0% — markedly lower, but
#     cannot rule out (consciously or not) being tuned against this system's
#     own detection logic, so it is disclosed as exploratory, not headlined.
# Both floors are DISCLOSED limitations, not targets: they only stop a future
# change from making either number silently worse. The hand_authored floor
# also doubles as the shared minimum every by_generation_method bucket must
# clear (see test_generation_method_recall_does_not_regress) since most of
# those buckets are too small (n<=10) for their own dedicated floor.
AI_FLUENT_RECALL_FLOOR_EXTERNAL = 0.60
AI_FLUENT_RECALL_FLOOR_HAND_AUTHORED = 0.30
