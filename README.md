# Fake News Screening (HDSS)

**English** | [Italiano](README.it.md)

A hybrid disinformation screening system: a calibrated SVM, a Bi-GRU and a
Bi-LSTM vote on English news text, backed by a *semantic* similarity lookup
against the training corpora and a human-review flag when the models
disagree. Originally developed as a university AI project, rebuilt here as a
clean, reproducible pipeline: **dataset analysis → models → Streamlit demo**.

Live demo: https://fake-news-screening.streamlit.app/

> **The honest headline:** the ensemble scores **94.6%** on a leakage-free
> in-domain test set and **80.0%** on 30 out-of-domain adversarial scenarios,
> with **zero false negatives** — it never waves a hoax through; every error
> is an over-cautious false positive on a true statement. The retrieval layer
> is used to *find* the closest known real/fake claims, not to assert truth
> from topical similarity — see *"Two very different uses of embeddings"* below
> for why that distinction matters and what it costs.

## The problem with "99% accuracy"

Early experiments on the ISOT corpus put *every* architecture above 98%
accuracy. [`notebooks/01_dataset_bias_analysis.ipynb`](notebooks/01_dataset_bias_analysis.ipynb)
documents why those numbers are a red flag rather than a result:

| Bias in the data | Effect |
|---|---|
| **Stylistic leakage** — fake articles average 2.16 `!`/`?` per article and 30% capitals in titles, real ones 0.17 and 6% | models learn punctuation, not content |
| **Source leakage** — 99.2% of "real" articles contain the `(Reuters)` dateline, 0.0% of fake ones | the label is literally written in the text |
| **Temporal blindness** — 2015–2017 US politics only, with fake/real volumes misaligned in time | anything post-2018 (COVID, elections) is out of domain |

<p align="center">
  <img src="reports/figures/style_leakage.png" width="48%" alt="Fake articles average 2.16 exclamation/question marks per article vs 0.17 for real, and 30% capital letters in titles vs 6% for real" />
  <img src="reports/figures/reuters_leakage.png" width="48%" alt="99.2% of real articles contain the (Reuters) dateline vs 0.0% of fake articles" />
</p>
<p align="center">
  <img src="reports/figures/temporal_window.png" width="70%" alt="Real and fake article volumes per month, 2015 to 2018, showing a narrow and misaligned time window" />
</p>

## What the system does about it

1. **Multi-dataset fusion** — ISOT + WELFake (quality-filtered: length, caps
   ratio, punctuation) + COVID-19 claims, deduplicated: 53,661 unique articles.
2. **Strict split protocol** — train/test split *before* any oversampling;
   the COVID slice is balanced and boosted ×3 on the training side only; all
   models share the same untouched test set (10,733 articles).
   Fixing this protocol alone moved the SVM from a claimed ~98% to a real 95.3%.
3. **Ensemble of cheap, transparent models** — TF-IDF + calibrated LinearSVC
   baseline, plus two light bidirectional RNNs (~1.3 MB each), served as
   TFLite models via the ~10 MB `ai-edge-litert` interpreter rather than the
   full TensorFlow runtime; final score is the simple average.
4. **Reference retrieval layer** — sentence-embedding similarity
   ([`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
   against snippets of the ~68k known real/fake articles: it matches a
   *reworded* claim, not just a literal one. This is *retrieval over what
   the system has already seen*, *not* fact-checking, and the demo shows the
   retrieved evidence explicitly.
5. **Claim-level retrieval** — the input is split into claim-like sentences
   and each claim is retrieved independently, so the UI can show, per claim,
   whether it matches a known false claim, matches known reporting, or has no
   close match — evidence labels, not truth judgements.
6. **Live retrieval** — the first few claims are also checked against free
    live sources, in order: Google Fact Check Tools (a real fact-check
    *verdict*, when an API key is configured), then Wikipedia (reliable,
    key-free topic *context*), with GDELT as a last-resort news search. A live
    fact-check verdict takes precedence for that claim; otherwise the committed
    corpus decides, so the system works fully offline too.
7. **Human-review flag** — when the three models disagree strongly
   (spread > 0.40), the verdict is marked low-confidence instead of being
   reported as certain.

## Pipeline & Figures

The full pipeline is documented in [PIPELINE.md](PIPELINE.md). It shows the
end-to-end flow from raw datasets to Streamlit deployment.

The reporting layer is summarized in [reports/README.md](reports/README.md),
which explains what each chart above proves and why it matters for the final
system. Taken together, the three figures document the failure modes that
forced the final design away from a plain accuracy-driven benchmark and
toward a retrieval-plus-review workflow.

## Pipeline summary

```mermaid
flowchart LR
    A[Raw datasets] --> B[Clean / deduplicate / filter]
    B --> C[Train-test split]
    C --> D[TF-IDF + tokenizer]
    D --> E[SVM + Bi-GRU/Bi-LSTM -> TFLite]
    E --> F[Semantic reference-corpus retrieval]
    F --> G[Ensemble + review flag]
    G --> H[Streamlit demo]
```

## Results (all measured, all reproducible)

**In-domain** — shared held-out test set, `python -m src.train` →
[`models/metrics.json`](models/metrics.json):

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) |
|---|---|---|---|---|
| SVM (TF-IDF, calibrated) | 95.3% | 94.8% | 94.9% | 94.8% |
| Bi-GRU | 92.9% | 93.0% | 91.0% | 92.0% |
| Bi-LSTM | 92.9% | 94.1% | 89.9% | 92.0% |
| **Ensemble (mean)** | **94.6%** | 94.5% | 93.3% | 93.9% |

**Out-of-domain** — 30 adversarial scenarios (plausible hoaxes, uncomfortable
truths), `python -m src.evaluate --adversarial` →
[`benchmarks/adversarial_results.json`](benchmarks/adversarial_results.json):

| Domain | Accuracy | False positives | False negatives | Flagged for review |
|---|---|---|---|---|
| Politics | 70% | 3 | 0 | 2 |
| COVID | 90% | 1 | 0 | 3 |
| Mixed | 80% | 2 | 0 | 3 |
| **Overall** | **80.0%** | 6 | 0 | 8 |

Every error is a **false positive on a true statement** ("Donald Trump won the
2016 election…" → FAKE): the 2015–2017 training window taught the classifiers
that short factual claims about US politics *look like* fake-news bait, and the
retrieval layer deliberately no longer "rescues" them by treating a
same-topic real article as proof (see the next section). This is the
temporal/stylistic bias surviving every mitigation — the reason the demo
presents itself as a screening aid, not a truth oracle. What matters for a
disinformation tool is the other column: **zero false negatives**, no hoax
waved through.

## Reporting Takeaways

The report charts are meant to answer three questions before anyone looks at
accuracy:

1. Is the dataset leaking the label through style?
2. Is the label leaking through source markers?
3. Is the temporal window too narrow to support generalization?

If any of those answers is "yes", the model metrics need to be read as
in-domain estimates only. That is why the portfolio now foregrounds the
adversarial benchmark and the retrieval/review pipeline instead of just the
headline accuracy number.

## Two very different uses of embeddings

TF-IDF, a linear SVM and two small RNNs look dated next to current text
classifiers — so both uses of transformer embeddings were tested on this
project, with opposite, equally instructive results.

**Classification: tested, rejected.** `experiments/` replaces the TF-IDF
baseline with sentence embeddings
([`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
plus a calibrated linear classifier, trained and evaluated on the *exact
same* fused dataset and split as `src.train`
(`experiments/embeddings_baseline.py`, `experiments/embeddings_adversarial.py`).

| | In-domain | Out-of-domain (30 scenarios) |
|---|---|---|
| Current ensemble (TF-IDF + SVM/GRU/LSTM) | 94.6% | 80.0% |
| MiniLM embeddings + linear classifier | 88.5% | 60% |

The embeddings-based classifier lost on both axes — sharpest on WELFake
(67.1% vs. 86.9%) and on the adversarial "mixed" domain. This is the measured
consequence of the leakage documented in
[`notebooks/01_dataset_bias_analysis.ipynb`](notebooks/01_dataset_bias_analysis.ipynb):
the fake/real split in these corpora is driven largely by surface style and
source markers (punctuation, capitalization, the `(Reuters)` dateline), and
TF-IDF is built to exploit exactly that literal signal. A semantic embedding
model is built to be invariant to it — so on this dataset, understanding
meaning *better* is a handicap for classification.

**Retrieval: tested, adopted.** Finding the closest *known* claim is a
different task from classification, and it is exactly what semantic
embeddings are good at: matching "the COVID shot alters your genetic code"
to a stored claim about the vaccine "permanently altering DNA" despite
almost no shared vocabulary — something the old TF-IDF reference layer,
built on literal term overlap, structurally could not do. `src/rag.py` now
embeds the ~68k-snippet reference corpus once (`REF_EMBEDDINGS_FILE`,
committed, ~46 MB) and compares queries against it by cosine similarity.
The model weights themselves (`models/embedding_model/`, ~88 MB) are also
committed rather than pulled from the Hugging Face Hub at runtime — Streamlit
Cloud containers restart from a clean filesystem on every redeploy, and this
layer runs on *every* prediction, not just live retrieval, so a Hub download
on cold start was a real "the app won't boot if the network hiccups" risk.

**The retrieval signal is deliberately asymmetric — and that asymmetry
matters more than the headline number.** An early version let *any* close
match, real or fake, sway the verdict. It scored a higher 83.3% adversarial
— but it did so partly by fabricating truth: a false claim shares its topic
with genuine reporting constantly ("the vaccine alters your DNA" sits right
next to real articles on COVID genetics), so the demo would show a green
"known REAL / SUPPORTED" panel *directly under a red FAKE headline*, and even
green-light a vaccine conspiracy. That is exactly the wrong signal for a
disinformation tool. So the reference layer is now asymmetric:

- Matching a known **fake** claim is genuine evidence of fakeness — it boosts
  the score from a modest similarity, and a near-verbatim match can override.
- Matching a known **real** article only asserts REAL when it is
  *near-verbatim* (`REF_OVERRIDE_THRESHOLD = 0.90`); mere topical proximity is
  surfaced as neutral evidence ("closest known snippet is real at 69%"), never
  as a verdict.

This costs about three points of adversarial accuracy (83.3% → 80.0%, one
true COVID statement no longer "rescued" by a same-topic real article) — a
cost worth paying: the panels can no longer contradict the headline, and the
demo never presents a false claim as supported. Embedding similarity simply
does not separate "same claim, reworded" from "same topic, different claim"
cleanly enough to be trusted as a truth signal, only as retrieved evidence.

**Why both were affordable at once:** the RNNs now run as TFLite models via
the ~10 MB `ai-edge-litert` interpreter instead of the full TensorFlow
runtime (~500+ MB just for the framework, regardless of model size).
Measured peak memory for the whole system — SVM, both RNNs, the reference
corpus, and the sentence-embedding model together — is **~600 MB**, against
a free-tier Streamlit Cloud ceiling of 1 GB. Running TensorFlow and PyTorch
side by side would not have fit; running neither RNNs nor embeddings would
have been a needless trade-off. Training still happens in full TensorFlow
(`requirements-train.txt`); only the deployed app needed to change.

## Scope within the information-disorder taxonomy

"Fake news" is a scientifically inadequate label: Wardle & Derakhshan's
*Information Disorder* framework (Council of Europe, 2017) distinguishes
**misinformation** (false, shared without harmful intent), **disinformation**
(false, intentionally harmful) and **malinformation** (genuine content used to
harm). A text classifier can only ever address the *content-falsity signal* of
the first two — it is blind to intent, and by construction to malinformation,
where the content is true. That is a second, structural reason (besides the
measured out-of-domain accuracy) why this system is framed as a **screening
aid inside a human process**, not an automated arbiter of truth.

The versioned adversarial benchmark follows the same logic that cognitive
security literature applies to institutions — *you cannot defend what you have
not tested*: the 30 scenarios are kept in the repo as a permanent, repeatable
stress test rather than a one-off experiment.

## Repository layout

```
├── app.py                  Streamlit demo (UI only)
├── src/
│   ├── config.py           every path, hyperparameter and threshold
│   ├── data.py             unified load / filter / fuse / split protocol
│   ├── train.py            trains SVM + GRU + LSTM, exports TFLite, writes metrics.json
│   ├── predict.py          ScreeningSystem: ensemble + heuristic + review flag
│   ├── evaluate.py         in-domain report & adversarial benchmark
│   ├── rag.py              reference-corpus retrieval (semantic embeddings)
│   ├── claim_rag.py        per-claim retrieval analysis
│   ├── external_retrieval.py  live evidence (Google Fact Check / Wikipedia / GDELT)
│   └── tokenizer.py        framework-independent tokenizer (no TF at serving time)
├── tests/                  pytest suite: split protocol, ensemble logic, retrieval
├── models/                 trained artifacts incl. TFLite RNNs (~8 MB) and the
│                           bundled embedding model (~88 MB, committed)
├── reference_corpus/       known real/fake snippets + embeddings (~55 MB)
├── benchmarks/             versioned scenarios + measured results
├── experiments/            tested-and-rejected alternatives (see below)
├── notebooks/              dataset bias analysis (the "why" of the design)
├── reports/figures/        exported charts
└── data/                   datasets (not committed — see data/README.md)
```

## Quickstart

```bash
# Python 3.10 or 3.11
pip install -r requirements.txt

# Run the demo with the committed models
streamlit run app.py

# Reproduce everything from scratch — needs the datasets (see data/README.md)
# AND TensorFlow, only used for training; the app itself does not need it:
pip install -r requirements-train.txt
python -m src.train                  # ~10 min on CPU
python -m src.evaluate               # in-domain metrics table
python -m src.evaluate --adversarial # out-of-domain benchmark

# Run the test suite (split protocol, ensemble logic, retrieval)
pip install -r requirements-dev.txt
python -m pytest tests/
```

## Deploy on Streamlit Cloud

This repository is already configured for a standard Streamlit Cloud deploy.

You can open the deployed app directly at
https://fake-news-screening.streamlit.app/.

1. Connect the GitHub repository `lauratonsi/Fake_News_Screening`.
2. Use `app.py` as the entry point.
3. Keep the default branch as `main`.
4. Let Streamlit install dependencies from `requirements.txt` (it includes a
   CPU-only PyTorch index for `torch`, so it does not pull in a multi-GB
   CUDA build).
5. In **Advanced settings**, set the Python version to **3.11**.
6. The app theme/server defaults are set in `.streamlit/config.toml`.

If the deployment succeeds, the demo should load the committed models from
`models/` and `reference_corpus/` and run without requiring retraining or
TensorFlow — see *"Why both were affordable at once"* above for the memory
budget behind that.

## Live retrieval: setup and honest expectations

The live layer (`src/external_retrieval.py`) queries free sources per claim,
in order of precedence:

1. **Google Fact Check Tools** — only if `GOOGLE_FACTCHECK_API_KEY` is set. The
   only source that returns an actual fact-check *verdict*, so it wins.
2. **Wikipedia** (MediaWiki search API, no key) — reliable, fast topic
   *context*. This is the dependable default that makes the "Live retrieval"
   panel actually show concrete evidence. It is context, never a verdict.
3. **GDELT** (no key) — a last-resort live *news* search. Its free shared
   endpoint is heavily rate-limited (HTTP 429) and flaky, so it only runs when
   the two above return nothing; it is kept for completeness, not relied upon.

Wikipedia returns something on-topic for essentially any claim, current or
historical, which is why it replaced GDELT as the default: a Federal Reserve
claim returns the *Federal Reserve* article, a 2016-election claim returns
*Donald Trump* / *Hillary Clinton 2016 presidential campaign*. This is
context to read, not a truth verdict — only the Google Fact Check path
asserts one.

**To enable the higher-quality Google Fact Check path:**
1. In Google Cloud Console, enable the "Fact Check Tools API" and create an
   API key.
2. Locally: `export GOOGLE_FACTCHECK_API_KEY=your-key-here` before
   `streamlit run app.py`.
3. On Streamlit Community Cloud: open the app's **Settings → Secrets** and
   add
   ```toml
   GOOGLE_FACTCHECK_API_KEY = "your-key-here"
   ```
   Streamlit Cloud exposes Secrets to the app as environment variables, so no
   code change is needed.

Without a key, the app still works exactly as documented above — Google
Fact Check is skipped and Wikipedia is the default live source.

## Honest limitations

- English only; the training corpora essentially stop in 2020 — current events are out of domain.
- The reference lookup recognises *known* claims (now including reworded
  ones — see above); it cannot verify genuinely new ones. Its top-1
  nearest-neighbor search can also confuse "same topic" with "same claim"
  for ambiguous inputs, which is why overriding the ensemble outright is
  reserved for near-verbatim matches (`REF_OVERRIDE_THRESHOLD = 0.90`).
- The RNNs are trained on a 5,000-article subsample (CPU budget); the SVM sees
  the full training set.
- Out-of-domain accuracy (80.0%) is the number that matters for real-world
  use, and it is why any deployment of a system like this needs a human in
  the loop.
