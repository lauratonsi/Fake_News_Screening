# Reports and Figures

This folder collects the static artifacts used in the portfolio and in the README.
The charts are not decorative: each one documents a specific data-quality problem
that shaped the final design.

## Figures

### [Reuters leakage](figures/reuters_leakage.png)
Shows that the ISOT "real" class is strongly associated with the `(Reuters)`
dateline. A classifier can exploit this source marker instead of learning the
content of the article.

### [Style leakage](figures/style_leakage.png)
Compares surface-style signals across the two classes. Fake articles in ISOT use
more punctuation and capitalized titles, which makes punctuation a shortcut for
classification.

### [Temporal window](figures/temporal_window.png)
Shows the narrow and misaligned 2015–2017 time window in the corpus. The model
has limited exposure to later domains such as COVID-era claims or post-2018
politics.

## How to regenerate

Run the dataset-bias notebook:

```bash
jupyter notebook notebooks/01_dataset_bias_analysis.ipynb
```

The notebook writes the PNG files into `reports/figures/` and documents the
consequences of the bias analysis for the final pipeline.

## Interpretation notes

- The figures describe dataset bias, not model performance.
- They are meant to explain why the system uses multi-dataset fusion,
  strict train/test separation, and a human-review flag.
- The out-of-domain benchmark in `benchmarks/adversarial_results.json` is the
  more important signal for real-world use.
