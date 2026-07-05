# Datasets

The datasets are not committed to the repository. Download them into this
folder before running `python -m src.train`:

| Dataset | Files expected here | Source |
|---|---|---|
| ISOT Fake News | `Fake.csv`, `True.csv` | [ISOT / University of Victoria](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) (also on [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)) |
| WELFake | `WELFake_Dataset.csv` | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| COVID-19 Fake News | `Covid_Fake_New/` (folder with the `ClaimFake*/ClaimReal*/NewsFake*/NewsReal*` CSV files) | [CoAID-style COVID collection on Kaggle](https://www.kaggle.com/datasets/arashnic/covid19-fake-news) |

Expected layout:

```
data/
├── Fake.csv
├── True.csv
├── WELFake_Dataset.csv
└── Covid_Fake_New/
    ├── ClaimFakeCOVID-19_5.csv
    ├── ClaimRealCOVID-19.csv
    └── ...
```

Label convention used throughout the project: `target = 1` fake, `target = 0` real.
Tweet/reply files in the COVID folder are ignored — only news/claim files are used.
