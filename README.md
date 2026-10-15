# Titanic Historical Record Linkage

**Reconstructing published Titanic passenger outcomes through deterministic cross-dataset matching.**

`Python` · `Pandas` · `Record Linkage` · `Data Provenance` · `Reproducibility`

[View the Kaggle Notebook](https://www.kaggle.com/code/kalidenaiarys/titanic-historical-record-linkage-1309-verified) · [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)

> **Methodology disclosure:** This is **not a machine-learning prediction model**. The Vanderbilt Titanic3 dataset includes published survival outcomes corresponding to Kaggle's test passengers. Matching records and looking up those outcomes is fundamentally different from predicting unknown labels. A perfect competition submission score, if separately documented, should **not** be described as 100% ML accuracy.

## At a glance

| Check | Result in the recorded Kaggle run |
| --- | ---: |
| Vanderbilt Titanic3 source records | 1,309 |
| Kaggle training records linked | 891 / 891 |
| Kaggle test records linked | 418 / 418 |
| Exact name + ticket matches (test) | 396 |
| Normalized name + exact ticket matches (test) | 22 |
| Unmatched / ambiguous test records | 0 / 0 |
| Agreement with known Kaggle training labels | 891 / 891 |

These are **record-linkage consistency checks**, not out-of-sample model-evaluation metrics. Figures were taken from the saved Kaggle notebook run dated 13 October 2026.

## Problem and approach

The Titanic beginner competition normally asks participants to infer passenger survival from features in a training dataset. This project instead explores the data-provenance issue that arises when the published historical dataset contains outcomes for the competition's test passengers.

1. Read Kaggle's `train.csv` (891 rows) and `test.csv` (418 rows), and Vanderbilt's `titanic3.csv` (1,309 rows).
2. Normalize passenger names by lowercasing and removing punctuation; combine each normalized name with its exact ticket identifier.
3. Require a unique cross-dataset key; perform a one-to-one join without fuzzy matching.
4. Check class, sex, siblings/spouses, and parents/children fields for identity conflicts.
5. Confirm that known training labels agree with Vanderbilt labels, and that source rows are disjoint across the two Kaggle splits.
6. Export a submission file, a per-record matching audit and a JSON verification report with SHA-256 digests.

## Reproduce locally

**Requirements:** Python 3.10+ and pandas (see `requirements.txt`).

1. Download `train.csv` and `test.csv` after accepting the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) rules.
2. Download [Vanderbilt Titanic3 CSV](https://hbiostat.org/data/repo/titanic3.csv).
3. Save all three files to a local `data/` directory (excluded from this repository).
4. Install dependencies and execute:

```bash
python -m pip install -r requirements.txt
python build_submission.py --data data --out output
```

Generated files in `output/`: `submission.csv`, `matching_audit.csv`, `verification.json`.

**Data are not bundled here**. Obtain them from their original publishers and observe applicable competition and dataset terms.

## Notebook

- [Clean and readable analysis notebook in this repository](notebooks/titanic_historical_record_linkage.ipynb). This version is designed for reproducible execution and does not bundle historical source files.
- [Executed Kaggle notebook with its saved outputs](https://www.kaggle.com/code/kalidenaiarys/titanic-historical-record-linkage-1309-verified). The GitHub notebook intentionally separates explanation from intermediate debugging steps.

## Results and verification

The script checks expected row counts, identity uniqueness, one-to-one matching, field consistency, train-label agreement, and input/output hashes. The recorded run found 396 exact test name+ticket matches and 22 additional test matches after name normalization. There were no ambiguous or unmatched test records.

The score of a separate competition submission cannot be confirmed from `verification.json` alone: the report intentionally stores `kaggle_score: null`. The published outcomes used by this approach are **external target labels**, not unseen ground truth inferred by a trained algorithm.

## Limitations

- No model is trained or evaluated with a held-out sample.
- A fully linked historical dataset directly reveals outcomes for test passengers; results are not evidence of generalization to new examples.
- Name normalization may collide or fail on other datasets. This script explicitly rejects ambiguous identities rather than guessing.
- The matching relies on particular Kaggle/Vanderbilt dataset versions; the verification report stores their input SHA-256 hashes.
- The approach illustrates data provenance, reproducibility, and evaluation leakage, **not** predictive AI performance.

## Links and attribution

- [Interactive Kaggle notebook](https://www.kaggle.com/code/kalidenaiarys/titanic-historical-record-linkage-1309-verified)
- [Kaggle Titanic dataset and rules](https://www.kaggle.com/competitions/titanic)
- [Vanderbilt Titanic3 data](https://hbiostat.org/data/repo/titanic3.csv)
- [Vanderbilt Titanic documentation](https://hbiostat.org/data/repo/titanic)

Historical dataset courtesy of the Vanderbilt University Department of Biostatistics.

## Reuse and scope

This repository publishes the matching implementation and verification metadata, rather than rehosting the Kaggle and Vanderbilt datasets. Reproduction requires the original source files and their respective use permissions. It should be evaluated as a record-linkage project, **not** a survival-prediction model.
