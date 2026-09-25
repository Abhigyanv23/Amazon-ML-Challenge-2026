# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source 1 business record, find all matching records in Source 2 and Source 3.
Scored by per-entity F0.5, macro-averaged over Source 1.

Uses **only** the competition-provided data. No external lookups, APIs, geocoding or scraping.

## Setup

Python 3.10+ (Windows PowerShell shown; use `python3` / `/` on Mac/Linux).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONIOENCODING="utf-8"
```

Place the data (not committed to git):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## Pipeline (run from repo root, in order)

| Step | Command | Output |
|---|---|---|
| 0. EDA (optional) | `python src\data_inspection.py --data-dir dataset` | report to stdout |
| 1. Holdout split (run once, frozen) | `python src\make_split.py` | `experiments/splits/*_s1_ids.txt` |
| 2. Scorer self-test | `python src\validate_local.py --selftest` | F0.5 = 1.0000 / 0.0558 |
| 3. Normalize | `python src\normalization.py --split train` then `--split test` | `experiments/cache/*.parquet` |
| 4. Blocking: holdout (recall) | `python src\blocking.py --split train --s1-set holdout --k 50 --tag v001` | candidates + `experiments/v001/blocking_holdout.json` |
| 5. Blocking: train sample | `python src\blocking.py --split train --s1-set train --sample 300000 --k 20 --tag v001` | training candidates |
| 6. Train matcher | `python src\matcher.py train --tag v001 --k 20` | `models/v001/lgbm.txt`, `experiments/v001/decision.json` |
| 7. Score holdout | `python src\matcher.py holdout --tag v001 --k 20` | `experiments/v001/holdout_metrics.json` |
| 8. Blocking: test | `python src\blocking.py --split test --k 20 --tag v001` | test candidates |
| 9. Predict test | `python src\matcher.py test --tag v001 --k 20` | `output/candidate_pairs.tsv`, `output/matching_results.tsv` |
| 10. Official check | `python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test` | must print PASS |

Diagnostics: `python src\normalization.py --demo | --eval`, `python src\diagnose_blocking.py --tag v001`.

## Method in one paragraph

Records are normalized (Unicode/accents, legal suffixes, street and state abbreviations, house numbers).
Candidates are generated per (country, state) block by cosine similarity of IDF-weighted tokens
(name words, address words, numbers), keeping the top-k per Source 1 record. A LightGBM classifier
scores each pair from string-similarity, address-agreement and blocking features. Matches are
decided by a probability threshold, a singleton gate and a one-Source-1-per-candidate rule, all tuned
for macro F0.5 on out-of-fold predictions. Details: `Documentation_template.md`.

## Validation discipline

- 20% of train Source 1 IDs form a frozen holdout (stratified by country × match count).
- Model training and threshold tuning use the train fold only; the holdout is scored once per version.
- Every artifact carries a version tag (`v001`, ...). See `experiments/EXPERIMENTS.md`.

## Repo layout

```
src/            pipeline code
experiments/    splits, per-version metrics, EXPERIMENTS.md (cache/ is git-ignored)
models/         trained models (git-ignored)
output/         submission TSVs (git-ignored)
utils/          organizer-provided validator
```

## Licences

LightGBM (MIT), rapidfuzz (MIT), sparse_dot_topn (Apache 2.0), scikit-learn/pandas/numpy/scipy (BSD).
No pretrained models are used.