# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source 1 business record, find all matching records in Source 2 and Source 3.
Scored by per-entity F0.5, macro-averaged over Source 1.

Uses **only** the competition-provided data. No external lookups, APIs, geocoding or scraping.

## Setup

Python 3.10+ (Windows PowerShell shown; use `python3` / `/` on Mac/Linux). 12 cores / 16 GB RAM tested.

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

## Pipeline (run from repo root, in order) — current best: v002 (v003 in progress)

| Step | Command | Output | Time |
|---|---|---|---|
| 0. EDA (optional) | `python src\data_inspection.py --data-dir dataset` | report to stdout | ~10 min |
| 1. Holdout split (run once, frozen) | `python src\make_split.py` | `experiments/splits/*_s1_ids.txt` | 1 min |
| 2. Scorer self-test | `python src\validate_local.py --selftest` | F0.5 = 1.0000 / 0.0558 | 1 min |
| 3. Normalize | `python src\normalization.py --split train` then `--split test` | `experiments/cache/*_source{1,2,3}.parquet` | ~5 min each |
| 4. Blocking: holdout (recall) | `python src\blocking.py --split train --s1-set holdout --tag v002` | candidates + `experiments/v002/blocking_holdout.json` | ~16 min |
| 5. Blocking: train sample | `python src\blocking.py --split train --s1-set train --sample 300000 --tag v002` | training candidates | ~16 min |
| 6. Train matcher (stage 1) | `python src\matcher.py train --tag v002` | `models/v002/lgbm.txt`, `experiments/v002/decision.json`, OOF probabilities | ~30 min |
| 7. Score holdout | `python src\matcher.py holdout --tag v002` | `experiments/v002/holdout_metrics.json` | ~8 min |
| 8. Blocking: test | `python src\blocking.py --split test --tag v002` | test candidates | ~35 min |
| 9. Predict test | `python src\matcher.py test --tag v002` | `output/candidate_pairs.tsv`, `output/matching_results.tsv` | ~28 min |
| 10. Local format check | `python src\check_submission.py` | PASS | 2 min |
| 11. Official check | `python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test --check-ids` | must print PASS | 2 min |

Stage 2 (v003, after steps 6, 7 and 9 have saved stage-1 probabilities):

```powershell
python src\stage2.py train   --stage1 v002 --tag v003
python src\stage2.py holdout --stage1 v002 --tag v003
python src\stage2.py test    --stage1 v002 --tag v003   # overwrites output/*.tsv; re-run steps 10-11
```

Stage 2 with transformer cross-encoders (v009, current best; needs `requirements-llm.txt` + CUDA torch).
GPT-2 small (g001) and XLM-RoBERTa-base (x001) are fine-tuned as pair classifiers on uncertain pairs only
(stage-1 p in [0.02, 0.98], top 8 per S1; 2 folds by S1). Their probabilities are stage-2 features.
Commands as run for v009 (AWS g5.2xlarge, A10G 24 GB; lower `--bs` on smaller GPUs):

```bash
python src/llm_rescore.py count   --stage1 v005ws1                  # band sizes
python src/llm_rescore.py train   --stage1 v005ws1 --tag g001 --bs 64 --max-train 0 --epochs 2
python src/llm_rescore.py holdout --tag g001
python src/llm_rescore.py test    --tag g001
python src/llm_rescore.py train   --stage1 v005ws1 --tag x001 --model xlm-roberta-base \
       --bs 64 --max-train 0 --epochs 2 --lr 2e-5 --bf16
python src/llm_rescore.py holdout --tag x001
python src/llm_rescore.py test    --tag x001
python src/stage2.py train   --stage1 v005ws1 --tag v009 --llm g001,x001
python src/stage2.py holdout --stage1 v005ws1 --tag v009
python src/stage2.py test    --stage1 v005ws1 --tag v009    # writes output/*.tsv
```

Timings on the A10G: each cross-encoder trains in ~25-30 min and scores the 2.4M test band pairs in
30-60 min; stage 2 takes ~40 min per split on 8 vCPU.

Blocking defaults: word channel k=20, trigram channel `--k-tri 10`, non-Latin address channel `--k-nl 5`.

Diagnostics (labels used only for analysis, never for training):
`python src\normalization.py --demo` / `--eval`, `python src\diagnose_blocking.py --tag v002`,
`python src\diagnose_country.py --tag v002` (unsupervised, test set).

## Method in one paragraph

Records are normalized (Unicode/accents, legal suffixes, street and state abbreviations, postcodes
separated from house numbers); missing states are filled from an address-component → state map
learned from the same split's Source 1 text. Candidates come from three channels per
(country, state) block — IDF-weighted word tokens, character trigrams of the name, and an
address-only channel for non-Latin names — keeping the top-k of each. A LightGBM classifier scores
each pair from string-similarity, address-agreement and blocking features. Matches are decided by a
probability threshold, a singleton gate and a one-Source-1-per-candidate rule, tuned for macro F0.5
on out-of-fold predictions. Details: `Documentation_template.md`.

## Validation discipline

- 20% of train Source 1 IDs form a frozen holdout (stratified by country × match count).
- Model training and threshold tuning use the train fold only; the holdout is scored once per version.
- Every artifact carries a version tag (`v001`, `v002`, ...). See `experiments/EXPERIMENTS.md`.
- Each submitted version: files copied to `experiments/<tag>/submission/`, commit tagged (`day1-sub1`, ...).

## Repo layout

```
src/            pipeline code
experiments/    splits, per-version metrics, EXPERIMENTS.md (cache/ is git-ignored)
models/         trained models (git-ignored)
output/         submission TSVs (git-ignored)
utils/          organizer-provided validator (unmodified)
```

## Licences

LightGBM (MIT), rapidfuzz (MIT), sparse_dot_topn (Apache 2.0), scikit-learn/pandas/numpy/scipy (BSD),
pyarrow (Apache 2.0). Stage-2 rescoring (v008/v009) uses pretrained checkpoints
GPT-2 small (MIT, 124M) and XLM-RoBERTa-base (MIT, 278M), fine-tuned on the provided training data only, via
transformers (Apache 2.0) and PyTorch (BSD). Weights are downloaded once at setup; no lookups at run time.