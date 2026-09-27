# Amazon ML Challenge 2026 — Business Entity Resolution

For every Source 1 business record, find all matching records in Source 2 and Source 3.
Scored by per-entity F0.5, macro-averaged over Source 1.

Uses **only** the competition-provided data. No external lookups, APIs, geocoding or scraping.
No pretrained models.

## Setup

Python 3.10+ (Windows PowerShell shown; on Linux use `python3` and `/`). Tested on 12 cores / 16 GB RAM
(Windows) and a Linux instance.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas pyarrow numpy scipy scikit-learn rapidfuzz lightgbm sparse_dot_topn catboost
$env:PYTHONIOENCODING="utf-8"
```

(`requirements.txt` pins the exact versions used; pandas must be 3.x.)

Place the data (not committed to git):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## Best confirmed version: v005w + GPT-2 stage 2 (public LB 0.9770)

Built on a parallel GPU instance; the LightGBM-only base (`v005w`, public LB 0.9690) is
`v005` plus a `name_overlap` feature. **The GPT-2 stage-2 pipeline's exact scripts,
checkpoint, and integration method still need to be added to `src/` and documented here and
in `Documentation_template.md` before packaging — see the "Open items" section of
`experiments/EXPERIMENTS.md`.** Until that's done, the reproducible instructions below build
`v005`/`v005w` (LightGBM-only, public LB 0.9683 / 0.9690), which is a safe fallback for the
submission package if the GPT-2 pipeline can't be fully documented/verified in time.

## Reproducible base: v005 (git tag `day2-sub2`, public LB 0.9683)

Run from the repo root at tag `day2-sub2` (`git checkout day2-sub2`), in order:

| Step | Command | Output | Time* |
|---|---|---|---|
| 1. Holdout split (committed, frozen — do not re-run) | `python src\make_split.py` | `experiments/splits/*_s1_ids.txt` | 1 min |
| 2. Scorer self-test | `python src\validate_local.py --selftest` | F0.5 = 1.0000 / 0.0558 | 1 min |
| 3. Normalize | `python src\normalization.py --split train` then `--split test` | `experiments/cache/*_source{1,2,3}.parquet` | ~5 min each |
| 4. Blocking: holdout | `python src\blocking.py --split train --s1-set holdout --tag v005` | candidates + recall report | ~26 min |
| 5. Blocking: train sample | `python src\blocking.py --split train --s1-set train --sample 300000 --tag v005` | training candidates | ~27 min |
| 6. Stage-1 matcher | `python src\matcher.py train --tag v005s1 --cand-tag v005` | model + OOF probabilities | ~50 min |
| 7. Stage-1 holdout | `python src\matcher.py holdout --tag v005s1 --cand-tag v005` | holdout metrics | ~20 min |
| 8. Stage-2 | `python src\stage2.py train --stage1 v005s1 --tag v005` | stage-2 model + decision | ~30 min |
| 9. Stage-2 holdout | `python src\stage2.py holdout --stage1 v005s1 --tag v005` | holdout F0.5 0.9773 | ~5 min |
| 10. Blocking: test | `python src\blocking.py --split test --tag v005` | test candidates (63.5M pairs) | ~55 min |
| 11. Test prediction | `python src\matcher.py test --tag v005s1 --cand-tag v005` then `python src\stage2.py test --stage1 v005s1 --tag v005` | `output/*.tsv` | ~2 h |
| 12. Checks | `python src\check_submission.py` and `python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test --check-ids` | both PASS | ~5 min |

*Times on the 12-core laptop.

### Other versions / tools (on `main`)
- **v006 (pruned candidates, 10.4/S1):** `python src\prune.py train --tag v005 --max-recall-loss 0.008`,
  `python src\prune.py apply --tag v005 --split {train --s1-set holdout | test --s1-set all}`, then
  matcher/stage2 with `--tag v006s1 --cand-tag v005p` (+ `--lr 0.05 --folds 5 --max-rounds 6000 --catboost --cb-gpu`).
- **Final fit on train + holdout:** add `--final-fit` (blocking `--s1-set all`, prune, matcher, stage2).
  Never run `holdout` mode on a final-fit model.
- **Per-country combination of two saved submissions:**
  `python src\combine_by_country.py --base experiments\<A>\submission --override experiments\<B>\submission --countries france`
- **Diagnostics (labels used for analysis only):** `normalization.py --demo/--eval`, `diagnose_blocking.py`,
  `diagnose_country.py`, `diagnose_france.py`, `error_analysis.py`, `adversarial.py`.
- **Feature cache:** `--cache-features` / `--reuse-features` on `matcher.py`.

## Method in one paragraph

Records are normalized (Unicode/accents, legal suffixes, street and state abbreviations, postcodes and
composite house numbers, Indic-script consonant skeletons); missing states are filled from an
address-component → state map learned from the same split's Source 1 text. Candidates come from six
channels per (country, state) block — IDF-weighted word tokens, name character trigrams, cross-script
skeleton, address-only, non-Latin address, and reverse (pool → S1) — top-k each, unioned. A LightGBM
stage-1 matcher scores pairs from string, address, house-number and blocking features; a stage-2
LightGBM rescores each candidate by its agreement with the S1's other confident candidates (with a
fallback to stage 1 for sparse groups). Matches are decided by OOF-tuned thresholds, a singleton gate,
and a one-S1-per-candidate rule. Details: `Documentation_template.md`; history: `experiments/EXPERIMENTS.md`.

## Validation discipline

- 20% of train Source 1 IDs form a frozen holdout (stratified by country × match count).
- Model training and threshold tuning use the train fold only; the holdout is scored once per version.
- Every artifact carries a version tag. Each submitted version: files copied to
  `experiments/<tag>/submission/`, counts re-checked with `check_submission.py --out-dir`, commit tagged.

## Repo layout

```
src/            pipeline code
experiments/    splits, per-version metrics, EXPERIMENTS.md, ROADMAP.md (cache/ is git-ignored)
models/         trained models (git-ignored)
output/         submission TSVs (git-ignored)
utils/          organizer-provided validator (unmodified)
```

## Licences

LightGBM (MIT), CatBoost (Apache 2.0), rapidfuzz (MIT), sparse_dot_topn (Apache 2.0),
scikit-learn/pandas/numpy/scipy (BSD), pyarrow (Apache 2.0). No pretrained models; all models
trained from scratch on the provided training data.