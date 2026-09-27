# Amazon ML Challenge 2026 — Business Entity Resolution (team Sunshine)

For every Source 1 business record, find all matching records in Source 2 and Source 3.
Scored by per-entity F0.5, macro-averaged over Source 1.

**Final submission: v009** — holdout F0.5 **0.9857**, public leaderboard **0.980893**.

Only the competition-provided data is used: no external lookups, APIs, geocoding or scraping.
Two pretrained checkpoints (GPT-2 small and XLM-RoBERTa-base, both MIT) are fine-tuned on the
provided training pairs; their weights are downloaded once at setup.

## Setup

Linux, Python 3.12 (tested 3.12.3). A CUDA GPU is needed for step 5 (cross-encoders); v009 was run
on an NVIDIA A10G (24 GB) with 8 vCPU / 32 GB RAM for steps 5-6. With a smaller GPU lower `--bs`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pinned; PyTorch from the CUDA 12.8 index
export HF_HOME=$PWD/.hf_cache            # optional: where GPT-2 / XLM-R weights are cached
```

Place the data (not included):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## Reproduce the submission end to end

One command, run from this folder: `bash src/run_all.sh`. It runs these steps in order (exact flags):

| Step | Command | Output |
|---|---|---|
| 1. Frozen holdout split | `python src/make_split.py` (seed 42; files included in `experiments/splits/`) | `experiments/splits/*_s1_ids.txt` |
| 2. Normalize | `python src/normalization.py --split train`, then `--split test` | `experiments/cache/*_source{1,2,3}.parquet` |
| 3. Blocking (v005w) | `python src/blocking.py --split train --s1-set holdout $BLK`<br>`python src/blocking.py --split train --s1-set train --sample 300000 $BLK`<br>`python src/blocking.py --split test $BLK`<br>with `BLK="--k 30 --k-tri 10 --k-nl 5 --k-sk 10 --k-ad 5 --k-rev 5 --tag v005w"` | candidate tables; `experiments/v005w/blocking_*.json` |
| 4. Stage 1 (v005ws1) | `python src/matcher.py train\|holdout\|test --tag v005ws1 --cand-tag v005w --k 30 --folds 5` | `models/v005ws1/`, stage-1 probabilities `experiments/cache/p1_*_v005ws1.parquet` |
| 5a. GPT-2 (g001) | `python src/llm_rescore.py train --stage1 v005ws1 --tag g001 --bs 64 --max-train 0 --epochs 2`, then `holdout --tag g001`, `test --tag g001` | `models/g001/`, `experiments/cache/llm_*_g001.parquet` |
| 5b. XLM-R (x001) | `python src/llm_rescore.py train --stage1 v005ws1 --tag x001 --model xlm-roberta-base --bs 64 --max-train 0 --epochs 2 --lr 2e-5 --bf16`, then `holdout --tag x001`, `test --tag x001` | `models/x001/`, `experiments/cache/llm_*_x001.parquet` |
| 6. Stage 2 (v009) | `python src/stage2.py train\|holdout\|test --stage1 v005ws1 --tag v009 --llm g001,x001` | `models/v009/`, **`output/matching_results.tsv`, `output/candidate_pairs.tsv`** |
| 7. Checks | `python src/check_submission.py`; `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` | must print PASS |

Run all commands from this folder with `PYTHONPATH=src` (set by `run_all.sh`). The `holdout` modes
score the frozen holdout and write `experiments/<tag>/holdout_metrics.json`; they are not needed for the
test output but reproduce the reported numbers. Measured times on the A10G machine: each cross-encoder
trains in 25-30 min (2 folds x 2 epochs) and scores the 2.4M uncertain test pairs in 30-60 min; stage 2
takes about 40 min per split on 8 vCPU.

## Method in one paragraph

Records are normalized (Unicode/accents, legal suffixes, street and state abbreviations, postcodes and
composite house numbers, an Indic-to-Latin consonant skeleton); missing states are filled from an
address-component -> state map learned from the same split's Source 1 text. Six unsupervised blocking
channels per (country, state) block — IDF-weighted words, name trigrams, skeleton trigrams, address-only,
non-Latin address, and a reverse channel (each S2/S3 record's best S1s) — give 68 candidates per S1 at
99.1% candidate recall. Stage 1 is a LightGBM pair classifier on 71 string/address/blocking features.
For the uncertain pairs (stage-1 p in [0.02, 0.98], top 8 per S1) two cross-encoders — GPT-2 small and
XLM-RoBERTa-base — are fine-tuned on the raw "name + address" text of both records. Stage 2 is a second
LightGBM combining stage-1 probability, group-consistency features (agreement with the S1's other
confident candidates) and the two cross-encoder probabilities. Matches are decided by a probability
threshold, a singleton gate and a one-Source-1-per-candidate rule, all tuned for macro F0.5 on
out-of-fold predictions. Details: `Documentation_template.md`.

## Validation discipline

- 20% of train Source 1 IDs form a frozen holdout (stratified by country x match count, seed 42).
- All models (stage 1, cross-encoders, stage 2) train on train-fold S1 only, with group-by-S1
  out-of-fold predictions feeding the next stage; thresholds are tuned on out-of-fold predictions.
  The holdout is scored once per version and never used for training or tuning.
- Every artifact carries a version tag; the full log is `experiments/EXPERIMENTS.md`.

## Layout

```
src/            pipeline code (run_all.sh = end-to-end)
experiments/    frozen split, per-version configs and metrics, EXPERIMENTS.md (cache/ is generated)
models/         trained models (generated)
output/         submission TSVs (generated)
utils/          organizer-provided validator (unmodified)
```

## Licences

Models: GPT-2 small (MIT, 124M parameters), XLM-RoBERTa-base (MIT, 278M), LightGBM (MIT). All within the
rule "MIT/Apache 2.0 licence, up to 8 billion parameters". Libraries: transformers, tokenizers,
safetensors, pyarrow (Apache 2.0); PyTorch, scikit-learn, pandas, numpy, scipy (BSD); rapidfuzz (MIT);
sparse_dot_topn (Apache 2.0); CatBoost (Apache 2.0, optional, unused in v009).
