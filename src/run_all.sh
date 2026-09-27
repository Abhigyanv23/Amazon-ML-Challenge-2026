#!/bin/bash
# End-to-end reproduction of the final submission (v009): data -> normalization -> blocking ->
# stage-1 matcher -> cross-encoders (GPT-2, XLM-R) -> stage-2 re-scorer -> output/*.tsv -> validation.
# Run from the package root (the folder that contains src/), with dataset/ laid out as in README.md.
# Flags are exactly those recorded in experiments/v005w, v005ws1, g001, x001 and v009.
set -euo pipefail
export PYTHONPATH=src PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
BLK="--k 30 --k-tri 10 --k-nl 5 --k-sk 10 --k-ad 5 --k-rev 5 --tag v005w"   # blocking v005w
S1="--tag v005ws1 --cand-tag v005w --k 30 --folds 5"                           # stage 1 v005ws1

mkdir -p output experiments/cache models

# 1. Frozen holdout split (seed 42; skipped when experiments/splits/ already exists)
[ -f experiments/splits/holdout_s1_ids.txt ] || $PY src/make_split.py

# 2. Normalization caches (experiments/cache/*_source{1,2,3}.parquet)
$PY src/normalization.py --split train
$PY src/normalization.py --split test

# 3. Blocking: holdout (for measurement), 300K train-fold S1 (for training), test
$PY src/blocking.py --split train --s1-set holdout $BLK
$PY src/blocking.py --split train --s1-set train --sample 300000 $BLK
$PY src/blocking.py --split test $BLK

# 4. Stage 1: pairwise LightGBM (saves stage-1 probabilities p1_* for stage 2)
$PY src/matcher.py train $S1
$PY src/matcher.py holdout $S1
$PY src/matcher.py test $S1

# 5. Cross-encoders on uncertain pairs (stage-1 p in [0.02, 0.98], top 8 per S1); needs a CUDA GPU
$PY src/llm_rescore.py train --stage1 v005ws1 --tag g001 --bs 64 --max-train 0 --epochs 2
$PY src/llm_rescore.py holdout --tag g001
$PY src/llm_rescore.py test --tag g001
$PY src/llm_rescore.py train --stage1 v005ws1 --tag x001 --model xlm-roberta-base \
    --bs 64 --max-train 0 --epochs 2 --lr 2e-5 --bf16
$PY src/llm_rescore.py holdout --tag x001
$PY src/llm_rescore.py test --tag x001

# 6. Stage 2: group-consistency + cross-encoder features; writes output/matching_results.tsv, candidate_pairs.tsv
$PY src/stage2.py train --stage1 v005ws1 --tag v009 --llm g001,x001
$PY src/stage2.py holdout --stage1 v005ws1 --tag v009
$PY src/stage2.py test --stage1 v005ws1 --tag v009

# 7. Format checks
$PY src/check_submission.py
if [ -f utils/validate_submission.py ]; then
  $PY utils/validate_submission.py --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
fi
