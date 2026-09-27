#!/bin/bash
# v009 = stage 2 with GPT-2 (g001) + XLM-R (x001) scores. GPU chain and CPU chain run in parallel.
# Assumes the /data layout from the AWS setup (venv /data/venv, logs /data/logs) and v005ws1 stage-1 files in experiments/cache.
cd ~/awsml26
export PYTHONPATH=src HF_HOME=/data/hf TMPDIR=/data/tmp PYTHONUNBUFFERED=1
PY=/data/venv/bin/python
L=/data/logs
wait_done() { until grep -q "^done in" "$1" 2>/dev/null; do grep -q Traceback "$1" 2>/dev/null && { echo "FAILED: $1"; exit 1; }; sleep 30; done; }

gpu() {
  $PY src/llm_rescore.py train --stage1 v005ws1 --tag x001 --model xlm-roberta-base \
      --bs 64 --max-train 0 --epochs 2 --lr 2e-5 --bf16 > $L/x001_train.log 2>&1 || exit 1
  $PY src/llm_rescore.py holdout --tag x001 > $L/x001_holdout.log 2>&1 || exit 1
  $PY src/llm_rescore.py test --tag x001 > $L/x001_test.log 2>&1 || exit 1
}

cpu() {
  wait_done $L/x001_train.log
  $PY src/stage2.py train --stage1 v005ws1 --tag v009 --llm g001,x001 --jobs 6 > $L/v009_train.log 2>&1 || exit 1
  wait_done $L/x001_holdout.log
  $PY src/stage2.py holdout --stage1 v005ws1 --tag v009 --jobs 6 > $L/v009_holdout.log 2>&1 || exit 1
  F=$($PY -c "import json;print(json.load(open('experiments/v009/holdout_metrics.json'))['F0.5_macro'])")
  echo "v009 holdout F0.5 = $F (v008 0.9835)"
  if ! $PY -c "import sys; sys.exit(0 if $F > 0.9835 else 1)"; then echo "NOT BETTER - keeping v008 outputs"; exit 0; fi
  wait_done $L/x001_test.log
  $PY src/stage2.py test --stage1 v005ws1 --tag v009 --jobs 6 > $L/v009_test.log 2>&1 || exit 1
  $PY src/check_submission.py > $L/v009_check.log 2>&1
  $PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test --check-ids >> $L/v009_check.log 2>&1
  tail -n 3 $L/v009_check.log
  mkdir -p /data/submissions/v009 && cp output/*.tsv /data/submissions/v009/
  aws s3 cp /data/submissions/v009/ s3://sunshine-aws-ml-26/v009/submission/ --recursive --only-show-errors
  aws s3 cp /data/models/x001/ s3://sunshine-aws-ml-26/v009/models/x001/ --recursive --only-show-errors
  aws s3 cp /data/models/v009/ s3://sunshine-aws-ml-26/v009/models/v009/ --recursive --only-show-errors
  echo "v009 DONE"
}

gpu > $L/v009_gpu_chain.log 2>&1 &
cpu 2>&1 | tee $L/v009_cpu_chain.log
wait
echo "ALL FINISHED $(date -u +%H:%M)"
