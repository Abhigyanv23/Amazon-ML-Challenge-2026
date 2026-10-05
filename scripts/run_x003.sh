#!/bin/bash
# x003 = xlm-roberta-large cross-encoder on the v005ws1 band (0.02-0.98, top 8): train (2-fold OOF) -> holdout -> test.
cd ~/awsml26
export PYTHONPATH=src HF_HOME=/data/hf TMPDIR=/data/tmp PYTHONUNBUFFERED=1
PY=/data/venv/bin/python; L=/data/logs
$PY src/llm_rescore.py train --stage1 v005ws1 --tag x003 --model xlm-roberta-large \
    --bs 32 --max-train 0 --epochs 2 --lr 1e-5 --bf16 > $L/x003_train.log 2>&1 || { echo FAILED train; exit 1; }
$PY src/llm_rescore.py holdout --tag x003 > $L/x003_holdout.log 2>&1 || { echo FAILED holdout; exit 1; }
$PY src/llm_rescore.py test --tag x003 > $L/x003_test.log 2>&1 || { echo FAILED test; exit 1; }
echo "x003 DONE $(date -u +%H:%M)"
