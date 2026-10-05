#!/bin/bash
# v010 stage-2 experiments on cached base features (holdout only; test once a winner is known).
#   v010a: --llm g001,x001w               (XLM-R scores on the wider band 0.005-0.995)
#   v010b: --llm g001,x001w --group-llm   (+ per-S1 cross-encoder group features)
cd ~/awsml26
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=/data/venv/bin/python; L=/data/logs
until grep -q "^exit" $L/x001w_a.log 2>/dev/null; do sleep 20; done
grep -q "^exit 0" $L/x001w_a.log || { echo "x001w scoring failed"; exit 1; }
until [ -f /data/cache/s2base_train_holdout_v005ws1/part0029.parquet ] && ! pgrep -f "stage2.py cache" >/dev/null; do sleep 20; done
for v in a b; do
  extra=""; [ $v = b ] && extra="--group-llm"
  $PY src/stage2.py train --stage1 v005ws1 --tag v010$v --llm g001,x001w $extra --jobs 6 > $L/v010${v}_train.log 2>&1 || { echo "FAILED v010$v train"; exit 1; }
  grep -- "-> using" $L/v010${v}_train.log
  $PY src/stage2.py holdout --stage1 v005ws1 --tag v010$v --jobs 6 > $L/v010${v}_holdout.log 2>&1 || { echo "FAILED v010$v holdout"; exit 1; }
  echo "v010$v holdout: $(grep F0.5_macro $L/v010${v}_holdout.log)   (v009 0.9857)"
done
echo "v010 stage-2 experiments DONE $(date -u +%H:%M)"
