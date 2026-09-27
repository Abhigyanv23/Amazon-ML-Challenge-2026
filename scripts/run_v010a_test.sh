#!/bin/bash
cd ~/awsml26
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=/data/venv/bin/python; L=/data/logs
echo "start $(date -u +%H:%M:%S)"
$PY src/stage2.py test --stage1 v005ws1 --tag v010a --p1-min 0.001 --jobs 6 > $L/v010a_test.log 2>&1 || { echo "FAILED test"; exit 1; }
echo "test done $(date -u +%H:%M:%S)"; grep -A5 "^TEST" $L/v010a_test.log
$PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids > $L/v010a_check.log 2>&1
tail -n 2 $L/v010a_check.log
mkdir -p /data/submissions/v010a && cp output/matching_results.tsv /data/submissions/v010a/ && echo "READY /data/submissions/v010a/matching_results.tsv $(date -u +%H:%M:%S)"
