#!/bin/bash
# Fast path to a v010a submission: test with --p1-min 0.001 as soon as v010a is trained.
cd ~/awsml26
export PYTHONPATH=src PYTHONUNBUFFERED=1
PY=/data/venv/bin/python; L=/data/logs
# guard: stop the queued full test-cache build and v010b (not needed now); regex [.] avoids matching this script
( for i in $(seq 360); do
    pkill -f "stage2[.]py cache --stage1 v005ws1 --splits test" && echo "stopped full test cache build"
    if grep -q "v010a holdout" $L/v010_s2.log 2>/dev/null; then
      pkill -f "run_v010[_]s2.sh"; pkill -f "tag v010[b]" && echo "stopped v010b"; break; fi
    sleep 10; done ) &
until grep -q "saved models/v010a" $L/v010a_train.log 2>/dev/null; do grep -q Traceback $L/v010a_train.log && exit 1; sleep 10; done
until grep -q "^exit" $L/x001w_test.log 2>/dev/null; do sleep 10; done
grep -q "^exit 0" $L/x001w_test.log || { echo "x001w test scoring failed"; exit 1; }
echo "v010a trained: $(grep -- '-> using' $L/v010a_train.log)  $(date -u +%H:%M)"
$PY src/stage2.py holdout --stage1 v005ws1 --tag v010a --p1-min 0.001 --jobs 2 > $L/v010a_holdout_sub.log 2>&1 &
$PY src/stage2.py test --stage1 v005ws1 --tag v010a --p1-min 0.001 --jobs 6 > $L/v010a_test.log 2>&1 || { echo "FAILED test"; exit 1; }
echo "test done $(date -u +%H:%M)"; tail -n 6 $L/v010a_test.log
$PY src/check_submission.py > $L/v010a_check.log 2>&1
$PY utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids >> $L/v010a_check.log 2>&1
grep -E "PASS|FAIL" $L/v010a_check.log
mkdir -p /data/submissions/v010a && cp output/*.tsv /data/submissions/v010a/ && echo "copied to /data/submissions/v010a $(date -u +%H:%M)"
wait
echo "holdout (subset mode): $(grep F0.5_macro $L/v010a_holdout_sub.log)"
