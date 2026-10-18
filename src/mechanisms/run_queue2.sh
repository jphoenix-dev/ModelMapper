#!/usr/bin/env bash
# Follow-ups after run_queue.sh: re-test the positive H1 result on fresh concepts (seed 1, 20 concepts), and the
# exploratory H3 check with 200 neurons zeroed. Waits for run_queue.sh to finish (one model job at a time).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" MECH_THREADS=6
PY=C:/dev/already-clean/.venv/Scripts/python.exe
OUT=D:/already-clean-data/mechanisms
until grep -q "queue: done" $OUT/queue.log; do sleep 20; done
echo "queue2: repeat K=200 $(date)"; REPEAT_K=200 $PY -X utf8 -u src/mechanisms/repeat_neurons.py > $OUT/repeat_k200_run.log 2>&1; echo "repeat_k200 exit $?"
echo "queue2: inhibitors seed 1 $(date)"; $PY -X utf8 -u src/mechanisms/inhibitors.py 20 1 > $OUT/inhibitors_seed1_run.log 2>&1; echo "inhibitors_seed1 exit $?"
echo "queue2: done $(date)"
