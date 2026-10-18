#!/usr/bin/env bash
# Run the remaining MECHANISMS.md tests one after another (this PC's commit charge allows ~2 model jobs at once).
# Waits for the pre-registered entropy run to finish first. Logs and outputs go to D:/already-clean-data/mechanisms/.
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" MECH_THREADS=6
PY=C:/dev/already-clean/.venv/Scripts/python.exe
OUT=D:/already-clean-data/mechanisms
until [ -f $OUT/entropy/causal.json ] || grep -q "Traceback" $OUT/entropy_run.log; do sleep 20; done
echo "queue: language $(date)"; $PY -X utf8 -u -m modelmapper experiment mechanisms.language_neurons > $OUT/language_run.log 2>&1; echo "language exit $?"
echo "queue: repeat $(date)"; $PY -X utf8 -u -m modelmapper experiment mechanisms.repeat_neurons > $OUT/repeat_run.log 2>&1; echo "repeat exit $?"
echo "queue: entropy v2 $(date)"; RULE=v2 $PY -X utf8 -u -m modelmapper experiment mechanisms.entropy_neurons > $OUT/entropy_v2_run.log 2>&1; echo "entropy_v2 exit $?"
echo "queue: rival_push $(date)"; $PY -X utf8 -u -m modelmapper experiment mechanisms.rival_push 400 > $OUT/rival_push_run.log 2>&1; echo "rival_push exit $?"
echo "queue: done $(date)"
