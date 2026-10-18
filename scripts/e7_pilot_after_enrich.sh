#!/usr/bin/env bash
# E7 inhibitors pilot (LOOP_V3.md 'Order' item 7), approved by James 18 Oct: run on the 5090 once the current enrich run
# has finished. Waits until no `enrich.py run` / `modelmapper enrich run` process is left AND the GPU has < 4 GB in use on two checks 2 minutes
# apart (so it never collides with a job someone starts next); gives up after 30 hours. Then: E7 on the 500-concept sample
# (loop_v3/e7_pilot/sample.json, made with ENRICH_ALL=1 ENRICH_SAMPLE_N=500), then the summary (I1-I5).
cd "$(dirname "$0")/.." || exit 1
export PYTHONDONTWRITEBYTECODE=1 MODEL_MAPPER_MODELS=C:/dev/models/hf ENRICH_NAME=e7_pilot ENRICH_STAGES=E7
PY=C:/dev/already-clean/.venv/Scripts/python.exe
OUT=D:/already-clean-data/loop_v3/e7_pilot
busy() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*enrich.py run*' -or \$_.CommandLine -like '*modelmapper enrich run*' }).Count" | tr -d '\r'
}
gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '\r '; }
deadline=$(( $(date +%s) + 30*3600 ))
echo "waiting for the enrich run to finish ($(date))"
while :; do
  if [ "$(date +%s)" -gt "$deadline" ]; then echo "gave up after 30 h ($(date))"; exit 2; fi
  n=$(busy); m=$(gpu_mb)
  if [ "${n:-1}" = "0" ] && [ "${m:-99999}" -lt 4000 ]; then
    sleep 120; n=$(busy); m=$(gpu_mb)
    if [ "${n:-1}" = "0" ] && [ "${m:-99999}" -lt 4000 ]; then break; fi
  fi
  sleep 120
done
echo "GPU free (enrich processes: $n, GPU memory used: $m MiB); starting E7 pilot ($(date))"
$PY -X utf8 -u -m modelmapper enrich run > $OUT/run.log 2>&1; echo "run exit $? ($(date))"
$PY -X utf8 -u -m modelmapper enrich summary > $OUT/summary.log 2>&1; echo "summary exit $? ($(date))"
cat $OUT/summary.txt
