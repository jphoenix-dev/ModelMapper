"""Loop v2 gate 1: the v2 core reproduces v1 exactly on the 12-concept reproduction set (cached specs, no Luna).

Rules (stated before running): for every concept, status, recall, answer form, k_mean, k_median, k_flip, drop curves
and random curves IDENTICAL to the validated v1 run (results/maploop_repro/map_loop_repro.json); top-30 neurons
identical in order and single drops identical (rank_save_repro.json); head knockout drops identical. Also the schema
check (gate 2): every recalled concept has provenance, null drops, roles for 30 neurons and residuals in its arrays.
Output: D:/already-clean-data/loop_v2/gate_repro/ (records + arrays) and gate_repro.txt. Exit 1 on any failure.
Usage: python src/loop_v2/gate_repro.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import concept_pipeline as cp  # noqa: E402
from config import DEFAULT_MODEL  # noqa: E402
from loop_v2.map_core import map_batch  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "loop_v2" / "gate_repro"


def main():
    rep = ROOT / "results/maploop_repro"
    concepts = json.loads((ROOT / "prompts/_map_loop_repro.json").read_text(encoding="utf-8"))["concepts"]["repro"]
    specs = {c: json.loads((rep / f"{cp.slug(c)}_spec.json").read_text(encoding="utf-8")) for c in concepts}
    v1 = json.loads((rep / "map_loop_repro.json").read_text(encoding="utf-8"))
    v1r = json.loads((rep / "rank_save_repro.json").read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True)
    if (OUT / "records.jsonl").exists():
        (OUT / "records.jsonl").rename(OUT / "records_previous.jsonl")
    recs = list(map_batch(pipe, specs, OUT, code_commit=commit))
    bad, notes = [], []
    for r in recs:
        c, row, rank = r["concept"], r["row"], r["rank"]; o = v1[c]
        for k in ("status", "recall", "answer_form", "k_mean", "k_median", "k_flip", "drops", "random_drops"):
            if row.get(k) != o.get(k):
                bad.append(f"{c}: {k} {o.get(k)} -> {row.get(k)}")
        if row["status"] != "ok":
            continue
        o_r = v1r[c]
        a = [(e["neuron"], e["single_drop"]) for e in rank["top"]]; b = [(e["neuron"], e["single_drop"]) for e in o_r["top"]]
        if a != b:
            bad.append(f"{c}: top-30 differs (first difference at {next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)})")
        if rank["head_knockout_per_prompt"] != o_r.get("head_knockout_per_prompt"):
            bad.append(f"{c}: head knockouts differ")
        v2 = r["v2"]
        if not v2.get("provenance", {}).get("code_commit") or len(v2.get("null_single_drops", {})) != 2 or len(v2.get("roles_top30", [])) != 30:
            bad.append(f"{c}: v2 fields missing")
        arr = np.load(OUT / "arrays" / v2["arrays"])
        if "resid_answer_pos" not in arr.files or arr["resid_answer_pos"].shape[0] != row["recall"]:
            bad.append(f"{c}: residuals missing or wrong shape")
        nd = max(abs(sum(v) / len(v)) for v in v2["null_single_drops"].values())
        notes.append(f"{c}: null |mean drop| max {nd:.3f}; roles {sorted(set(x['role'] for x in v2['roles_top30']))}; "
                     f"time " + " ".join(f"{k}={v}" for k, v in v2["timing_s"].items()))
    msg = f"loop v2 gate 1+2 ({len(recs)} concepts): {'PASS' if not bad else 'FAIL'}\n" + "\n".join(bad + notes)
    (OUT / "gate_repro.txt").write_text(msg + "\n", encoding="utf-8"); print(msg)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
