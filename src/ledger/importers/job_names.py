"""Convenience job names for neurons (map session; JOB_NAMES.md) -> ledger pack, experiment `job_names`.

One row per neuron Luna was asked about: value_text = the name ('' if none), value = Jev's blind p on the held-out concept,
noise_floor = the control p (another neuron's name, same question), threshold 0.5, passes = confidence high and p >= 0.5
(the rule fixed before the run). concept = the held-out concept. Names are labels for convenience, not the neuron's
true meaning (James, 18 Oct) - every claim says so.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import PackWriter, read_jsonl, thing  # noqa: E402

D = Path(os.environ.get("MAP_JOB_NAMES", "D:/already-clean-data/job_names"))
MAP = Path(os.environ.get("MAP_WORKTREE", "C:/dev/already-clean-map"))
SPEC = MAP / "explainer/gen/map/JOB_NAMES.md"
CONCEPTS = D / "concepts_used.json"          # frozen copy of the map build the 'held' indexes refer to (live order changes)


def main(force=False):
    inputs = [D / "names.cache.jsonl", D / "summary.json", SPEC, CONCEPTS, MAP / "explainer/gen/map/job_names.py"]
    with PackWriter("job_names", inputs=inputs, code=[__file__], force=force) as pw:
        if pw.skip:
            print("job_names: unchanged, skipped"); return
        summ = json.loads((D / "summary.json").read_text(encoding="utf-8"))
        pw.experiment("job_names", title="Convenience job names for neurons (Luna), blind-checked (Jev)",
                      method=SPEC.read_text(encoding="utf-8"), status="exploratory (pass rule fixed before the run)", status_class="exploratory",
                      date="2026-10-18", script="explainer/gen/map/job_names.py (branch research/map-page)",
                      reproduce="python explainer/gen/map/job_names.py --workers 12 --budget 5   (in the map worktree)",
                      data_files=["data:job_names/names.cache.jsonl", "data:job_names/summary.json"],
                      caveats=["A name is a short label to help read the map, not the neuron's true meaning (James, 18 Oct).",
                               "Only neurons used by >= 2 run-1 concepts are eligible (one is held out for the check).",
                               "Word readouts before layer 20 are unreliable (LAYERS_AND_JOBS S1/S2); Luna was told so."],
                      docs=["JOB_NAMES.md (map worktree)"], params={"rule": summ.get("rule"), "summary": summ})
        cj = json.loads(CONCEPTS.read_text(encoding="utf-8")); I = {k: i for i, k in enumerate(cj["cols"])}
        cname = [r[I["name"]] for r in cj["rows"]]
        C = {}
        for r in read_jsonl(D / "names.cache.jsonl"):
            kind, unit = r["k"].split("|", 1); C.setdefault(unit, {})[kind] = r["v"]
        bad = 0
        for unit, v in sorted(C.items()):
            lu = v.get("luna") or {}; jv = v.get("jev") or {}; jc = v.get("jev_ctrl") or {}
            held = cname[lu["held"]] if isinstance(lu.get("held"), int) and lu["held"] < len(cname) else None
            if jv and held not in jv.get("options", []):
                bad += 1; held_ok = False
            else:
                held_ok = True
            p = jv.get("p_held"); pc = jc.get("p_held")
            passes = (lu.get("confidence") == "high" and p is not None and p >= 0.5) if lu.get("name") else False
            pw.evidence(experiment="job_names", unit=unit, concept=thing(pw, held) if held and held_ok else None, evidence_type="job_name",
                        metric="luna_name_blind_check", value=p, value_text=lu.get("name") or "", noise_floor=pc, threshold=0.5, passes=passes,
                        context=f"confidence={lu.get('confidence')}",
                        details={"confidence": lu.get("confidence"), "basis": lu.get("basis"), "held_out": held, "held_index": lu.get("held"),
                                 "held_resolved_ok": held_ok, "options": jv.get("options"), "control_name": jc.get("name"),
                                 "control_options": jc.get("options"), "p_control": pc, "prompt_chars": lu.get("prompt_chars"),
                                 "note": "convenience label, not the neuron's true meaning"},
                        source=f"job_names/names.cache.jsonl#{unit}")
        print(f"job_names: {len(C)} neurons; held-out concept not among the judge's options (index drift): {bad}")
        if bad:
            raise SystemExit("held-out indexes do not match the judge's options: the map's concepts.json changed since the run")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
