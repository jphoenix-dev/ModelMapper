"""Cross-activation: do a concept's top neurons also fire for OTHER concepts? (no API; step 2 of the coding test)

Union of every recalled concept's top-10 measured neurons (rank_save_*.json, all sets). For each concept B of one set
(its own recall prompts), record each union neuron's activation, averaged over B's prompts:
  ans   at the answer position (last prompt token) - where the rankings measured it
  peak  the largest activation in the neuron's OWN direction (sign it takes on its own concept) over every position
        except the first (attention sink), i.e. anywhere in the question or template
Run once per set (CONCEPT_OUT_DIR = that set's folder, so its cached prompts are used); results/cross_activation_<tag>.json
Analysis is in cross_activation_summary.py.
Usage: RECALL_VARIANT=V3 ANSWER_DISCOVERY=0 CONCEPT_OUT_DIR=<dir> python src/cross_activation.py <tag> "<concept>" [...]
"""
import glob
import json
import os
import sys
from pathlib import Path

import torch

import concept_pipeline as cp
from config import DEFAULT_MODEL
from neurons import DEVICE, mlps
from pipeline_fast import FastPipeline

ROOT = Path(__file__).resolve().parents[1]
TOP = 10


def union():
    units, sign = [], {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" not in v:                       # prompt-count mismatch: not ranked (rank_save.py)
                print(f"skipping {c}: {v.get('mismatch')}", flush=True); continue
            for e in v["top"][:TOP]:
                u = e["neuron"]
                if u not in sign:
                    units.append(u); sign[u] = []
                sign[u].append(1.0 if e["act_answer_pos"] >= 0 else -1.0)
    return units, sign


def main():
    tag = sys.argv[1]
    assert os.environ.get("ANSWER_DISCOVERY") == "0", "set ANSWER_DISCOVERY=0 (no API calls)"
    units, signs = union()
    by_layer = {}
    for i, u in enumerate(units):
        l, n = (int(x) for x in u[1:].split(":"))
        by_layer.setdefault(l, []).append((i, n))
    # a neuron that is top-10 for several concepts could take opposite signs; use the majority, and record conflicts
    sg = torch.tensor([1.0 if sum(s) >= 0 else -1.0 for s in (signs[u] for u in units)], device=DEVICE)
    conflicts = sum(1 for u in units if len(set(signs[u])) > 1)
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m = pipe.model
    out_p = ROOT / f"results/cross_activation_{tag}.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"units": units, "sign_conflicts": conflicts, "concepts": {}}
    assert out["units"] == units, "the neuron union changed since this file was started - delete it and rerun"
    for concept in [c.strip() for c in sys.argv[2:]]:
        if concept in out["concepts"]:
            continue
        spec, tid, recall, _ = pipe.make_prompts(concept)
        ans = torch.zeros(len(units), device=DEVICE); peak = torch.zeros(len(units), device=DEVICE)
        for q, _ in recall:
            ids = pipe.tok(q, return_tensors="pt").to(DEVICE).input_ids
            st = {}
            hk = [mlps(m)[l].down_proj.register_forward_pre_hook(lambda mod, a, l=l: st.__setitem__(l, a[0][0].float()))
                  for l in by_layer]
            try:
                with torch.no_grad():
                    m(input_ids=ids)
            finally:
                for h in hk:
                    h.remove()
            for l, items in by_layer.items():
                idx = torch.tensor([i for i, _ in items], device=DEVICE); ns = torch.tensor([n for _, n in items], device=DEVICE)
                a = st[l][:, ns]                                  # [T, k]
                ans[idx] += a[-1] / len(recall)
                peak[idx] += (a[1:] * sg[idx]).max(0).values * sg[idx] / len(recall)
        out["concepts"][concept] = {"ans": [round(v, 3) for v in ans.tolist()], "peak": [round(v, 3) for v in peak.tolist()]}
        print(f"{concept:<30} done ({len(recall)} prompts)", flush=True)
        out_p.write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
