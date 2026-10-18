"""CR-3 Jev calibration (PREREG CR-3, P4b): Jev judges auto vs default held-out replies on 10 recalled concepts.

Concepts: 10 drawn at random (seed 3) from CR-3's recalled concepts. Auto replies come from wave_free.json
(all_heldout). Default replies (R1 top-4 late-layer attributed neurons, 4 neurons, 1x, add, reply-only) are
regenerated: greedy decoding makes this identical to what wave_free scored. Jev asks the pipeline's two questions
(lenient on-topic, fluent); joint = both >= 0.5. Writes results/cr3_jev.json.

Usage: CONCEPT_OUT_DIR=results/concepts_cr3 python src/cr3_jev.py
"""
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from modelmapper.core.settings import REPO_ROOT

os.environ["RECALL_VARIANT"] = "V3"
from modelmapper.core import concept_pipeline as cp  # noqa: E402
from modelmapper.core import jev  # noqa: E402
from modelmapper.core.pipeline_fast import FastPipeline  # noqa: E402

ROOT = REPO_ROOT
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))


def judge(concept, r):
    q = {"on": {"type": "noul", "instructions": f"Does this text mention, describe, or seem preoccupied with {concept}?"},
         "fluent": {"type": "noul", "instructions": "Is this text fluent, grammatical English that makes sense?"}}
    a = jev.ask(r or "(empty)", q)
    return a["on"]["noul"] >= 0.5 and a["fluent"]["noul"] >= 0.5


def main():
    d = json.loads((cp.OUT / "wave_free.json").read_text(encoding="utf-8"))
    wave = json.loads((ROOT / "prompts/concepts_cr3.json").read_text(encoding="utf-8"))["concepts"]
    people = set(wave["people"])
    ok = sorted(c for c, v in d.items() if v.get("status") == "ok")
    sample = random.Random(3).sample(ok, min(10, len(ok)))
    pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
    rows = {}
    for c in sample:
        spec, tid, recall, _ = pipe.make_prompts(c)
        A = pipe.attribute([q for q, _ in recall], tid)
        units = [u for u in A["top"][:200] if u[0] >= pipe.L // 2][:4]
        prompts = FRIENDLY[10:] if c in people else HELD
        peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
        default = pipe.gen_rows(prompts, units, torch.stack([peaks] * len(prompts)),
                                torch.ones(len(prompts), len(units), dtype=torch.bool, device="cuda"))
        auto = d[c]["steer"]["all_heldout"]
        with ThreadPoolExecutor(16) as ex:
            ja = list(ex.map(lambda r: judge(c, r), auto)); jd = list(ex.map(lambda r: judge(c, r), default))
        rows[c] = {"auto": sum(ja) / len(ja), "default": sum(jd) / len(jd)}
        print(f"{c:<28} Jev joint: auto {rows[c]['auto']:.2f} vs default {rows[c]['default']:.2f}", flush=True)
    out = {"n": len(rows), "concepts": rows, "auto_mean": sum(r["auto"] for r in rows.values()) / len(rows),
           "default_mean": sum(r["default"] for r in rows.values()) / len(rows)}
    print(f"mean: auto {out['auto_mean']:.3f} vs default {out['default_mean']:.3f} | Jev {jev.usage}, ~${jev.cost():.3f}")
    (ROOT / "results/cr3_jev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
