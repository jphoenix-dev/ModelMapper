"""Regenerate the fixed-default (R1, 4 neurons, 1x, add, reply-only) held-out replies for the 59 concepts so the
auto-vs-default comparison can be judged by Jev on BOTH sides (auto_steer_scale.py saved only auto's replies).
Writes results/default_heldout.json {tag: [replies]}."""
import json
import os
from pathlib import Path

import torch

os.environ["RECALL_VARIANT"] = "V3"
os.environ["ANSWER_DISCOVERY"] = "0"
import concept_pipeline as cp  # noqa: E402
import pipeline_fast  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HELD = json.loads((ROOT / "prompts/neutral_heldout.json").read_text(encoding="utf-8"))
auto = json.loads((ROOT / "results/auto_steer_scale.json").read_text(encoding="utf-8"))
pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
out = {}
for tag in auto:
    d, concept = tag.split("/", 1)
    cp.OUT = ROOT / "results" / d; pipeline_fast.OUT = cp.OUT
    spec, tid, recall, _ = pipe.make_prompts(concept)
    A = pipe.attribute([q for q, _ in recall], tid)
    units = [u for u in A["top"][:200] if u[0] >= pipe.L // 2][:4]
    peaks = torch.tensor([pipe.steer_edits([u], 1.0, A["last_act"])[0][2] for u in units], device="cuda")
    out[tag] = pipe.gen_rows(HELD, units, torch.stack([peaks] * len(HELD)),
                             torch.ones(len(HELD), len(units), dtype=torch.bool, device="cuda"))
    print(tag, flush=True)
(ROOT / "results/default_heldout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
