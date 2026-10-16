"""Post-hoc, EXPLORATORY (after CR-3 scoring): why did P5 fail? Unsteered name rate for CR-3 people on the same
held-out person-friendly prompts, plus a strict key (surname only, word boundary) re-score of the saved steered replies."""
import json
import os
import re
from pathlib import Path

import torch

os.environ["RECALL_VARIANT"] = "V3"
import concept_pipeline as cp  # noqa: E402
from pipeline_fast import FastPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRIENDLY = json.loads((ROOT / "prompts/neutral_person_friendly.json").read_text(encoding="utf-8"))[10:]
STRICT = {"Cleopatra": "cleopatra", "Abraham Lincoln": "lincoln", "Mahatma Gandhi": "gandhi", "Martin Luther King Jr.": "luther king"}
d = json.loads((cp.OUT / "wave_free.json").read_text(encoding="utf-8"))
pipe = FastPipeline("Qwen/Qwen3-8B", mode="add", reply_only=True)
units = [(30, 0)]
base = pipe.gen_rows(FRIENDLY, units, torch.zeros(len(FRIENDLY), 1, device="cuda"),
                     torch.zeros(len(FRIENDLY), 1, dtype=torch.bool, device="cuda"))
for c, k in STRICT.items():
    s = d[c]["steer"]; keys = s["keys"]
    hit = lambda r, ks: any(re.search(r"\b" + re.escape(x) + r"\b", r.lower()) for x in ks)  # noqa: E731
    print(f"{c:<24} unsteered name rate (loose keys {keys}): {sum(hit(r, keys) for r in base)/len(base):.2f} | strict '{k}': "
          f"{sum(hit(r, [k]) for r in base)/len(base):.2f} || steered held-out (saved) loose {s['heldout']:.2f}, strict-key "
          f"name rate {sum(hit(r, [k]) for r in s['all_heldout'])/len(s['all_heldout']):.2f}")
print("unsteered sample:", [r[:90] for r in base[:3]])
