"""Shuffled-judge control: steer with concept A's best setting, ask Jev whether replies are about concept B.

If Jev says "yes, about B" for B != A more than ~10% of the time, its on-topic judgement is too lenient
(e.g. it says yes to any weird or obsessive text) and nudgeability scores can't be trusted.

Usage: python src/shuffled_judge.py <model> [tag=clamp] [pairs=5]
"""
import json
import random
import sys
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
from modelmapper.core.concept_pipeline import Pipeline  # noqa: E402

model = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "clamp"
n_pairs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
rows = [json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "results/concepts").glob(f"*__{tag}.json")]
rows = [r for r in rows if r["status"] == "ok"]
rng = random.Random(0)
pipe = Pipeline(model, mode=rows[0]["steer_mode"], reply_only=rows[0]["reply_only"])
out = []
for _ in range(n_pairs):
    a, b = rng.sample(rows, 2)
    best = a["steering_best"]
    late = [(t["layer"], t["neuron"]) for t in a["top_neurons"] if t["layer"] >= pipe.L // 2][:best["n"]]
    signs = {(t["layer"], t["neuron"]): t["sign_on_recall"] for t in a["top_neurons"]}
    import torch
    last_act = torch.zeros(pipe.L, pipe.W)
    for (l, n) in late:
        last_act[l, n] = -1.0 if signs[(l, n)] == "neg" else 1.0
    on_a, fl, replies = pipe.run_steer(late, best["strength"], last_act, a["concept"])
    on_b, _ = pipe.judge(replies, b["concept"])
    out.append({"steered_with": a["concept"], "judged_as": b["concept"], "on_A": on_a, "on_B": on_b, "fluent": fl})
    print(f"steer {a['concept']!r:<28} -> about A {on_a:.0%} | about B={b['concept']!r}: {on_b:.0%}")
(ROOT / "results/concepts" / f"shuffled_judge__{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
