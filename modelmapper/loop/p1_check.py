"""Run 1 self-check, P1-style (GPU; no API): do a concept's top-5 neurons (found on Luna's recall questions) also break
recall on INDEPENDENT prompts - cloze sentences from its Wikipedia intro with the name blanked (concept_atlas.cloze)?
Sample: 200 mapped run-1 concepts with >= 2 cloze prompts that Qwen recalls (seed 91; named and common both).
Per concept, one pass per group (rows_logodds: as-is copy in the same pass): top-5 zeroed at every position vs 5
random neurons from the same layers. Criterion (P1, pre-registered for the earlier maps): top-5 median drop clearly
above random (P1: +5.59 vs -0.003); flips = share of recalled prompts whose answer P falls below 0.5.
Usage: python src/loop_v2/p1_check.py <run_dir> -> <run_dir>/p1_check.json, p1_check.txt
"""
import glob
import json
import os
import random
import statistics as stx
import sys
from pathlib import Path

from modelmapper.core import concept_pipeline as cp  # noqa: E402
from modelmapper.experiments.concept_atlas import cloze, rows_logodds  # noqa: E402
from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import load  # noqa: E402
from modelmapper.core.settings import DATA_DIR

run = Path(sys.argv[1])


def main():
    F = {x["concept"]: x for x in json.loads((DATA_DIR / "loop_v2/frame/frame.json").read_text(encoding="utf-8"))["draw"]}
    V = json.loads((DATA_DIR / "loop_v2/frame/vital.json").read_text(encoding="utf-8"))
    recs = {}
    for b in glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line); recs[r["concept"]] = r
    ok = [c for c, r in recs.items() if r["row"]["status"] == "ok" and len(cloze(c, V["intro"].get(c, ""))[1]) >= 2]
    rng = random.Random(91); rng.shuffle(ok)
    model, tok = load(DEFAULT_MODEL); W = model.config.intermediate_size
    out, n_try = [], 0
    for c in ok:
        if len(out) >= 200:
            break
        n_try += 1; r = recs[c]; name, ps = cloze(c, V["intro"][c])
        ids = r["rank"]["answer_ids"]
        chats = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)) for p in ps]
        base = rows_logodds(model, tok, [[(ch, ids, None) for ch in chats]])[0]
        rec_i = [i for i, x in enumerate(base) if 1 / (1 + 2.718281828 ** (-x)) >= cp.MIN_P]
        if len(rec_i) < 2:
            continue
        top5 = [(int(u[1:].split(":")[0]), int(u.split(":")[1])) for u in [e["neuron"] for e in r["rank"]["top"][:5]]]
        rnd5 = [(L, rng.randrange(W)) for L, _ in top5]
        # all 5 units zeroed together in row 1 of a 2-row pass; row 0 is the as-is copy (same-pass baseline)
        import torch
        from modelmapper.core.neurons import mlps, DEVICE
        def multi(units):
            hks = []
            by = {}
            for L, n in units:
                by.setdefault(L, []).append(n)
            for L, ns in by.items():
                idx = torch.tensor(ns, device=DEVICE)
                def h(mod, a, idx=idx):
                    y = a[0].clone(); y[1:, :, idx] = 0.0; return (y,)       # row 0 stays as is (same-pass baseline)
                hks.append(mlps(model)[L].down_proj.register_forward_pre_hook(h))
            return hks
        drops = {}
        for tag, units in (("top5", top5), ("random5", rnd5)):
            vals = []
            for i in rec_i:
                g = [(chats[i], ids, None), (chats[i], ids, None)]           # [as is, zeroed] in one pass
                hks = multi(units)
                try:
                    v = rows_logodds(model, tok, [g])[0]
                finally:
                    for h in hks:
                        h.remove()
                vals.append((v[0], v[1]))
            drops[tag] = vals
        d5 = [a - b for a, b in drops["top5"]]; dr = [a - b for a, b in drops["random5"]]
        flips = sum(1 / (1 + 2.718281828 ** (-b)) < 0.5 for a, b in drops["top5"] if a > 0) / max(1, sum(a > 0 for a, _ in drops["top5"]))
        out.append({"concept": c, "kind": F[c].get("kind"), "section": F[c]["section"], "n_prompts": len(rec_i),
                    "drop_top5": stx.mean(d5), "drop_random5": stx.mean(dr), "flip_share": flips})
    md = stx.median
    L = [f"P1-style check on run 1: {len(out)} concepts tested ({n_try} tried; needs >= 2 recalled cloze prompts)",
         f"  top-5 zeroed: median drop {md(o['drop_top5'] for o in out):+.2f} log-odds; random-5: {md(o['drop_random5'] for o in out):+.3f}",
         f"  concepts where top-5 flips >= half the recalled cloze prompts: {sum(o['flip_share'] >= 0.5 for o in out)}/{len(out)}",
         "  (P1 on the earlier maps: +5.59 vs -0.003; flips in 66%)"]
    for kd in ("named", "common"):
        rs = [o for o in out if o["kind"] == kd]
        if rs:
            L.append(f"  {kd:<7} n={len(rs)}: median drop {md(o['drop_top5'] for o in rs):+.2f} vs random {md(o['drop_random5'] for o in rs):+.3f}; "
                     f"flips >= half in {sum(o['flip_share'] >= 0.5 for o in rs)}/{len(rs)}")
    txt = "\n".join(L)
    (run / "p1_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (run / "p1_check.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
