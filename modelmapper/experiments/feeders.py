"""Do shared neurons have the same EARLIER feeders in each concept that uses them? (GPU; no API)

James (18 Oct): a neuron shared by several concepts - same reason to fire each time (same upstream neurons), or
different inputs per concept (polysemantic on the input side)?
Sample (seed 111): 80 'topic-sized' shared neurons (top-10 of 3-10 maps, layer >= 24) and 20 widely shared ones
(>= 100 maps; 6 random users each). For each (neuron N, user concept c): N's activation at the answer position (last
prompt token), mean over c's recall prompts; attribution of every earlier-layer MLP neuron = activation x gradient of
N's activation, summed over positions (earlier neurons at any word can reach N through attention); its 20 strongest
feeders (|attribution|).
Compare (Jaccard of top-20 feeder sets):
  same N, two of its users                       -> shared reason?
  same concept, two different shared neurons      -> how much is just 'this concept's general feeders'
  different N, different concept (random pairs)  -> chance
Prediction (stated before running): topic neurons share feeders across their users well above chance (and above the
same-concept baseline); widely shared machinery neurons share fewer, with feeders varying by concept.
Output: <run>/feeders.jsonl (every (N, c): top-50 feeders with attribution), feeders.txt.  Usage: python src/loop_v2/feeders.py <run>
"""
import json
import random
import statistics as stx
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import torch

from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import DEVICE, load, mlps  # noqa: E402

run = Path(sys.argv[1])


class Stop(Exception):
    pass


def main():
    S = json.loads((run / "shared_neurons.json").read_text(encoding="utf-8"))
    import glob
    recs = {}
    for b in glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["row"]["status"] == "ok":
                recs[r["concept"]] = r["rank"]["prompts"]
    rng = random.Random(111)
    topic = sorted(u for u, o in S.items() if o["n_users"] <= 10 and o["layer"] >= 24)
    wide = sorted(u for u, o in S.items() if o["n_users"] >= 100)
    pick = [(u, [x["concept"] for x in S[u]["users"]]) for u in rng.sample(topic, min(80, len(topic)))]
    pick += [(u, rng.sample([x["concept"] for x in S[u]["users"]], 6)) for u in rng.sample(wide, min(20, len(wide)))]
    model, tok = load(DEFAULT_MODEL)
    for p in model.parameters():
        p.requires_grad_(False)
    out_p = run / "feeders.jsonl"; f = open(out_p, "w", encoding="utf-8")
    feeders = {}
    for k, (u, users) in enumerate(pick):
        L, n = int(u[1:].split(":")[0]), int(u.split(":")[1])
        for c in users:
            prompts = recs[c]
            enc = [tok.encode(p) for p in prompts]; T = max(map(len, enc)); pad = tok.pad_token_id or 0
            x = torch.full((len(enc), T), pad, device=DEVICE); mask = torch.zeros_like(x)
            for i, e in enumerate(enc):
                x[i, :len(e)] = torch.tensor(e, device=DEVICE); mask[i, :len(e)] = 1
            last = torch.tensor([len(e) - 1 for e in enc], device=DEVICE)
            acts = {}

            def keep(l):
                def h(mod, a):
                    a0 = a[0]
                    if not a0.requires_grad:
                        a0 = a0.detach().requires_grad_(True); acts[l] = a0; return (a0,)
                    a0.retain_grad(); acts[l] = a0
                return h

            def target(mod, a):
                acts["target"] = a[0][torch.arange(len(enc), device=DEVICE), last, n]; raise Stop
            hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]
            hks.append(mlps(model)[L].down_proj.register_forward_pre_hook(target))
            try:
                with torch.enable_grad():
                    try:
                        model.model(input_ids=x, attention_mask=mask)
                    except Stop:
                        pass
                    acts["target"].float().mean().backward()
            finally:
                for h in hks:
                    h.remove()
            m3 = mask[..., None].float()
            att = torch.stack([(acts[l] * acts[l].grad * m3).float().sum((0, 1)) for l in range(L)])     # [L, W]
            flat = att.flatten(); top = flat.abs().topk(50)
            fl = [(f"L{int(i) // att.shape[1]}:{int(i) % att.shape[1]}", round(float(flat[i]), 5)) for i in top.indices]
            feeders[(u, c)] = [a for a, _ in fl[:20]]
            f.write(json.dumps({"neuron": u, "concept": c, "target_act_mean": float(acts["target"].float().mean()), "feeders_top50": fl}, ensure_ascii=False) + "\n")
            f.flush()
        if k % 10 == 0:
            print(f"{k + 1}/{len(pick)} neurons", flush=True)
    jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
    groups = {"topic": [u for u, _ in pick if u in topic], "wide": [u for u, _ in pick if u in wide]}
    same_n = {g: [stx.mean(jac(feeders[(u, a)], feeders[(u, b)]) for a, b in combinations([c for (uu, c) in feeders if uu == u], 2))
                  for u in us] for g, us in groups.items()}
    by_c = defaultdict(list)
    for (u, c) in feeders:
        by_c[c].append(u)
    same_c = [jac(feeders[(a, c)], feeders[(b, c)]) for c, us in by_c.items() for a, b in combinations(us, 2)]
    keys = list(feeders); chance = []
    while len(chance) < 2000:
        a, b = rng.sample(keys, 2)
        if a[0] != b[0] and a[1] != b[1]:
            chance.append(jac(feeders[a], feeders[b]))
    md = stx.median
    L_ = [f"Feeder overlap (Jaccard of top-20 earlier-layer feeders), {len(feeders)} (neuron, concept) pairs:",
          f"  same neuron, two of its users - topic neurons (n={len(same_n['topic'])}): median {md(same_n['topic']):.3f}, mean {stx.mean(same_n['topic']):.3f}",
          f"  same neuron, two of its users - widely shared (n={len(same_n['wide'])}): median {md(same_n['wide']):.3f}, mean {stx.mean(same_n['wide']):.3f}",
          f"  same concept, two different shared neurons (n={len(same_c)}): median {md(same_c) if same_c else float('nan'):.3f}",
          f"  chance (different neuron, different concept; n=2000): median {md(chance):.3f}, mean {stx.mean(chance):.3f}",
          f"  topic neurons above the chance median: {sum(v > md(chance) for v in same_n['topic'])}/{len(same_n['topic'])}"]
    txt = "\n".join(L_)
    (run / "feeders.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()
