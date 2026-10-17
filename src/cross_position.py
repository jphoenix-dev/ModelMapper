"""Cross-position path: do a concept's upstream key neurons, firing at EARLIER words, feed its 'say it' neuron at the
answer word through specific attention heads? (GPU; no API)

James (17 Oct), after indirect_path.py showed the key neurons form a feed-forward assembly at the answer position.
Pairs: in each concept with a 'say' neuron S (types from indirect_path.jsonl), every other top-5 neuron U in an earlier
layer. For every recall prompt (answer position = last prompt token, P = prompt length):
  1. one batch, identical rows, stopped after the last layer needed:
       row 0 as is | per U: 'early' = U zeroed at positions 0..P-2 (every word BEFORE the answer position);
       'ans' = U zeroed at the answer position only; 'null' = an ACTIVATION-MATCHED null zeroed at 0..P-2 (same layer,
       outside the concept's top-30, max |activation| over 0..P-2 closest to U's)
     captured at the answer position: S's activation; every head's output (o_proj input, per head) in layers <= L_S;
     every MLP's output in layers < L_S
  2. attribution (linear, 'attribution patching'): g = d act_S / d (residual entering layer L_S's MLP norm), from a
     separate batch-1 pass with gradients. Each head's change in output at the answer position (row 'early' - row 0),
     mapped through its slice of o_proj, dotted with g = its share of S's change; likewise each MLP. Their sum is
     checked against S's actual change.
  3. validation by real patching: into the as-is run, copy the 'early' run's answer-position outputs of (a) the top-3
     heads by attribution, (b) 3 random heads from layers L_U+1..L_S; recovered = how much of S's 'early' drop each
     reproduces.
Gate: heads in layers <= L_U must not change at all (U's earlier-word write cannot reach them).
Predictions (stated before running; exploratory):
  (c1) the earlier-words part is real: S's relative drop with U zeroed at earlier words >= 3x the null's (median over
       pairs), sign test p < 0.05.
  (c2) it runs through few heads: patching the top-3 heads recovers >= 50% of that drop (median over pairs with a drop
       >= 5%), vs <= 10% for 3 random heads.
  (c3) exploratory: some heads recur as carriers across concepts (reported with counts vs a uniform expectation).
Method change after the 3-concept trial (before the full run): S's activation is bf16 (resolution 0.25 at ~40) and the
earlier-word drops are often that small, and batch-1 vs batch-2 as-is values differed by up to ~0.5. So each U's
'early' row is repeated inside the patching batch (recovery = within one batch), and (c2) is also read from the
attribution, which is not quantised: the top-3 heads' share of the summed head attribution.
Output: results/cross_position.jsonl (per concept, prompt and pair: everything above, all head attributions kept).
Usage: python src/cross_position.py [max_concepts=all]
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

from config import DEFAULT_MODEL
from indirect_path import concepts
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cross_position.jsonl"


class Stop(Exception):
    pass


def lay(u):
    return int(u.split(":")[0][1:])


def nid(u):
    return int(u.split(":")[1])


def main():
    maxc = int(sys.argv[1]) if len(sys.argv) > 1 else None
    C = concepts()
    IP = {}
    for l in (ROOT / "results/indirect_path.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l); IP[r["concept"]] = dict(zip(r["units"], r["types"]))
    work = []
    for c, ty in IP.items():
        say = [u for u, t in ty.items() if t == "say"]
        pairs = [(u, s) for s in say for u in ty if u != s and lay(u) < lay(s)]
        if pairs:
            work.append((c, pairs))
    done = set()
    if OUT.exists():
        done = {json.loads(l)["concept"] for l in OUT.read_text(encoding="utf-8").splitlines()}
    work = [w for w in work if w[0] not in done]
    if maxc:
        work = work[:maxc]
    model, tok = load(DEFAULT_MODEL)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    H = model.config.num_attention_heads; D = model.config.head_dim; W = model.config.intermediate_size
    layers = model.model.layers
    f = open(OUT, "a", encoding="utf-8"); t0 = time.time(); npairs = 0; gate_bad = 0
    print(f"{len(work)} concepts with a 'say' neuron and an earlier key neuron; {sum(len(p) for _, p in work)} pairs", flush=True)
    for ci, (c, pairs) in enumerate(work):
        v = C[c]; top30 = v["top30"]
        Us = sorted({u for u, _ in pairs}, key=lay); Ss = sorted({s for _, s in pairs}, key=lay)
        Lmax = max(lay(s) for s in Ss)
        rng = random.Random(f"xpos|{c}")
        out_prompts = []
        for pi, prompt in enumerate(v["prompts"]):
            ids = torch.tensor(tok.encode(prompt), device=DEVICE)[None]; P = ids.shape[1]
            # base pass: U layers' activations at every position (for the nulls)
            acts = {}
            hks = [mlps(model)[L].down_proj.register_forward_pre_hook(lambda m, a, L=L: acts.__setitem__(L, a[0][0].float().clone()))
                   for L in sorted({lay(u) for u in Us})]
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            finally:
                for h in hks:
                    h.remove()
            nulls = {}
            for u in Us:
                L, n = lay(u), nid(u)
                mx = acts[L][:P - 1].abs().max(0).values
                d = (mx - mx[n]).abs(); d[n] = float("inf")
                for t in top30:
                    if lay(t) == L:
                        d[nid(t)] = float("inf")
                nulls[u] = (L, int(d.argmin()))
            # gradient of each S's activation w.r.t. the residual entering its MLP norm (answer position)
            grads = {}
            for s in Ss:
                Ls, ns = lay(s), nid(s); st = {}

                def pre(m, a):
                    x = a[0].detach().requires_grad_(True); st["x"] = x; return (x,)

                def cap(m, a):
                    st["act"] = a[0][0, -1, ns]; raise Stop
                h1 = layers[Ls].post_attention_layernorm.register_forward_pre_hook(pre)
                h2 = mlps(model)[Ls].down_proj.register_forward_pre_hook(cap)
                try:
                    with torch.enable_grad():
                        try:
                            model(input_ids=ids)
                        except Stop:
                            pass
                        st["act"].float().backward()
                finally:
                    h1.remove(); h2.remove()
                grads[s] = st["x"].grad[0, -1].float().clone()
            # batch 1: as is + per U (early, ans, null)
            rows = [None]
            for u in Us:
                rows += [("early", u), ("ans", u), ("null", u)]
            R = len(rows); z = {}; mo = {}; sa = {}

            def kill_hook(L):
                def h(m, a):
                    x = a[0].clone()
                    for r, spec in enumerate(rows):
                        if spec is None:
                            continue
                        kind, u = spec
                        LL, nn = nulls[u] if kind == "null" else (lay(u), nid(u))
                        if LL != L:
                            continue
                        if kind == "ans":
                            x[r, -1, nn] = 0.0
                        else:
                            x[r, :-1, nn] = 0.0
                    for s in Ss:
                        if lay(s) == L:
                            sa[s] = x[:, -1, nid(s)].float().clone()
                    if L == Lmax:
                        raise Stop
                    return (x,)
                return h
            kl = sorted({lay(u) for u in Us} | {nulls[u][0] for u in Us} | {lay(s) for s in Ss})
            hks = [mlps(model)[L].down_proj.register_forward_pre_hook(kill_hook(L)) for L in kl]
            hks += [layers[L].self_attn.o_proj.register_forward_pre_hook(lambda m, a, L=L: z.__setitem__(L, a[0][:, -1].float().clone()))
                    for L in range(Lmax + 1)]
            hks += [layers[L].mlp.register_forward_hook(lambda m, a, o, L=L: mo.__setitem__(L, o[:, -1].float().clone()))
                    for L in range(Lmax)]
            try:
                with torch.no_grad():
                    try:
                        model(input_ids=ids.repeat(R, 1))
                    except Stop:
                        pass
            finally:
                for h in hks:
                    h.remove()
            rowi = {spec: i for i, spec in enumerate(rows) if spec is not None}
            recs = []; patches = []
            for u, s in pairs:
                Lu, Ls = lay(u), lay(s); g = grads[s]; re_ = rowi[("early", u)]
                s0 = sa[s][0].item(); sg = 1.0 if s0 > 0 else -1.0
                gate = max(float((z[L][re_] - z[L][0]).abs().max()) for L in range(Lu + 1))
                gate_bad += gate > 0
                head_attr = []
                for L in range(Lu + 1, Ls + 1):
                    Wo = layers[L].self_attn.o_proj.weight.float()             # [4096, H*D]
                    dz = (z[L][re_] - z[L][0]).view(H, D)
                    contrib = torch.einsum("hd,ohd->ho", dz, Wo.view(-1, H, D))   # [H, 4096]
                    head_attr.append((contrib @ g).tolist())
                mlp_attr = [float((mo[L][re_] - mo[L][0]) @ g) for L in range(Lu, Ls)]
                drop_c = [(-sg * a, Lu + 1 + i, h) for i, row in enumerate(head_attr) for h, a in enumerate(row)]
                drop_c.sort(reverse=True)
                top3 = [(L, h) for _, L, h in drop_c[:3]]
                pool = [(L, h) for L in range(Lu + 1, Ls + 1) for h in range(H) if (L, h) not in top3]
                rnd3 = rng.sample(pool, 3)
                recs.append({"U": u, "S": s, "s0": s0, "s_early": sa[s][re_].item(), "s_ans": sa[s][rowi[("ans", u)]].item(),
                             "s_null": sa[s][rowi[("null", u)]].item(), "null": f"L{nulls[u][0]}:{nulls[u][1]}",
                             "head_attr": [[round(x, 5) for x in row] for row in head_attr], "mlp_attr": [round(x, 5) for x in mlp_attr],
                             "attr_sum": sum(sum(r) for r in head_attr) + sum(mlp_attr), "top3": top3, "rnd3": rnd3, "gate_maxdz": gate})
                patches += [(len(recs) - 1, "top3", top3, re_), (len(recs) - 1, "rnd3", rnd3, re_)]
            # batch 2: real patching into the as-is run, plus each U's 'early' row again so that every recovery ratio is
            # measured inside one batch (trial: batch-1 vs batch-2 as-is values differ by up to ~0.5, the size of the drops)
            early_rows = {u: 1 + len(patches) + i for i, u in enumerate(Us)}
            R2 = 1 + len(patches) + len(Us); sa2 = {}

            def kill2(L):
                def h(m, a):
                    x = a[0].clone()
                    for u, r in early_rows.items():
                        if lay(u) == L:
                            x[r, :-1, nid(u)] = 0.0
                    for s in Ss:
                        if lay(s) == L:
                            sa2[s] = x[:, -1, nid(s)].float().clone()
                    if L == Lmax:
                        raise Stop
                    return (x,)
                return h

            def patch_hook(L):
                def h(m, a):
                    x = a[0].clone()
                    for r, (_, _, heads, src) in enumerate(patches, start=1):
                        for (LL, hh) in heads:
                            if LL == L:
                                x[r, -1, hh * D:(hh + 1) * D] = z[L][src, hh * D:(hh + 1) * D].to(x.dtype)
                    return (x,)
                return h

            def s_hook(L):
                def h(m, a):
                    for s in Ss:
                        if lay(s) == L:
                            sa2[s] = a[0][:, -1, nid(s)].float().clone()
                    if L == Lmax:
                        raise Stop
                return h
            hks = [layers[L].self_attn.o_proj.register_forward_pre_hook(patch_hook(L)) for L in range(Lmax + 1)]
            hks += [mlps(model)[L].down_proj.register_forward_pre_hook(kill2(L)) for L in sorted({lay(u) for u in Us} | {lay(s) for s in Ss})]
            try:
                with torch.no_grad():
                    try:
                        model(input_ids=ids.repeat(R2, 1))
                    except Stop:
                        pass
            finally:
                for h in hks:
                    h.remove()
            for r, (k, kind, _, _) in enumerate(patches, start=1):
                s = recs[k]["S"]; recs[k][f"s_patch_{kind}"] = sa2[s][r].item(); recs[k]["s0_batch2"] = sa2[s][0].item()
                recs[k]["s_early_batch2"] = sa2[s][early_rows[recs[k]["U"]]].item()
            npairs += len(recs)
            out_prompts.append({"prompt_i": pi, "pairs": recs})
        f.write(json.dumps({"concept": c, "prompts": out_prompts}) + "\n"); f.flush()
        if ci % 20 == 0:
            el = time.time() - t0
            print(f"{ci + 1}/{len(work)} concepts, {npairs} pair-prompts, {el / 60:.1f} min; gate (heads at or before L_U changed) {gate_bad}", flush=True)
    print(f"done: {npairs} pair-prompts; gate failures {gate_bad}")


if __name__ == "__main__":
    main()
