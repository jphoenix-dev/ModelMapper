"""Do numeric-fact neurons code magnitude (a number line), or just 'this digit'? (GPU; no API)

James (18 Oct), follow-up to numeric.py, whose zero-vs-double test could not see graded coding (Qwen is > 0.9999 sure of
87% of digits; doubling only sharpened the same digit). Two fixes:
  (a) a DOSE SERIES that pushes much harder: each neuron's activation multiplied by -4, -2, -1, 0, 0.5, 1 (as-is), 2, 4 at
      every position; the full float64 distribution over the 10 digit tokens recorded at every answer position;
  (b) LESS-CERTAIN numbers: Qwen's own greedy number (as the target) for questions it is unsure about - heights of all 30
      landmarks, year construction began, people's death years, plus the numeric.py candidates it failed - kept where some
      digit has P(top digit) < 0.9 on that prompt.
Neurons: numeric.py facts - top-3 per fact (drop > 1.0, from maps.jsonl); uncertain facts - attribution of the summed digit
log-odds of Qwen's own answer over all positions -> top-100 -> zeroed alone -> top-3. Controls: for every fact, 3 random
neurons from the same layers as its top-3.
Measures, at the digit position the neuron hurts most when zeroed (q*; correct digit c):
  where lost probability goes   the digits gaining probability (vs as-is) at factors 0, -2, -4: share going to c-1 / c+1
                                vs the uniform expectation (#neighbours / 9), and the mass-weighted mean |d - c|
  graded dose response          expected digit E = sum d p(d) across the 8 factors: 'graded' if Spearman |rho| >= 0.9 and
                                range >= 0.3 digits
  Qwen's own uncertainty        uncertain facts as-is: share of the non-top-digit mass on the top digit's neighbours
  unembedding geometry          cosine between the 10 digit tokens' output vectors by |d - d'| (whether output geometry alone
                                would favour neighbours)
Predictions (stated before running):
  A1  lost probability goes preferentially to neighbouring digits (median neighbour share > uniform; >= 60% of top neurons
      above uniform at factor 0 or -2)
  A2  >= 30% of top neurons give a graded expected-digit dose response; < 5% of random layer-matched neurons do
  A3  Qwen's own uncertainty sits mostly on neighbouring digits (median share > uniform)
Output: D:/already-clean-data/numeric/mag_*.  Usage: python src/loop_v2/numeric_mag.py uncertain | dose | summary
"""
import json
import math
import random
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT

from modelmapper.experiments import facts as F1  # noqa: E402
from modelmapper.experiments import numeric as N  # noqa: E402

OUT = N.OUT
DIG = N.DIG
FACTORS = [-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0]
EXTRA = {"landmark": {"height": N.RELS["landmark"]["height"],
                      "start": ["In which year did construction of {X} begin?", "Construction of {X} started in which year?",
                                "What year was construction of {X} started?"]},
         "person": {"death": ["In which year did {X} die?", "{X} died in which year?", "What year did {X} die?"]}}


def chat_ids(tok, q):
    text = tok.apply_chat_template([{"role": "user", "content": q + N.SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def dist_pass(model, tok, rows):
    """rows: [(prompt_ids, answer_ids, edits)], edits = [(layer, neuron, factor)] applied at every position. One pass.
    Returns per row a list over answer positions of the float64 log-probs of the 10 digit tokens (full-vocab softmax)."""
    import torch
    from modelmapper.core.neurons import DEVICE, mlps
    seqs = [p + a[:-1] for p, a, _ in rows]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
    x = torch.full((len(rows), T), pad, device=DEVICE); mask = torch.zeros_like(x)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=DEVICE); mask[i, :len(s)] = 1
    by = defaultdict(list)
    for i, (_, _, ed) in enumerate(rows):
        for (l, n, fac) in (ed or []):
            by[l].append((i, n, fac))
    hks = []
    for l, items in by.items():
        def h(mod, a, items=items):
            y = a[0].clone()
            for i, n, fac in items:
                y[i, :, n] = y[i, :, n] * fac
            return (y,)
        hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
    try:
        with torch.no_grad():
            hs = model.model(input_ids=x, attention_mask=mask).last_hidden_state
    finally:
        for hk in hks:
            hk.remove()
    dig = torch.tensor(DIG, device=DEVICE); out = []
    for i, (p, a, _) in enumerate(rows):
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=DEVICE)
        lp = model.lm_head(hs[i, pos]).double().log_softmax(-1)[:, dig]
        out.append(lp.tolist())
    return out


def uncertain():
    import torch
    E = json.loads((OUT.parent / "facts2" / "entities.json").read_text(encoding="utf-8"))
    kept = {(f["entity"], f["relation"]) for f in json.loads((OUT / "facts.json").read_text(encoding="utf-8"))}
    cands = []
    for key in E:
        kind, e = key.split("|", 1)
        rels = dict(EXTRA.get(kind, {}))
        for r, t in N.RELS[kind].items():
            if (e, r) not in kept:
                rels.setdefault(r, t)
        cands += [(kind, e, r, t) for r, t in rels.items()]
    model, tok = F1.setup(); res = []
    for kind, e, r, tpls in cands:
        ps = [chat_ids(tok, t.format(X=e)) for t in tpls]
        prompts, answers = [], []
        for p in ps:
            with torch.no_grad():
                g = model.generate(torch.tensor([p], device=model.device), max_new_tokens=8, do_sample=False)[0, len(p):].tolist()
            a = []
            for t_ in g:
                if t_ in DIG:
                    a.append(t_)
                else:
                    break
            if 2 <= len(a) <= 4:
                prompts.append(p); answers.append(a)
        if not prompts:
            continue
        d = dist_pass(model, tok, [(p, a, None) for p, a in zip(prompts, answers)])
        ptop = [[math.exp(max(row)) / sum(math.exp(v) for v in row) for row in rr] for rr in d]
        keep = [i for i, pt in enumerate(ptop) if min(pt) < 0.9]
        if keep:
            res.append({"kind": kind, "entity": e, "relation": r, "prompts": [prompts[i] for i in keep], "answers": [answers[i] for i in keep],
                        "greedy": [tok.decode(answers[i]) for i in keep], "p_top_digit": [[round(x, 4) for x in ptop[i]] for i in keep]})
    (OUT / "mag_uncertain.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cands)} candidate questions -> {len(res)} uncertain facts ({sum(len(r['prompts']) for r in res)} prompts); "
          f"{dict(Counter((r['kind'], r['relation']) for r in res))}")


def _attr_top(model, tok, prompts, answers, k=100):
    import torch
    from modelmapper.core.neurons import DEVICE, mlps
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    attr = torch.zeros(L, W, device=DEVICE)
    for p, ans in zip(prompts, answers):
        acts = {}

        def keep(l):
            def h(mod, a):
                a0 = a[0].detach().requires_grad_(True); acts[l] = a0; return (a0,)
            return h
        hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]
        ids = torch.tensor([p + ans[:-1]], device=DEVICE); n = len(p)
        try:
            with torch.enable_grad():
                lp = model(input_ids=ids).logits[0, n - 1:].double().log_softmax(-1)
                c = lp[torch.arange(len(ans), device=DEVICE), torch.tensor(ans, device=DEVICE)]
                (c - torch.log(-torch.expm1(c))).sum().backward()
        finally:
            for h in hks:
                h.remove()
        for l in range(L):
            attr[l] += (acts[l][0].float() * acts[l].grad[0].float()).sum(0)
    return [divmod(int(i), W) for i in attr.flatten().topk(k).indices]


def _lo(row_lp, c):
    lp = row_lp[c]; return lp - math.log(-math.expm1(lp)) if lp < 0 else float("inf")


def dose():
    Fs = {(f["entity"], f["relation"]): f for f in json.loads((OUT / "facts.json").read_text(encoding="utf-8"))}
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    U = json.loads((OUT / "mag_uncertain.json").read_text(encoding="utf-8"))
    model, tok = F1.setup(); W = model.config.intermediate_size; rng = random.Random(211)
    jobs = []
    for m in M:
        fa = Fs[(m["entity"], m["relation"])]
        top = [x["neuron"] for x in m["top"][:3] if x["full"] > N.THR]
        jobs.append(("certain", fa["kind"], fa["entity"], fa["relation"], fa["prompts"], [fa["answer_ids"]] * len(fa["prompts"]),
                     [(int(u[1:].split(":")[0]), int(u.split(":")[1])) for u in top]))
    for u in U:
        cand = _attr_top(model, tok, u["prompts"], u["answers"])
        P = len(u["prompts"]); drops = {}
        for c0 in range(0, len(cand), 24):
            rows = [(p, a, None) for p, a in zip(u["prompts"], u["answers"])]
            for (l, n) in cand[c0:c0 + 24]:
                rows += [(p, a, [(l, n, 0.0)]) for p, a in zip(u["prompts"], u["answers"])]
            d = dist_pass(model, tok, rows); base = d[:P]
            for j, ln in enumerate(cand[c0:c0 + 24]):
                ab = d[P * (j + 1):P * (j + 2)]
                drops[ln] = stx.mean(sum(_lo(b[q], a[q] - DIG[0]) - _lo(z[q], a[q] - DIG[0]) for q in range(len(a)))
                                     for b, z, a in zip(base, ab, u["answers"]))
        top = [ln for ln in sorted(drops, key=lambda k: -drops[k])[:3] if drops[ln] > N.THR]
        jobs.append(("uncertain", u["kind"], u["entity"], u["relation"], u["prompts"], u["answers"], top))
    f = open(OUT / "mag_dose.jsonl", "w", encoding="utf-8")
    for k, (grp, kind, e, r, prompts, answers, top) in enumerate(jobs):
        units = [("top", ln) for ln in top] + [("random", (l, rng.randrange(W))) for l, _ in top]
        P = len(prompts)
        for typ, (l, n) in units:
            rows = [(p, a, [(l, n, fac)]) for fac in FACTORS for p, a in zip(prompts, answers)]
            d = dist_pass(model, tok, rows)
            f.write(json.dumps({"group": grp, "kind": kind, "entity": e, "relation": r, "type": typ, "neuron": f"L{l}:{n}", "layer": l,
                                "answers": [[t - DIG[0] for t in a] for a in answers], "factors": FACTORS,
                                "logp": [[d[fi * P + pi] for pi in range(P)] for fi in range(len(FACTORS))]},   # full float64: rounding zeroed near-certain log-probs
                               ensure_ascii=False) + "\n")
        f.flush()
        if k % 20 == 0:
            print(f"dose {k + 1}/{len(jobs)} {grp} {e} {r}: {len(top)} top neurons", flush=True)


def _probs(lp):
    m = max(lp); ex = [math.exp(v - m) for v in lp]; s = sum(ex); return [v / s for v in ex]


def _spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r
    a, b = rk(x), rk(y); ma, mb = stx.mean(a), stx.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b)); den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return num / den if den else 0.0


def summary():
    import torch
    R = [json.loads(l) for l in (OUT / "mag_dose.jsonl").read_text(encoding="utf-8").splitlines()]
    md = stx.median; i1, i0 = FACTORS.index(1.0), FACTORS.index(0.0)
    recs = []
    for r in R:
        P = len(r["answers"])
        # q*: the position whose correct digit loses most log-odds when zeroed (mean over prompts)
        Q = min(len(a) for a in r["answers"])
        drop = [stx.mean(_lo(r["logp"][i1][p][q], r["answers"][p][q]) - _lo(r["logp"][i0][p][q], r["answers"][p][q]) for p in range(P)) for q in range(Q)]
        q = max(range(Q), key=lambda j: drop[j])
        out = {"group": r["group"], "type": r["type"], "neuron": r["neuron"], "entity": r["entity"], "relation": r["relation"], "q": q, "drop0": drop[q]}
        for fac in (0.0, -2.0, -4.0):
            fi = FACTORS.index(fac); sh, dist, uni, top_d = [], [], [], []
            for p in range(P):
                c = r["answers"][p][q]; b = _probs(r["logp"][i1][p][q]); z = _probs(r["logp"][fi][p][q])
                gain = [max(0.0, z[d] - b[d]) if d != c else 0.0 for d in range(10)]
                if sum(gain) < 1e-9:
                    continue
                nb = [d for d in (c - 1, c + 1) if 0 <= d <= 9]
                sh.append(sum(gain[d] for d in nb) / sum(gain)); uni.append(len(nb) / 9)
                dist.append(sum(gain[d] * abs(d - c) for d in range(10)) / sum(gain))
                top_d.append(abs(max(range(10), key=lambda d: gain[d]) - c))
            if sh:
                out[f"nb_share_{fac}"] = stx.mean(sh); out[f"nb_uniform_{fac}"] = stx.mean(uni); out[f"dist_{fac}"] = stx.mean(dist)
                out[f"topgain_dist_{fac}"] = stx.mean(top_d)
        Es = [stx.mean(sum(d * pr for d, pr in enumerate(_probs(r["logp"][fi][p][q]))) for p in range(P)) for fi in range(len(FACTORS))]
        out["E"] = Es; out["rho"] = _spearman(FACTORS, Es); out["E_range"] = max(Es) - min(Es)
        out["c_mean"] = stx.mean(r["answers"][p][q] for p in range(P))
        out["graded"] = abs(out["rho"]) >= 0.9 and out["E_range"] >= 0.3
        recs.append(out)
    L = [f"MAGNITUDE (numeric_mag.py): {len(R)} neuron x fact dose series (factors {FACTORS}); "
         f"{dict(Counter((x['group'], x['type']) for x in recs))}"]
    uniform_d = lambda c: sum(abs(d - c) for d in range(10) if d != c) / 9
    for grp in ("certain", "uncertain"):
        for typ in ("top", "random"):
            xs = [x for x in recs if x["group"] == grp and x["type"] == typ]
            if not xs:
                continue
            L.append(f"\n[{grp} facts, {typ} neurons, n={len(xs)}]  median drop when zeroed at q*: {md(x['drop0'] for x in xs):.2f}")
            for fac in (0.0, -2.0, -4.0):
                ys = [x for x in xs if f"nb_share_{fac}" in x]
                if ys:
                    above = sum(x[f"nb_share_{fac}"] > x[f"nb_uniform_{fac}"] for x in ys)
                    L.append(f"   factor {fac:+}: lost probability to neighbouring digits: median share {md(x[f'nb_share_{fac}'] for x in ys):.0%} vs uniform "
                             f"{md(x[f'nb_uniform_{fac}'] for x in ys):.0%}; above uniform {above}/{len(ys)} ({above / len(ys):.0%}); mass-weighted |d-c| median "
                             f"{md(x[f'dist_{fac}'] for x in ys):.2f} (uniform ~{uniform_d(5):.1f}-{uniform_d(0):.1f}); biggest gainer |d-c| median {md(x[f'topgain_dist_{fac}'] for x in ys):.1f}")
            g = sum(x["graded"] for x in xs)
            L.append(f"   graded dose response (|rho| >= 0.9 and E range >= 0.3): {g}/{len(xs)} ({g / len(xs):.0%}); E range median {md(x['E_range'] for x in xs):.2f}")
            gr = [x for x in xs if x["graded"]]
            if gr:
                tw = sum((x["E"][-1] - x["E"][FACTORS.index(0.0)]) * (x["c_mean"] - x["E"][FACTORS.index(0.0)]) > 0 for x in gr)
                L.append(f"      of the graded ones, raising the neuron moves E TOWARDS the target digit in {tw}/{len(gr)} (push-pull for its own digit, "
                         f"not an abstract 'bigger' direction, if ~all)")
    # Qwen's own uncertainty on uncertain facts (as-is)
    U = json.loads((OUT / "mag_uncertain.json").read_text(encoding="utf-8"))
    model_free = []
    seen = set()
    for r in R:
        if r["group"] != "uncertain" or (r["entity"], r["relation"]) in seen:
            continue
        seen.add((r["entity"], r["relation"]))
        for p in range(len(r["answers"])):
            for q in range(len(r["answers"][p])):
                pr = _probs(r["logp"][i1][p][q]); t = max(range(10), key=lambda d: pr[d])
                if pr[t] < 0.9:
                    rest = sum(pr) - pr[t]; nb = [d for d in (t - 1, t + 1) if 0 <= d <= 9]
                    rnd = [d for d in (0, 5) if d != t]
                    model_free.append((sum(pr[d] for d in nb) / rest, len(nb) / 9, q, sum(pr[d] for d in rnd) / rest, len(rnd) / 9))
    if model_free:
        above = sum(x[0] > x[1] for x in model_free)
        L.append(f"\nQWEN'S OWN UNCERTAINTY (uncertain facts as-is, digit slots with P(top) < 0.9: {len(model_free)}): share of the other digits' mass on the top digit's "
                 f"neighbours median {md(x[0] for x in model_free):.0%} vs uniform {md(x[1] for x in model_free):.0%}; above uniform {above}/{len(model_free)}")
        for q in range(4):
            ys = [x for x in model_free if x[2] == q]
            if ys:
                L.append(f"   digit position {q}: n={len(ys)} median neighbour share {md(x[0] for x in ys):.0%} | share on round digits 0/5 {md(x[3] for x in ys):.0%} "
                         f"(uniform {md(x[4] for x in ys):.0%})")
    # unembedding geometry
    from modelmapper.core.config import DEFAULT_MODEL, model_path
    from safetensors import safe_open
    import glob
    Wu = None
    for fp in glob.glob(str(Path(model_path(DEFAULT_MODEL)) / "*.safetensors")):
        with safe_open(fp, "pt") as fh:
            if "lm_head.weight" in fh.keys():
                Wu = fh.get_slice("lm_head.weight")[DIG[0]:DIG[-1] + 1].float()
    if Wu is not None:
        Wn = Wu / Wu.norm(dim=1, keepdim=True); C = (Wn @ Wn.T).tolist()
        byd = {k: stx.mean(C[a][b] for a in range(10) for b in range(10) if abs(a - b) == k) for k in range(1, 10)}
        L.append("\nUNEMBEDDING geometry (cosine between digit tokens' output vectors by |d - d'|): " + ", ".join(f"{k}: {v:.3f}" for k, v in byd.items()))
    txt = "\n".join(L)
    (OUT / "mag_rows.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    (OUT / "mag_summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"uncertain": uncertain, "dose": dose, "summary": summary}[sys.argv[1]]()
