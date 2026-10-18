"""H1: inhibitor neurons - neurons whose knockout RAISES the answer (MECHANISMS.md H1). CPU, Qwen3-0.6B.

Why: every 8B concept map knocked out only the top 200 neurons by POSITIVE attribution, so neurons that push against
the answer were never tested. This checks on 0.6B whether such neurons exist, beyond a random-neuron null, on held-out
prompts.

Data reused: loop v2 run 1's recall prompts and answer forms (8B maps; same tokenizer and chat template as 0.6B).
Procedure:
  screen    concepts in a fixed shuffled order (seed 0); 0.6B exact log-odds of the run-1 answer form on the first 6
            recall prompts (skip the concept if < 2 are in reach), then on all; a prompt is "in reach" if log-odds > -2
            (P > 0.12; changed from > 0 after a smoke test found 0.6B confident on almost no run-1 concept, before any
            knockout ran); keep the concept if >= 6 prompts are in reach; stop at N_CONCEPTS kept (every screened
            concept is saved, kept or not)
  split     recalled prompts: even index -> A (selection), odd -> B (held-out measurement)
  attribute on A: activation x gradient of log P(answer), summed over positions and prompts, all 28 x 3,072 neurons
  select    top 20 (most positive), bottom 20 (most negative), random 20 (uniform, outside the top/bottom 200), matched 20
            (one random neuron from each bottom neuron's layer, outside the top/bottom 200)
  knock out each neuron alone (zero at every position) on B (and on A for comparison); record the change in mean exact
            log-odds of the answer (+ = answer went UP), the change in entropy at the first answer position, and the
            change in log-prob of the clean run's strongest rival token there
Pass rule (stated before running): inhibitor = a bottom neuron whose held-out rise exceeds the 99th percentile of the
pooled random + matched null rises. H1 holds if >= 30% of concepts have >= 1 inhibitor.
Predictions (MECHANISMS.md, before running):
  P1a >= 30% of concepts have >= 1 inhibitor
  P1b >= 50% of inhibitors in layers 21-27 (last quarter)
  P1c >= 3 neurons are inhibitors for >= 10% of kept concepts (generic dampeners)
  P1d median held-out rise of each concept's best inhibitor >= 0.3 log-odds
Output: D:/already-clean-data/mechanisms/inhibitors/{screen.json, per_concept.jsonl, summary.json}
Usage: python -X utf8 src/mechanisms/inhibitors.py [N_CONCEPTS=40] [SEED=0]
"""
import json
import random
import statistics as stx
import sys
import time
from collections import Counter
from pathlib import Path

from modelmapper.experiments.mechanisms.common import OUT, RUN1, exact_logodds, load06, mlp_hook, save  # noqa: E402
import torch  # noqa: E402

ARGS = [a for a in sys.argv[1:] if a != "summary"]
N_CONCEPTS = int(ARGS[0]) if len(ARGS) > 0 else 40
SEED = int(ARGS[1]) if len(ARGS) > 1 else 0
K = 20
REACH = -2.0
D = OUT / ("inhibitors" if SEED == 0 else f"inhibitors_seed{SEED}")
D.mkdir(parents=True, exist_ok=True)


def run1_concepts():
    out = []
    for f in sorted(RUN1.glob("b*/records.jsonl")):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            row = r.get("row") or {}
            if row.get("recall_prompts") and row.get("answer_form"):
                out.append({"concept": r["concept"], "answer": row["answer_form"], "prompts": row["recall_prompts"],
                            "source": f"{f.relative_to(RUN1.parents[3]).as_posix()}"})
    return out


def forward_rows(model, tok, rows, rival=None):
    """rows: [(prompt_ids, ans_ids, kills[(layer, neuron)])] -> [(logodds, entropy0, logprob0_of_rival, top0_id)]."""
    seqs = [p + a[:-1] for p, a, _ in rows]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
    x = torch.full((len(rows), T), pad); m = torch.zeros_like(x)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    by = {}
    for i, (_, _, kills) in enumerate(rows):
        for l, n in kills:
            by.setdefault(l, []).append((i, n))
    hks = []
    for l, items in by.items():
        ri = torch.tensor([i for i, _ in items]); ni = torch.tensor([n for _, n in items])

        def h(mod, a, ri=ri, ni=ni):
            y = a[0].clone(); y[ri, :, ni] = 0.0
            return (y,)
        hks.append(mlp_hook(model, l, h))
    try:
        hs = model.model(input_ids=x, attention_mask=m).last_hidden_state
    finally:
        for hk in hks:
            hk.remove()
    out = []
    for i, (p, a, _) in enumerate(rows):
        lg = model.lm_head(hs[i, len(p) - 1: len(p) - 1 + len(a)])
        lo = exact_logodds(lg, a)
        lp0 = lg[0].double().log_softmax(-1)
        ent = -(lp0.exp() * lp0).sum().item()
        rv = rival[i] if rival is not None else None
        out.append((lo, ent, lp0[rv].item() if rv is not None else None, int(lp0.argmax())))
    return out


def attribute(model, tok, prompts, ans):
    L = len(model.model.layers); acts = {}
    hooks = []
    for l in range(L):
        def keep(mod, a, l=l):
            a[0].retain_grad(); acts[l] = a[0]
        hooks.append(mlp_hook(model, l, keep))
    hooks.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: o.requires_grad_(True)))
    attr = torch.zeros(L, model.config.intermediate_size, dtype=torch.float64)
    try:
        for p in prompts:
            with torch.enable_grad():
                ids = torch.tensor([p + ans[:-1]])
                lp = model(input_ids=ids).logits[0, len(p) - 1:].float().log_softmax(-1)
                s = lp[torch.arange(len(ans)), torch.tensor(ans)].sum()
                s.backward()
            attr += torch.stack([(acts[l][0] * acts[l].grad[0]).sum(0).double() for l in range(L)])
            acts.clear()
    finally:
        for h in hooks:
            h.remove()
    return attr


def main():
    t0 = time.time()
    model, tok = load06()
    W = model.config.intermediate_size; L = model.config.num_hidden_layers
    cands = run1_concepts(); random.Random(SEED).shuffle(cands)
    print(f"{len(cands)} run-1 concepts with recall prompts; screening for {N_CONCEPTS}", flush=True)
    screen, kept = [], []
    for c in cands:
        ans = tok.encode(c["answer"]); P = [tok.encode(p) for p in c["prompts"]]
        lo = [r[0] for r in forward_rows(model, tok, [(p, ans, []) for p in P[:6]])]
        if sum(v > REACH for v in lo) >= 2:
            lo += [r[0] for r in forward_rows(model, tok, [(p, ans, []) for p in P[6:]])] if len(P) > 6 else []
        rec = [i for i, v in enumerate(lo) if v > REACH]
        screen.append({"concept": c["concept"], "answer": c["answer"], "n_prompts": len(P), "n_recalled": len(rec),
                       "logodds": [round(v, 4) for v in lo], "kept": len(rec) >= 6})
        if len(rec) >= 6:
            kept.append((c, ans, [P[i] for i in rec]))
        if len(screen) % 50 == 0:
            print(f"  screened {len(screen)}, kept {len(kept)} ({time.time() - t0:.0f}s)", flush=True)
        if len(kept) == N_CONCEPTS:
            break
    save(D.relative_to(OUT).as_posix() + "/screen.json", screen)
    print(f"screened {len(screen)}, kept {len(kept)} ({time.time() - t0:.0f}s)", flush=True)

    fo = (D / "per_concept.jsonl").open("w", encoding="utf-8")
    for ci, (c, ans, P) in enumerate(kept):
        A, B = P[0::2], P[1::2]
        attr = attribute(model, tok, A, ans)
        flat = attr.flatten(); order = flat.argsort()
        bottom = [divmod(i.item(), W) for i in order[:K]]
        top = [divmod(i.item(), W) for i in order.flip(0)[:K]]
        excl = {divmod(i.item(), W) for i in torch.cat([order[:200], order[-200:]])}
        rng = random.Random(1000 * SEED + ci)
        rand, matched = [], []
        while len(rand) < K:
            u = (rng.randrange(L), rng.randrange(W))
            if u not in excl and u not in rand:
                rand.append(u)
        for l, _ in bottom:
            while True:
                u = (l, rng.randrange(W))
                if u not in excl and u not in matched:
                    matched.append(u); break
        groups = {"top": top, "bottom": bottom, "random": rand, "matched": matched}
        rec = {"concept": c["concept"], "answer": c["answer"], "source": c["source"], "nA": len(A), "nB": len(B)}
        for half, PP in (("B", B), ("A", A)):
            # strongest rival = the top token at the first answer position that is not the answer's first token
            rival = []
            for p in PP:
                ids = torch.tensor([p])
                lg = model(input_ids=ids).logits[0, -1].double()
                lg[ans[0]] = -1e9
                rival.append(int(lg.argmax()))
            clean = forward_rows(model, tok, [(p, ans, []) for p in PP], rival=rival)
            c_lo = stx.mean(r[0] for r in clean); c_ent = stx.mean(r[1] for r in clean); c_rv = stx.mean(r[2] for r in clean)
            rec[f"clean_{half}"] = {"logodds": [round(r[0], 4) for r in clean], "entropy0": [round(r[1], 4) for r in clean],
                                    "rival_tokens": [tok.decode([r]) for r in rival]}
            for g, units in groups.items():
                rows, idx = [], []
                for u in units:
                    for j, p in enumerate(PP):
                        rows.append((p, ans, [u])); idx.append(j)
                res = []
                for s in range(0, len(rows), 48):
                    res += forward_rows(model, tok, rows[s:s + 48], rival=[rival[j] for j in idx[s:s + 48]])
                per = []
                for k, u in enumerate(units):
                    rr = res[k * len(PP):(k + 1) * len(PP)]
                    per.append({"unit": f"L{u[0]}:{u[1]}", "attr": round(attr[u].item(), 5),
                                "rise": round(stx.mean(r[0] for r in rr) - c_lo, 4),
                                "rise_per_prompt": [round(r[0] - cl[0], 4) for r, cl in zip(rr, clean)],
                                "d_entropy0": round(stx.mean(r[1] for r in rr) - c_ent, 4),
                                "d_rival_logprob": round(stx.mean(r[2] for r in rr) - c_rv, 4)})
                rec[f"{g}_{half}"] = per
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n"); fo.flush()
        best = max(x["rise"] for x in rec["bottom_B"])
        print(f"[{ci + 1}/{len(kept)}] {c['concept'][:30]:30s} nB={len(B)} best bottom rise {best:+.3f}  "
              f"top drop {min(x['rise'] for x in rec['top_B']):+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    fo.close()
    summarise()


def summarise():
    recs = [json.loads(l) for l in (D / "per_concept.jsonl").open(encoding="utf-8")]
    null = sorted(x["rise"] for r in recs for g in ("random_B", "matched_B") for x in r[g])
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
    thr = q(null, 0.99)
    inhib, best, layers = [], [], []
    for r in recs:
        hits = [x for x in r["bottom_B"] if x["rise"] > thr]
        inhib.append(hits); best.append(max(x["rise"] for x in r["bottom_B"]))
        layers += [int(x["unit"][1:].split(":")[0]) for x in hits]
    n = len(recs); with_inh = sum(1 for h in inhib if h)
    share = Counter(x["unit"] for h in inhib for x in h)
    # same counts for the null groups, as a control on the counting itself (a random neuron 'passing' by chance)
    null_conc = sum(1 for r in recs if any(x["rise"] > thr for x in r["random_B"]))
    top_drop = [min(x["rise"] for x in r["top_B"]) for r in recs]
    # does the raise replicate on the selection half A? (sign agreement of bottom neurons' rises A vs B)
    agree = [((a["rise"] > 0) == (b["rise"] > 0)) for r in recs for a, b in zip(r["bottom_A"], r["bottom_B"])]
    hit_ent = [x["d_entropy0"] for h in inhib for x in h]; hit_rv = [x["d_rival_logprob"] for h in inhib for x in h]
    S = {"n_concepts": n, "null_n": len(null), "null_p99_rise": thr, "null_p95_rise": q(null, 0.95), "null_max": null[-1],
         "P1a_frac_concepts_with_inhibitor": with_inh / n, "P1a_pass": with_inh / n >= 0.30,
         "random_group_frac_concepts_over_thr (control)": null_conc / n,
         "n_inhibitors": len(layers), "inhibitor_layers": sorted(Counter(layers).items()),
         "P1b_frac_in_layers_21_27": (sum(1 for l in layers if l >= 21) / len(layers)) if layers else None,
         "P1c_units_inhibitor_for_ge10pct": [(u, k) for u, k in share.most_common() if k >= 0.1 * n],
         "P1c_top_shared": share.most_common(10),
         "P1d_median_best_rise": stx.median(best), "best_rise_per_concept": [round(b, 3) for b in best],
         "median_top_neuron_best_drop": stx.median(top_drop),
         "bottom_rise_sign_agreement_A_vs_B": sum(agree) / len(agree),
         "inhibitors_median_d_entropy0": stx.median(hit_ent) if hit_ent else None,
         "inhibitors_frac_lower_rival": (sum(1 for v in hit_rv if v < 0) / len(hit_rv)) if hit_rv else None,
         "inhibitors_median_d_rival_logprob": stx.median(hit_rv) if hit_rv else None}
    S["P1b_pass"] = S["P1b_frac_in_layers_21_27"] is not None and S["P1b_frac_in_layers_21_27"] >= 0.5
    S["P1c_pass"] = len(S["P1c_units_inhibitor_for_ge10pct"]) >= 3
    S["P1d_pass"] = S["P1d_median_best_rise"] >= 0.3
    save(D.relative_to(OUT).as_posix() + "/summary.json", S)
    print(json.dumps({k: v for k, v in S.items() if k != "best_rise_per_concept"}, indent=1))


if __name__ == "__main__":
    if "summary" in sys.argv[1:]:
        summarise()
    else:
        main()
