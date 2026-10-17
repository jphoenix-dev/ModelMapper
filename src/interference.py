"""Interference test: when a concept neuron fires OFF-topic, does its push help or hurt Qwen's predictions there?
(GPU; no API)

James (17 Oct): many triggers, one job - harmless reuse, a leak the network tolerates, or a push it uses everywhere?
Contexts: exactly the 12 random firings per neuron that luna_firing_judge.py showed Luna (rebuilt from its seeds and
checked against the saved snippets), each labelled A (about the concept), B (related) or C (unrelated); concept
neurons and their paired random CONTROL neurons (judged against the same concept, mostly C).
Per context, one batch of 3 identical sequences (the document up to the firing token + the next T = 16 tokens, fewer
at the end of a document - recorded): row 0 as is, row 1 the neuron zeroed at the firing token, row 2 a random
same-layer neuron zeroed there (noise null; seeded per context). Saved per row and position: log P(actual next
token) for the T predictions from the firing token on. delta = loss(zeroed) - loss(as is): > 0 the neuron's push HELPED
predict the real text, < 0 it HURT (zeroing it helps).
Readings fixed before running: on-concept (A) expected > 0. Off-topic (C) of concept neurons: ~0 (vs null) = harmless;
> 0 = used there too; < 0 = interference. Compared with control neurons' (C) firings, a random neuron's typical effect.
Gate (checked on the run's own data before any reading): every rebuilt snippet equals the saved one; the as-is
activation at the firing token, in the neuron's own direction (corpus_profile stores sign x activation), matches
corpus_profile's within 10% (or 0.1); the zeroed row's activation is 0.
Output: results/interference.jsonl (one line per context, resumable), results/interference_gate.txt.
Usage: python src/interference.py [max_units=all]
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
N_CTX, HALF, T = 12, 20, 16
OUT = ROOT / "results/interference.jsonl"


def contexts():
    """Rebuild luna_firing_judge's (unit, doc, pos) choices exactly."""
    rows = json.loads((ROOT / "results/corpus_profile_units.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "results/corpus_profile_summary.json").read_text(encoding="utf-8"))
    concept_rows = [i for i, r in enumerate(rows) if not r["control"] and r["neuron"] in prof and prof[r["neuron"]]["n"] >= N_CTX]
    pair = {r["paired_with"]: i for i, r in enumerate(rows) if r["control"]}
    jobs = [(i, i) for i in concept_rows] + [(pair[rows[i]["neuron"]], i) for i in concept_rows]
    z = np.load(ROOT / "results/corpus_profile.npz"); unit, doc, pos, val = z["unit"], z["doc"], z["pos"].astype(np.int64), z["val"]
    order = np.argsort(unit, kind="stable"); unit, doc, pos, val = unit[order], doc[order], pos[order], val[order]
    bounds = np.searchsorted(unit, np.arange(len(rows) + 1))
    J = json.loads((ROOT / "results/luna_firing_judge.json").read_text(encoding="utf-8"))["jobs"]
    out = []
    for show, name in jobs:
        key = f"{rows[show]['neuron']}|{'control' if rows[show]['control'] else 'concept'}|{rows[name]['neuron']}"
        j = J.get(key)
        if not j or not j.get("valid"):
            continue
        sel = np.arange(bounds[show], bounds[show + 1])
        ch = random.Random(f"21|{key}").sample(list(sel), N_CTX)
        out.append({"key": key, "neuron": rows[show]["neuron"], "control": rows[show]["control"], "concept": j["concept"],
                    "sgn": 1.0 if rows[show]["own_act"] >= 0 else -1.0,
                    "labels": j["labels"], "snippets": j["snippets"],
                    "ctx": [(int(doc[k]), int(pos[k]), float(val[k])) for k in ch]})
    return out


def main():
    max_units = int(sys.argv[1]) if len(sys.argv) > 1 else None
    U = contexts()
    random.Random(5).shuffle(U)                                  # concept and control units interleaved
    if max_units:
        U = U[:max_units]
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    model, tok = load(DEFAULT_MODEL); W = model.config.intermediate_size
    # gate 1: snippets rebuilt exactly
    bad = 0
    for u in U:
        for (di, p, _), s in zip(u["ctx"], u["snippets"]):
            g = starts[di] + p; lo = max(starts[di], g - HALF); hi = min(starts[di] + lens[di], g + HALF + 1)
            s2 = (tok.decode(toks[lo:g]) + "[[" + tok.decode(toks[g:g + 1]) + "]]" + tok.decode(toks[g + 1:hi])).replace("\n", " ")
            bad += s2 != s
    gate = [f"gate 1 - rebuilt snippets differing from the saved ones: {bad} of {sum(len(u['ctx']) for u in U)}"]
    print(gate[-1], flush=True)
    assert bad == 0
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done.add((r["key"], r["i"]))
    f = open(OUT, "a", encoding="utf-8"); t0 = time.time(); n = 0; act_bad = 0; zero_bad = 0
    for ui, u in enumerate(U):
        L, nn = (int(x) for x in u["neuron"][1:].split(":"))
        for i, ((di, p, v), lab) in enumerate(zip(u["ctx"], u["labels"])):
            if (u["key"], i) in done:
                continue
            end = min(p + 1 + T, int(lens[di]))
            ids = torch.tensor(toks[starts[di]:starts[di] + end].astype(np.int64), device=DEVICE)[None].repeat(3, 1)
            rnd = random.Random(f"{u['key']}|{i}").randrange(W)
            while rnd == nn:
                rnd = random.Random(f"{u['key']}|{i}|x").randrange(W)
            seen = {}

            def h(m, a):
                x = a[0].clone(); seen["act"] = x[0, p, nn].float().item()
                x[1, p, nn] = 0.0; x[2, p, rnd] = 0.0
                seen["act_after"] = x[1, p, nn].float().item(); seen["rnd_act"] = a[0][2, p, rnd].float().item()
                return (x,)
            hk = mlps(model)[L].down_proj.register_forward_pre_hook(h)
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[:, p:end - 1].double().log_softmax(-1)
            finally:
                hk.remove()
            tgt = ids[0, p + 1:end]
            lp = lg.gather(2, tgt[None, :, None].expand(3, -1, 1))[..., 0].tolist()
            # profile values are in the neuron's own direction
            # (17 Oct full run: an edit had put the two counter updates below inside this comment, so the run's printed
            # gate 2/3 counts were 0 regardless; per-context act_ok was recorded correctly - 38/22,476 mismatches - and
            # zeroing was verified afterwards from the outputs: the zeroed row differs from as-is in 98.8%, the rest
            # being layer-35 neurons whose single-position change is below bf16 logit resolution)
            ok_act = abs(u["sgn"] * seen["act"] - v) <= max(0.1, 0.1 * abs(v))
            act_bad += not ok_act; zero_bad += seen["act_after"] != 0.0
            f.write(json.dumps({"key": u["key"], "i": i, "neuron": u["neuron"], "control": u["control"], "concept": u["concept"],
                                "label": lab, "doc": di, "pos": p, "T": end - 1 - p, "act": round(seen["act"], 4), "profile_act": round(v, 4),
                                "act_ok": ok_act, "rnd_neuron": rnd, "rnd_act": round(seen["rnd_act"], 4),
                                "logp": [[round(x, 5) for x in row] for row in lp]}) + "\n")
            n += 1
        f.flush()
        if ui % 25 == 0:
            el = time.time() - t0
            print(f"{ui + 1}/{len(U)} units, {n} contexts, {el / 60:.1f} min ({el / max(1, n):.3f} s/context); "
                  f"activation mismatches {act_bad}, zeroing failures {zero_bad}", flush=True)
    gate.append(f"gate 2 - activation at the firing token vs corpus_profile (10% or 0.1): mismatches {act_bad} of {n} new contexts")
    gate.append(f"gate 3 - zeroed row's activation not 0: {zero_bad}")
    (ROOT / "results/interference_gate.txt").write_text("\n".join(gate) + "\n", encoding="utf-8")
    print("\n".join(gate))


if __name__ == "__main__":
    main()
