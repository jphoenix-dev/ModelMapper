"""Corpus profile of every concept neuron: where does it fire strongly in ordinary web text? (no API; step 3)

Neurons: the union of every recalled concept's top-10 measured neurons (rank_save_*.json), each with the sign and
size of its activation at the answer position on its own concept (the largest |act| if it is top-10 for several).
Control: for each, one random neuron from the same layer (seeded), given the SAME sign and threshold.
Text: exactly the 10M tokens / 20,359 documents the atlas scanned (data/atlas_<model>_docs.npz), same 1,024 windows,
token 0 of each window skipped (attention sink), as build_atlas.py.
Recorded: EVERY firing with sign * activation >= THRESH * |own activation| (THRESH = 0.5) - no top-K, no sampling.
If the total ever passed MAX_FIRINGS the run stops with an error rather than dropping any (none dropped silently).
Output: results/corpus_profile.npz (neuron index, doc, position, value for every firing) + results/corpus_profile_units.json
Usage: python src/corpus_profile.py
"""
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]
TOP, THRESH, CTX, BATCH_TOKENS, MAX_FIRINGS = 10, 0.5, 1024, 6144, 200_000_000


def units():
    best = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f:
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" not in v:                       # prompt-count mismatch: not ranked (rank_save.py)
                print(f"skipping {c}: {v.get('mismatch')}", flush=True); continue
            for e in v["top"][:TOP]:
                u = e["neuron"]; a = e["act_answer_pos"]
                best.setdefault(u, {"concepts": []})["concepts"].append(c)
                if abs(a) > abs(best[u].get("own_act", 0.0)):
                    best[u]["own_act"] = a
    rows = []
    rng = random.Random(11)
    taken = {tuple(int(x) for x in u[1:].split(":")) for u in best}
    for u, v in best.items():
        l, n = (int(x) for x in u[1:].split(":"))
        rows.append({"neuron": u, "layer": l, "index": n, "own_act": v["own_act"], "concepts": v["concepts"], "control": False})
        while True:
            r = rng.randrange(12288)
            if (l, r) not in taken:
                taken.add((l, r)); break
        rows.append({"neuron": f"L{l}:{r}", "layer": l, "index": r, "own_act": v["own_act"], "concepts": [], "control": True,
                     "paired_with": u})
    return rows


def main():
    rows = units()
    (ROOT / "results/corpus_profile_units.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    model, tok = load(DEFAULT_MODEL)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    by_layer = {}
    for i, r in enumerate(rows):
        by_layer.setdefault(r["layer"], []).append(i)
    L_idx = {l: torch.tensor([rows[i]["index"] for i in ii], device=DEVICE) for l, ii in by_layer.items()}
    L_row = {l: torch.tensor(ii, device=DEVICE) for l, ii in by_layer.items()}
    L_sg = {l: torch.tensor([1.0 if rows[i]["own_act"] >= 0 else -1.0 for i in ii], device=DEVICE) for l, ii in by_layer.items()}
    L_th = {l: torch.tensor([THRESH * abs(rows[i]["own_act"]) for i in ii], device=DEVICE) for l, ii in by_layer.items()}
    found = {"unit": [], "doc": [], "pos": [], "val": []}; total = 0
    state = {}

    def hook(l):
        def h(mod, args):
            a = args[0][:, :, L_idx[l]].float() * L_sg[l]          # [B, T, k] in each neuron's own direction
            hit = (a >= L_th[l]) & state["valid"][:, :, None]
            b, t, k = hit.nonzero(as_tuple=True)
            if len(b):
                found["unit"].append(L_row[l][k].cpu().numpy().astype(np.int32))
                found["doc"].append((state["doc0"] + b).cpu().numpy().astype(np.int32))
                found["pos"].append(t.cpu().numpy().astype(np.int16))
                found["val"].append(a[b, t, k].cpu().numpy().astype(np.float16))
                state["n"] += len(b)
        return h
    hk = [mlps(model)[l].down_proj.register_forward_pre_hook(hook(l)) for l in by_layer]
    t0 = time.time(); doc = 0; seen = 0; state["n"] = 0
    try:
        while doc < len(lens):
            batch = []
            while doc < len(lens) and sum(map(len, batch)) < BATCH_TOKENS:
                batch.append(toks[starts[doc]:starts[doc] + lens[doc]]); doc += 1
            T = max(map(len, batch))
            ids = torch.zeros((len(batch), T), dtype=torch.long); mask = torch.zeros_like(ids)
            for i, x in enumerate(batch):
                ids[i, :len(x)] = torch.as_tensor(x.astype(np.int64)); mask[i, :len(x)] = 1
            valid = mask.bool(); valid[:, 0] = False
            state["valid"] = valid.to(DEVICE); state["doc0"] = doc - len(batch)
            with torch.no_grad():
                model.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
            seen += int(valid.sum())
            if state["n"] > MAX_FIRINGS:
                raise RuntimeError(f"more than {MAX_FIRINGS:,} firings recorded - stopping rather than dropping any")
            if doc % 2000 < len(batch):
                rate = seen / (time.time() - t0)
                print(f"{doc}/{len(lens)} docs, {seen/1e6:.2f}M tokens, {state['n']:,} firings, {rate:,.0f} tok/s", flush=True)
    finally:
        for h in hk:
            h.remove()
    np.savez_compressed(ROOT / "results/corpus_profile.npz", tokens_scanned=seen,
                        **{k: np.concatenate(v) if v else np.zeros(0) for k, v in found.items()})
    print(f"done: {seen:,} tokens, {state['n']:,} firings, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
