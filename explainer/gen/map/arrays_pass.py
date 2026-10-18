"""Per-concept views computed from run 1's saved arrays (cached; resumable).

    python -I -X utf8 explainer/gen/map/arrays_pass.py [--limit N] [--no-lens]

For every mapped run-1 concept it reads two arrays from the concept's .npz:
  attr               (36, 12288)   activation x gradient of the answer's log-odds, every neuron
  resid_answer_pos   (q, 37, 4096) the residual stream at the answer position, per question, embeddings + 36 layers
and writes one line per concept to cache/run1_arrays.jsonl:
  light.layer_share  share of total |attribution| in each layer (how the concept 'lights up' the network, by depth)
  light.strip        36 rows x 96 bins (128 neurons each), max |attribution| per bin, sqrt-scaled to one hex digit
  lens.rank          logit lens: median rank of the answer's first token per layer (final norm + unembedding applied
                     to the residual), i.e. at which depth the answer becomes readable from the stream
  lens.p             mean probability of that token per layer
The logit lens needs the model's final norm and lm_head weights (read from the local safetensors; GPU if available).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, MODELS, RUN1, read_jsonl  # noqa: E402

BINS = 96


def unembed():
    import torch
    from safetensors import safe_open
    d = MODELS / "Qwen3-8B"
    wm = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((d / "config.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for name in ("lm_head.weight", "model.norm.weight"):
        with safe_open(str(d / wm[name]), framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    dt = torch.float16 if dev == "cuda" else torch.float32
    return out["lm_head.weight"].to(dev, dt), out["model.norm.weight"].to(dev, torch.float32), cfg["rms_norm_eps"], dev


def lens(resid, aid, W, g, eps, dev):
    import torch
    x = torch.from_numpy(resid.astype(np.float32)).to(dev)            # (q, 37, d)
    q, L1, d = x.shape
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    xn = x * rms * g
    # HF hidden_states = [embeddings, output of layer 0, ..., output of layer 35], and transformers overwrites the last
    # one with last_hidden_state, which already has the final norm applied (capture_outputs(tie_last_hidden_states=True))
    already = True
    xn[:, -1] = x[:, -1]
    ranks = np.zeros((q, L1), dtype=np.int64); probs = np.zeros((q, L1))
    for i in range(q):
        lg = (xn[i].to(W.dtype) @ W.T).float()                          # (37, V)
        a = lg[:, aid]
        ranks[i] = ((lg > a[:, None]).sum(-1) + 1).cpu().numpy()
        probs[i] = torch.softmax(lg, -1)[:, aid].cpu().numpy()
    return ([int(v) for v in np.median(ranks, 0)], [round(float(v), 4) for v in probs.mean(0)], already)


def strip(attr):
    a = np.abs(attr.astype(np.float32))
    share = a.sum(1); share = share / share.sum()
    b = a.reshape(a.shape[0], BINS, -1).max(-1)
    b = np.sqrt(b / (b.max() + 1e-12))
    q = np.clip(np.rint(b * 15), 0, 15).astype(int)
    return {"layer_share": [round(float(v), 4) for v in share], "strip": "".join("0123456789abcdef"[v] for v in q.ravel()),
            "bins": BINS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-lens", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="recompute everything (overwrites the cache)")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    out_p = CACHE / "run1_arrays.jsonl"
    if args.fresh and out_p.exists():
        out_p.write_text("", encoding="utf-8")
    done = {r["concept"] for r in read_jsonl(out_p)} if out_p.exists() else set()
    todo = []
    for bf in sorted((RUN1 / "results" / "results" / "map").glob("b*/records.jsonl")):
        for rec in read_jsonl(bf):
            if rec.get("rank") and rec["concept"] not in done and (rec.get("v2") or {}).get("arrays"):
                todo.append((rec["concept"], bf.parent / "arrays" / rec["v2"]["arrays"], rec["rank"]["answer_ids"][0]))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} cached, {len(todo)} to do")
    W = g = eps = dev = None
    if not args.no_lens:
        W, g, eps, dev = unembed()
        print("unembedding on", dev)
    t0 = time.time(); flags = []
    with open(out_p, "a", encoding="utf-8") as fh:
        for i, (concept, path, aid) in enumerate(todo):
            try:
                z = np.load(path)
                row = {"concept": concept, "light": strip(z["attr"])}
                if W is not None:
                    r, p, already = lens(z["resid_answer_pos"], aid, W, g, eps, dev)
                    row["lens"] = {"rank": r, "p": p, "token_id": int(aid)}
                    flags.append(already)
            except Exception as e:                                    # a missing/corrupt array: skip, keep going
                print("skip", concept, repr(e)[:200]); continue
            fh.write(json.dumps(row) + "\n")
            if i % 200 == 0:
                fh.flush()
                print(f"{i + 1}/{len(todo)}  {time.time() - t0:.0f}s  final-norm-already-applied: "
                      f"{sum(flags)}/{len(flags)}", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
