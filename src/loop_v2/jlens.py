"""J-lens for Qwen3-8B (after Gurnee, Sofroniew, Pearce, Lindsey et al. 2026, 'Verbalizable Representations Form a Global
Workspace in Language Models', transformer-circuits.pub/2026/workspace) - and every MLP neuron read through it. (GPU; no API)

J_l = the Jacobian of the final residual stream (pre-norm, after the last layer) with respect to the residual stream after
layer l, averaged over source positions t >= 1, later positions t' >= t, and web-text windows. The lens reads a vector v at
layer l as softmax(W_U . (g * J_l v)) (g = final RMSNorm weight; RMSNorm's scale does not change token rankings).
Estimation: one backward pass from sum_{t'} h_L[t', j] gives row j of sum_{t<=t'} J(t', t) for ALL layers at once; batch
rows = different output dims j of the same window; 4096 dims per window. No lm_head / logits in the estimation passes.

Stages
  estimate  [N windows x T tokens from FineWeb (atlas docs), rows B]  -> J sums per split half (A = even windows, B = odd),
            checkpointed every few windows (resumable); timing + GPU busy % + peak memory in jlens_log.jsonl
  validate  held-out windows: at each layer, rank of the tokens that actually come next (t+1..t+16) under the J-lens vs the
            logit lens; final layer check (J_last = I); split-half reliability (top-20 token overlap, J_A vs J_B, of
            random neurons' output vectors)
  neurons   every neuron's output vector (down_proj column) read through the J-lens of its layer and through the logit lens:
            top / bottom 30 tokens + logits + z of the top logit -> jlens_neurons_L<l>.npz
  tests     the predictions below                                                                -> jlens_tests.txt
Predictions (stated before running):
  J1  the J-lens is valid on Qwen3-8B: in middle layers (12-30) the tokens that actually come next rank better under the J-lens
      than under the logit lens (median best rank over t+1..t+16), and the two agree at the last layer.
  J2  (tests LAYERS_AND_JOBS S2, 'early neurons do not write words') the facts2 direct entity neurons (median layer 8) carry
      their entity in the J-lens: >= 30% have a word of their entity's name or facts among their top-50 J-lens tokens, vs
      <= 5% of random same-layer neurons and <= 10% under the logit lens.
  J3  J-lens and logit-lens readouts agree for late neurons (layers 28-35: median top-15 overlap >= 50%) and not for early ones
      (layers 0-19: < 20%).
  J4  L31:8930 and L28:9936 (era neurons) read as their era under the J-lens too (era / period / history tokens in the top 15).
Output: D:/already-clean-data/jlens/.  Usage: python src/loop_v2/jlens.py estimate [--windows N --start a --end b] | merge <files> | validate | neurons | tests
(shards for rented GPUs: estimate --windows 256 --start 32 --end 64 -> j_sums_32_64.pt; JL_ROWS=4 on 24 GB cards)
"""
import json
import os
import random
import statistics as stx
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src" / "loop_v2"))
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "jlens"
DOCS = Path(os.environ.get("AC_ATLAS_DOCS", "C:/dev/already-clean/data/atlas_Qwen3-8B_docs.npz"))
T = int(os.environ.get("JL_TOKENS", "128"))
B = int(os.environ.get("JL_ROWS", "16"))
CKPT_EVERY = 8                                                    # a 4.8 GB checkpoint idles the GPU; not too often


def windows(n, seed, skip=()):
    """n windows of T tokens (+1 for validation targets) from distinct FineWeb docs (atlas docs), deterministic."""
    d = np.load(DOCS); toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    rng = random.Random(seed); order = [i for i in range(len(lens)) if lens[i] >= T + 17 and i not in skip]; rng.shuffle(order)
    out = []
    for i in order[:n]:
        o = rng.randrange(0, lens[i] - T - 16)
        out.append((int(i), toks[starts[i] + o: starts[i] + o + T + 16].astype(np.int64)))
    return out


class GpuBusy:
    """Samples GPU utilisation every second (nvidia-smi) in a thread - the 'GPU busy %' lesson."""
    def __init__(self):
        self.samples, self._stop = [], False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                self.samples.append(int(r.stdout.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)

    def take(self):
        s, self.samples = self.samples, []
        return round(sum(s) / len(s), 1) if s else None


def setup():
    import facts as F1
    from config import DEFAULT_MODEL, model_path
    if not os.path.isdir(model_path(DEFAULT_MODEL)):        # never fall through to a silent 16 GB hub download (18 Oct)
        raise SystemExit(f"no local copy of {DEFAULT_MODEL}: set MODEL_MAPPER_MODELS (e.g. C:/dev/models/hf); the research "
                         "worktree has no .env")
    return F1.setup()


def layer_out(o):
    return o[0] if isinstance(o, tuple) else o


def estimate(n_windows, start=0, end=None):
    """Windows [start, end) of the deterministic draw of n_windows (windows(n)[:k] == windows(k)): a shard for rented GPUs
    writes j_sums_<start>_<end>.pt; `merge` adds shards (and the 5090 pilot, windows 0-31) into j_sums.pt."""
    end = n_windows if end is None else end
    model, tok = setup(); L, D = model.config.num_hidden_layers, model.config.hidden_size
    OUT.mkdir(parents=True, exist_ok=True)
    shard = not (start == 0 and end == n_windows)
    ck = OUT / (f"j_sums_{start}_{end}.pt" if shard else "j_sums.pt")
    if ck.exists():
        st = torch.load(ck, weights_only=False); JA, JB, pairs = st["A"], st["B"], st["pairs"]
        done = st.get("next", st["done"]); print(f"resuming at window {done}")
    else:
        JA = torch.zeros(L, D, D); JB = torch.zeros(L, D, D); done = start; pairs = {"A": 0, "B": 0}
    W = windows(n_windows, seed=301)
    layers = model.model.layers; dev = next(model.parameters()).device
    cur = torch.zeros(L, D, D, device=dev)                         # this window's sum (float32, on GPU)
    log = open(OUT / "jlens_log.jsonl", "a", encoding="utf-8"); gb = GpuBusy()
    hs = {}

    def keep(l):
        def h(mod, a, o):
            x = layer_out(o)
            if l < L - 1:
                x.retain_grad()
            hs[l] = x
        return h
    hk = [layers[l].register_forward_hook(keep(l)) for l in range(L)]        # once, not per pass (overhead lesson)
    rows = torch.arange(B, device=dev); maxd = int(os.environ.get("JL_MAX_DIMS", D))   # JL_MAX_DIMS: benchmarking only
    for w in range(done, end):
        t0 = time.time(); cur.zero_()
        ids = torch.tensor(W[w][1][:T], device=dev)[None].repeat(B, 1)
        for j0 in range(0, maxd, B):
            if j0 == B:
                torch.cuda.synchronize(); t1 = time.time(); gb.take()
            hs.clear()
            emb = model.model.embed_tokens(ids).detach().requires_grad_(True)
            with torch.enable_grad():
                model.model(inputs_embeds=emb)                      # final norm runs but lm_head never does
                js = torch.arange(j0, j0 + B, device=dev)
                obj = hs[L - 1][rows, 1:, :].gather(2, js[:, None, None].expand(B, T - 1, 1)).sum()
                obj.backward()
            # rows j0..j0+B of sum_{1<=t<=t'} J(t', t), all layers in one op
            cur[:L - 1, j0:j0 + B] += torch.stack([hs[l].grad[:, 1:, :].sum(1) for l in range(L - 1)]).float()
        cur[L - 1] = torch.eye(D, device=dev) * (T - 1)            # last layer: J(t', t) = I iff t' = t
        torch.cuda.synchronize(); per_pass = (time.time() - t1) / max(1, (min(D, int(os.environ.get("JL_MAX_DIMS", D))) // B) - 1)
        if os.environ.get("JL_MAX_DIMS"):
            for h in hk:
                h.remove()
            print({"bench": True, "B": B, "T": T, "s_per_pass": round(per_pass, 3), "tokens_per_s": round(B * T / per_pass),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "gpu_busy_pct": gb.take()}, flush=True)
            return
        half = "A" if w % 2 == 0 else "B"
        if not all(torch.isfinite(cur[l]).all() for l in range(L)):   # 18 Oct: 3 rented shards came back all-NaN; stop, don't poison the sum
            # (per layer: a whole-tensor isfinite allocates 2.25 GiB and OOMed 24 GB GPUs)
            log.write(json.dumps({"window": w, "doc": W[w][0], "nonfinite": True, "gpu": torch.cuda.get_device_name()}) + "\n"); log.flush()
            raise SystemExit(f"window {w}: non-finite J (GPU {torch.cuda.get_device_name()}); shard stopped, checkpoint kept at window {done}")
        (JA if half == "A" else JB).add_(cur.cpu()); pairs[half] += (T - 1) * T // 2
        done = w + 1
        rec = {"window": w, "doc": W[w][0], "half": half, "s": round(time.time() - t0, 1), "gpu_busy_pct": gb.take(),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "T": T, "B": B}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
        if (done - start) % CKPT_EVERY == 0 or done == end:
            tmp = ck.with_suffix(".tmp"); torch.save({"A": JA, "B": JB, "done": done, "next": done, "start": start, "end": end, "pairs": pairs, "T": T,
                                                      "windows": [x[0] for x in W[start:done]]}, tmp)
            tmp.replace(ck)


def merge(paths):
    """Sum shard files (and/or the pilot j_sums.pt) into OUT/j_sums.pt; the previous j_sums.pt is kept as j_sums_pilot.pt."""
    m = None                                                          # one shard in memory at a time (each is 4.8 GB)
    for p in paths:
        st = torch.load(p, weights_only=False, mmap=True)          # mapped, not read: only the running sum is in RAM
        if "merged_from" in st:                                    # an earlier merge's output (e.g. a re-run's shards merged in the cloud)
            if st["done"] != len(st["windows"]):
                raise SystemExit(f"{p}: merged file with done {st['done']} != {len(st['windows'])} windows")
        elif st.get("next", st["done"]) != st["end"] or len(st["windows"]) != st["end"] - st["start"]:
            raise SystemExit(f"{p}: incomplete shard ({len(st['windows'])} of {st['start']}-{st['end']})")
        if not all(torch.isfinite(st[h][l]).all() for h in "AB" for l in range(st[h].shape[0])):
            raise SystemExit(f"{p}: non-finite values (excluded shards are listed in RESULTS.md)")
        if m is None:
            m = {"A": st["A"].clone(), "B": st["B"].clone(), "pairs": dict(st["pairs"]), "T": st["T"], "windows": list(st["windows"])}
        else:
            if st["T"] != m["T"]:
                raise SystemExit(f"{p}: T {st['T']} != {m['T']}")
            m["A"] += st["A"]; m["B"] += st["B"]; m["windows"] += st["windows"]
            for h in "AB":
                m["pairs"][h] += st["pairs"][h]
        del st
    wins = m["windows"]
    if len(wins) != len(set(wins)):
        raise SystemExit("overlapping windows across shards")
    if (OUT / "j_sums.pt").exists() and not (OUT / "j_sums_pilot.pt").exists():
        (OUT / "j_sums.pt").rename(OUT / "j_sums_pilot.pt")
    m.update({"done": len(wins), "next": len(wins), "merged_from": [str(p) for p in paths]})
    torch.save(m, OUT / "j_sums.pt"); print(f"merged {len(paths)} files: {len(wins)} windows")


def compare(pa, pb):
    """Reproduction check: the same windows computed on two machines / row counts (James, 18 Oct: 'are we definite results
    will repro?'). Pass rule (fixed before any comparison): per layer relative Frobenius difference < 1% AND median top-20
    token overlap >= 0.95 for 256 random neurons per sampled layer (split-half across DIFFERENT text gave 0.25-1.0).
    If it fails, rented results are not merged with the 5090 pilot; windows 0-31 are recomputed on the rented GPUs too."""
    a, b = torch.load(pa, weights_only=False), torch.load(pb, weights_only=False)
    if a["windows"] != b["windows"]:
        raise SystemExit(f"different windows: {a['windows'][:4]} vs {b['windows'][:4]}")
    JA = a["A"] + a["B"]; JB = b["A"] + b["B"]; L = JA.shape[0]
    rel = [float((JA[l] - JB[l]).norm() / JB[l].norm()) for l in range(L)]
    model, tok = setup(); dev = next(model.parameters()).device
    g = model.model.norm.weight.float(); WU = model.lm_head.weight.float(); rng = random.Random(5); ov = {}
    for l in range(0, L, 5):
        Wd = model.model.layers[l].mlp.down_proj.weight.float(); V = Wd[:, rng.sample(range(Wd.shape[1]), 256)]
        ta = (WU @ (g[:, None] * (JA[l].to(dev) @ V))).topk(20, dim=0).indices
        tb = (WU @ (g[:, None] * (JB[l].to(dev) @ V))).topk(20, dim=0).indices
        ov[l] = stx.median(len(set(ta[:, i].tolist()) & set(tb[:, i].tolist())) / 20 for i in range(V.shape[1]))
    ok = max(rel) < 0.01 and min(ov.values()) >= 0.95
    txt = (f"compare {Path(pa).name} vs {Path(pb).name} ({len(a['windows'])} windows): relative difference per layer max {max(rel):.4%} "
           f"(median {stx.median(rel):.4%}); top-20 readout overlap " + ", ".join(f"L{l} {v:.2f}" for l, v in ov.items()) +
           f"\n  REPRODUCES (rel < 1% in every layer, overlap >= 0.95): {'YES' if ok else 'NO'}")
    with open(OUT / "jlens_compare.txt", "a", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)


def load_J(which="all"):
    st = torch.load(OUT / "j_sums.pt", weights_only=False)
    if which == "A":
        return st["A"] / st["pairs"]["A"], st
    if which == "B":
        return st["B"] / st["pairs"]["B"], st
    return (st["A"] + st["B"]) / (st["pairs"]["A"] + st["pairs"]["B"]), st


def validate():
    model, tok = setup(); L = model.config.num_hidden_layers; dev = next(model.parameters()).device
    J, st = load_J(); g = model.model.norm.weight.float(); WU = model.lm_head.weight
    held = windows(24, seed=401, skip=set(st["windows"]))
    res = {l: {"J": [], "logit": []} for l in range(L)}
    for _, w in held:
        ids = torch.tensor(w, device=dev)[None]; hs = {}
        hk = [model.model.layers[l].register_forward_hook(lambda m, a, o, l=l: hs.__setitem__(l, layer_out(o).detach())) for l in range(L)]
        try:
            with torch.no_grad():
                model.model(input_ids=ids[:, :T])
        finally:
            for h in hk:
                h.remove()
        for l in range(L):
            Jl = J[l].to(dev)
            for t in range(8, T - 1, 8):
                fut = ids[0, t + 1:t + 17]
                h = hs[l][0, t].float()
                for name, v in (("J", Jl @ h), ("logit", h)):
                    lg = (WU.float() @ (g * v / v.norm()))
                    rank = (lg[None, :] > lg[fut][:, None]).sum(1)       # rank of each future token
                    res[l][name].append(int(rank.min()))
    lines = ["J1 validity - median best rank of the next 16 actual tokens (lower = better), held-out web text:"]
    for l in range(L):
        lines.append(f"  layer {l:2d}: J-lens {stx.median(res[l]['J']):8.0f} | logit lens {stx.median(res[l]['logit']):8.0f}")
    mid = [l for l in range(12, 31)]
    better = sum(stx.median(res[l]["J"]) < stx.median(res[l]["logit"]) for l in mid)
    lines.append(f"  J-lens better in {better}/{len(mid)} middle layers (12-30); last layer equal: {stx.median(res[L - 1]['J']) == stx.median(res[L - 1]['logit'])}")
    # split-half reliability on random neurons' output vectors
    JA, _ = load_J("A"); JB, _ = load_J("B"); rng = random.Random(7); ov = {}
    for l in range(0, L, 5):
        Wd = model.model.layers[l].mlp.down_proj.weight.float()
        ns = rng.sample(range(Wd.shape[1]), 64); V = Wd[:, ns]
        ta = (WU.float() @ (g[:, None] * (JA[l].to(dev) @ V))).topk(20, dim=0).indices
        tb = (WU.float() @ (g[:, None] * (JB[l].to(dev) @ V))).topk(20, dim=0).indices
        ov[l] = stx.median(len(set(ta[:, i].tolist()) & set(tb[:, i].tolist())) / 20 for i in range(64))
    lines.append("split-half reliability (median top-20 overlap, J from even vs odd windows, 64 random neurons): " + ", ".join(f"L{l} {v:.2f}" for l, v in ov.items()))
    txt = "\n".join(lines); (OUT / "jlens_validate.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)
    (OUT / "jlens_validate.json").write_text(json.dumps({"ranks": res, "split_half": ov}), encoding="utf-8")


def neurons():
    model, tok = setup(); L = model.config.num_hidden_layers; dev = next(model.parameters()).device
    J, _ = load_J(); g = model.model.norm.weight.float(); WU = model.lm_head.weight.float()
    for l in range(L):
        Wd = model.model.layers[l].mlp.down_proj.weight.float(); Jl = J[l].to(dev); N = Wd.shape[1]
        out = {k: [] for k in ("j_top", "j_top_lg", "j_bot", "j_bot_lg", "j_z", "l_top", "l_top_lg", "l_bot", "l_bot_lg", "l_z")}
        for c0 in range(0, N, 1024):
            V = Wd[:, c0:c0 + 1024]
            for pre, M in (("j", Jl @ V), ("l", V)):
                lg = WU @ (g[:, None] * M)                                    # [vocab, chunk]
                mu, sd = lg.mean(0), lg.std(0)
                t = lg.topk(30, dim=0); b = (-lg).topk(30, dim=0)
                out[f"{pre}_top"].append(t.indices.T.cpu()); out[f"{pre}_top_lg"].append(t.values.T.half().cpu())
                out[f"{pre}_bot"].append(b.indices.T.cpu()); out[f"{pre}_bot_lg"].append((-b.values).T.half().cpu())
                out[f"{pre}_z"].append(((t.values[0] - mu) / sd).cpu())
        np.savez_compressed(OUT / f"jlens_neurons_L{l}.npz", **{k: torch.cat(v).numpy() for k, v in out.items()})
        print(f"layer {l} done", flush=True)


def tests():
    from transformers import AutoTokenizer
    from config import DEFAULT_MODEL, model_path
    import re
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    Z = {l: np.load(OUT / f"jlens_neurons_L{l}.npz") for l in range(36)}
    dec = lambda ids: [tok.decode([int(i)]).strip().lower() for i in ids]
    L = []
    # J3 overlap by layer
    rng = random.Random(11); ov = {}
    for l in range(36):
        ns = rng.sample(range(12288), 200)
        ov[l] = stx.median(len(set(Z[l]["j_top"][n, :15]) & set(Z[l]["l_top"][n, :15])) / 15 for n in ns)
    early = stx.median(ov[l] for l in range(20)); late = stx.median(ov[l] for l in range(28, 36))
    L.append("J3 J-lens vs logit-lens top-15 overlap (200 random neurons per layer): " + ", ".join(f"L{l} {v:.2f}" for l, v in ov.items()))
    L.append(f"   layers 0-19 median {early:.2f} (pred < 0.20) | layers 28-35 median {late:.2f} (pred >= 0.50) -> {'HOLDS' if early < 0.2 and late >= 0.5 else 'FAILS'}")
    # J2 entity neurons
    F2 = Path("D:/already-clean-data/facts2")
    E = json.loads((F2 / "entities.json").read_text(encoding="utf-8"))
    words = {}
    for key, facts in E.items():
        e = key.split("|", 1)[1]
        ws = set(w.lower() for w in re.findall(r"\w{3,}", e))
        for v in (facts or {}).values():
            ws |= set(w.lower() for w in re.findall(r"\w{3,}", str(v)))
        words[e] = ws - {"the", "and", "century", "of"}
    units = json.loads((F2 / "entity_units.json").read_text(encoding="utf-8"))

    def hit(ids, ws):
        for t in dec(ids):
            if len(t) >= 3 and (t in ws or (len(t) >= 4 and any(w.startswith(t) for w in ws))):
                return True
        return False
    jh, lh, rh, ex = 0, 0, 0, []
    for u, e in units:
        l, n = int(u[1:].split(":")[0]), int(u.split(":")[1]); ws = words.get(e, set()); r = rng.randrange(12288)
        a = hit(Z[l]["j_top"][n, :30], ws) or hit(Z[l]["j_bot"][n, :20], ws); b = hit(Z[l]["l_top"][n, :30], ws) or hit(Z[l]["l_bot"][n, :20], ws)
        c = hit(Z[l]["j_top"][r, :30], ws) or hit(Z[l]["j_bot"][r, :20], ws)
        jh += a; lh += b; rh += c
        if a and len(ex) < 12:
            ex.append(f"{u} {e}: {dec(Z[l]['j_top'][n, :10])}")
    n = len(units)
    L.append(f"\nJ2 entity neurons (facts2, n={n}): entity word in top-30 / bottom-20 J-lens tokens {jh / n:.0%} | logit lens {lh / n:.0%} | random same-layer neuron (J) {rh / n:.0%}")
    L.append(f"   prediction (J >= 30%, random <= 5%, logit <= 10%): {'HOLDS' if jh / n >= 0.3 and rh / n <= 0.05 and lh / n <= 0.1 else 'FAILS'}")
    L += ["   " + x for x in ex]
    L.append("\nJ4 era neurons, J-lens top 15 (and logit lens):")
    for u in ("L31:8930", "L28:9936", "L28:8912", "L33:12258"):
        l, n_ = int(u[1:].split(":")[0]), int(u.split(":")[1])
        L.append(f"   {u} J top: {dec(Z[l]['j_top'][n_, :15])} | J bottom: {dec(Z[l]['j_bot'][n_, :8])}")
        L.append(f"   {' ' * len(u)} logit top: {dec(Z[l]['l_top'][n_, :10])}")
    txt = "\n".join(L); (OUT / "jlens_tests.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    a = sys.argv[1]
    arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    nw = arg("--windows", 32)
    {"estimate": lambda: estimate(nw, arg("--start", 0), arg("--end", nw)), "validate": validate, "neurons": neurons, "tests": tests,
     "merge": lambda: merge([Path(p) for p in sys.argv[2:]]), "compare": lambda: compare(sys.argv[2], sys.argv[3])}[a]()
