"""H3: repeat / induction neurons - neurons that fire when the text repeats, and help copy it. (MECHANISMS.md H3)
CPU, Qwen3-0.6B.

Sequences: 40 random tokens R (drawn uniformly from tokens seen >= 3 times in 200 FineWeb windows), then
  repeat  = R + R          (the second half repeats the first: the model can copy)
  control = R + R'         (R' = 40 fresh random tokens: same first half, same positions, nothing to copy)
The first halves are identical, so any difference is caused by the repeat (checked: first-half activations must match).
Per neuron: mean activation over the second half (positions 40-79), paired difference repeat - control per sequence,
effect size d = mean(diff) / sd(diff). Selection on seeds 0-29, confirmation on seeds 30-59.
Causal: on seeds 30-59, zero the top 20 confirmed repeat neurons (largest selection d among those with held-out d >= 2)
in the second half only; change in loss on the second copy (predicting tokens 41-79). Null: 20 random neurons drawn
from the same layers (one per selected neuron). Also measured on the control sequences (nothing to copy).
Pass rule (before running): >= 20 neurons with held-out d >= 2; zeroing the top 20 raises second-copy loss by >= 3x the
random set's rise.
Predictions (MECHANISMS.md, before running):
  P3a >= 20 neurons with held-out d >= 2
  P3b their median layer is between 8 and 20 (of 28)
  P3c zeroing the top 20 raises second-copy loss >= 3x the random set and by >= 0.05 nats/token
Output: D:/already-clean-data/mechanisms/repeat/{summary.json, neurons.json}
Usage: python -X utf8 src/mechanisms/repeat_neurons.py

REPEAT_K=200 (EXPLORATORY, added after the pre-registered K=20 run: 3,688 neurons confirmed but zeroing 20 raised
second-copy loss by only 0.015 nats): same test zeroing the top K; output repeat_k200/. Asks whether copying is spread
over many neurons rather than absent.
"""
import os
import random
import statistics as stx
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load06, mlp_hook, save, web_windows  # noqa: E402
import torch  # noqa: E402

N, HALF = 60, 40
K = int(os.environ.get("REPEAT_K", "20"))
OD = "repeat" if K == 20 else f"repeat_k{K}"


def main():
    t0 = time.time()
    model, tok = load06()
    L = model.config.num_hidden_layers; W = model.config.intermediate_size
    cnt = Counter(web_windows(tok, 200, 128, seed=7).flatten().tolist())
    vocab = sorted(t for t, c in cnt.items() if c >= 3)
    rep, ctl = [], []
    for s in range(N):
        rng = random.Random(s)
        R = [rng.choice(vocab) for _ in range(HALF)]; R2 = [rng.choice(vocab) for _ in range(HALF)]
        rep.append(R + R); ctl.append(R + R2)
    rep, ctl = torch.tensor(rep), torch.tensor(ctl)

    store = {}
    hks = [mlp_hook(model, l, lambda m, a, l=l: store.__setitem__(l, a[0])) for l in range(L)]
    M = torch.zeros(2, N, L, W); first_diff = 0.0; FHs = []; CHECK = (0, L // 2, L - 1)
    for gi, X in enumerate((rep, ctl)):
        for bi, s in enumerate(range(0, N, 10)):
            model.model(input_ids=X[s:s + 10])
            for l in range(L):
                M[gi, s:s + 10, l] = store[l][:, HALF:].mean(1)
            if gi == 0:
                FHs.append({l: store[l][:, :HALF].clone() for l in CHECK})
            else:
                first_diff = max(first_diff, max((store[l][:, :HALF] - FHs[bi][l]).abs().max().item() for l in CHECK))
    for h in hks:
        h.remove()
    print("activations done", time.time() - t0, "max first-half difference (must be ~0):", first_diff, flush=True)
    diff = M[0] - M[1]                                   # [N, L, W]
    dA = diff[:30].mean(0) / diff[:30].std(0).clamp(min=1e-6)
    dB = diff[30:].mean(0) / diff[30:].std(0).clamp(min=1e-6)
    conf_pos = ((dA >= 2) & (dB >= 2)); conf_neg = ((dA <= -2) & (dB <= -2))
    heldout_pos = (dB >= 2)
    layers_pos = conf_pos.nonzero()[:, 0].tolist()
    print("confirmed up:", int(conf_pos.sum()), "down:", int(conf_neg.sum()), "held-out d>=2 (any selection):", int(heldout_pos.sum()), flush=True)

    sel = [(l, n) for l, n in conf_pos.nonzero().tolist()]
    sel.sort(key=lambda u: -dA[u].item()); top = sel[:K]
    rng = random.Random(99); rand = []
    for l, _ in top:
        while True:
            u = (l, rng.randrange(W))
            if not conf_pos[u] and u not in rand:
                rand.append(u); break

    def loss2(X, kill):
        by = {}
        for l, n in kill:
            by.setdefault(l, []).append(n)
        hk = []
        for l, ns in by.items():
            idx = torch.tensor(ns)

            def h(m, a, idx=idx):
                y = a[0].clone(); y[:, HALF:, idx] = 0.0
                return (y,)
            hk.append(mlp_hook(model, l, h))
        try:
            out = []
            for s in range(0, len(X), 10):
                lg = model(input_ids=X[s:s + 10]).logits[:, HALF:-1].float()
                out.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), X[s:s + 10, HALF + 1:].reshape(-1),
                                                             reduction="none").view(lg.shape[0], -1).mean(1))
        finally:
            for x in hk:
                x.remove()
        return torch.cat(out)

    held = slice(30, N)
    res = {}
    for name, X in (("repeat", rep[held]), ("control", ctl[held])):
        c = loss2(X, []); t = loss2(X, top); r = loss2(X, rand)
        res[name] = {"clean_loss": c.mean().item(), "top_rise": (t - c).mean().item(), "random_rise": (r - c).mean().item(),
                     "top_rise_per_seq": (t - c).tolist(), "random_rise_per_seq": (r - c).tolist()}
        print(name, {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)}, flush=True)
    rr = res["repeat"]
    S = {"n_seq": N, "half": HALF, "vocab_size_used": len(vocab), "first_half_max_abs_diff": first_diff,
         "n_confirmed_up": int(conf_pos.sum()), "n_confirmed_down": int(conf_neg.sum()),
         "n_heldout_d_ge2": int(heldout_pos.sum()),
         "confirmed_up_by_layer": sorted(Counter(layers_pos).items()),
         "P3a_pass": int(conf_pos.sum()) >= 20,
         "P3b_median_layer": stx.median(layers_pos) if layers_pos else None,
         "P3b_pass": bool(layers_pos) and 8 <= stx.median(layers_pos) <= 20,
         "top": [{"unit": f"L{l}:{n}", "dA": dA[l, n].item(), "dB": dB[l, n].item(),
                    "mean_act_repeat": M[0, :, l, n].mean().item(), "mean_act_control": M[1, :, l, n].mean().item()} for l, n in top],
         "random": [f"L{l}:{n}" for l, n in rand],
         "causal": res,
         "P3c_ratio": rr["top_rise"] / rr["random_rise"] if rr["random_rise"] > 0 else None,
         "P3c_pass": rr["top_rise"] >= 0.05 and rr["top_rise"] >= 3 * max(rr["random_rise"], 1e-9)}
    save(f"{OD}/summary.json", S)
    save(f"{OD}/neurons.json", {"confirmed_up": [{"unit": f"L{l}:{n}", "dA": round(dA[l, n].item(), 3), "dB": round(dB[l, n].item(), 3)} for l, n in sel],
                                 "confirmed_down": [{"unit": f"L{l}:{n}", "dA": round(dA[l, n].item(), 3), "dB": round(dB[l, n].item(), 3)}
                                                    for l, n in conf_neg.nonzero().tolist()]})
    print({k: v for k, v in S.items() if k not in ("top", "causal", "random")})


if __name__ == "__main__":
    main()
