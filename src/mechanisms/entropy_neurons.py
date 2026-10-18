"""H2: confidence (entropy) neurons - late neurons that change how sure the model is, not what it says.
(MECHANISMS.md H2; Stolfo et al. 2024 'Confidence regulation neurons'.) CPU, Qwen3-0.6B.

Weights: logits = W_U (g * r / rms(r)), r = residual before the final norm, g = the norm's weights. A neuron writes
a * w into r (w = its column of down_proj). Its push on the logits is W_U (g * w). The unembedding W_U has a few
near-null directions (tiny singular values): writing there changes almost no logit directly, but it changes rms(r)
and so scales ALL logits - i.e. the model's confidence.
  null fraction = |V_bottom^T (g*w)|^2 / |g*w|^2, V_bottom = W_U's bottom 2% right-singular vectors (20 of 1,024)
  candidates    = top 10 by |w| x null fraction among the last 3 layers (25-27)
Causal (web text, FineWeb, windows 0-63 of 128 tokens; means from windows 64-95): each neuron alone set to its mean
activation (mean ablation) at every position; at 32 sampled positions per window (32..125 step 3) compare with the
clean run:
  total   effect = full forward with the ablation
  direct  effect = clean residual + the neuron's own change (mean - a) * w, normalised with the CLEAN rms
                   (what the neuron would do if the final norm did not re-scale)
  metrics = mean |change in entropy| (nats), mean change in entropy, mean change in loss, top-1 agreement with clean
Nulls: 30 random neurons from layers 25-27 (outside the candidates); and a sharper control: the 10 largest-|w| neurons in
layers 25-27 whose null fraction is below the median (big, but writing in ordinary directions).
Pass rule (before running): a confidence neuron has mean |dH_total| > 5x the random 95th percentile AND
mean |dH_total| >= 2x mean |dH_direct|.
Predictions (MECHANISMS.md, before running):
  P2a >= 1 candidate in the last layer (27) passes
  P2b passing neurons' null fraction >= 10x the median neuron's (layers 25-27)
  P2c passing neurons keep top-1 agreement with the clean run >= 95% (they change sureness, not the word)
Output: D:/already-clean-data/mechanisms/entropy/{weights.json, causal.json}
Usage: python -X utf8 src/mechanisms/entropy_neurons.py          (RULE=v1, the pre-registered run)

RULE=v2 (EXPLORATORY, designed after v1's first candidates turned out to be near-silent neurons - sd of activation
0.03-0.6 - so they write almost nothing anywhere): candidates = top 10 by sd(activation) x |w| x null fraction, i.e.
the typical size of what the neuron actually writes into the null directions; control = top 10 by sd x |w| among
neurons with null fraction below the median; same causal test, same pass rule; output entropy_v2/.
Usage: RULE=v2 python -X utf8 src/mechanisms/entropy_neurons.py
"""
import os
import random
import statistics as stx
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load06, mlp_hook, save, web_windows  # noqa: E402
import torch  # noqa: E402

LAYERS = [25, 26, 27]
POS = list(range(32, 126, 3))
RULE = os.environ.get("RULE", "v1")
OD = "entropy" if RULE == "v1" else "entropy_v2"


def main():
    t0 = time.time()
    model, tok = load06()
    WU = model.lm_head.weight.float()                       # [V, d] (tied with the embedding in 0.6B)
    g = model.model.norm.weight.float()
    d = WU.shape[1]
    evals, V = torch.linalg.eigh(WU.T @ WU)                 # ascending; singular values = sqrt(evals)
    nb = max(1, round(0.02 * d)); Vb = V[:, :nb]
    rows = []
    for l in LAYERS:
        Wd = model.model.layers[l].mlp.down_proj.weight.float()   # [d, W]
        gw = g[:, None] * Wd
        nf = ((Vb.T @ gw) ** 2).sum(0) / (gw ** 2).sum(0)
        nrm = Wd.norm(dim=0)
        for n in range(Wd.shape[1]):
            rows.append({"unit": f"L{l}:{n}", "layer": l, "n": n, "norm": nrm[n].item(), "null_frac": nf[n].item()})
    med_nf = stx.median(r["null_frac"] for r in rows); med_norm = stx.median(r["norm"] for r in rows)

    X = web_windows(tok, 96, 128, seed=0)
    Xm, Xe = X[64:], X[:64]
    # mean (and sd) activation of every neuron in LAYERS over the mean-windows
    means, sds = {}, {}
    store = {}
    hks = [mlp_hook(model, l, lambda m, a, l=l: store.__setitem__(l, a[0])) for l in LAYERS]
    acc = {l: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for l in LAYERS}; cnt = 0
    acc2 = {l: torch.zeros(model.config.intermediate_size, dtype=torch.float64) for l in LAYERS}
    for s in range(0, len(Xm), 8):
        model.model(input_ids=Xm[s:s + 8])
        for l in LAYERS:
            acc[l] += store[l].double().sum((0, 1)); acc2[l] += store[l].double().pow(2).sum((0, 1))
        cnt += Xm[s:s + 8].numel()
    for h in hks:
        h.remove()
    for l in LAYERS:
        means[l] = (acc[l] / cnt).float()
        sds[l] = (acc2[l] / cnt - (acc[l] / cnt) ** 2).clamp(min=0).sqrt().float()
    for r in rows:
        r["sd_act"] = sds[r["layer"]][r["n"]].item()
        r["score"] = r["norm"] * r["null_frac"] * (r["sd_act"] if RULE == "v2" else 1.0)
    cands = sorted(rows, key=lambda r: -r["score"])[:10]
    low = [r for r in rows if r["null_frac"] < med_nf]
    ctrl = sorted(low, key=lambda r: -(r["norm"] * (r["sd_act"] if RULE == "v2" else 1.0)))[:10]
    cset = {r["unit"] for r in cands + ctrl}
    rng = random.Random(0)
    rand = rng.sample([r for r in rows if r["unit"] not in cset], 30)
    save(f"{OD}/weights.json", {"n_bottom_dirs": nb, "singular_values_bottom": evals[:nb].clamp(min=0).sqrt().tolist(),
                                  "singular_values_median": evals.clamp(min=0).sqrt().median().item(),
                                  "median_null_frac": med_nf, "median_norm": med_norm,
                                  "candidates": cands, "big_ordinary_controls": ctrl,
                                  "top50_by_score": sorted(rows, key=lambda r: -r["score"])[:50],
                                  "top50_by_null_frac": sorted(rows, key=lambda r: -r["null_frac"])[:50]})
    print("weights done", time.time() - t0, [(c["unit"], round(c["null_frac"], 3), round(c["norm"], 2)) for c in cands], flush=True)


    # clean pass on the eval windows: residual before the final norm at sampled positions, and neuron activations there
    pre = {}
    hn = model.model.norm.register_forward_pre_hook(lambda m, a: pre.__setitem__("r", a[0]))
    P = torch.tensor(POS)

    def run(kill=None):
        """-> residual before final norm at sampled positions [N, d], activations at those positions per layer."""
        R, A = [], {l: [] for l in LAYERS}
        hk = []
        st = {}
        for l in LAYERS:
            hk.append(mlp_hook(model, l, lambda m, a, l=l: st.__setitem__(l, a[0][:, P])))
        if kill is not None:
            l, n = kill

            def h(m, a, l=l, n=n):
                y = a[0].clone(); y[..., n] = means[l][n]
                return (y,)
            hk.append(mlp_hook(model, l, h))
        try:
            for s in range(0, len(Xe), 8):
                model.model(input_ids=Xe[s:s + 8])
                R.append(pre["r"][:, P].reshape(-1, pre["r"].shape[-1]).clone())
                for ll in LAYERS:
                    A[ll].append(st[ll].reshape(-1, st[ll].shape[-1]).clone())
        finally:
            for x in hk:
                x.remove()
        return torch.cat(R), {ll: torch.cat(v) for ll, v in A.items()}

    eps = model.model.norm.variance_epsilon
    tgt = Xe[:, P + 1].reshape(-1)

    def readout(R, rms=None):
        """entropy, loss, top-1 for each row; rms = fixed scale (direct effect) or None (recomputed)."""
        rr = R.float()
        rm = rms if rms is not None else rr.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        H, Lo, T1 = [], [], []
        for s in range(0, len(rr), 256):
            lg = ((rr[s:s + 256] / rm[s:s + 256]) * g) @ WU.T
            lp = lg.log_softmax(-1)
            H.append(-(lp.exp() * lp).sum(-1)); Lo.append(-lp.gather(1, tgt[s:s + 256, None])[:, 0]); T1.append(lp.argmax(-1))
        return torch.cat(H), torch.cat(Lo), torch.cat(T1)

    R0, A0 = run()
    rms0 = R0.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    H0, L0, T0 = readout(R0)
    print("clean done", time.time() - t0, "mean entropy", H0.mean().item(), "loss", L0.mean().item(), flush=True)
    res = {}
    for grp, units in (("candidate", cands), ("big_ordinary_control", ctrl), ("random", rand)):
        for r in units:
            l, n = r["layer"], r["n"]
            R1, _ = run((l, n))
            H1, L1, T1 = readout(R1)
            w = model.model.layers[l].mlp.down_proj.weight[:, n].float()
            Rd = R0.float() + (means[l][n] - A0[l][:, n])[:, None] * w[None, :]
            Hd, Ld, Td = readout(Rd, rms=rms0)
            res[r["unit"]] = {"group": grp, "layer": l, "norm": r["norm"], "null_frac": r["null_frac"],
                              "mean_abs_dH_total": (H1 - H0).abs().mean().item(), "mean_dH_total": (H1 - H0).mean().item(),
                              "mean_abs_dH_direct": (Hd - H0).abs().mean().item(), "mean_dH_direct": (Hd - H0).mean().item(),
                              "mean_dloss_total": (L1 - L0).mean().item(), "mean_dloss_direct": (Ld - L0).mean().item(),
                              "top1_agree_total": (T1 == T0).float().mean().item(),
                              "corr_dH_with_clean_act_minus_mean": torch.corrcoef(torch.stack([(H1 - H0), (A0[l][:, n] - means[l][n])]))[0, 1].item(),
                              "mean_act": means[l][n].item(), "sd_act": A0[l][:, n].std().item()}
            print(grp, r["unit"], {k: round(v, 4) for k, v in res[r["unit"]].items() if isinstance(v, float)}, f"({time.time() - t0:.0f}s)", flush=True)
    hn.remove()
    rnd = sorted(v["mean_abs_dH_total"] for v in res.values() if v["group"] == "random")
    p95 = rnd[int(0.95 * len(rnd))]
    passed = [u for u, v in res.items() if v["group"] == "candidate" and v["mean_abs_dH_total"] > 5 * p95
              and v["mean_abs_dH_total"] >= 2 * v["mean_abs_dH_direct"]]
    S = {"n_tokens": len(H0), "clean_mean_entropy": H0.mean().item(), "clean_mean_loss": L0.mean().item(),
         "random_p95_abs_dH": p95, "passed": passed,
         "P2a_pass": any(res[u]["layer"] == 27 for u in passed),
         "P2b_null_frac_ratio_to_median": {u: res[u]["null_frac"] / med_nf for u in passed},
         "P2b_pass": bool(passed) and all(res[u]["null_frac"] >= 10 * med_nf for u in passed),
         "P2c_top1_agree": {u: res[u]["top1_agree_total"] for u in passed},
         "P2c_pass": bool(passed) and all(res[u]["top1_agree_total"] >= 0.95 for u in passed),
         "per_neuron": res}
    save(f"{OD}/causal.json", S)
    print({k: v for k, v in S.items() if k != "per_neuron"})


if __name__ == "__main__":
    main()
