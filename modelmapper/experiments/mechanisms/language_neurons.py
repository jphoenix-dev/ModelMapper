"""H4: language-identity neurons - neurons that track which language is being written. (MECHANISMS.md H4)
CPU, Qwen3-0.6B.

Data: src/mechanisms/parallel_sentences.json - 40 everyday sentences with the same meaning in English, French, German,
Spanish and Chinese, so content is matched and only the language differs.
Per neuron and sentence: mean activation over the sentence's tokens. Per neuron and language: AUC of that mean for
the language vs the other four. Selection on sentences 0-19, confirmation on sentences 20-39 (different content).
Confirmed = AUC >= 0.95 in both halves (neuron fires MORE for the language).
Causal: for each language, zero its top 20 confirmed neurons (by selection-half effect size) at every position; change in
loss per token on held-out sentences 20-39 of EVERY language. Null: 20 random neurons from the same layers.
Pass rule (before running): for >= 3 of 5 languages, own-language loss rise >= 3x the mean rise on the other languages
AND >= 3x the random set's own-language rise.
Predictions (MECHANISMS.md, before running):
  P4a every language has >= 10 confirmed neurons (held-out AUC >= 0.95)
  P4b their median layer is <= 10 (language is visible in the tokens, so it should be read early)
  P4c the causal rule passes for >= 3 of 5 languages
Output: D:/already-clean-data/mechanisms/language/{summary.json, neurons.json}
Usage: python -X utf8 src/mechanisms/language_neurons.py
"""
import json
import random
import statistics as stx
import sys
import time
from collections import Counter
from pathlib import Path

from modelmapper.experiments.mechanisms.common import load06, mlp_hook, save  # noqa: E402
import torch  # noqa: E402

LANGS = ["en", "fr", "de", "es", "zh"]
K = 20


def auc(X, y):
    """X [n, m], y bool [n] -> AUC per column (prob. a positive outranks a negative)."""
    r = X.argsort(0).argsort(0).double() + 1
    npos = y.sum().item(); nneg = len(y) - npos
    return (r[y].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    t0 = time.time()
    data = json.loads((Path(__file__).resolve().parent / "parallel_sentences.json").read_text(encoding="utf-8"))
    model, tok = load06()
    L = model.config.num_hidden_layers; W = model.config.intermediate_size
    ids = {lg: [tok.encode(s) for s in data[lg]] for lg in LANGS}
    store = {}
    hks = [mlp_hook(model, l, lambda m, a, l=l: store.__setitem__(l, a[0])) for l in range(L)]
    M = torch.zeros(len(LANGS), 40, L * W)
    for li, lg in enumerate(LANGS):
        for i, s in enumerate(ids[lg]):
            model.model(input_ids=torch.tensor([s]))
            M[li, i] = torch.cat([store[l][0].mean(0) for l in range(L)])
    for h in hks:
        h.remove()
    print("activations", time.time() - t0, flush=True)
    halves = {"A": slice(0, 20), "B": slice(20, 40)}
    A = {}
    for h, sl in halves.items():
        X = M[:, sl].reshape(-1, L * W)
        lab = torch.arange(len(LANGS)).repeat_interleave(20)
        A[h] = torch.stack([auc(X, lab == li) for li in range(len(LANGS))])      # [5, L*W]
    XA = M[:, halves["A"]]
    neurons, top = {}, {}
    for li, lg in enumerate(LANGS):
        conf = ((A["A"][li] >= 0.95) & (A["B"][li] >= 0.95)).nonzero()[:, 0]
        own = XA[li].mean(0); oth = torch.cat([XA[j] for j in range(len(LANGS)) if j != li]).mean(0)
        sd = torch.cat([XA[j] for j in range(len(LANGS))]).std(0).clamp(min=1e-6)
        dsel = (own - oth) / sd
        order = conf[dsel[conf].argsort(descending=True)]
        neurons[lg] = [{"unit": f"L{int(u) // W}:{int(u) % W}", "aucA": round(A["A"][li, u].item(), 3),
                        "aucB": round(A["B"][li, u].item(), 3), "d_sel": round(dsel[u].item(), 3)} for u in order]
        top[lg] = [divmod(int(u), W) for u in order[:K]]
        print(lg, "confirmed", len(conf), "median layer", stx.median([int(u) // W for u in conf]) if len(conf) else None, flush=True)
    rng = random.Random(5)
    rand = {lg: [] for lg in LANGS}
    for lg in LANGS:
        for l, _ in top[lg]:
            while True:
                u = (l, rng.randrange(W))
                if u not in top[lg] and u not in rand[lg]:
                    rand[lg].append(u); break

    def lang_loss(kill):
        by = {}
        for l, n in kill:
            by.setdefault(l, []).append(n)
        hk = []
        for l, ns in by.items():
            idx = torch.tensor(ns)

            def h(m, a, idx=idx):
                y = a[0].clone(); y[..., idx] = 0.0
                return (y,)
            hk.append(mlp_hook(model, l, h))
        out = {}
        try:
            for lg in LANGS:
                v = []
                for s in ids[lg][20:]:
                    x = torch.tensor([s]); lgt = model(input_ids=x).logits[0, :-1].float()
                    v.append(torch.nn.functional.cross_entropy(lgt, x[0, 1:]).item())
                out[lg] = v
        finally:
            for x in hk:
                x.remove()
        return out

    clean = lang_loss([])
    causal = {}
    for lg in LANGS:
        if not top[lg]:
            causal[lg] = None; continue
        kt, kr = lang_loss(top[lg]), lang_loss(rand[lg])
        rise_t = {o: stx.mean(a - b for a, b in zip(kt[o], clean[o])) for o in LANGS}
        rise_r = {o: stx.mean(a - b for a, b in zip(kr[o], clean[o])) for o in LANGS}
        other = stx.mean(rise_t[o] for o in LANGS if o != lg)
        causal[lg] = {"n_killed": len(top[lg]), "rise_top": rise_t, "rise_random": rise_r, "own_over_other": rise_t[lg] / other if other > 0 else None,
                      "own_over_random": rise_t[lg] / rise_r[lg] if rise_r[lg] > 0 else None,
                      "passes": rise_t[lg] >= 3 * max(other, 1e-9) and rise_t[lg] >= 3 * max(rise_r[lg], 1e-9),
                      "per_sentence_top": {o: [round(a - b, 4) for a, b in zip(kt[o], clean[o])] for o in LANGS}}
        print(lg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in causal[lg].items() if k in ("own_over_other", "own_over_random", "passes")},
              {o: round(v, 3) for o, v in rise_t.items()}, flush=True)
    layers = {lg: [int(n["unit"][1:].split(":")[0]) for n in neurons[lg]] for lg in LANGS}
    S = {"n_confirmed": {lg: len(neurons[lg]) for lg in LANGS},
         "median_layer": {lg: (stx.median(layers[lg]) if layers[lg] else None) for lg in LANGS},
         "layer_hist": {lg: sorted(Counter(layers[lg]).items()) for lg in LANGS},
         "clean_loss": {lg: stx.mean(clean[lg]) for lg in LANGS},
         "P4a_pass": all(len(neurons[lg]) >= 10 for lg in LANGS),
         "P4b_pass": all(layers[lg] and stx.median(layers[lg]) <= 10 for lg in LANGS),
         "P4c_n_languages_pass": sum(1 for lg in LANGS if causal[lg] and causal[lg]["passes"]),
         "causal": causal, "random_sets": {lg: [f"L{l}:{n}" for l, n in rand[lg]] for lg in LANGS}}
    S["P4c_pass"] = S["P4c_n_languages_pass"] >= 3
    save("language/summary.json", S)
    save("language/neurons.json", neurons)
    print({k: v for k, v in S.items() if k not in ("causal", "random_sets", "layer_hist")})


if __name__ == "__main__":
    main()
