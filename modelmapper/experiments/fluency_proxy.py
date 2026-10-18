"""Can a free, local check replace Jev's fluency judge? (budget relief; validated against ~12k cached Jev verdicts)

For each reply Jev judged ("Is this text fluent, grammatical English that makes sense?", P(yes) >= 0.5 = fluent):
  lp     mean token log-probability of the reply under the UNSTEERED model (reply text alone)
  uniq   distinct-token fraction (repetition loops score low; LMs often find loops *likely*, so lp alone may fail)
  nonascii  share of non-ASCII characters (code-switching into other scripts)
Reports AUC of each and of a simple combination, on a random half, then checks it on the other half.

Usage: python src/fluency_proxy.py [max_replies=4000]
"""
import json
import random
import sys
from pathlib import Path

import torch

from modelmapper.core.config import DEFAULT_MODEL  # noqa: E402
from modelmapper.core.neurons import load, DEVICE  # noqa: E402
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT


def auc(pos, neg):
    s = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, rank_sum = 1, 0.0
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        rank_sum += sum(avg for k in range(i, j + 1) if s[k][1] == 1)
        i = j + 1
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    n_max = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    rows = []
    for line in (ROOT / "results/people_fluency.cache.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["k"].strip():
            rows.append((d["k"], d["v"]))
    rng = random.Random(0); rng.shuffle(rows); rows = rows[:n_max]
    model, tok = load(DEFAULT_MODEL)
    feats = []
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            batch = rows[i:i + 16]
            enc = tok([t for t, _ in batch], return_tensors="pt", padding=True, truncation=True, max_length=200).to(DEVICE)
            lp = model(**enc).logits.float().log_softmax(-1)
            ids = enc.input_ids; mask = enc.attention_mask
            tok_lp = lp[:, :-1].gather(-1, ids[:, 1:, None])[..., 0] * mask[:, 1:]
            mean_lp = (tok_lp.sum(1) / mask[:, 1:].sum(1).clamp(min=1)).tolist()
            for (t, v), m, row_ids, row_mask in zip(batch, mean_lp, ids.tolist(), mask.tolist()):
                toks = [x for x, k in zip(row_ids, row_mask) if k]
                uniq = len(set(toks)) / max(1, len(toks))
                nonascii = sum(ord(c) > 127 for c in t) / max(1, len(t))
                feats.append({"lp": m, "uniq": uniq, "nonascii": nonascii, "fluent": v >= 0.5})
    half = len(feats) // 2
    a, b = feats[:half], feats[half:]
    for name, f in (("lp", lambda x: x["lp"]), ("uniq", lambda x: x["uniq"]), ("-nonascii", lambda x: -x["nonascii"])):
        print(f"{name:<10} AUC {auc([f(x) for x in a if x['fluent']], [f(x) for x in a if not x['fluent']]):.3f}")
    # simple combined rule, thresholds fitted on half A, tested on half B
    best = None
    for tu in [i / 100 for i in range(20, 80, 2)]:
        for tl in [-i / 10 for i in range(5, 60, 2)]:
            for tn in (0.01, 0.03, 0.1, 1.0):
                pred = [x["uniq"] >= tu and x["lp"] >= tl and x["nonascii"] < tn for x in a]
                acc = sum(p == x["fluent"] for p, x in zip(pred, a)) / len(a)
                if best is None or acc > best[0]:
                    best = (acc, tu, tl, tn)
    acc, tu, tl, tn = best
    pred = [x["uniq"] >= tu and x["lp"] >= tl and x["nonascii"] < tn for x in b]
    tp = sum(p and x["fluent"] for p, x in zip(pred, b)); fp = sum(p and not x["fluent"] for p, x in zip(pred, b))
    fn = sum(not p and x["fluent"] for p, x in zip(pred, b))
    base = sum(x["fluent"] for x in b) / len(b)
    print(f"rule: uniq >= {tu}, lp >= {tl}, nonascii < {tn} | fit accuracy {acc:.3f} | held-out accuracy "
          f"{sum(p == x['fluent'] for p, x in zip(pred, b)) / len(b):.3f} (always-'fluent' baseline {max(base, 1-base):.3f}) | "
          f"precision {tp / max(1, tp + fp):.3f} recall {tp / max(1, tp + fn):.3f} | n={len(feats)}")
    (ROOT / "results/fluency_proxy.json").write_text(json.dumps({"rule": {"uniq": tu, "lp": tl, "nonascii": tn}, "fit_acc": acc}), encoding="utf-8")


if __name__ == "__main__":
    main()
