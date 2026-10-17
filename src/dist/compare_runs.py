"""Compare two runs' outputs (e.g. 5090 vs a rented GPU, or 2 shards merged vs 1 shard): are they the same within
bf16 tolerance?

  observe (merged):  per neuron and pole, overlap of the top-K (doc, position) sets; value differences on the shared
                     entries; mean / variance / share-above-threshold differences. Same for attention heads.
  ablate:            rows matched on (neuron, kind, doc, position): KL and activation differences, overlap of the
                     top-20 raised tokens, and of the static writer readout.
Prints a summary and writes it as JSON with --out.
Usage: python src/dist/compare_runs.py <store A> <store B> [--k 10] [--out file.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dist.storage import open_store  # noqa: E402
from dist.manifest import task_key  # noqa: E402


def overlap(da, pa, db, pb, k):
    """Mean share of shared (doc, pos) entries between rows of two top-k lists (first k columns)."""
    ca = da[:, :k].astype(np.int64) * 4096 + pa[:, :k]; cb = db[:, :k].astype(np.int64) * 4096 + pb[:, :k]
    shared = np.array([len(np.intersect1d(x, y)) for x, y in zip(ca, cb)])
    return shared / k, ca, cb


def value_diff(ca, va, cb, vb):
    """Absolute and relative differences of values at entries present in both lists."""
    absd, rel = [], []
    for x, u, y, w in zip(ca, va, cb, vb):
        _, i, j = np.intersect1d(x, y, return_indices=True)
        d = np.abs(u[i] - w[j]); absd.append(d); rel.append(d / np.maximum(np.abs(u[i]), 1e-6))
    absd, rel = np.concatenate(absd), np.concatenate(rel)
    return {"max_abs": float(absd.max(initial=0)), "p99_rel": float(np.quantile(rel, 0.99)) if len(rel) else None,
            "median_rel": float(np.median(rel)) if len(rel) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--k", type=int, default=None, help="compare the first k entries (default: all)")
    ap.add_argument("--out")
    args = ap.parse_args()
    A, B = open_store(args.a), open_store(args.b)
    res = {"a": args.a, "b": args.b, "layers": {}}
    mk = task_key("merge:observe")
    worst = {"top_overlap_min_layer_mean": 1.0}
    l = 0
    while (x := A.get_npz(f"{mk}/L{l:02d}.npz")) is not None:
        y = B.get_npz(f"{mk}/L{l:02d}.npz")
        if y is None:
            break
        k = args.k or x["top_v"].shape[1]
        r = {}
        for pole in ("top", "bot"):
            ov, ca, cb = overlap(x[f"{pole}_d"], x[f"{pole}_p"], y[f"{pole}_d"], y[f"{pole}_p"], k)
            r[pole] = {"overlap_mean": float(ov.mean()), "overlap_min": float(ov.min()),
                       "share_identical_sets": float((ov == 1).mean()),
                       **value_diff(ca, x[f"{pole}_v"][:, :k], cb, y[f"{pole}_v"][:, :k])}
            worst["top_overlap_min_layer_mean"] = min(worst["top_overlap_min_layer_mean"], r[pole]["overlap_mean"])
        for s in ("mean", "var", "frac_pos", "frac_abs_gt"):
            d = np.abs(x[s] - y[s]); r[s + "_max_abs_diff"] = float(d.max())
        r["mean_rel_diff_p99"] = float(np.quantile(np.abs(x["mean"] - y["mean"]) / np.sqrt(np.maximum(x["var"], 1e-12)), 0.99))
        res["layers"][l] = r
        l += 1
    hx, hy = A.get_npz(f"{mk}/heads.npz"), B.get_npz(f"{mk}/heads.npz")
    if hx is not None and hy is not None:
        L, H, K = hx["top_d"].shape; k = args.k or K
        ov, ca, cb = overlap(hx["top_d"].reshape(-1, K), hx["top_p"].reshape(-1, K), hy["top_d"].reshape(-1, K), hy["top_p"].reshape(-1, K), k)
        res["heads"] = {"overlap_mean": float(ov.mean()), "overlap_min": float(ov.min()),
                        **value_diff(ca, hx["top_v"].reshape(-1, K)[:, :k], cb, hy["top_v"].reshape(-1, K)[:, :k])}

    res["ablate"] = {}
    for t in A.get_json("manifest.json")["tasks"]:
        if not t["id"].startswith("ablate:"):
            continue
        p = task_key(t["id"])
        u, v = A.get_npz(f"{p}/ablate.npz"), B.get_npz(f"{p}/ablate.npz")
        if u is None or v is None:
            continue
        key = lambda z: {(int(n), int(k), int(d), int(q), int(md)): i for i, (n, k, d, q, md) in  # noqa: E731
                         enumerate(zip(z["neuron"], z["kind"], z["doc"], z["pos"], z["mode"] if "mode" in z else [0] * len(z["kl"])))}
        ka, kb = key(u), key(v)
        both = sorted(set(ka) & set(kb))
        ia = np.array([ka[x] for x in both]); ib = np.array([kb[x] for x in both])
        kl_a, kl_b = u["kl"][ia], v["kl"][ib]
        up = np.mean([len(np.intersect1d(u["up_ids"][i], v["up_ids"][j])) / u["up_ids"].shape[1] for i, j in zip(ia, ib)])
        wa, wb = A.get_npz(f"{p}/writer.npz"), B.get_npz(f"{p}/writer.npz")
        wov = np.mean([len(np.intersect1d(x, y)) / len(x) for x, y in zip(wa["up_ids"], wb["up_ids"])])
        res["ablate"][t["id"]] = {
            "rows_a": len(ka), "rows_b": len(kb), "rows_matched": len(both),
            "kl_max_abs_diff": float(np.abs(kl_a - kl_b).max()),
            "kl_rel_diff_median": float(np.median(np.abs(kl_a - kl_b) / np.maximum(kl_a, 1e-9))),
            "kl_spearman": float(np.corrcoef(np.argsort(np.argsort(kl_a)), np.argsort(np.argsort(kl_b)))[0, 1]),
            "act_max_abs_diff": float(np.abs(u["act"][ia] - v["act"][ib]).max()),
            "up20_overlap_mean": float(up), "writer_up20_overlap_mean": float(wov)}
    res["summary"] = {
        "layers": len(res["layers"]),
        "neuron_top_overlap_mean": float(np.mean([r[p]["overlap_mean"] for r in res["layers"].values() for p in ("top", "bot")])),
        "neuron_sets_identical_share": float(np.mean([r[p]["share_identical_sets"] for r in res["layers"].values() for p in ("top", "bot")])),
        "neuron_value_max_abs_diff": float(max(r[p]["max_abs"] for r in res["layers"].values() for p in ("top", "bot"))),
        "neuron_value_median_rel_diff": float(np.median([r[p]["median_rel"] for r in res["layers"].values() for p in ("top", "bot")])),
        **worst}
    print(json.dumps({k: res[k] for k in ("summary", "heads", "ablate") if k in res}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
