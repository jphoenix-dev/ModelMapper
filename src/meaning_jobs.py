"""Stage D of the meaning test: what JOB does the neuron do in each firing context? Luna-free. (GPU; no API)

James: judge "same meaning" by what the neuron does in Qwen, not by what a labeller calls it. For every valid unit
in meaning_judge.json and each of its 30 firing contexts: run Qwen on the context up to the firing token, once as is
and once with the neuron's activation set to 0 at that position (spliced units: the neuron the snippet came from).
Saved per context (everything, per James's save-all rule):
  act       the neuron's activation there
  total     change in the final residual stream (pre-final-norm) at that position, fp16 [4096]  (base - ablated)
  direct    the neuron's own write there, act x down_proj column - recomputable, saved for convenience
  indirect  total - direct: how the rest of the network reacts (where different 'jobs' would show)
  kl        KL(base || ablated) of the next-token distribution at that position
  up/down   the 20 tokens whose log-prob the neuron raises most / lowers most (ids and deltas)
Analysis (Luna-free clusters in Qwen's own context space vs these effects; agreement with Luna's groups; spliced
calibration) is in meaning_jobs_summary.py.
Output: results/meaning_jobs.npz (arrays) + results/meaning_jobs.json (per-context scalars, token lists). Resumable.
Usage: python src/meaning_jobs.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_MODEL
from neurons import DEVICE, load, mlps

ROOT = Path(__file__).resolve().parents[1]


def main():
    J = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    units = {k: v for k, v in J.items() if v.get("valid")}
    model, tok = load(DEFAULT_MODEL)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out_p = ROOT / "results/meaning_jobs.json"; arr_p = ROOT / "results/meaning_jobs.npz"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    arrays = dict(np.load(arr_p)) if arr_p.exists() and out else {}
    for ui, (key, v) in enumerate(units.items()):
        if key in out:
            continue
        rows, T, D, I = [], [], [], []
        for (di, p), src in zip(v["doc_pos"], v["source"]):
            u = v["units"][src]; L, n = (int(x) for x in u[1:].split(":"))
            ids = torch.tensor(toks[starts[di]:starts[di] + p + 1].astype(np.int64), device=DEVICE)[None]
            st = {}

            def run(kill):
                hk = [model.model.norm.register_forward_pre_hook(lambda m, a: st.__setitem__("hf", a[0][0, -1].float()))]

                def h(m, a):
                    st["act"] = a[0][0, -1, n].float().item()
                    if kill:
                        x = a[0].clone(); x[0, -1, n] = 0.0; return (x,)
                hk.append(mlps(model)[L].down_proj.register_forward_pre_hook(h))
                try:
                    with torch.no_grad():
                        lg = model(input_ids=ids).logits[0, -1].float()
                finally:
                    for x in hk:
                        x.remove()
                return lg, st["hf"].clone(), st["act"]
            lb, hb, act = run(False); la, ha, _ = run(True)
            total = hb - ha
            direct = act * mlps(model)[L].down_proj.weight[:, n].float()
            lpb, lpa = lb.log_softmax(-1), la.log_softmax(-1)
            kl = (lpb.exp() * (lpb - lpa)).sum().item()
            dlp = lpb - lpa
            up = dlp.topk(20); dn = (-dlp).topk(20)
            rows.append({"act": round(act, 4), "kl": round(kl, 6), "src": src,
                         "up": [[tok.decode([i]), round(x, 4)] for i, x in zip(up.indices.tolist(), up.values.tolist())],
                         "down": [[tok.decode([i]), round(-x, 4)] for i, x in zip(dn.indices.tolist(), dn.values.tolist())]})
            T.append(total.half().cpu().numpy()); D.append(direct.half().cpu().numpy()); I.append((total - direct).half().cpu().numpy())
        out[key] = {"kind": v["kind"], "units": v["units"], "contexts": rows}
        arrays[f"{ui}|total"] = np.stack(T); arrays[f"{ui}|direct"] = np.stack(D); arrays[f"{ui}|indirect"] = np.stack(I)
        out[key]["array_index"] = ui
        print(f"{key[:55]:<55} median KL {np.median([r['kl'] for r in rows]):.4f} | top up {rows[0]['up'][:3]}", flush=True)
        if len(out) % 10 == 0:
            out_p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8"); np.savez_compressed(arr_p, **arrays)
    out_p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8"); np.savez_compressed(arr_p, **arrays)


if __name__ == "__main__":
    main()
