"""Gate for batched knockouts (KNOCK_BATCH > 0) in map_loop.py. Rules stated before the run (17 Oct):
  - status, recall, answer form, k_mean, k_median, k_flip IDENTICAL to the original pipeline (wave_free.json, coverage-2)
  - drop curves (drops, random_drops) within 0.05 at every k
  - top-10 neurons the same SET as rank_save_fame2.json; top-30 overlap >= 28 (near-tie order swaps reported)
  - single-head knockout drops within 0.05 of the validated unbatched run (rank_save_repro.json), per head and prompt
bf16 kernels may round differently with batch size, hence tolerances rather than identity for curves and order.
Usage: python src/map_loop_check_batched.py <tag>   (exit 1 if any rule fails) -> results/map_loop_check_batched.txt
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    tag = sys.argv[1]
    new = jl(f"results/maploop_repro/map_loop_{tag}.json"); wf = jl("results/concepts_fame2/wave_free.json")
    rk = jl("results/concepts_fame2/rank_save_fame2.json"); nrk = jl(f"results/maploop_repro/rank_save_{tag}.json")
    ref_heads = jl("results/maploop_repro/rank_save_repro.json")
    bad, notes = [], []
    for c, r in new.items():
        o = wf[c]
        for k in ("status", "recall", "answer_form", "k_mean", "k_median", "k_flip"):
            if r.get(k) != o.get(k):
                bad.append(f"{c}: {k} {o.get(k)} -> {r.get(k)}")
        for k in ("drops", "random_drops"):
            if k in o and r.get(k) is not None:
                d = max(abs(a - b) for a, b in zip(o[k], r[k]))
                if d > 0.05:
                    bad.append(f"{c}: {k} differ by {d:.3f}")
        if c in rk and c in nrk:
            a = [e["neuron"] for e in rk[c]["top"]]; b = [e["neuron"] for e in nrk[c]["top"]]
            if set(a[:10]) != set(b[:10]):
                bad.append(f"{c}: top-10 set differs")
            ov = len(set(a) & set(b))
            if ov < 28:
                bad.append(f"{c}: top-30 overlap {ov}")
            if a != b:
                notes.append(f"{c}: top-30 order differs (overlap {ov})")
        if c in ref_heads and c in nrk:
            hr, hn = ref_heads[c]["head_knockout_per_prompt"], nrk[c]["head_knockout_per_prompt"]
            if set(hr) != set(hn):
                bad.append(f"{c}: different top-50 heads")
            else:
                d = max(abs(x - y) for h in hr for x, y in zip(hr[h], hn[h]))
                if d > 0.05:
                    bad.append(f"{c}: head knockout drops differ by {d:.3f}")
    L = [f"BATCHED GATE ({tag}, {len(new)} concepts): {'PASS' if not bad else 'FAIL'}"] + [f"  {b}" for b in bad] + [f"  note: {n}" for n in notes]
    try:
        meta_u = jl("results/maploop_repro/map_loop_repro.json"); meta_b = new
        t = lambda d, k: sum(r["timing_s"].get(k, 0) for r in d.values() if r["status"] == "ok")
        for k in ("G2_single_knockouts", "G2_head_knockouts"):
            L.append(f"  {k}: unbatched {t(meta_u, k):.0f} s -> batched {t(meta_b, k):.0f} s (x{t(meta_u, k) / max(t(meta_b, k), 1e-9):.1f})")
    except Exception as e:  # noqa: BLE001
        L.append(f"  timing comparison unavailable: {e!r}")
    txt = "\n".join(L)
    (ROOT / "results/map_loop_check_batched.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
