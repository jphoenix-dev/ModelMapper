"""Fan-out smoke test (fanout.py) -> ledger pack. v1 = smoke.jsonl (raw slopes), v2 = smoke2.jsonl (slopes in units of
each late neuron's natural spread, minus the random targets' mean).

Per record (target, kind, mode last|all, background set web|own) and each of its top-20 driven late neurons: a 'fanout'
row unit = target -> unit2 = driven neuron. v2: value = z (signed slope per dose unit / natural sd at dose 0),
noise_floor = the random targets' mean |z| for that late neuron (= |z| - specific; only for non-random targets),
details = slope, specific, sign consistency, R^2, pushed tokens. v1: value = raw slope (activation per dose unit), no
per-edge baseline (v1 had none). rank = position in the saved top-20 (v2 ranked by 'specific', v1 by |slope|).
Per record also: 'fanout' rows on the target alone for fan_strength (max |mean slope|) and split_half_top20_jaccard
(web only), with noise_floor = the median over the random targets for the same version / mode / set; and a
'readout_tokens' row with the next-token words whose log-prob rises / falls most with dose.
Concept-specific targets carry concept = their concept.
"""
import json
import statistics as stx
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, read_jsonl, thing  # noqa: E402

FO = DATA_ROOT / "fanout"
FILES = [("v1", FO / "smoke.jsonl"), ("v2", FO / "smoke2.jsonl")]
TXT2 = FO / "smoke2.txt"
EXP = {"v1": "fanout_smoke_v1", "v2": "fanout_smoke_v2"}
DOC = "RESULTS.md#18 Oct 2026: fan-out readout for early / mid neurons - smoke test (research branch)"
DOSES = [-4, -2, -1, 0, 1, 2, 4, 8]


def words(ws, k=10):
    return " | ".join(str(w[0]).strip() or repr(w[0]) for w in ws[:k])


def main(force=False):
    with PackWriter("fanout", inputs=[p for _, p in FILES] + [TXT2], code=[__file__], force=force) as pw:
        if pw.skip:
            print("fanout: unchanged, skipped"); return
        common = dict(status="exploratory", date="2026-10-18", script="src/loop_v2/fanout.py", docs=[DOC])
        method = ("Hold the input fixed and vary only the target neuron through a dose series d x sign x its web-text peak, "
                  "d in {-4,-2,-1,0,1,2,4,8} (0 = zeroed), at the last position or at every position. Read every neuron of "
                  "layers max(L+1, 24)..35 and the next-token log-probs at the last position; least-squares slope vs dose per "
                  "background, R^2, sign consistency across backgrounds, split-half (two halves of 8 web texts) top-20 Jaccard. "
                  "Backgrounds: 16 random web snippets; for concept-specific targets also the concept's own recall prompts (8). "
                  "29 targets: 11 concept-specific early neurons, 8 widely shared early neurons, 10 random (layers 4-23, seed 141).")
        pw.experiment(EXP["v1"], title="Fan-out smoke test v1: which late neurons move when one early / mid neuron is varied (raw slopes)",
                      method=method + " v1 ranks driven neurons by |mean raw slope|.",
                      reproduce="FANOUT_TAG=smoke python src/loop_v2/fanout.py  (v1 code: before the z / random-baseline change)",
                      data_files=["data:fanout/smoke.jsonl"],
                      caveats=["v1 rankings are swamped by generic last-layer hub neurons (L35:9510, L35:8724, L35:11333, L35:9490) that respond to any perturbation.",
                               "v1's text summary was overwritten by a naming bug; the jsonl is intact.",
                               "No per-edge random baseline in v1; target-level noise floors are the random targets' medians (importer)."],
                      params={"doses": DOSES, "n_web": 16, "ctx_tokens": 24, "late_from": 24, "top_saved": 20, "seed": 141}, **common)
        pw.experiment(EXP["v2"], title="Fan-out smoke test v2: specific fan-out (slope in natural-spread units minus random targets)",
                      method=method + (" v2 ranks driven neurons by 'specific' fan-out: |slope / natural sd at dose 0| minus the mean of the "
                                       "same quantity over the 10 random targets (run first as the baseline)."),
                      reproduce="python src/loop_v2/fanout.py   (FANOUT_TAG defaults to smoke2)",
                      data_files=["data:fanout/smoke2.jsonl", "data:fanout/smoke2.txt"],
                      caveats=["Smoke test: 29 targets, 16 web texts.",
                               "Driven neurons are mostly in layers 24-27 and themselves unreadable by their pushed words.",
                               "Random targets' own rows have no baseline (their 'specific' = |z|)."],
                      params={"doses": DOSES, "n_web": 16, "ctx_tokens": 24, "late_from": 24, "top_saved": 20, "seed": 141,
                              "noise_floor": "per edge: random targets' mean |z| for the same late neuron (mode, set); per target: median over random targets",
                              "summary_text": TXT2.read_text(encoding="utf-8")}, **common)
        for ver, path in FILES:
            R = list(read_jsonl(path)); exp = EXP[ver]
            base = defaultdict(lambda: defaultdict(list))
            for r in R:
                if r["kind"] == "random":
                    base[(r["mode"], r["set"])]["fan_strength"].append(r["fan_strength"])
                    if r["split_half_jaccard"] is not None:
                        base[(r["mode"], r["set"])]["split_half_top20_jaccard"].append(r["split_half_jaccard"])
            for i, r in enumerate(R):
                u = r["target"]; cid = thing(pw, r["concept"]) if r["concept"] else None
                ctx = f"version={ver};kind={r['kind']};mode={r['mode']};set={r['set']};doses=-4..8xpeak"
                src = f"fanout/{path.name}#{i}"
                for k, x in enumerate(r["fanout_top20"]):
                    if ver == "v2":
                        # random targets have no baseline; nor does the 'own' set (no random target ran on own prompts)
                        nf = abs(x["z"]) - x["specific"] if r["kind"] != "random" and r["set"] == "web" else None
                        pw.evidence(experiment=exp, unit=u, unit2=x["neuron"], concept=cid, evidence_type="fanout", metric="slope_z_natural_sd",
                                    value=x["z"], uom="natural_sd_per_dose", noise_floor=nf, rank=k + 1, n=16 if r["set"] == "web" else 8,
                                    context=ctx, source=src, details={kk: vv for kk, vv in x.items() if kk != "neuron"})
                    else:
                        pw.evidence(experiment=exp, unit=u, unit2=x["neuron"], concept=cid, evidence_type="fanout", metric="slope_raw",
                                    value=x["slope"], uom="activation_per_dose", rank=k + 1, n=16 if r["set"] == "web" else 8,
                                    context=ctx, source=src, details={kk: vv for kk, vv in x.items() if kk != "neuron"})
                det = {"sign": r["sign"], "peak": r["peak"], "kind": r["kind"], "concept": r["concept"]}
                for m_ in ("fan_strength", "split_half_top20_jaccard"):
                    v = r["fan_strength"] if m_ == "fan_strength" else r["split_half_jaccard"]
                    if v is None:
                        continue
                    b = base[(r["mode"], r["set"])][m_]
                    pw.evidence(experiment=exp, unit=u, concept=cid, evidence_type="fanout", metric=m_, value=v,
                                uom="activation_per_dose" if m_ == "fan_strength" else "fraction",
                                noise_floor=stx.median(b) if b and r["kind"] != "random" else None, context=ctx, source=src, details=det)
                pw.evidence(experiment=exp, unit=u, concept=cid, evidence_type="readout_tokens", metric="fanout_next_token_slope_up",
                            value_text=words(r["words_up"]), context=ctx, source=src,
                            details={"words_up": r["words_up"], "words_down": r["words_down"]})


if __name__ == "__main__":
    main(force="--force" in sys.argv)
