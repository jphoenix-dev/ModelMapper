"""Summarise corpus_profile.npz: is each concept neuron about its concept in ordinary web text? (no model, no API)

Keywords per concept (fixed rule, as wave_free.py): words of >= 4 letters from the concept name, its answer and the
model's discovered answer forms (spec files), minus wave_free.GENERIC; matched case-insensitively as whole words, optionally plural (-s/-es).
Related words (e.g. 'lunar' for the Moon) are NOT matched, so 'on-concept' is a lower bound.
A firing is ON-CONCEPT if one of its owner's keywords occurs within +-WINDOW tokens (WINDOW = 64).
Per concept neuron (|own act| >= 1; the rest counted separately):
  rate        strong firings (>= 0.5 x own activation, own direction) per million tokens
  on          share of its firings that are on-concept
  base        share of ALL scanned tokens within +-WINDOW of an owner keyword (chance level for 'on')
  ctrl_on     the same 'on' share for its paired random same-layer neuron at the same threshold
  other       share near another of our concepts' keywords but not its own; 'none' = near none of our concepts
  lift        on / base (how much more than chance its firings sit near its concept)
Usage: python src/corpus_profile_summary.py   -> results/corpus_profile_summary.txt / .json
"""
import glob
import json
import re
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import concept_pipeline as cp
from config import DEFAULT_MODEL, model_path
from families import family_map
from wave_free import GENERIC

ROOT = Path(__file__).resolve().parents[1]
WINDOW, MIN_OWN = 64, 1.0
TAG = "_new" if __import__("os").environ.get("CP_TAG") == "new" else ""     # PREREG_MAP P5: profile of NEW text


def keywords():
    kw = {}
    for f in sorted(glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json"))):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        d = Path(f).parent
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" not in v:
                continue
            spec = json.loads((d / f"{cp.slug(c)}_spec.json").read_text(encoding="utf-8"))
            words = [w.lower() for w in re.findall(r"[A-Za-zé]{4,}", spec["answer"] + " " + c + " " + " ".join(spec.get("answer_forms_extra") or []))]
            kw[c] = sorted({w for w in words if w not in GENERIC} or set(words))
    return kw


def main():
    rows = json.loads((ROOT / "results/corpus_profile_units.json").read_text(encoding="utf-8"))
    z = np.load(ROOT / ("results/corpus_profile_new.npz" if TAG else "results/corpus_profile.npz"))
    unit, doc, pos, n_tok = z["unit"], z["doc"], z["pos"].astype(np.int32), int(z["tokens_scanned"])
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    d = np.load(ROOT / ("results/corpus_new_docs.npz" if TAG else f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz"))
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    kw = keywords(); fam = family_map()
    all_kw = sorted({w for ws in kw.values() for w in ws})
    pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(all_kw, key=len, reverse=True)) + r")(?:s|es)?\b", re.I)
    # one boolean mask over ALL scanned tokens per concept: True within +-WINDOW of one of its keywords (vectorised)
    N = int(lens.sum()); word_pos = defaultdict(list)                    # keyword -> global token positions
    for di in range(len(lens)):
        ids = toks[starts[di]:starts[di] + lens[di]].tolist()
        pieces = tok.batch_decode([[t] for t in ids])
        text = "".join(pieces); offs = np.cumsum([0] + [len(p) for p in pieces])
        for mt in pat.finditer(text):
            word_pos[mt.group(1).lower()].append(starts[di] + int(np.searchsorted(offs, mt.start(), side="right") - 1))
    doc_of = np.repeat(np.arange(len(lens)), lens)
    def mask_for(words):
        m = np.zeros(N + 1, np.int32)
        for w in words:
            for g in word_pos.get(w, ()):
                d0 = doc_of[g]; lo = max(starts[d0] + 1, g - WINDOW); hi = min(starts[d0] + lens[d0], g + WINDOW + 1)
                m[lo] += 1; m[hi] -= 1
        return np.cumsum(m[:N]) > 0
    near = {c: mask_for(ws) for c, ws in kw.items()}
    any_near = np.zeros(N, bool)
    for v in near.values():
        any_near |= v
    first_tok = np.zeros(N, bool); first_tok[starts] = True                # token 0 of each window is never scanned
    global ANY_BASE
    ANY_BASE = float(any_near[~first_tok].mean())
    order = np.argsort(unit, kind="stable"); unit, doc, pos = unit[order], doc[order], pos[order]
    gidx = starts[doc] + pos
    bounds = np.searchsorted(unit, np.arange(len(rows) + 1))
    pair = {rr["paired_with"]: k for k, rr in enumerate(rows) if rr["control"]}
    res = {}
    for i, r in enumerate(rows):
        if r["control"]:
            continue
        owners = [c for c in r["concepts"] if c in kw]
        if not owners:
            continue
        own_m = np.zeros(N, bool)
        for c in owners:
            own_m |= near[c]
        base = own_m[~first_tok].mean()
        def prof(k, other):
            g = gidx[bounds[k]:bounds[k + 1]]; n = len(g)
            if n == 0:
                return {"n": 0, "rate": 0.0, "on": None, "other": None, "none": None}
            on = own_m[g].mean()
            if not other:
                return {"n": n, "rate": n / n_tok * 1e6, "on": float(on)}
            oth = (any_near[g] & ~own_m[g]).mean()
            return {"n": n, "rate": n / n_tok * 1e6, "on": float(on), "other": float(oth), "none": float(1 - on - oth)}
        p = prof(i, True); c_ = prof(pair[r["neuron"]], False)
        res[r["neuron"]] = {**p, "own_act": r["own_act"], "owners": owners, "families": sorted({fam.get(c, "?") for c in owners}),
                            "base": float(base), "lift": (p["on"] / base) if p["on"] is not None and base > 0 else None,
                            "ctrl": c_, "layer": r["layer"]}
    act = {u: v for u, v in res.items() if abs(v["own_act"]) >= MIN_OWN}
    fired = {u: v for u, v in act.items() if v["n"] > 0}
    L = [f"Corpus profile over {n_tok:,} web tokens: {len(res)} concept neurons ({len(act)} with |own act| >= {MIN_OWN}; "
         f"{len(act) - len(fired)} of those never reach half their own-concept activation anywhere in the text)."
         f" Firing = >= 0.5 x own activation, own direction. On-concept = owner keyword within +-{WINDOW} tokens.\n"]
    md = lambda xs: stx.median(xs) if xs else float("nan")
    L.append(f"firings per million tokens: median {md([v['rate'] for v in fired.values()]):.1f} "
             f"(paired random neurons, same threshold: {md([v['ctrl']['rate'] for v in act.values()]):.1f})")
    L.append(f"share of firings on-concept: median {md([v['on'] for v in fired.values()]):.2f} vs chance (base) "
             f"{md([v['base'] for v in fired.values()]):.3f} vs paired random neurons "
             f"{md([v['ctrl']['on'] for v in fired.values() if v['ctrl']['on'] is not None]):.3f}")
    L.append(f"lift (on / chance): median {md([v['lift'] for v in fired.values() if v['lift']]):.1f}; neurons with lift >= 5: "
             f"{sum((v['lift'] or 0) >= 5 for v in fired.values())}/{len(fired)}; on-concept >= 50%: {sum(v['on'] >= 0.5 for v in fired.values())}/{len(fired)}")
    L.append(f"firings near ANOTHER of our concepts only: median {md([v['other'] for v in fired.values()]):.2f}; near none of our "
             f"{len(kw)} concepts: median {md([v['none'] for v in fired.values()]):.2f}")
    L.append(f"  chance for that: share of all scanned tokens near ANY of our concepts' keywords = {ANY_BASE:.2f} "
             f"(so 'other' and 'none' are only interpretable against this)")
    L.append("\nby layer band: neurons firing | median rate/M | median on-concept | median lift | median 'none'")
    for lo, hi in ((0, 16), (16, 24), (24, 30), (30, 36)):
        vv = [v for v in fired.values() if lo <= v["layer"] < hi]
        if vv:
            L.append(f"  L{lo}-{hi - 1}: {len(vv):>4} | {md([v['rate'] for v in vv]):8.1f} | {md([v['on'] for v in vv]):.2f} | "
                     f"{md([v['lift'] for v in vv if v['lift']]):6.1f} | {md([v['none'] for v in vv]):.2f}")
    L.append("\nmost concept-specific (on-concept share, >= 20 firings):")
    for u, v in sorted(((u, v) for u, v in fired.items() if v["n"] >= 20), key=lambda t: -t[1]["on"])[:12]:
        L.append(f"  {u:<10} {', '.join(v['owners'])[:40]:<40} n={v['n']:<6} on {v['on']:.2f} lift {v['lift'] or 0:.0f}")
    L.append("\nleast concept-specific (>= 20 firings):")
    for u, v in sorted(((u, v) for u, v in fired.items() if v["n"] >= 20), key=lambda t: t[1]["on"])[:12]:
        L.append(f"  {u:<10} {', '.join(v['owners'])[:40]:<40} n={v['n']:<6} on {v['on']:.2f} other {v['other']:.2f} none {v['none']:.2f}")
    txt = "\n".join(L)
    (ROOT / f"results/corpus_profile_summary{TAG}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / f"results/corpus_profile_summary{TAG}.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
