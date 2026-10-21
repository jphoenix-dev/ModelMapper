"""Combine a labelling run into one record per neuron, with a triage and a summary (the research run's v3_combine.py).

Inputs, all in the research run's file formats:
  exams      {board: intelligibility_<tag>.json}  board = gate / apos (+ side) / aneg (- side)   (modelmapper.exam)
  gens       {board: gen_check_<tag>.json}        generative checks of the exam-verified labels   (modelmapper.gencheck)
  jobtype    jobtype.npz                          weights-only job type per side                   (modelmapper.jobtype)
  knockouts  {neuron: (n_concepts, max_drop)} + the null drops, optional (knockout_from_ledger reads them from a ledger)
Per neuron and board: the exam bin, verified (bin "intelligible", or "decoys-too-easy (intelligible)": verified at the
edge, flagged at_edge, lower confidence), the label, the list exam, the AUCs, and the generative check (None where it
didn't run). Counting labels at the edge as verified was decided after the v3 / v3x runs, on the corpus exam's evidence
(they pass it nearly as often as other verified labels); the runs' own files say verified = "intelligible" only. Job type per side: amplifier if cos(read, write) >= 0.2, eraser if <= -0.2,
else redirector (JOB_CUT: the rule the v3 results use; jobtype.py also gives a per-layer null threshold).
Triage: clear (gate verified) / one-sided (gate not, a side verified) / pattern-not-named (no verified label, but the gate's
list exam passes) / unexplained. Crossed with knockouts: needed-but-not-named, candidate junk, nameable-not-needed.
Knockout profile: effect if the largest drop >= max(0.301, the null bar), no effect if below the null bar, else weak; the
null bar is the pooled null drops' 1 - 0.05 / n_concepts quantile (one bar per neuron, Bonferroni over its concepts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BOARDS = ("gate", "apos", "aneg")
EDGE_BIN = "decoys-too-easy (intelligible)"          # both exams passed, but a control scored above 0.65
VERIFIED_BINS = ("intelligible", EDGE_BIN)
JOB_CUT = 0.2
EFFECT = 0.301
# knockout experiments that measure something else (enrichment / inhibitor stages), left out of the profile as in v3
E7_LIKE = ("loop_v4_E7", "loop_v3_e7_pilot", "mech_inhibitors_06b", "mech_inhibitors_06b_retest")


def _load(x) -> dict:
    if x is None:
        return {}
    if isinstance(x, (str, Path)):
        p = Path(x)
        if not p.exists():
            return {}
        x = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in x.items() if not k.startswith("_")}


def job_type(c: float, cut: float = JOB_CUT) -> str:
    return "amplifier" if c >= cut else "eraser" if c <= -cut else "redirector"


def band(layer: int, n_layers: int) -> str:
    """Thirds of the depth: early / middle / late (Qwen3-8B: 0-11, 12-23, 24-35)."""
    return "early" if layer < n_layers // 3 else "middle" if layer < 2 * n_layers // 3 else "late"


def knockout_profile(n_concepts, max_drop, nulls) -> tuple[str, int | None, float | None]:
    if not n_concepts:
        return "untested", None, None
    thr = float(np.quantile(nulls, 1 - 0.05 / max(1, n_concepts)))
    prof = "effect" if max_drop >= max(EFFECT, thr) else "no effect" if max_drop < thr else "weak"
    return prof, int(n_concepts), round(float(max_drop), 3)


def knockout_from_ledger(ledger: str | Path, neurons: list[str]) -> tuple[dict, np.ndarray]:
    """{neuron: (n_concepts, max_drop)} and the pooled null drops, from a project ledger (needs duckdb)."""
    import duckdb
    c = duckdb.connect(str(ledger), read_only=True)
    ko = {r[0]: (r[1], r[2]) for r in c.execute(
        f"""SELECT unit_key, count(DISTINCT concept_id), max(value) FROM evidence
            WHERE evidence_type = 'knockout_drop' AND metric = 'mean_logodds_drop' AND experiment_id NOT IN {E7_LIKE!r}
            AND unit_key IN (SELECT unnest(?)) GROUP BY unit_key""", [list(neurons)]).fetchall()}
    nulls = np.array([r[0] for r in c.execute(
        f"""SELECT value FROM evidence WHERE evidence_type = 'null_knockout_drop' AND metric = 'mean_logodds_drop'
            AND experiment_id NOT IN {E7_LIKE!r}""").fetchall()], float)
    return ko, nulls


def combine(neurons: list[str], exams: dict, gens: dict | None = None, jobtype=None, knockouts: dict | None = None,
            nulls=None, n_layers: int | None = None, sets: dict | None = None) -> dict:
    """{"neurons": {neuron: record}, "summary": {...}}. sets: optional {neuron: label}, e.g. "new" / "bridge"."""
    ex = {b: _load(exams.get(b)) for b in BOARDS}
    gen = {b: _load((gens or {}).get(b)) for b in BOARDS}
    jt = None if jobtype is None else (np.load(jobtype) if isinstance(jobtype, (str, Path)) else jobtype)
    layers = [int(u[1:].split(":")[0]) for u in neurons]
    n_layers = n_layers or (jt["c_pos"].shape[0] if jt is not None else max(layers) + 1)
    rec = {}
    for u, L in zip(neurons, layers):
        i = int(u.split(":")[1])
        r = {"layer": L}
        if sets:
            r["set"] = sets.get(u)
        for b in BOARDS:
            e = ex[b].get(u)
            r[b] = None if e is None else {"bin": e["bin"], "verified": e["bin"] in VERIFIED_BINS, "at_edge": e["bin"] == EDGE_BIN, "label": e.get("description"),
                                           "list_pass": e.get("list_pass"), "auc": e.get("auc_own"), "auc_control": e.get("auc_control"),
                                           "gen_pass": gen[b][u]["pass"] if u in gen[b] else None}
        if jt is not None:
            cp, cn = float(jt["c_pos"][L, i]), float(jt["c_neg"][L, i])
            r["jobtype"] = {"apos": job_type(cp), "c_pos": round(cp, 3), "aneg": job_type(cn), "c_neg": round(cn, 3)}
        k, n, mx = (knockout_profile(*(knockouts or {}).get(u, (None, None)), nulls) if knockouts is not None
                    else ("untested", None, None))
        r["knockout"] = {"profile": k, "n_concepts": n, "max_drop": mx}
        g = r["gate"] or {}
        sv = [b for b in ("apos", "aneg") if (r[b] or {}).get("verified")]
        read = "clear" if g.get("verified") else "one-sided" if sv else "pattern-not-named" if g.get("list_pass") else "unexplained"
        named = read in ("clear", "one-sided")
        r["triage"] = {"read": read, "verified_sides": sv,
                       "with_knockout": ("needed, not yet named" if not named and k == "effect" else "candidate junk" if not named and k == "no effect"
                                         else "nameable, not needed" if named and k == "no effect" else None)}
        rec[u] = r
    return {"neurons": rec, "summary": summarise(rec, ex, n_layers)}


def summarise(rec: dict, ex: dict, n_layers: int) -> dict:
    share = lambda xs: round(float(np.mean(xs)), 3) if len(xs) else None
    V = lambda r, b: bool((r[b] or {}).get("verified"))
    out = {"n": len(rec)}
    for b in BOARDS:
        rs = [r for r in rec.values() if r[b] is not None]
        out[b] = {"n": len(rs), "verified": share([V(r, b) for r in rs]), "at_edge": share([bool((r[b] or {}).get("at_edge")) for r in rs]),
                  "verified_and_generative": share([V(r, b) and r[b]["gen_pass"] is True for r in rs]),
                  "by_band": {bd: share([V(r, b) for r in rs if band(r["layer"], n_layers) == bd]) for bd in ("early", "middle", "late")},
                  "control_auc_median": (round(float(np.median([e["auc_control"] for e in ex[b].values() if e.get("auc_control") is not None])), 3)
                                         if ex[b] else None)}
    out["any_label_verified"] = share([any(V(r, b) for b in BOARDS) for r in rec.values()])
    out["triage"] = {k: sum(1 for r in rec.values() if r["triage"]["read"] == k) for k in ("clear", "one-sided", "pattern-not-named", "unexplained")}
    out["triage_with_knockout"] = {k: sum(1 for r in rec.values() if r["triage"]["with_knockout"] == k)
                                   for k in ("needed, not yet named", "candidate junk", "nameable, not needed")}
    out["knockout_tested"] = sum(1 for r in rec.values() if r["knockout"]["profile"] != "untested")
    return out


def pseudo_summary(exams: dict, gens: dict | None = None) -> dict:
    """The falsifier: how often fake neurons get a verified label (and pass the generative check too)."""
    out = {}
    for b in BOARDS:
        e, g = _load(exams.get(b)), _load((gens or {}).get(b))
        if not e:
            continue
        v = [k for k, r in e.items() if r.get("bin") in VERIFIED_BINS]
        out[b] = {"n": len(e), "verified": round(len(v) / len(e), 3), "at_edge": round(sum(1 for k in v if e[k].get("bin") == EDGE_BIN) / len(e), 3),
                  "verified_and_generative": round(sum(1 for k in v if g.get(k, {}).get("pass")) / len(e), 3)}
    return out


def save(result: dict, path: str | Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
