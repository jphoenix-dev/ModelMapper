"""The scorecard: triage and job-type rules on hand-made records, and a reproduction of the research run's v3_results.json."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from modelmapper import scorecard as sc


def exam(bin_, list_pass=None, auc=0.8, ctrl=0.5, desc="x"):
    return {"bin": bin_, "list_pass": list_pass, "auc_own": auc, "auc_control": ctrl, "description": desc}


def test_triage_and_job_types():
    ex = {"gate": {"L0:1": exam("intelligible"), "L0:2": exam("list-only", True), "L0:3": exam("stable-not-named", False),
                   "L1:4": exam("stable-not-named", False)},
          "apos": {"L0:1": exam("intelligible"), "L0:2": exam("panel-misses"), "L0:3": exam("panel-misses"), "L1:4": exam("intelligible")},
          "aneg": {}}
    gen = {"gate": {"L0:1": {"pass": True}}, "apos": {"L0:1": {"pass": False}, "L1:4": {"pass": True}}}
    jt = {"c_pos": np.array([[0, 0.25, -0.3, 0.0, 0], [0, 0, 0, 0, 0.19]]), "c_neg": np.array([[0, -0.2, 0.1, 0.2, 0], [0, 0, 0, 0, -0.5]])}
    ko = {"L0:2": (1, 1.5), "L0:3": (4, 0.0), "L0:1": (2, 0.0)}
    nulls = np.linspace(0, 0.2, 1001)
    out = sc.combine(["L0:1", "L0:2", "L0:3", "L1:4"], ex, gen, jt, ko, nulls, n_layers=2)
    r = out["neurons"]
    assert [r[u]["triage"]["read"] for u in r] == ["clear", "pattern-not-named", "unexplained", "one-sided"]
    assert r["L0:1"]["gate"]["gen_pass"] is True and r["L0:1"]["apos"]["gen_pass"] is False and r["L0:1"]["aneg"] is None
    assert r["L0:1"]["jobtype"] == {"apos": "amplifier", "c_pos": 0.25, "aneg": "eraser", "c_neg": -0.2}
    assert r["L1:4"]["jobtype"]["apos"] == "redirector"
    assert r["L0:2"]["knockout"]["profile"] == "effect" and r["L0:2"]["triage"]["with_knockout"] == "needed, not yet named"
    assert r["L0:3"]["triage"]["with_knockout"] == "candidate junk"
    assert r["L0:1"]["triage"]["with_knockout"] == "nameable, not needed"
    assert out["summary"]["triage"] == {"clear": 1, "one-sided": 1, "pattern-not-named": 1, "unexplained": 1}
    assert out["summary"]["gate"]["verified"] == 0.25


REF = Path(os.environ.get("MODEL_MAPPER_V3_REF", ""))


@pytest.mark.repro
@pytest.mark.skipif(not (REF.name and (REF / "v3_results.json").exists()), reason="set MODEL_MAPPER_V3_REF to the research run's folder")
def test_reproduces_v3_results():
    ref = json.loads((REF / "v3_results.json").read_text(encoding="utf-8"))
    allv3 = json.loads((REF / "v3_neurons.json").read_text())["neurons"]
    merged = lambda pat: {b: {**sc._load(REF / pat.format(run="v3", b=b)), **sc._load(REF / pat.format(run="v3pilot", b=b))} for b in sc.BOARDS}
    ledger = os.environ.get("MODEL_MAPPER_LEDGER", str(REF.parent.parent / "ledger" / "ledger.duckdb"))
    ko, nulls = sc.knockout_from_ledger(ledger, allv3) if Path(ledger).exists() else (None, None)
    out = sc.combine(allv3, merged("intelligibility_{run}_{b}.json"), merged("gen_check_{run}_{b}_surf.json"), REF / "jobtype_v3.npz",
                     ko, nulls, sets={u: ("bridge" if k >= 950 else "new") for k, u in enumerate(allv3)})
    diffs, read = [], {}
    for u in allv3:
        a, b = out["neurons"][u], ref["neurons"][u]
        for key in ("gate", "apos", "aneg", "jobtype", "set"):
            want = b.get(key)
            if key in sc.BOARDS and want is not None:          # the one change since the run: verified at the edge
                want = {**want, "verified": want["verified"] or want["bin"] == sc.EDGE_BIN, "at_edge": want["bin"] == sc.EDGE_BIN}
            if a.get(key) != want:
                diffs.append((u, key))
        g, sv = (a["gate"] or {}), [x for x in ("apos", "aneg") if (a[x] or {}).get("verified")]
        read[u] = "clear" if g.get("verified") else "one-sided" if sv else "pattern-not-named" if g.get("list_pass") else "unexplained"
        same = all((b[x] or {}).get("bin") != sc.EDGE_BIN for x in sc.BOARDS)
        if a["triage"]["read"] != (b["triage"]["read"] if same else read[u]):
            diffs.append((u, "triage"))
        if ko is not None and a["knockout"] != b["knockout"]:
            diffs.append((u, "knockout"))
    assert not diffs, f"{len(diffs)} differences, e.g. {diffs[:5]}"
    if all((ref["neurons"][u][x] or {}).get("bin") != sc.EDGE_BIN for u in allv3 for x in sc.BOARDS):
        assert out["summary"]["triage"] == ref["scorecard"]["triage_counts"]
