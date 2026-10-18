"""Verify the ledger against its source files (lossless check): recount source records independently and compare values.

  python src/ledger/verify.py        -> prints per-source checks; exit code 1 if any check fails
"""
import glob
import json
import random
import sys
from pathlib import Path

import duckdb

from modelmapper.ledger.core import DATA_ROOT, LEDGER_DIR, concept_id, read_jsonl  # noqa: E402

FAIL = []


def check(name, ok, msg):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {msg}")
    if not ok:
        FAIL.append(name)


def q(con, sql, p=()):
    return con.execute(sql, p).fetchall()


def run1(con, rng):
    print("run 1")
    files = sorted(glob.glob(str(DATA_ROOT / "loop_v2/run1/results/**/records.jsonl"), recursive=True))
    recs = [r for f in files for r in read_jsonl(f)]
    n_c = len(recs); n_ok = sum((r.get("row") or {}).get("status") == "ok" for r in recs)
    n_cand = sum(len((r.get("rank") or {}).get("candidates_all200") or []) for r in recs if (r.get("row") or {}).get("status") == "ok")
    n_head = sum(len((r.get("rank") or {}).get("head_knockout_mean") or {}) for r in recs if (r.get("row") or {}).get("status") == "ok")
    got = dict(q(con, "SELECT evidence_type || '/' || coalesce(unit_kind,'-'), count(*) FROM evidence WHERE experiment_id='loop_v2_run1' GROUP BY 1"))
    check("concepts", got.get("recall/-") == n_c, f"source {n_c} concept records, ledger recall rows {got.get('recall/-')}")
    check("candidate knockouts", got.get("knockout_drop/neuron") == n_cand, f"source {n_cand}, ledger {got.get('knockout_drop/neuron')}")
    check("head knockouts", got.get("knockout_drop/head") == n_head, f"source {n_head}, ledger {got.get('knockout_drop/head')}")
    bad = 0; tested = 0
    for r in rng.sample([r for r in recs if (r.get("row") or {}).get("status") == "ok"], 40):
        cid = concept_id("thing", r["concept"])
        led = dict(q(con, "SELECT unit_key, value FROM evidence WHERE experiment_id='loop_v2_run1' AND concept_id=? AND evidence_type='knockout_drop' AND unit_kind='neuron'", [cid]))
        for x in r["rank"]["candidates_all200"]:
            tested += 1; v = led.get(x["neuron"])
            if v is None or abs(v - x["single_drop"]) > 1e-9:
                bad += 1
                if bad <= 5:
                    print(f"     mismatch {r['concept']} {x['neuron']}: source {x['single_drop']} ledger {v}")
    check("candidate values (40 random concepts)", bad == 0, f"{tested} values compared, {bad} mismatches")


def era(con):
    print("era")
    rows = json.loads((DATA_ROOT / "era/era_rows.json").read_text(encoding="utf-8"))
    exp = sum(1 for r in rows for k, v in r.items() if k not in ("u", "label", "is_control", "n_in", "sign") and v is not None)
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='era_formats' AND evidence_type='activation_auc'")[0][0]
    check("AUC rows", got == exp, f"source {exp}, ledger {got}")
    C = list(read_jsonl(DATA_ROOT / "era/era_causal.jsonl")); exp = sum(len(c["drops"]) for c in C)
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='era_formats' AND evidence_type='knockout_drop' AND metric='logodds_drop'")[0][0]
    check("causal rows", got == exp, f"source {exp}, ledger {got}")
    A = list(read_jsonl(DATA_ROOT / "era/era_acts.jsonl")); exp = sum(len(a["acts"]) for a in A)
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='era_formats' AND evidence_type='activation' AND metric='activation'")[0][0]
    check("activation rows", got == exp, f"source {exp}, ledger {got}")
    v = q(con, "SELECT value FROM evidence WHERE experiment_id='era_formats' AND unit_key='L31:8930' AND evidence_type='activation_auc' AND context='format=year'")[0][0]
    src = next(r["year"] for r in rows if r["u"] == "L31:8930")
    check("L31:8930 year AUC", abs(v - src) < 1e-12, f"source {src:.6f}, ledger {v:.6f}")


def numeric(con):
    print("numeric")
    M = list(read_jsonl(DATA_ROOT / "numeric/maps.jsonl")); exp = sum(len(m["top"]) for m in M)
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='numeric_facts' AND context LIKE '%test=own_map%'")[0][0]
    check("map rows", got == exp, f"source {exp}, ledger {got}")
    S = list(read_jsonl(DATA_ROOT / "numeric/share.jsonl")); exp = sum(len(s["per_pos"]) for s in S)
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='numeric_facts' AND context LIKE '%test=share%'")[0][0]
    check("share rows", got == exp, f"source {exp}, ledger {got}")
    R = json.loads((DATA_ROOT / "numeric/mag_rows.json").read_text(encoding="utf-8"))
    got = q(con, "SELECT count(*) FROM evidence WHERE experiment_id='numeric_magnitude' AND evidence_type='dose_response'")[0][0]
    check("dose rows", got == len(R), f"source {len(R)}, ledger {got}")


def facts(con):
    for name in ("facts", "facts2"):
        print(name)
        G = list(read_jsonl(DATA_ROOT / name / "grid.jsonl")); exp = sum(len(g["drops"]) for g in G)
        got = q(con, f"SELECT count(*) FROM evidence WHERE experiment_id='{name}_grid' AND context LIKE '%test=grid%'")[0][0]
        check("grid rows", got == exp, f"source {exp}, ledger {got}")
        M = list(read_jsonl(DATA_ROOT / name / "maps.jsonl")); exp = sum(len(m["top"]) for m in M)
        got = q(con, f"SELECT count(*) FROM evidence WHERE experiment_id='{name}_grid' AND context LIKE '%test=own_map%'")[0][0]
        check("map rows", got == exp, f"source {exp}, ledger {got}")


def generic(con):
    print("all packs")
    for pj in sorted((LEDGER_DIR / "packs").glob("*/pack.json")):
        meta = json.loads(pj.read_text(encoding="utf-8"))
        sup = q(con, "SELECT superseded FROM ledger_meta")[0][0]
        if meta["pack_id"] in json.loads(sup):
            continue
        for t, info in meta.get("tables", {}).items():
            got = q(con, f"SELECT count(*) FROM {t}_raw WHERE pack_id = ?", [meta["pack_id"]])[0][0]
            check(f"{meta['pack_id']} {t}", got == info["rows"], f"pack.json {info['rows']}, ledger {got}")


def main():
    con = duckdb.connect(str(LEDGER_DIR / "ledger.duckdb"), read_only=True); rng = random.Random(7)
    run1(con, rng); era(con); numeric(con); facts(con); generic(con)
    print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
