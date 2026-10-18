"""Query the ledger (read-only).

  python src/ledger/query.py unit L31:8930 [--all]        everything known about a unit (salient first; --all = every row)
  python src/ledger/query.py concept "Eiffel Tower"       a concept's units by strength, its facts, claims
  python src/ledger/query.py find <text>                  concepts whose name / alias matches
  python src/ledger/query.py shared --min 10              units with passing evidence for >= N concepts
  python src/ledger/query.py experiments                  experiments with status and row counts
  python src/ledger/query.py sql "<SQL>"                  any SQL (tables: evidence, concepts, units, experiments, artifacts,
                                                          claims, unit_concept, unit_summary, concept_summary, layer_summary, head_summary)
Library: from ledger.query import connect, unit, concept
"""
import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ledger.core import LEDGER_DIR, concept_id, parse_unit  # noqa: E402


def connect():
    return duckdb.connect(str(LEDGER_DIR / "ledger.duckdb"), read_only=True)


def _rows(con, sql, params=()):
    cur = con.execute(sql, params); cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def unit(u, all_rows=False, con=None):
    con = con or connect(); uid = parse_unit(u)["unit_id"]
    s = _rows(con, "SELECT * FROM unit_summary WHERE unit_id = ?", [uid])
    where = "" if all_rows else "AND (passes OR evidence_type IN ('role','readout_tokens','label','job_class','activation_auc'))"
    ev = _rows(con, f"""SELECT e.experiment_id, e.evidence_type, e.metric, e.value, e.value_text, e.noise_floor, e.passes, e.context,
                               c.label AS concept, c2.label AS concept2, e.unit2_key
                        FROM evidence e LEFT JOIN concepts c ON c.concept_id = e.concept_id LEFT JOIN concepts c2 ON c2.concept_id = e.concept2_id
                        WHERE (e.unit_id = ? OR e.unit2_id = ?) {where}
                        ORDER BY e.experiment_id, e.evidence_type, e.value DESC NULLS LAST""", [uid, uid])
    cl = _rows(con, "SELECT claim_id, grade, status, text FROM claims WHERE unit_id = ?", [uid])
    return {"summary": s[0] if s else None, "evidence": ev, "claims": cl}


def concept(name, con=None):
    con = con or connect()
    cid = name if ":" in name else concept_id("thing", name)
    c = _rows(con, "SELECT * FROM concept_summary WHERE concept_id = ?", [cid])
    ev = _rows(con, """SELECT experiment_id, evidence_type, metric, count(*) AS n, count(*) FILTER (WHERE passes) AS n_passing
                       FROM evidence WHERE concept_id = ? OR concept2_id = ? GROUP BY ALL ORDER BY 1, 2""", [cid, cid])
    return {"concept": c[0] if c else None, "evidence_overview": ev}


def _print(x):
    print(json.dumps(x, indent=1, ensure_ascii=False, default=str))


def main(argv):
    if not argv:
        print(__doc__); return
    con = connect(); cmd, rest = argv[0], argv[1:]
    if cmd == "unit":
        r = unit(rest[0], "--all" in rest, con)
        s = r["summary"] or {}
        print(f"{rest[0]}  layer {s.get('layer')}  concepts passing {s.get('n_concepts_passing')}/{s.get('n_concepts')}  roles {s.get('role_votes')}")
        print(f"  readout: {s.get('readout_tokens')}")
        for e in r["evidence"][:200 if "--all" not in rest else None]:
            print(f"  [{e['experiment_id']}] {e['evidence_type']}/{e['metric']} = {e['value'] if e['value'] is not None else e['value_text']}"
                  f"  {e['concept'] or ''}{' -> ' + e['unit2_key'] if e['unit2_key'] else ''}  ({e['context'] or ''}){'  PASS' if e['passes'] else ''}")
        for c in r["claims"]:
            print(f"  claim {c['claim_id']} [{c['grade']}; {c['status']}]: {c['text']}")
    elif cmd == "concept":
        _print(concept(" ".join(rest), con))
    elif cmd == "find":
        q = "%" + " ".join(rest).lower() + "%"
        _print(_rows(con, "SELECT concept_id, label, kind, qid FROM concepts WHERE lower(label) LIKE ? OR lower(aliases) LIKE ? LIMIT 50", [q, q]))
    elif cmd == "shared":
        k = int(rest[rest.index("--min") + 1]) if "--min" in rest else 3
        _print(_rows(con, "SELECT unit_key, layer, n_concepts_passing, role_votes, readout_tokens FROM unit_summary WHERE n_concepts_passing >= ? "
                          "ORDER BY n_concepts_passing DESC LIMIT 100", [k]))
    elif cmd == "experiments":
        _print(_rows(con, "SELECT x.experiment_id, x.title, x.status, count(e.evidence_id) AS rows FROM experiments x LEFT JOIN evidence e USING (experiment_id) "
                          "GROUP BY ALL ORDER BY 1"))
    elif cmd == "sql":
        _print(_rows(con, " ".join(rest)))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
