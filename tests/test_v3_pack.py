"""export-pack: the ledger's third-party format and validation rules (fast), and, marked repro, the same evidence rows as
the research ledger holds for the research run's own files.

Repro tests need (each skips without its own):
  MODEL_MAPPER_V3_REF      the research v3 run's folder (intelligibility_v3_gate.json, gen_check_v3_gate_surf.json)
  MODEL_MAPPER_LEDGER      the built research ledger (ledger.duckdb) and the duckdb package; run 1's records are read from
                           <ledger folder>/../loop_v2/run1/results/results/map/b0001/records.jsonl
  MODEL_MAPPER_RESEARCH    the research checkout (src/ledger/pack.py), run with MODEL_MAPPER_RESEARCH_PYTHON (default:
                           this Python; it needs pyarrow and yaml)
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_record
from modelmapper import pack
from modelmapper.cli import main


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def exam_settings(board, pseudo=False, focus=None):
    return {"board": board, "n_test": 30, "n_ctrl": 5, "n_judge": 1, "floor": 10.0, "reuse_desc": "", "focus": focus, "hub_x": 4.0,
            "judge": "jev", "pseudo": pseudo, "draws": 3}


def label_run(d: Path):
    """A finished label run's files: gate and apos exams, the gate's verified subset, both generative checks, a pseudo exam."""
    recs = {"L0:1": {"layer": 0, "bin": "intelligible", "auc_own": 0.91, "auc_control": 0.5, "list_auc_own": 0.8, "label_pass": True,
                     "list_pass": True, "kind": "meaning", "stability": 0.3, "reach": None, "description": "dogs", "draws": [{}]},
            "L1:3": {"layer": 1, "bin": "no-generalisation", "auc_own": 0.6, "auc_control": 0.55, "description": "?", "stable": True},
            "L2:4": {"layer": 2, "bin": "no-pattern", "auc_own": None, "description": None},          # skipped, as by jobs_v1
            "P0:L3:gate": {"bin": "intelligible", "auc_own": 0.9, "description": "x"}}                  # not a real unit
    write(d / "intelligibility_run_c0_gate.json", {"_summary": {"exam": exam_settings("gate", focus=["L0:1"]), "n": 3}, **recs})
    write(d / "intelligibility_run_c0_apos.json", {"_summary": {"exam": exam_settings("apos"), "n": 1}, "L0:1": recs["L0:1"]})
    write(d / "intelligibility_run_c0_gate_ver.json", {"_summary": {"subset_of": "run_c0_gate", "n": 1}, "L0:1": recs["L0:1"]})
    write(d / "intelligibility_run_pq_gate.json", {"_summary": {"exam": exam_settings("gate", pseudo=True), "n": 1},
                                                   "P0:L3:gate": recs["P0:L3:gate"]})
    gen = {"bin": "intelligible", "layer": 0, "label": "dogs", "tokens": [" dog"], "auc": 0.8, "pass": True, "median_z_own": 1.2,
           "single_token_share": 1.0, "z_own": [1.2]}
    for b in ("gate", "apos"):
        write(d / f"gen_check_run_c0_{b}_surf.json", {"_summary": {"tag": f"run_c0_{b}_surf", "labels": f"run_c0_{b}_ver", "pseudo": "",
                                                                   "control": f"surface:run_c0_{b}_ver", "n": 1}, "L0:1": gen})
    return d


def rows(d):
    return list(pack._read_jsonl(d / "evidence.jsonl"))


def exps(d):
    return {e["experiment_id"]: e for e in pack._read_jsonl(d / "experiments.jsonl")}


# ---------------------------------------------------------------- the copied ledger rules
def test_settings_hash_is_the_ledgers():
    # jobs_v1_v3_gen_gate's settings_hash in the research ledger
    assert pack.shash({"control": "surface-only label", "board": "gate", "process": "generative_check"}) == "08cd1dd6ec03"
    assert pack.shash({"b": 1, "a": 2}) == pack.shash({"a": 2, "b": 1})


def test_unit_ids_as_the_ledger():
    assert pack.parse_unit("L31:8930")["unit_id"] == "qwen3-8b/L31/N8930"
    assert pack.parse_unit("L0_N492")["unit_key"] == "L0:492"
    assert pack.parse_unit("L23H10")["unit_id"] == "qwen3-8b/L23/H10"
    assert pack.parse_unit("qwen3-0.6b/L3/N9", "qwen3-8b")["model"] == "qwen3-0.6b"
    with pytest.raises(ValueError):
        pack.parse_unit("L36:1")                                   # Qwen3-8B has layers 0-35
    assert pack.parse_unit("L40:99999", "olmo-2-7b")["layer"] == 40    # no range table for other models (as the ledger)
    with pytest.raises(ValueError):
        pack.parse_unit("neuron 5")
    assert pack.concept_id("thing", "The Eiffel Tower") == "thing:eiffel-tower"
    assert pack.concept_id("thing", "Café Müller") == "thing:cafe-muller"
    assert pack.model_slug("Qwen/Qwen3-8B") == "qwen3-8b"


# ---------------------------------------------------------------- labelling runs
def test_label_run_rows(tmp_path):
    src = label_run(tmp_path / "labels")
    res = pack.export_pack([src], tmp_path / "pk", contributor="Ada Lovelace", model="Qwen/Qwen3-8B", log=lambda *_: None)
    assert res["importer"] == "mm_ada_lovelace" and res["model"] == "qwen3-8b"
    E = exps(tmp_path / "pk")
    assert set(E) == {"mm_ada_lovelace_run_c0_gate", "mm_ada_lovelace_run_c0_apos", "mm_ada_lovelace_run_pq_gate",
                      "mm_ada_lovelace_gen_run_c0_gate_surf", "mm_ada_lovelace_gen_run_c0_apos_surf"}          # _ver skipped
    R = rows(tmp_path / "pk")
    gate = [r for r in R if r["experiment"] == "mm_ada_lovelace_run_c0_gate"]
    assert [r["unit"] for r in gate] == ["L0:1", "L1:3"]                      # no-description and pseudo rows left out
    g = gate[0]
    sh = E["mm_ada_lovelace_run_c0_gate"]["params"]["settings_hash"]
    assert g == {"experiment": "mm_ada_lovelace_run_c0_gate", "unit": "L0:1", "evidence_type": "label_exam", "metric": "label_auc",
                 "value": 0.91, "value_text": "dogs", "uom": "AUC", "noise_floor": 0.5, "threshold": 0.75, "passes": True,
                 "context": "process=gate_exam;board=gate",
                 "details": {"bin": "intelligible", "auc_control": 0.5, "list_auc_own": 0.8, "label_pass": True, "list_pass": True,
                             "kind": "meaning", "stability": 0.3, "reach": None, "settings_hash": sh},
                 "source": "labels/intelligibility_run_c0_gate.json#L0:1"}
    assert gate[1]["passes"] is False and gate[1]["details"]["stable"] is True
    s = dict(exam_settings("gate", focus=["L0:1"]), process="gate_exam")
    assert sh == pack.shash({k: v for k, v in s.items() if k not in ("focus", "reuse_desc", "batches")})
    assert sh == pack.shash({k: v for k, v in dict(s, focus=["L9:9"], reuse_desc="other").items() if k not in ("focus", "reuse_desc")})
    apos = [r for r in R if r["experiment"] == "mm_ada_lovelace_run_c0_apos"]
    assert apos[0]["context"] == "process=side_exam;board=apos"
    assert not [r for r in R if r["experiment"] == "mm_ada_lovelace_run_pq_gate"]          # the falsifier: experiment only
    gen = [r for r in R if r["experiment"] == "mm_ada_lovelace_gen_run_c0_apos_surf"][0]
    assert gen["evidence_type"] == "label_check" and gen["metric"] == "generative_auc" and gen["passes"] is True
    assert gen["context"] == "process=generative_check;control=surface-only label"
    assert gen["details"] == {"bin": "intelligible", "median_z_own": 1.2, "single_token_share": 1.0, "tokens": [" dog"],
                              "settings_hash": pack.shash({"control": "surface-only label", "board": "apos", "process": "generative_check"})}
    p = E["mm_ada_lovelace_gen_run_c0_apos_surf"]["params"]
    assert p["board"] == "apos" and p["process"] == "generative_check" and p["control"] == "surface-only label"
    prov = json.loads((tmp_path / "pk" / "pack.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["package"].startswith("modelmapper ") and prov["model"] == "Qwen/Qwen3-8B" and prov["judge_model"] == "jev"
    assert {"dtype", "device", "labeller_model", "quick", "calibrated", "check"} <= set(prov)
    assert p["provenance"] == prov
    assert pack.validate(tmp_path / "pk") == ([], [])


def test_provenance_from_the_run_and_the_check_report(tmp_path):
    src = label_run(tmp_path / "labels")
    write(src / "run_results.json", {"neurons": {}, "provenance": {"dtype": "bfloat16", "device": "cuda", "quick": True, "calibrated": False}})
    write(tmp_path / "report.json", {"overall": "PASS", "version": "0.2.0", "when": "2026-10-21", "steps": "full",
                                     "items": [{"id": "sweep1", "name": "sweep", "status": "PASS"}, {"id": "exam1", "status": "SKIPPED"}]})
    pack.export_pack([src], tmp_path / "pk", contributor="x", model="Qwen/Qwen3-0.6B", check_report=tmp_path / "report.json",
                     provenance={"labeller_model": "gpt-6-luna"}, log=lambda *_: None)
    meta = json.loads((tmp_path / "pk" / "pack.json").read_text(encoding="utf-8"))
    p = meta["provenance"]
    assert (p["dtype"], p["device"], p["quick"], p["calibrated"], p["labeller_model"]) == ("bfloat16", "cuda", True, False, "gpt-6-luna")
    assert p["check"]["overall"] == "PASS" and p["check"]["steps"] == {"sweep1": "PASS", "exam1": "SKIPPED"}
    assert meta["model"] == "qwen3-0.6b" and meta["importer"] == "mm_x" and meta["contributor"] == "x"
    assert any("quick sweep" in c for c in next(iter(exps(tmp_path / "pk").values()))["caveats"])


def test_importer_names(tmp_path):
    assert pack.importer_for("Ada Lovelace") == "mm_ada_lovelace"
    assert pack.importer_for("x", "my_lab_oct") == "my_lab_oct"
    for bad in ("jobs_v1", "loop_v2_run1", "jobs", "loop", "Bad Name"):
        with pytest.raises(pack.PackError):
            pack.importer_for("x", bad)
    with pytest.raises(pack.PackError):
        pack.export_pack([label_run(tmp_path / "l")], tmp_path / "pk", contributor="x", log=lambda *_: None)    # no model given
    with pytest.raises(pack.PackError):
        pack.export_pack([tmp_path / "missing"], tmp_path / "pk", contributor="x", model="m", log=lambda *_: None)


# ---------------------------------------------------------------- map records
def test_map_records_rows(records_dir, tmp_path):
    (records_dir / "records.jsonl").write_text(json.dumps(make_record("Bitcoin")) + "\n" + json.dumps(make_record("Ether", status="not_recalled"))
                                               + "\n", encoding="utf-8")
    res = pack.export_pack([records_dir], tmp_path / "pk", contributor="x", log=lambda *_: None)
    assert res["model"] == "qwen3-0.6b"                                 # from the records' provenance
    R = rows(tmp_path / "pk")
    E = exps(tmp_path / "pk")
    (eid, e), = E.items()
    assert eid == "mm_x_maps_qwen3_0_6b"
    floor = e["params"]["noise_floor"]["value"]
    assert floor == 0.01 and e["params"]["noise_floor"]["n"] == 6     # |mean| of [0, .02], [0, 0] in each of 3 records, pooled
    by = {}
    for r in R:
        by.setdefault(r["concept"], []).append(r)
    assert set(by) == {"thing:bitcoin", "thing:youtube", "thing:ether"}   # Bitcoin's <slug>.json wins over its records.jsonl line
    assert [r["evidence_type"] for r in by["thing:ether"]] == ["recall"] and by["thing:ether"][0]["value_text"] == "not_recalled"
    b = by["thing:bitcoin"]
    count = {}
    for r in b:
        count[(r["evidence_type"], r["metric"])] = count.get((r["evidence_type"], r["metric"]), 0) + 1
    assert count == {("recall", "n_recall_prompts"): 1, ("concept_stat", "k_mean"): 1,   # (make_record has no row baseline)
                     ("concept_stat", "k_median"): 1, ("concept_stat", "k_flip"): 1, ("concept_stat", "k_curve"): 1,
                     ("knockout_drop", "mean_logodds_drop"): 40 + 2, ("null_knockout_drop", "mean_logodds_drop"): 2,
                     ("role", "weights_role"): 1, ("readout_tokens", "direct_logit_top30"): 1}
    ko = [r for r in b if r["evidence_type"] == "knockout_drop" and ":" in r["unit"]]
    assert [r["rank"] for r in ko] == list(range(1, 41)) and ko[0]["value"] == 4.0 and ko[0]["unit"] == "L0:100"
    assert ko[0]["noise_floor"] == ko[0]["threshold"] == floor and ko[0]["passes"] is True and ko[-1]["passes"] is True
    assert ko[0]["context"] == "positions=all;prompts=recall" and set(ko[0]["details"]) == {"attr_rank", "act_answer_pos", "per_prompt"}
    heads = [r for r in b if r["evidence_type"] == "knockout_drop" and "H" in r["unit"]]
    assert [(r["unit"], r["rank"]) for r in heads] == [("L3H1", 1), ("L19H29", 2)]
    nul = [r for r in b if r["evidence_type"] == "null_knockout_drop"]
    assert nul[0]["value"] == 0.01 and nul[0]["n"] == 2 and nul[0]["context"].endswith("control=random_same_layer")
    rd = [r for r in b if r["evidence_type"] == "readout_tokens"][0]
    assert rd["value_text"] == "Bitcoin | BTC"
    C = list(pack._read_jsonl(tmp_path / "pk" / "concepts.jsonl"))
    assert {c["concept_id"] for c in C} == set(by) and all(c["kind"] == "thing" for c in C)
    assert pack.validate(tmp_path / "pk") == ([], [])


def test_guard_calls_are_kept(tmp_path):
    rec = make_record("Bitcoin")
    for c in rec["rank"]["candidates_all200"]:
        c.update(margin=0.1, call="uncalibrated")
    rec["guards"] = {"band": {"value": None, "source": "none"}, "bar_rank": 30, "bar": 1.1}
    write(tmp_path / "m" / "bitcoin.json", rec)
    pack.export_pack([tmp_path / "m"], tmp_path / "pk", contributor="x", log=lambda *_: None)
    R = rows(tmp_path / "pk")
    ko = [r for r in R if r["evidence_type"] == "knockout_drop" and ":" in r["unit"]][0]
    assert ko["details"]["call"] == "uncalibrated" and ko["details"]["margin"] == 0.1
    assert [r for r in R if r["evidence_type"] == "recall"][0]["details"]["guards"]["bar"] == 1.1


def test_label_run_model_and_parts(tmp_path):
    # the laptop trial: a production run (describe once, judge in slices, merge) with the model only in its results
    src = label_run(tmp_path / "labels")
    for extra in ("intelligibility_run_c0_gate_desc.json", "intelligibility_run_c0_gate_b0.json", "intelligibility_run_c0_gate_b1.json",
                  "intelligibility_run_c0_gate_hedge.json"):
        write(src / extra, {"_summary": {"exam": exam_settings("gate"), "n": 1}, "L1:3": {"bin": "no-generalisation", "auc_own": 0.6,
                                                                                         "auc_control": 0.5, "description": "?"}})
    write(src / "run_ab12_results.json", {"provenance": {"model": "Qwen/Qwen3-0.6B", "package": "modelmapper 0.2.0"}, "neurons": {}})
    res = pack.export_pack([src], tmp_path / "pk", contributor="Ada", log=lambda *_: None)
    assert res["model"] == "qwen3-0.6b"
    E = exps(tmp_path / "pk")
    assert not [e for e in E if e.endswith(("_desc", "_b0", "_b1", "_hedge"))]            # the merged exam holds them
    assert "mm_ada_run_c0_gate" in E


def test_one_model_per_pack(tmp_path):
    write(tmp_path / "a" / "x.json", make_record("X", model="Qwen/Qwen3-0.6B"))
    write(tmp_path / "b" / "y.json", make_record("Y", model="Qwen/Qwen3-8B"))
    with pytest.raises(pack.PackError):
        pack.export_pack([tmp_path / "a", tmp_path / "b"], tmp_path / "pk", contributor="x", log=lambda *_: None)
    with pytest.raises(pack.PackError):
        pack.export_pack([tmp_path / "a"], tmp_path / "pk", contributor="x", model="Qwen/Qwen3-8B", log=lambda *_: None)


def test_cli(records_dir, tmp_path, capsys):
    assert main(["export-pack", str(records_dir), "--out", str(tmp_path / "pk"), "--contributor", "Ada"]) == 0
    assert {p.name for p in (tmp_path / "pk").iterdir()} == {"pack.json", "experiments.jsonl", "evidence.jsonl", "concepts.jsonl"}
    assert main(["export-pack", str(label_run(tmp_path / "l")), "--out", str(tmp_path / "pk2"), "--contributor", "Ada"]) == 2
    assert "--model" in capsys.readouterr().err


def test_no_duckdb_or_yaml_needed(records_dir, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)                   # any import of them now fails
    monkeypatch.setitem(sys.modules, "yaml", None)
    pack.export_pack([records_dir, label_run(tmp_path / "l")], tmp_path / "pk", contributor="x", model="Qwen/Qwen3-0.6B", log=lambda *_: None)
    assert pack.validate(tmp_path / "pk")[0] == []


# ---------------------------------------------------------------- validation (research pack.py validate + PackWriter's checks)
@pytest.fixture
def good(tmp_path):
    d = tmp_path / "pk"
    write(d / "pack.json", {"importer": "mm_t", "contributor": "t", "model": "qwen3-8b"})
    pack._dump(d / "experiments.jsonl", [{"experiment_id": "mm_t_a", "title": "a", "method": "m", "status": "exploratory", "date": "2026-10-21"}])
    pack._dump(d / "evidence.jsonl", [{"experiment": "mm_t_a", "unit": "L1:2", "evidence_type": "label_exam", "metric": "label_auc", "value": 0.8}])
    pack._dump(d / "concepts.jsonl", [{"concept_id": "thing:x", "label": "x", "kind": "thing"}])
    assert pack.validate(d) == ([], [])
    return d


def _ev(d, *rows_):
    pack._dump(d / "evidence.jsonl", list(rows_))
    return pack.validate(d)


def test_validate_pack_json_and_experiments(good):
    write(good / "pack.json", {"contributor": "t"})
    assert "pack.json: 'importer' is required" in pack.validate(good)[0]
    write(good / "pack.json", {"importer": "mm_t"})
    pack._dump(good / "experiments.jsonl", [])
    assert "experiments.jsonl: at least one experiment is required" in pack.validate(good)[0]
    pack._dump(good / "experiments.jsonl", [{"experiment_id": "mm_t_a"}, {"experiment_id": "mm_t_a"}])
    assert any("duplicate experiment_id" in p for p in pack.validate(good)[0])


def test_validate_evidence(good):
    base = {"experiment": "mm_t_a", "unit": "L1:2", "evidence_type": "label_exam", "metric": "label_auc"}
    assert any("unknown fields ['colour']" in p for p in _ev(good, dict(base, colour=1))[0])
    for k in ("experiment", "evidence_type", "metric"):
        assert any(f"'{k}' is required" in p for p in _ev(good, {kk: v for kk, v in base.items() if kk != k})[0])
    assert any("not in experiments.jsonl" in p for p in _ev(good, dict(base, experiment="other"))[0])
    p, w = _ev(good, dict(base, evidence_type="my_new_type"))
    assert p == [] and w == ["new evidence_type 'my_new_type' - add it to vocab.yaml"]
    assert any("not a unit id" in p for p in _ev(good, dict(base, unit="neuron 3"))[0])
    assert any("out of range" in p for p in _ev(good, dict(base, unit="L99:1"))[0])
    assert any("duplicate row key" in p for p in _ev(good, base, dict(base, value=1.0))[0])
    assert _ev(good, base, dict(base, value=1.0, key="second"))[0] == []
    assert _ev(good, base, dict(base, source="f#2"))[0] == []           # the source keeps rows apart


def test_validate_concepts(good):
    pack._dump(good / "concepts.jsonl", [{"concept_id": "colour:red", "label": "red", "kind": "colour"}, {"concept_id": "thing:y", "kind": "thing"}])
    p, w = pack.validate(good)
    assert p == ["concepts.jsonl:2: 'label' is required"] and w == ["new concept kind 'colour' - add it to vocab.yaml"]


def test_ledger_row_normalises_as_packwriter():
    r = pack.ledger_row({"experiment": "e", "unit": "L23H10", "evidence_type": "t", "metric": "m", "value": 1, "n": 3.0, "passes": 1,
                         "value_text": 5, "details": {"a": float("nan")}}, "qwen3-8b")
    assert (r["unit_id"], r["unit_kind"], r["layer"], r["value"], r["n"], r["passes"], r["value_text"]) == ("qwen3-8b/L23/H10", "head", 23, 1.0, 3, True, "5")
    assert r["details"] == '{"a": NaN}'


# ---------------------------------------------------------------- repro: the research run's files -> the ledger's rows
REF = Path(os.environ.get("MODEL_MAPPER_V3_REF", ""))
LEDGER = Path(os.environ.get("MODEL_MAPPER_LEDGER", ""))
RESEARCH = Path(os.environ.get("MODEL_MAPPER_RESEARCH", ""))
V3_FILES = ("intelligibility_v3_gate.json", "gen_check_v3_gate_surf.json")
COLS = ("unit_id", "unit_key", "unit_kind", "layer", "concept_id", "context", "evidence_type", "metric", "value", "value_text", "uom",
        "n", "noise_floor", "threshold", "passes", "rank", "details")
need_v3 = pytest.mark.skipif(not (REF.name and all((REF / f).exists() for f in V3_FILES)),
                             reason="set MODEL_MAPPER_V3_REF to the research run's folder")
need_ledger = pytest.mark.skipif(not (LEDGER.name and LEDGER.exists()), reason="set MODEL_MAPPER_LEDGER to the research ledger.duckdb")
need_research = pytest.mark.skipif(not (RESEARCH.name and (RESEARCH / "src" / "ledger" / "pack.py").exists()),
                                   reason="set MODEL_MAPPER_RESEARCH to the research checkout")


@pytest.fixture(scope="module")
def v3_pack(tmp_path_factory):
    out = tmp_path_factory.mktemp("v3pack")
    pack.export_pack([REF / f for f in V3_FILES], out, contributor="repro", model="Qwen/Qwen3-8B", log=lambda *_: None)
    return out


def _same(a, b, k):
    if k == "details" and a is not None and b is not None:
        a, b = json.dumps(json.loads(a), sort_keys=True), json.dumps(json.loads(b), sort_keys=True)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _ledger(sql):
    duckdb = pytest.importorskip("duckdb")
    with duckdb.connect(str(LEDGER), read_only=True) as c:
        return c.sql(sql).fetchall()


def _compare(ledger_rows, ours, key):
    led = sorted((dict(zip(COLS, r)) for r in ledger_rows), key=key)
    mine = sorted(ours, key=key)
    assert len(led) == len(mine) and len(led) > 0
    diffs = [(key(a), k, a[k], b[k]) for a, b in zip(led, mine) for k in COLS if not _same(a[k], b[k], k)]
    assert not diffs, f"{len(diffs)} differences, e.g. {diffs[:3]}"
    return len(led)


@pytest.mark.repro
@need_v3
@need_ledger
@pytest.mark.parametrize("ours,theirs", [("mm_repro_v3_gate", "jobs_v1_v3_gate"), ("mm_repro_gen_v3_gate_surf", "jobs_v1_v3_gen_gate")])
def test_v3_rows_equal_the_ledgers(v3_pack, ours, theirs):
    mine = [pack.ledger_row(e, "qwen3-8b") for e in rows(v3_pack) if e["experiment"] == ours]
    n = _compare(_ledger(f"select {', '.join(COLS)} from evidence where experiment_id = '{theirs}'"), mine, key=lambda r: r["unit_id"])
    assert n == {"jobs_v1_v3_gate": 950, "jobs_v1_v3_gen_gate": 420}[theirs]
    (p,), = _ledger(f"select params from experiments where experiment_id = '{theirs}'")
    assert exps(v3_pack)[ours]["params"]["settings_hash"] == json.loads(p)["settings_hash"]


@pytest.mark.repro
@need_ledger
def test_run1_record_rows_equal_the_ledgers(tmp_path):
    src = LEDGER.parent.parent / "loop_v2" / "run1" / "results" / "results" / "map" / "b0001" / "records.jsonl"
    if not src.exists():
        pytest.skip(f"no run 1 records at {src}")
    with open(src, encoding="utf-8") as f:
        lines = [next(f) for _ in range(2)]
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "records.jsonl").write_text("".join(lines), encoding="utf-8")
    (p,), = _ledger("select params from experiments where experiment_id = 'loop_v2_run1'")
    floor = json.loads(p)["noise_floor"]["value"]                       # pooled over all of run 1, not these 2 records
    pack.export_pack([tmp_path / "run1"], tmp_path / "pk", contributor="repro", map_noise_floor=floor, log=lambda *_: None)
    mine = [pack.ledger_row(e, "qwen3-8b") for e in rows(tmp_path / "pk")]
    key = lambda r: (r["concept_id"], r["evidence_type"], r["metric"], r["unit_id"] or "")   # noqa: E731
    rel = src.relative_to(LEDGER.parent.parent).as_posix()
    led = _ledger(f"select {', '.join(COLS)} from evidence where source in ('{rel}#0', '{rel}#1')")
    assert _compare(led, mine, key) > 500


def _research(cmd, d, env=None):
    py = os.environ.get("MODEL_MAPPER_RESEARCH_PYTHON") or sys.executable
    r = subprocess.run([py, "-X", "utf8", str(RESEARCH / "src" / "ledger" / "pack.py"), cmd, str(d)], cwd=str(RESEARCH),
                       capture_output=True, text=True, env={**os.environ, **(env or {})}, timeout=600)
    if "No module named" in r.stderr:
        pytest.skip(f"the research Python lacks a module: {r.stderr.strip().splitlines()[-1]}")
    return r


@pytest.mark.repro
@need_research
def test_research_tooling_accepts_a_map_pack(records_dir, tmp_path):
    pack.export_pack([records_dir], tmp_path / "pk", contributor="repro", log=lambda *_: None)
    r = _research("validate", tmp_path / "pk")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stdout + r.stderr
    r = _research("from-jsonl", tmp_path / "pk", env={"LEDGER_DIR": str(tmp_path / "ledger")})     # a scratch ledger, never ours
    assert r.returncode == 0 and "pack mm_repro__" in r.stdout, r.stdout + r.stderr
    assert list((tmp_path / "ledger" / "packs").glob("mm_repro__*/evidence.parquet"))


@pytest.mark.repro
@need_research
@need_v3
def test_research_tooling_accepts_the_v3_pack(v3_pack, tmp_path):
    r = _research("validate", v3_pack)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stdout + r.stderr
    r = _research("from-jsonl", v3_pack, env={"LEDGER_DIR": str(tmp_path / "ledger")})
    assert r.returncode == 0 and "evidence 1,370" in r.stdout, r.stdout + r.stderr
