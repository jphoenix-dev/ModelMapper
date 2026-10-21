"""`modelmapper export-pack`: turn your results into a contribution pack for the research ledger (LEDGER.md,
"Contributing"): the third-party JSON-lines format that the ledger's `python src/ledger/pack.py from-jsonl <dir>` imports.

    from modelmapper.pack import export_pack
    export_pack(["labels/run1", "maps/Qwen3-8B"], "my_pack", contributor="Ada Lovelace", model="Qwen/Qwen3-8B")

    modelmapper export-pack labels/run1 maps/Qwen3-8B --out my_pack --contributor "Ada Lovelace" [--model ID]

Sources (folders, searched recursively, or single files):
  labelling runs   intelligibility_<tag>.json (the exam, as `label` / `exam` write it) and gen_check_<tag>.json (the
                   generative check). Verified subsets (*_ver.json), describe-only and partial files are skipped.
  map records      map-concept / map-batch records (FORMAT.md): <slug>.json files and records.jsonl in one folder per
                   model. A concept's <slug>.json is used when there is one (it is the latest); records.jsonl adds the
                   concepts that have none (its last line per concept).
Output, in out_dir:
  pack.json          importer, contributor, model, inputs, provenance (package version, model, dtype, device, labeller
                     and judge models, quick or calibrated, the `check` result if given)
  experiments.jsonl  one experiment per exam file, per generative-check file and per folder of map records
  evidence.jsonl     the rows the ledger's own importers write from the same files (below)
  concepts.jsonl     the concepts of the map records

THE ROWS ARE OURS. Each row is what the research importer writes from the same file, field for field:
  exam            src/ledger/importers/jobs_v1.py, parser "intelligibility": evidence_type label_exam, metric label_auc,
                  value auc_own, value_text the description, noise_floor auc_control, threshold 0.75, passes bin ==
                  intelligible, details = the exam's scalar fields (SCALARS) + settings_hash, context process=...;board=...
                  Pseudo-neuron exams (the falsifier) are experiments with no rows (their summary is in params), as there.
  generative      jobs_v1.py parser "gen": label_check / generative_auc, threshold 0.75, passes = the check's pass,
                  details bin, median_z_own, single_token_share, tokens, settings_hash.
  settings_hash   jobs_v1.shash of the exam settings (+ process; minus focus, reuse_desc and batches), so the same
                  settings hash the same in your pack and in ours: the ledger's `coverage` view pools them.
  map records     src/ledger/importers/run1.py: per concept, recall (n_recall_prompts), concept_stat (baseline_mean_logodds,
                  k_mean, k_median, k_flip, k_curve), knockout_drop for every candidate (rank by drop, noise_floor =
                  threshold = the pooled 99th percentile of |mean drop| of the folder's random-neuron controls, run 1's
                  rule) and every knocked-out head, null_knockout_drop for each random control, role and readout_tokens
                  for the top 30. Added only when the record has them (records from this package, not run 1's): each
                  candidate's guard call and margin in its knockout details, the record's guards block in the recall
                  details.
Experiment ids are <importer>_<tag>, with importer = importer_name or "mm_<contributor slug>", so they never collide
with ours (our importer names are reserved). A new pack under the same importer name SUPERSEDES that importer's
earlier pack in the ledger (LEDGER.md): use a new importer_name for a contribution that should sit beside an earlier one.

Nothing here needs duckdb or yaml. Copied from the research ledger (src/ledger/, 21 Oct 2026), and kept in step with it:
parse_unit, map_slug, ascii_slug and concept_id (core.py); shash, real and SCALARS (importers/jobs_v1.py); the row
builders (importers/jobs_v1.py and importers/run1.py); validate (pack.py, plus the checks PackWriter makes on import);
the evidence types and concept kinds (vocab.yaml, names only).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics as stx
import time
import unicodedata
from pathlib import Path

FORMAT = "ledger-1.0 third-party JSON lines (LEDGER.md, Contributing)"

# ------------------------------------------------------------------ copied from research src/ledger/core.py
MODELS = {"qwen3-8b": {"hf_name": "Qwen/Qwen3-8B", "n_layers": 36, "d_mlp": 12288, "n_heads": 32}}
KIND_LETTER = {"neuron": "N", "head": "H"}
LETTER_KIND = {v: k for k, v in KIND_LETTER.items()}
_SHORT_N = re.compile(r"^L(\d+)[:_]N?(\d+)$")       # 'L31:8930', 'L0_N492'
_SHORT_H = re.compile(r"^L(\d+)H(\d+)$")            # 'L23H10'
_FULL = re.compile(r"^([^/]+)/L(\d+)/([A-Z])(\d+)$")  # 'qwen3-8b/L31/N8930'
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def parse_unit(u, model="qwen3-8b"):
    """Any unit spelling -> dict(model, kind, layer, index, unit_id, unit_key) (core.parse_unit)."""
    if isinstance(u, (tuple, list)):
        layer, index = int(u[0]), int(u[1]); kind = u[2] if len(u) > 2 else "neuron"
    else:
        s = str(u).strip()
        if m := _FULL.match(s):
            model, layer, kind, index = m.group(1), int(m.group(2)), LETTER_KIND[m.group(3)], int(m.group(4))
        elif m := _SHORT_N.match(s):
            layer, index, kind = int(m.group(1)), int(m.group(2)), "neuron"
        elif m := _SHORT_H.match(s):
            layer, index, kind = int(m.group(1)), int(m.group(2)), "head"
        else:
            raise ValueError(f"not a unit id: {u!r}")
    spec = MODELS.get(model)
    if spec:
        if not 0 <= layer < spec["n_layers"]:
            raise ValueError(f"layer out of range: {u!r}")
        if not 0 <= index < (spec["d_mlp"] if kind == "neuron" else spec["n_heads"]):
            raise ValueError(f"index out of range: {u!r}")
    key = f"L{layer}:{index}" if kind == "neuron" else f"L{layer}H{index}"
    return {"model": model, "kind": kind, "layer": layer, "index": index,
            "unit_id": f"{model}/L{layer}/{KIND_LETTER[kind]}{index}", "unit_key": key}


def map_slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower().replace("&", " and ")).strip("-")
    return s[:80] or "concept"


def ascii_slug(name):
    return map_slug(unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode())


def concept_id(kind, label):
    lab = _ARTICLE.sub("", str(label).strip()) if kind == "thing" else label
    return f"{kind}:{ascii_slug(lab)}"


# ------------------------------------------------------------------ copied from research src/ledger/importers/jobs_v1.py
SCALARS = ("bin", "auc_control", "list_auc_own", "list_auc_control", "label_pass", "list_pass", "kind", "stability", "reach",
           "confidence", "coverage", "p_perm", "stable", "second_draw", "n_hub_dropped", "draw1_margin")
UNHASHED = ("focus", "reuse_desc", "batches")      # per-run fields: kept in params, left out of the fingerprint


def shash(x):
    """The settings fingerprint (jobs_v1.shash): sha256 of the sorted-key JSON, first 12 hex."""
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]


def real(k):
    return isinstance(k, str) and k.startswith("L") and k[1:].split(":")[0].isdigit() and ":" in k


# ------------------------------------------------------------------ copied from research src/ledger/pack.py and vocab.yaml
EV_KEYS = {"experiment", "evidence_type", "metric", "unit", "unit2", "concept", "concept2", "context", "value", "value_text", "uom", "n",
           "noise_floor", "threshold", "passes", "rank", "details", "source", "key"}
EXP_KEYS = ("experiment_id", "title", "method", "status", "date", "script", "reproduce", "data_files", "caveats", "docs", "params",
            "description", "code_commit")
EVIDENCE_TYPES = {"recall", "concept_stat", "knockout_drop", "null_knockout_drop", "attribution", "activation", "activation_auc", "role",
                  "readout_tokens", "position_split", "job_class", "specificity", "sharing", "causal_route", "feeder", "fanout",
                  "dose_response", "digit_shift", "label", "label_exam", "knockout_explanation", "label_check", "web_firing_share",
                  "search_verify", "forced_readout", "specificity_gradient", "shared_use", "feeder_overlap", "job_name", "causal_link",
                  "graph_edge", "confidence_effect", "repeat_selectivity", "group_knockout", "language_auc", "language_necessity",
                  "necessary_call", "mean_ablation_drop", "rival_push"}
CONCEPT_KINDS = {"thing", "era", "relation", "value", "digit", "format"}
# the research ledger's importer names (its packs folder, 21 Oct 2026): a contributor may not use them
RESERVED = {"concept_atlas", "concept_search", "curated_claims", "earlier_maps", "era_formats", "facts2_grid", "facts_grid", "fanout",
            "forced_readout", "indirect_path", "jlens", "job_names", "jobs_v1", "labels", "links_run1", "loop_v2_run1", "loop_v3",
            "loop_v4", "mechanisms", "mechanisms_8b", "numeric_facts", "research_claims", "scales", "search_verify", "shared_neurons",
            "specificity", "web_firings", "jobs", "loop_v2", "run1"}

# ------------------------------------------------------------------ the map records (research src/ledger/importers/run1.py)
KS = [1, 2, 3, 5, 10, 20, 50]
LABEL_PREFIXES = ("intelligibility_", "gen_check_", "gen_words_", "graded_surface_", "vocab_sweep_", "hub_tokens_")


class PackError(ValueError):
    pass


def model_slug(name: str) -> str:
    """'Qwen/Qwen3-8B' -> 'qwen3-8b' (the ledger's model id; already a slug -> unchanged)."""
    return str(name).strip().split("/")[-1].lower()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_slug(s).lower()).strip("_") or "x"


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _split(d):
    s = d.pop("_summary", None) if isinstance(d, dict) else None
    return d, s


# ------------------------------------------------------------------ finding the files
def _scan(sources):
    """-> (exam files, gen-check files, {folder: {"json": [...], "jsonl": [...]}}, results / provenance files)."""
    exams, gens, maps, extra = [], [], {}, []
    files = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and not ({"arrays", "spec_cache"} & set(f.relative_to(p).parts[:-1])))
        elif p.exists():
            files.append(p)
        else:
            raise PackError(f"no such file or folder: {s}")
    seen = set()
    for f in files:
        r = f.resolve()
        if r in seen:
            continue
        seen.add(r)
        n = f.name
        if n.endswith(".partial.json") or f.suffix not in (".json", ".jsonl"):
            continue
        if n.startswith("intelligibility_") and n.endswith(".json"):
            if not f.stem.endswith(("_ver", "_hedge")):
                exams.append(f)
        elif n.startswith("gen_check_") and n.endswith(".json"):
            gens.append(f)
        elif n == "records.jsonl":
            maps.setdefault(f.parent, {"json": [], "jsonl": []})["jsonl"].append(f)
        elif n.endswith("_results.json") or n == "provenance.json":
            extra.append(f)
        elif n.endswith(".json") and not n.startswith(LABEL_PREFIXES):
            maps.setdefault(f.parent, {"json": [], "jsonl": []})["json"].append(f)
    # a production label run describes once, judges in slices and merges (label.board_exam): the merged exam holds every
    # verdict, so its describe-only and slice files are left out rather than exported twice
    merged = {(f.parent, f.stem) for f in exams}
    part = lambda f: re.match(r"(.+)_(desc|b\d+)$", f.stem)                     # noqa: E731
    exams = [f for f in exams if not (part(f) and (f.parent, part(f).group(1)) in merged)]
    return exams, gens, maps, extra


def _is_record(r) -> bool:
    return isinstance(r, dict) and "concept" in r and isinstance(r.get("row"), dict)


# ------------------------------------------------------------------ provenance
def check_summary(report) -> dict | None:
    """A `modelmapper check` report.json (path or dict) -> {overall, version, when, steps: {id: status}}."""
    if report is None:
        return None
    d = _read(Path(report)) if isinstance(report, (str, Path)) else report
    return {"overall": d.get("overall"), "version": d.get("version"), "when": d.get("when"), "steps_run": d.get("steps"),
            "steps": {i.get("id") or i.get("name"): i.get("status") for i in d.get("items", [])}}


def _found_provenance(extra_files, records) -> dict:
    """Provenance written by the run itself: a provenance block in <prefix>_results.json or provenance.json, else the map
    records' own provenance (the first one)."""
    got = {}
    for f in extra_files:
        try:
            d = _read(f)
        except (OSError, ValueError):
            continue
        p = d.get("provenance") if isinstance(d, dict) and f.name != "provenance.json" else d
        if isinstance(p, dict):
            got.update({k: v for k, v in p.items() if v is not None})
    for r in records:
        p = (r.get("v2") or {}).get("provenance") or {}
        for k in ("model", "dtype", "device", "package", "gpu", "torch", "transformers"):
            if p.get(k) is not None:
                got.setdefault(k, p[k])
    return got


def build_provenance(model, given, found, judge=None, check=None) -> dict:
    import modelmapper
    p = {"package": f"modelmapper {modelmapper.__version__}", "model": model, "dtype": None, "device": None, "labeller_model": None,
         "judge_model": judge, "quick": None, "calibrated": None, "check": check}
    p.update({k: v for k, v in found.items() if v is not None})
    p.update({k: v for k, v in (given or {}).items() if v is not None})
    p["exported_by"] = f"modelmapper {modelmapper.__version__} export-pack"
    return {k: ("unknown" if v is None and k not in ("check", "quick", "calibrated") else v) for k, v in p.items()}


# ------------------------------------------------------------------ labelling rows (jobs_v1.py, parsers "intelligibility" and "gen")
def _date(f: Path) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(f.stat().st_mtime))


def _exp(eid, title, method, f, rel, params, description, caveats=()):
    return {"experiment_id": eid, "title": title, "method": method, "status": "exploratory", "date": _date(f),
            "script": "modelmapper (pip package)", "reproduce": "modelmapper label / exam / gencheck (LABELLING.md)",
            "data_files": [rel], "caveats": list(caveats), "docs": ["LABELLING.md"], "params": params, "description": description}


def exam_rows(f: Path, eid: str, rel: str, prov: dict, caveats=()):
    """One exam file -> (experiment, evidence rows), as jobs_v1.py imports intelligibility_<tag>.json."""
    d, s = _split(_read(f))
    if not s or not isinstance(s.get("exam"), dict) or s.get("describe_only"):
        return None, []
    settings = dict(s["exam"])
    board = settings.get("board", "gate")
    process = "gate_exam" if board == "gate" else "side_exam"
    pseudo = bool(settings.get("pseudo"))
    settings["process"] = process
    sh = shash({k: v for k, v in settings.items() if k not in UNHASHED})
    ctx = f"process={process};board={board}"
    units = sorted(k for k in d if real(k))
    params = {"process": process, "board": board, "settings": settings, "settings_hash": sh, "source_file": rel, "summary": s,
              "units": "pseudo-neurons (the falsifier)" if pseudo else f"{len(units)} neurons", "provenance": prov}
    exp = _exp(eid, f"modelmapper labelling: {process} ({f.stem[len('intelligibility_'):]})",
               f"{process}; modelmapper's port of the v3 label exam (LABELLING.md). Settings fingerprint {sh}.", f, rel, params,
               f"{process} run {f.stem}", caveats)
    rows = []
    if pseudo:                                       # jobs_v1 imports pseudo runs as experiments only (parser "summary")
        return exp, rows
    for k, r in d.items():
        if not real(k) or r.get("auc_own") is None and not r.get("description"):
            continue
        rows.append({"experiment": eid, "unit": k, "evidence_type": "label_exam", "metric": "label_auc", "value": r.get("auc_own"),
                     "value_text": r.get("description"), "uom": "AUC", "noise_floor": r.get("auc_control"), "threshold": 0.75,
                     "passes": r.get("bin") == "intelligible", "context": ctx,
                     "details": {kk: r.get(kk) for kk in SCALARS if kk in r} | {"settings_hash": sh}, "source": f"{rel}#{k}"})
    return exp, rows


def gen_board(f: Path, s: dict) -> str:
    """The board a generative check ran on: its summary, else its exam's settings (labels '<tag>_ver' ->
    intelligibility_<tag>.json beside it), else the tag's ending, else gate."""
    if s.get("board"):
        return s["board"]
    lab = str(s.get("labels") or "")
    tag = lab[:-4] if lab.endswith("_ver") else lab
    ex = f.parent / f"intelligibility_{tag}.json"
    if tag and ex.exists():
        with open(ex, encoding="utf-8") as fh:        # the summary is the file's first key: read only the head
            head = fh.read(1 << 16)
        m = re.search(r'"board"\s*:\s*"(gate|apos|aneg)"', head)
        if m:
            return m.group(1)
        try:
            b = ((_read(ex).get("_summary") or {}).get("exam") or {}).get("board")
        except (OSError, ValueError):
            b = None
        if b:
            return b
    for b in ("apos", "aneg", "gate"):
        if re.search(rf"_{b}(_|$)", f.stem):
            return b
    return "gate"


def gen_rows(f: Path, eid: str, rel: str, prov: dict, caveats=()):
    """One generative-check file -> (experiment, rows), as jobs_v1.py imports gen_check_<tag>.json."""
    d, s = _split(_read(f))
    if s is None:
        return None, []
    process = "generative_check"
    control = "surface-only label" if str(s.get("control", "")).startswith("surface") or f.stem.endswith("_surf") else "another neuron's label"
    board = gen_board(f, s)
    settings = {"control": control, "board": board, "process": process}
    sh = shash({k: v for k, v in settings.items() if k not in UNHASHED})
    ctx = f"process={process};control={control}"
    params = {"process": process, "board": board, "settings": settings, "settings_hash": sh, "source_file": rel, "summary": s,
              "control": control, "provenance": prov}
    exp = _exp(eid, f"modelmapper labelling: {process} ({f.stem[len('gen_check_'):]})",
               f"{process}; modelmapper's port of the v3 generative check (LABELLING.md). Settings fingerprint {sh}.", f, rel,
               params, f"{process} run {f.stem}", caveats)
    rows = []
    if s.get("pseudo"):                               # pseudo-neuron checks: experiment only, as jobs_v1 (parser "summary")
        return exp, rows
    for k, r in d.items():
        if real(k):
            rows.append({"experiment": eid, "unit": k, "evidence_type": "label_check", "metric": "generative_auc", "value": r.get("auc"),
                         "value_text": r.get("label"), "uom": "AUC", "threshold": 0.75, "passes": r.get("pass"), "context": ctx,
                         "details": {"bin": r.get("bin"), "median_z_own": r.get("median_z_own"), "single_token_share": r.get("single_token_share"),
                                     "tokens": r.get("tokens"), "settings_hash": sh}, "source": f"{rel}#{k}"})
    return exp, rows


# ------------------------------------------------------------------ map rows (run1.py)
def map_records(folder: Path, files: dict, root_label) -> list[tuple[str, dict]]:
    """[(source locator, record)]: each concept's <slug>.json, plus records.jsonl's last line for concepts without one."""
    out, have = {}, set()
    for f in sorted(files["json"]):
        try:
            r = _read(f)
        except (OSError, ValueError):
            continue
        if _is_record(r):
            out[r["concept"]] = (f"{root_label(f)}#0", r); have.add(r["concept"])
    for f in files["jsonl"]:
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if line.strip():
                    r = json.loads(line)
                    if _is_record(r) and r["concept"] not in have:
                        out.pop(r["concept"], None); out[r["concept"]] = (f"{root_label(f)}#{i}", r)
    return list(out.values())


def pooled_floor(records) -> tuple[float | None, int]:
    """run1.py's noise floor: the 99th percentile of |mean drop| over every random-neuron control, pooled."""
    nulls = [abs(stx.mean(v)) for _, r in records for v in ((r.get("v2") or {}).get("null_single_drops") or {}).values() if v]
    nulls.sort()
    return (nulls[int(0.99 * len(nulls))] if nulls else None), len(nulls)


def map_rows(records, eid: str, floor):
    """The run1.py rows for each record -> (concepts {id: concept}, rows)."""
    concepts, rows = {}, []

    def ev(**kw):
        rows.append({"experiment": eid, **kw})
    for src, r in records:
        c = r["concept"]; row = r.get("row") or {}; rank = r.get("rank") or {}; v2 = r.get("v2") or {}
        cid = concept_id("thing", c)
        cc = concepts.setdefault(cid, {"concept_id": cid, "label": c, "kind": "thing", "aliases": [], "qid": None, "attributes": {}})
        for a in (row.get("answer_form"), row.get("answer")):
            if a and a != cc["label"] and a not in cc["aliases"]:
                cc["aliases"].append(a)
        prov = v2.get("provenance"); n = row.get("recall")
        det = {"answer": row.get("answer"), "answer_form": row.get("answer_form"), "answer_form_scores": row.get("answer_form_scores"),
               "prompts": row.get("recall_prompts"), "provenance": prov, "timing_s": v2.get("timing_s")}
        if r.get("guards") is not None:                    # this package's guards (not in run 1's records)
            det["guards"] = r["guards"]
        ev(concept=cid, evidence_type="recall", metric="n_recall_prompts", value=n, n=20, threshold=5, passes=(n or 0) >= 5,
           value_text=row.get("status"), source=src, details=det)
        if row.get("status") != "ok":
            continue
        for m_ in ("baseline_mean_logodds", "k_mean", "k_median", "k_flip"):
            if row.get(m_) is not None:
                ev(concept=cid, evidence_type="concept_stat", metric=m_, value=row[m_], n=n, source=src,
                   details={"base_per_prompt": row.get("base_per_prompt")} if m_ == "baseline_mean_logodds" else None)
        ev(concept=cid, evidence_type="concept_stat", metric="k_curve", value=None, n=n, source=src,
           details={"ks": KS, "drops": row.get("drops"), "random_drops": row.get("random_drops")})
        ppd = rank.get("per_prompt_drop_all200") or {}
        cands = sorted(rank.get("candidates_all200") or [], key=lambda x: -(x.get("single_drop") or -1e9))
        for k, x in enumerate(cands):
            d = {"attr_rank": x.get("attr_rank"), "act_answer_pos": x.get("act_answer_pos"), "per_prompt": ppd.get(x["neuron"])}
            d.update({g: x[g] for g in ("call", "margin") if g in x})          # this package's guard calls
            ev(unit=x["neuron"], concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop", value=x.get("single_drop"),
               uom="logodds", n=n, noise_floor=floor, threshold=floor, passes=None if floor is None else (x.get("single_drop") or 0) > floor,
               rank=k + 1, context="positions=all;prompts=recall", details=d, source=src)
        for u, pp in (v2.get("null_single_drops") or {}).items():
            ev(unit=u, concept=cid, evidence_type="null_knockout_drop", metric="mean_logodds_drop", value=stx.mean(pp) if pp else None,
               uom="logodds", n=len(pp or []), noise_floor=floor, context="positions=all;prompts=recall;control=random_same_layer",
               details={"per_prompt": pp}, source=src)
        hk = rank.get("head_knockout_mean") or {}; hpp = rank.get("head_knockout_per_prompt") or {}
        hattr = {h: j + 1 for j, h in enumerate(rank.get("heads_top50_by_attr") or [])}
        for k, (h, d) in enumerate(sorted(hk.items(), key=lambda kv: -kv[1])):
            ev(unit=h, concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop", value=d, uom="logodds", n=n, rank=k + 1,
               context="positions=all;prompts=recall", details={"attr_rank": hattr.get(h), "per_prompt": hpp.get(h)}, source=src)
        for x in v2.get("roles_top30") or []:
            ev(unit=x["neuron"], concept=cid, evidence_type="role", metric="weights_role", value_text=x.get("role"),
               details={"answer_rank": x.get("answer_rank"), "answer_z": x.get("answer_z")}, source=src)
            if x.get("job_tokens"):
                ev(unit=x["neuron"], concept=cid, evidence_type="readout_tokens", metric="direct_logit_top30",
                   value_text=" | ".join(t.strip() or repr(t) for t in x["job_tokens"][:10]), details={"tokens": x["job_tokens"]}, source=src)
    return concepts, rows


# ------------------------------------------------------------------ the pack
def importer_for(contributor: str, importer_name: str | None = None) -> str:
    name = importer_name or f"mm_{_slug(contributor)}"
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", name):
        raise PackError(f"importer name {name!r}: use lowercase letters, digits, _ and -")
    if any(r == name or r.startswith(name + "_") or name.startswith(r + "_") for r in RESERVED):
        raise PackError(f"importer name {name!r} is (or overlaps) one of the research ledger's own importers; choose another")
    return name


def _dump(path: Path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=True) + "\n")


def export_pack(sources, out_dir, *, contributor: str, model: str | None = None, importer_name: str | None = None,
                provenance: dict | None = None, check_report=None, map_noise_floor: float | None = None, log=print) -> dict:
    """Write a ledger contribution pack (pack.json, experiments.jsonl, evidence.jsonl, concepts.jsonl) to out_dir.

    sources: labelling-run folders / map-record folders / single files. model: the model the results are from (Hugging
    Face id or ledger slug); default: the map records' own provenance (an error if none says). provenance: fields to
    record that the files don't hold (dtype, device, labeller_model, judge_model, quick, calibrated, ...).
    check_report: a `modelmapper check` report.json, summarised into the provenance. map_noise_floor: use this knockout
    noise floor instead of pooling the folder's own random controls (e.g. run 1's 0.1005, to compare with the ledger).
    Returns a summary {out, importer, model, experiments, evidence, concepts, warnings}."""
    if not str(contributor or "").strip():
        raise PackError("a contributor name is required")
    importer = importer_for(contributor, importer_name)
    exams, gens, maps, extra = _scan(sources)
    roots = [Path(s).resolve() for s in sources]

    def rel(f: Path) -> str:
        r = f.resolve()
        for root in roots:
            base = root if root.is_dir() else root.parent
            try:
                sub = r.relative_to(base).as_posix()
            except ValueError:
                continue
            return base.name if sub == "." else f"{base.name}/{sub}"
        return f.name
    recs_by_folder = {folder: map_records(folder, files, rel) for folder, files in sorted(maps.items())}
    recs_by_folder = {k: v for k, v in recs_by_folder.items() if v}
    all_recs = [r for v in recs_by_folder.values() for _, r in v]
    if not (exams or gens or all_recs):
        raise PackError("nothing to export: no intelligibility_*.json, gen_check_*.json or map records in " + ", ".join(map(str, sources)))
    rec_models = {(r.get("v2") or {}).get("provenance", {}).get("model") for r in all_recs} - {None}
    for f in extra:                                         # `modelmapper label` writes the model into its results' provenance
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m = (d.get("provenance") or {}).get("model") if f.name.endswith("_results.json") else d.get("model")
        if isinstance(m, str) and m:
            rec_models.add(m)
    if model is None:
        if len(rec_models) == 1:
            model = rec_models.pop()
        else:
            raise PackError("give the model (--model): " + ("the records name several: " + ", ".join(sorted(rec_models)) if rec_models
                                                           else "the labelling files don't record it"))
    mslug = model_slug(model)
    wrong = sorted(m for m in rec_models if model_slug(m) != mslug)
    if wrong:
        raise PackError(f"map records from another model ({', '.join(wrong)}) than {model}: one pack per model")
    judge = None
    for f in exams[:1]:
        try:
            judge = (_split(_read(f))[1] or {}).get("exam", {}).get("judge")
        except (OSError, ValueError):
            pass
    prov = build_provenance(model, provenance, _found_provenance(extra, all_recs), judge=judge, check=check_summary(check_report))
    caveats = []
    if prov.get("quick"):
        caveats.append("quick sweep (a seeded 20,000-token subset of the vocabulary): not calibrated against the research run")
    if mslug not in MODELS:
        caveats.append(f"{model}: a model the research ledger has no reference run for; untested combinations are marked in provenance")

    experiments, evidence, concepts, warnings, ids = [], [], {}, [], set()

    def add(exp, rows):
        if exp is None:
            return
        if exp["experiment_id"] in ids:
            raise PackError(f"two sources give experiment {exp['experiment_id']}: export them in separate packs")
        ids.add(exp["experiment_id"]); experiments.append(exp); evidence.extend(rows)
    for f in exams:
        exp, rows = exam_rows(f, f"{importer}_{f.stem[len('intelligibility_'):]}", rel(f), prov, caveats)
        if exp is None:
            warnings.append(f"{f.name}: not a finished exam (no exam settings in its summary), skipped")
        add(exp, rows)
    for f in gens:
        exp, rows = gen_rows(f, f"{importer}_gen_{f.stem[len('gen_check_'):]}", rel(f), prov, caveats)
        if exp is None:
            warnings.append(f"{f.name}: no summary, skipped")
        add(exp, rows)
    for folder, recs in recs_by_folder.items():
        floor, n_nulls = pooled_floor(recs)
        rule = "pooled 99th pct |mean drop| of random controls"
        if map_noise_floor is not None:
            floor, rule = float(map_noise_floor), "given (export_pack map_noise_floor)"
        eid = f"{importer}_maps_{_slug(folder.name)}"
        cs, rows = map_rows(recs, eid, floor)
        for cid, c in cs.items():
            if cid in concepts:
                concepts[cid]["aliases"] += [a for a in c["aliases"] if a not in concepts[cid]["aliases"] and a != concepts[cid]["label"]]
            else:
                concepts[cid] = c
        statuses = {}
        for _, r in recs:
            statuses[(r.get("row") or {}).get("status")] = statuses.get((r.get("row") or {}).get("status"), 0) + 1
        ncand = max((len((r.get("rank") or {}).get("candidates_all200") or []) for _, r in recs), default=0)
        params = {"process": "map_concept", "n_candidates": ncand, "ks": KS, "min_recall": 5,
                  "noise_floor": {"rule": rule, "value": floor, "n": n_nulls}, "records": len(recs), "status": statuses,
                  "source_folder": rel(folder), "provenance": prov}
        f0 = Path(folder)
        exp = {"experiment_id": eid, "title": f"modelmapper map-concept records ({folder.name})",
               "method": ("map-concept / map-batch (METHOD.md): recall questions, attribution, the top candidates each zeroed alone "
                          "and ranked by the drop in the answer's log-odds, k curves, random same-layer controls, head knockouts, "
                          "weights-based roles. Rows as the research run-1 importer writes them."),
               "status": "exploratory", "date": _date(f0), "script": "modelmapper map-concept / map-batch", "reproduce": "modelmapper map-batch <concepts> (README)",
               "data_files": sorted({s.split("#")[0] for s, _ in recs}), "caveats": caveats + ["Format-specific: every question asks to 'Answer with just the name'."],
               "docs": ["METHOD.md", "FORMAT.md"], "params": params, "description": f"{len(recs)} concept records from {folder.name}"}
        add(exp, rows)
    if not experiments:
        raise PackError("nothing to export: " + "; ".join(warnings))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = {"importer": importer, "contributor": contributor, "model": mslug,
            "inputs": sorted({rel(f) for f in exams + gens} | {s.split("#")[0] for v in recs_by_folder.values() for s, _ in v}),
            "supersedes": [], "format": FORMAT, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provenance": prov}
    (out / "pack.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    _dump(out / "experiments.jsonl", experiments)
    _dump(out / "evidence.jsonl", evidence)
    if concepts:
        _dump(out / "concepts.jsonl", concepts.values())
    elif (out / "concepts.jsonl").exists():
        (out / "concepts.jsonl").unlink()
    problems, warns = validate(out)
    if problems:
        raise PackError("the pack does not validate:\n" + "\n".join(problems))
    res = {"out": str(out), "importer": importer, "model": mslug, "experiments": len(experiments), "evidence": len(evidence),
           "concepts": len(concepts), "warnings": warnings + warns}
    log(f"pack {out}: {len(experiments)} experiments, {len(evidence)} evidence rows, {len(concepts)} concepts (importer {importer}, model {mslug})")
    for w in res["warnings"]:
        log(f"  warning: {w}")
    return res


# ------------------------------------------------------------------ validation
def _read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def ledger_row(e: dict, model: str) -> dict:
    """An evidence.jsonl row as the ledger stores it (core.PackWriter.evidence, without the pack-dependent ids): units
    parsed, numbers as float, details as JSON text. Raises ValueError where PackWriter would."""
    u = parse_unit(e["unit"], model) if e.get("unit") is not None else None
    u2 = parse_unit(e["unit2"], model) if e.get("unit2") is not None else None
    num = lambda x: None if x is None else float(x)   # noqa: E731
    return {"experiment_id": e.get("experiment"), "model_id": model,
            "unit_id": u and u["unit_id"], "unit_key": u and u["unit_key"], "unit_kind": u and u["kind"], "layer": u and u["layer"],
            "unit2_id": u2 and u2["unit_id"], "unit2_key": u2 and u2["unit_key"],
            "concept_id": e.get("concept"), "concept2_id": e.get("concept2"), "context": e.get("context"),
            "evidence_type": e.get("evidence_type"), "metric": e.get("metric"), "value": num(e.get("value")),
            "value_text": None if e.get("value_text") is None else str(e["value_text"]), "uom": e.get("uom"),
            "n": None if e.get("n") is None else int(e["n"]), "noise_floor": num(e.get("noise_floor")), "threshold": num(e.get("threshold")),
            "passes": None if e.get("passes") is None else bool(e["passes"]), "rank": None if e.get("rank") is None else int(e["rank"]),
            "details": None if e.get("details") is None else json.dumps(e["details"], ensure_ascii=False, allow_nan=True),
            "source": e.get("source")}


def row_key(e: dict, model: str) -> str:
    """The evidence row key PackWriter derives (its id is sha256 of pack id + this): unique within a pack."""
    u = parse_unit(e["unit"], model)["unit_id"] if e.get("unit") is not None else None
    u2 = parse_unit(e["unit2"], model)["unit_id"] if e.get("unit2") is not None else None
    return e.get("key") or "|".join(str(x) for x in (e.get("experiment"), u, u2, e.get("concept"), e.get("concept2"), e.get("context"),
                                                     e.get("evidence_type"), e.get("metric"), e.get("source")))


def validate(d) -> tuple[list[str], list[str]]:
    """(problems, warnings) for a pack folder. The research pack.py validate's rules, plus what PackWriter checks on
    import (experiment ids unique, units parse, numbers convert, row keys unique, concepts complete)."""
    d = Path(d); problems, warnings = [], []
    try:
        meta = json.loads((d / "pack.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"pack.json: {e}"], []
    if not meta.get("importer"):
        problems.append("pack.json: 'importer' is required")
    model = meta.get("model", "qwen3-8b")
    exp_list = list(_read_jsonl(d / "experiments.jsonl")) if (d / "experiments.jsonl").exists() else []
    exps = set()
    for i, e in enumerate(exp_list):
        if not e.get("experiment_id"):
            problems.append(f"experiments.jsonl:{i + 1}: 'experiment_id' is required"); continue
        if e["experiment_id"] in exps:
            problems.append(f"experiments.jsonl:{i + 1}: duplicate experiment_id {e['experiment_id']!r}")
        exps.add(e["experiment_id"])
        if extra := set(e) - set(EXP_KEYS):
            warnings.append(f"experiments.jsonl: fields {sorted(extra)} are not imported (put them in 'params')")
    if not exps:
        problems.append("experiments.jsonl: at least one experiment is required")
    keys = set()
    for i, e in enumerate(_read_jsonl(d / "evidence.jsonl") if (d / "evidence.jsonl").exists() else []):
        if extra := set(e) - EV_KEYS:
            problems.append(f"evidence.jsonl:{i + 1}: unknown fields {sorted(extra)} (put extra data in 'details')")
        for k in ("experiment", "evidence_type", "metric"):
            if not e.get(k):
                problems.append(f"evidence.jsonl:{i + 1}: '{k}' is required")
        if e.get("experiment") not in exps:
            problems.append(f"evidence.jsonl:{i + 1}: experiment {e.get('experiment')!r} not in experiments.jsonl")
        if e.get("evidence_type") and e["evidence_type"] not in EVIDENCE_TYPES:
            warnings.append(f"new evidence_type {e['evidence_type']!r} - add it to vocab.yaml")
        try:
            ledger_row(e, model)
            k = row_key(e, model)
            if k in keys:
                problems.append(f"evidence.jsonl:{i + 1}: duplicate row key {k} (pass key= to disambiguate)")
            keys.add(k)
        except (ValueError, TypeError, KeyError) as x:
            problems.append(f"evidence.jsonl:{i + 1}: {x}")
        if len(problems) > 50:
            break
    if (d / "concepts.jsonl").exists():
        for i, c in enumerate(_read_jsonl(d / "concepts.jsonl")):
            for k in ("concept_id", "label", "kind"):
                if not c.get(k):
                    problems.append(f"concepts.jsonl:{i + 1}: '{k}' is required")
            if c.get("kind") not in CONCEPT_KINDS:
                warnings.append(f"new concept kind {c.get('kind')!r} - add it to vocab.yaml")
    return problems, sorted(set(warnings))


# ------------------------------------------------------------------ CLI
def add_parser(sub):
    x = sub.add_parser("export-pack", help="turn your results into a contribution pack for the research ledger",
                       description="Labelling runs (intelligibility_*.json, gen_check_*.json) and map-concept / map-batch "
                                   "records -> a ledger contribution pack (pack.json, experiments.jsonl, evidence.jsonl, "
                                   "concepts.jsonl) with the same rows the research importers write (modelmapper/pack.py).")
    x.add_argument("sources", nargs="+", help="folders (searched recursively) or files")
    x.add_argument("--out", required=True, help="the pack folder to write")
    x.add_argument("--contributor", required=True, help="your name or team (goes in pack.json)")
    x.add_argument("--model", help="the model the results are from (default: what the map records say; needed for labelling runs)")
    x.add_argument("--importer", help="importer name (default mm_<contributor>); experiment ids are <importer>_<tag>")
    x.add_argument("--check-report", help="a `modelmapper check` report.json to record in the provenance")
    x.add_argument("--provenance", help='JSON of fields the files don\'t hold, e.g. \'{"dtype": "bfloat16", "labeller_model": "gpt-6-luna"}\'')
    return x


def cli(a) -> int:
    import sys
    try:
        export_pack(a.sources, a.out, contributor=a.contributor, model=a.model, importer_name=a.importer,
                    provenance=json.loads(a.provenance) if a.provenance else None, check_report=a.check_report)
    except PackError as e:
        print(f"export-pack: {e}", file=sys.stderr)
        return 2
    return 0
