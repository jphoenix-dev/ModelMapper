"""`modelmapper check --labels`: does labelling behave on this machine and this model as it did on ours? Three steps,
reported by `modelmapper check` (selfcheck.py) as PASS / WARN / FAIL / SKIPPED next to our values:

  sweep1     (the model; no API) a fixed slice of the vocabulary sweep: the SWEEP1_TOKENS tokens that
             sweep.allowed_tokens draws with its fixed seed, in panel contexts 0 and 1, slot reading, with full gate
             (z_g) and up (z_u) profiles for 6 marker neurons. Compared with our stored profiles, per profile (6 neurons
             x 2 contexts x gate / up = 24): the correlation and the largest absolute difference, also as a share of that
             profile's spread (sd over the tokens). PASS: every correlation >= SWEEP1_CORR and every difference <=
             SWEEP1_SD x sd; WARN: correlations >= SWEEP1_WARN_CORR (same neurons, numbers outside the band); else
             FAIL (another model, tokenizer, panel or a broken reading). A different token slice is a FAIL.
             The band: Qwen3-0.6B on one CPU, bfloat16 against float32, differs by at most 0.25 sd (correlation >=
             0.9992); float32 with batch 64 against batch 1024 is identical once stored as float16 (CALIBRATION.md).
  exam1      (judge only) about 20 neurons' gate exams judged again from shipped inputs: their descriptions, test
             lists (favourites and surface-matched decoys, every draw), the control neurons' descriptions and shown
             lists (exam.fixed_inputs / exam.rejudge). PASS when >= EXAM1_SAME of 20 final bins equal ours, else WARN.
  falsifier  (labeller + judge, about $0.50) on the model under check: a quick seeded sweep (FALSIFIER_TOKENS tokens),
             then FALSIFIER_N random real neurons and FALSIFIER_N fake ones (pseudo.py's up set: random gate and up
             directions) labelled for real by label.label_run (exams on all three parts, generative checks). PASS when,
             on every part, at most FAKE_MAX fakes get a verified label that also passes the generative check (the
             falsifier column of LABELLING.md) and the median control AUC of the real neurons' exams is within
             CTRL_BAND. The real neurons' rates are reported, not judged: there is no bar for an unknown model.

References ship in modelmapper/data/check/ (sweep1_<model>.npz, exam1_<model>.json.gz) for Qwen3-0.6B and Qwen3-8B.
A model without them gets sweep1 and exam1 SKIPPED and the report says labelling is uncalibrated for it (internal
controls only). The tested combination is labeller gpt-6-luna with the Jev judge, labelling Qwen3-8B and Qwen3-0.6B;
anything else is reported as untested. The build_* functions made the shipped files (our side; run once).
"""
from __future__ import annotations

import gzip
import json
import platform
import time
from pathlib import Path

import numpy as np

from modelmapper.settings import short_name

DATA = Path(__file__).resolve().parent / "data" / "check"
TESTED_LABELLER, TESTED_JUDGE = "gpt-6-luna", "jev"
TESTED_MODELS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-0.6B")
SWEEP1_TOKENS, SWEEP1_CONTEXTS, SWEEP1_READ = 512, (0, 1), "slot"
SWEEP1_CORR, SWEEP1_SD, SWEEP1_WARN_CORR = 0.995, 0.5, 0.98
EXAM1_SAME = 18                 # of 20: final bins equal to ours for a PASS, else WARN
FALSIFIER_TOKENS, FALSIFIER_N, FALSIFIER_SEED = 3000, 20, 2026
FAKE_MAX, CTRL_BAND = 1, (0.40, 0.60)
FAKE_FAIL, CTRL_FAIL = 3, (0.30, 0.70)     # beyond these the falsifier FAILs; in between it WARNs (draws vary)
LUNA_CAP, JEV_CAP = 0.50, 0.50          # falsifier spend caps (USD; label_run stops before new work past them)
TAG = "fq"                              # the falsifier's sweep tag
# the 6 marker neurons per model: Qwen3-0.6B, the answer key's three markers and three neurons of the key's Bitcoin map
# (CALIBRATION.md); Qwen3-8B, the v3 run's four check rows (sweep.CHECK_V3) and the sample's first neurons in layers
# 16 and 35 (filling the gaps between them)
MARKERS = {"Qwen/Qwen3-0.6B": ["L4:2581", "L9:1456", "L23:487", "L23:2069", "L23:2392", "L27:81"],
           "Qwen/Qwen3-8B": ["L0:1951", "L7:11335", "L16:11480", "L25:7850", "L31:10454", "L35:8545"]}
SHORT = {short_name(m): m for m in TESTED_MODELS}


def full_name(model: str) -> str:
    """'Qwen3-8B' -> 'Qwen/Qwen3-8B' for the models we ship references for; anything else unchanged."""
    return SHORT.get(model, model)


def ref_path(model: str, kind: str) -> Path:
    return DATA / (f"sweep1_{short_name(model)}.npz" if kind == "sweep1" else f"exam1_{short_name(model)}.json.gz")


def load_sweep_ref(model: str, data=None) -> dict | None:
    """The shipped sweep1 reference for this model, or None: tokens, sample, c<k>_z / c<k>_u [tokens, 6] and meta."""
    p = (Path(data) if data else DATA) / ref_path(model, "sweep1").name
    if not p.exists():
        return None
    with np.load(p) as f:
        ref = {k: f[k] for k in f.files if k != "meta"}
        ref["meta"] = json.loads(str(f["meta"]))
    ref["sample"] = [str(x) for x in ref["sample"]]
    return ref


def load_exam_ref(model: str, data=None) -> dict | None:
    p = (Path(data) if data else DATA) / ref_path(model, "exam1").name
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def combination(model: str, labeller_model: str, judge_backend: str) -> tuple[bool, str]:
    """(tested?, text) for the report."""
    ok = labeller_model == TESTED_LABELLER and judge_backend == TESTED_JUDGE and full_name(model) in TESTED_MODELS
    txt = f"labeller {labeller_model}, judge {'Jev' if judge_backend == 'jev' else judge_backend}, model {short_name(model)}"
    return ok, txt + (": the tested combination" if ok else
                      ": untested (tested: labeller gpt-6-luna and judge Jev, labelling on Qwen3-8B and Qwen3-0.6B)")


# ------------------------------------------------------------------------------------------------ sweep1
def sweep_slice(model, tok, out_dir, sample, log=lambda *_: None) -> dict:
    """Run the sweep1 slice on a loaded model; -> {tokens, sample, c<k>_z, c<k>_u} (float16, as the sweep stores them)."""
    from modelmapper import sweep
    sweep.run_sweep(model, tok, out_dir, "sweep1", sample=list(sample), max_tokens=SWEEP1_TOKENS, contexts=list(SWEEP1_CONTEXTS),
                    log=log)
    got = {}
    for c in SWEEP1_CONTEXTS:
        with np.load(Path(out_dir) / f"vocab_sweep_sweep1_c{c}.npz") as f:
            got["tokens"] = f["tokens"].astype(np.int64); got["sample"] = [str(x) for x in f["sample"]]
            got[f"c{c}_z"] = f[f"{SWEEP1_READ}_sample_z"]; got[f"c{c}_u"] = f[f"{SWEEP1_READ}_sample_u"]
    return got


def compare_profiles(ref: dict, got: dict) -> dict:
    """Per profile (neuron x context x gate / up): correlation, largest |difference| and that over the profile's sd."""
    if list(ref["sample"]) != list(got["sample"]):
        return {"error": f"different marker neurons ({got['sample']} against {ref['sample']})"}
    if not np.array_equal(np.asarray(ref["tokens"], np.int64), np.asarray(got["tokens"], np.int64)):
        return {"error": f"a different token slice ({len(got['tokens'])} tokens; the tokenizer differs from ours)"}
    rows = []
    for c in SWEEP1_CONTEXTS:
        for part in ("z", "u"):
            A = np.asarray(ref[f"c{c}_{part}"], np.float32); B = np.asarray(got[f"c{c}_{part}"], np.float32)
            for j, s in enumerate(ref["sample"]):
                a, b = A[:, j], B[:, j]; d = float(np.abs(a - b).max()); sd = float(a.std())
                corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else (1.0 if d == 0 else 0.0)
                rows.append({"neuron": s, "context": c, "part": "gate" if part == "z" else "up", "corr": corr, "max_abs": d,
                             "max_sd": d / sd if sd > 0 else (0.0 if d == 0 else float("inf"))})
    return {"rows": rows, "n": len(rows), "corr_min": min(r["corr"] for r in rows), "max_abs": max(r["max_abs"] for r in rows),
            "max_sd": max(r["max_sd"] for r in rows)}


def sweep1_status(cmp: dict) -> str:
    if "error" in cmp:
        return "FAIL"
    if cmp["corr_min"] >= SWEEP1_CORR and cmp["max_sd"] <= SWEEP1_SD:
        return "PASS"
    return "WARN" if cmp["corr_min"] >= SWEEP1_WARN_CORR else "FAIL"


def sweep1_value(cmp: dict) -> str:
    if "error" in cmp:
        return cmp["error"]
    worst = max(cmp["rows"], key=lambda r: r["max_sd"])
    return (f"max abs difference {cmp['max_abs']:.3f} ({cmp['max_sd']:.2f} sd, {worst['neuron']} {worst['part']} context "
            f"{worst['context']}); min correlation {cmp['corr_min']:.4f} over {cmp['n']} profiles x {SWEEP1_TOKENS} tokens")


# ------------------------------------------------------------------------------------------------ exam1
def exam1_compare(ref: dict, res: dict) -> tuple[int, int, list[str]]:
    """(same final bins, neurons, one line per neuron that differs)"""
    same, rows = 0, []
    for s, r in ref["neurons"].items():
        a, b = res[s]["bin"], r["ours"]["bin"]
        same += a == b
        if a != b:
            rows.append(f"{s}: {a} (ours {b}; AUC {res[s].get('auc_own')} / {r['ours'].get('auc_own')})")
    return same, len(ref["neurons"]), rows


def exam1_status(same: int, n: int) -> str:
    return "PASS" if same >= EXAM1_SAME * n / 20 else "WARN"


# ------------------------------------------------------------------------------------------------ falsifier
def falsifier_neurons(n_layers: int, width: int, n: int = FALSIFIER_N, seed: int = FALSIFIER_SEED) -> list[str]:
    """n random real neurons (seeded; drawn uniformly over every layer's neurons)."""
    flat = np.random.default_rng(seed).choice(n_layers * width, n, replace=False)
    return [f"L{int(f) // width}:{int(f) % width}" for f in flat]


def _hub_counts(npz_path: Path, reads, V: int, per_layer, per_read: list, names: list, k: int):
    with np.load(npz_path) as f:
        for r in reads:
            ti = f[f"{r}_top_i"]
            per_read.append(np.bincount(ti.reshape(-1), minlength=V)); names.append(f"c{k}:{r}")
            for l in range(ti.shape[0]):
                per_layer[l] += np.bincount(ti[l].reshape(-1), minlength=V)


def _shrink(npz_path: Path, keep: list[tuple[int, int]]):
    """Rewrite one context's sweep file keeping the favourites of `keep` only (zeros elsewhere, compressed): every
    other array (statistics, pseudo profiles, tokens) unchanged. A 3,000-token sweep of Qwen3-0.6B is ~4.5 GB in full."""
    with np.load(npz_path) as f:
        arrs = {k: f[k] for k in f.files}
    ul, ui = np.array([l for l, _ in keep]), np.array([i for _, i in keep])
    for key, a in arrs.items():
        if a.ndim == 3 and any(key.endswith(f"_{ls}_{x}") for ls in ("top", "apos", "aneg") for x in ("i", "v", "g", "u")):
            z = np.zeros_like(a); z[ul, ui] = a[ul, ui]; arrs[key] = z
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrs); tmp.replace(npz_path)


def quick_sweep(model, tok, sweep_dir, keep, pseudo_set, *, n_tokens=FALSIFIER_TOKENS, log=print) -> dict:
    """The falsifier's sweep: every panel context, n_tokens seeded tokens, real neurons and the pseudo set in one pass.
    Context by context, the hub-token counts are taken from the full favourites, then the file keeps only `keep`'s
    favourites. Writes hub_tokens_<TAG>.npz as sweep.hub_tokens does. Skips contexts already done (resumable)."""
    from modelmapper import sweep
    D = Path(sweep_dir); D.mkdir(parents=True, exist_ok=True)
    lay = model.model.layers; NL = len(lay); V = model.get_input_embeddings().weight.shape[0]
    ctxs = sweep.panel_contexts(tok); per_layer = np.zeros((NL, V), np.int64); per_read, names = [], []
    hub = D / f"hub_tokens_{TAG}.npz"
    if hub.exists() and (D / f"vocab_sweep_{TAG}.json").exists():
        return json.loads((D / f"vocab_sweep_{TAG}.json").read_text(encoding="utf-8"))
    meta = None
    for k, c in enumerate(ctxs):
        meta = sweep.run_sweep(model, tok, D, TAG, pseudo=pseudo_set, max_tokens=n_tokens, contexts=[k], log=log)
        p = D / f"vocab_sweep_{TAG}_c{k}.npz"
        _hub_counts(p, c["read"], V, per_layer, per_read, names, k)
        _shrink(p, keep)
    np.savez_compressed(hub, per_read=np.stack(per_read).astype(np.int32), per_layer=per_layer.astype(np.int32), readings=np.array(names))
    return meta


def falsifier_run(model, tok, model_name: str, out_dir, *, n: int = FALSIFIER_N, n_tokens: int = FALSIFIER_TOKENS,
                  luna_cap: float = LUNA_CAP, jev_cap: float = JEV_CAP, log=print) -> dict:
    """The falsifier on a loaded model: returns its summary (also written to falsifier.json in out_dir)."""
    from modelmapper import label, pseudo
    from modelmapper.weights import Weights
    D = Path(out_dir); D.mkdir(parents=True, exist_ok=True); t0 = time.time()
    lay = model.model.layers; NL, NI = len(lay), lay[0].mlp.gate_proj.weight.shape[0]
    real = falsifier_neurons(NL, NI, n)
    pq = D / "pseudo_up.npz"
    if pq.exists():
        up = pseudo.load(pq)
    else:
        W = Weights(model_name)
        gs = pseudo.make_gate_set(W, [int(s[1:].split(":")[0]) for s in real], n=n, log=lambda *_: None)
        up = pseudo.make_up_set(W, gs); pseudo.save(up, pq)
    fakes = [str(x) for x in up["names"]]
    t1 = time.time()
    quick_sweep(model, tok, D / "sweep", [tuple(map(int, s[1:].split(":"))) for s in real], up, n_tokens=n_tokens, log=log)
    t2 = time.time()
    res = label.label_run(real, D / "labels", "fz", D / "sweep", TAG, chunk=n, pseudo_sweep_tag=TAG, pseudo_set=pq,
                          pseudo_focus=fakes, model=model, tok=tok, model_name=model_name, luna_cap=luna_cap, jev_cap=jev_cap,
                          log=log)
    t3 = time.time()
    summ = summarise_falsifier(res, D / "labels", n)
    summ.update({"model": model_name, "real_neurons": real, "fake_neurons": fakes, "n_tokens": n_tokens,
                 "seconds": {"pseudo": round(t1 - t0), "sweep": round(t2 - t1), "label": round(t3 - t2), "total": round(t3 - t0)}})
    (D / "falsifier.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return summ


def summarise_falsifier(res: dict, labels_dir, n: int) -> dict:
    """Counts from a label_run result: fakes verified (exam) and verified + generative per part, the real neurons'
    rates, and the median control AUC over the real neurons' exams (every part)."""
    D = Path(labels_dir); boards = ("gate", "apos", "aneg")
    fake = {b: {"n": v["n"], "verified": round(v["verified"] * v["n"]), "verified_and_generative": round(v["verified_and_generative"] * v["n"])}
            for b, v in (res.get("pseudo") or {}).items()}
    ctl, ctl_by = [], {}
    for b in boards:
        vals = []
        for f in sorted(D.glob(f"intelligibility_fz_c*_{b}.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            vals += [r["auc_control"] for k, r in d.items() if not k.startswith("_") and r.get("auc_control") is not None]
        ctl += vals; ctl_by[b] = round(float(np.median(vals)), 3) if vals else None
    s = res.get("summary", {})
    real = {b: {k: s.get(b, {}).get(k) for k in ("n", "verified", "verified_and_generative")} for b in boards if b in s}
    return {"fakes": fake, "real": real, "ctrl_auc_median": round(float(np.median(ctl)), 3) if ctl else None,
            "ctrl_auc_median_by_part": ctl_by, "spend_usd": res.get("spend_usd"), "n": n}


def falsifier_status(summ: dict) -> str:
    worst = max([v["verified_and_generative"] for v in summ["fakes"].values()] or [99])
    c = summ.get("ctrl_auc_median")
    if c is None or not summ["fakes"]:
        return "FAIL"
    if worst <= FAKE_MAX and CTRL_BAND[0] <= c <= CTRL_BAND[1]:
        return "PASS"
    return "FAIL" if worst >= FAKE_FAIL or not CTRL_FAIL[0] <= c <= CTRL_FAIL[1] else "WARN"


def falsifier_value(summ: dict) -> str:
    n = summ["n"]
    fk = ", ".join(f"{b} {v['verified_and_generative']}/{v['n']}" for b, v in summ["fakes"].items())
    fe = ", ".join(f"{b} {v['verified']}" for b, v in summ["fakes"].items())
    rl = ", ".join(f"{b} {v['verified']:.0%} ({v['verified_and_generative']:.0%} with the generative check)"
                   for b, v in summ["real"].items() if v.get("verified") is not None)
    sp = summ.get("spend_usd") or {}
    return (f"fakes verified + generative: {fk} (exam alone: {fe}); control AUC median {summ['ctrl_auc_median']}; "
            f"real neurons (reported only): {rl}; {n} + {n} neurons, {summ['n_tokens']:,} tokens; "
            f"spend labeller ${sp.get('labeller', 0):.3f}, judge ${sp.get('judge', 0):.3f}")


# ------------------------------------------------------------------------------------------------ building the references (ours)
def build_sweep_ref_research(sweep_files: dict, tokenizer: str, sample: list[str], out: Path, meta: dict) -> Path:
    """Qwen3-8B: the research run's stored profiles for the slice's tokens and the markers. sweep_files: {k: path of
    vocab_sweep_<tag>_c<k>.npz}; each key is read alone (memory-mapped), only the slice's rows and the markers' columns."""
    from transformers import AutoTokenizer

    from modelmapper import sweep
    from modelmapper.exam import _Npz
    from modelmapper.settings import model_path
    tok = AutoTokenizer.from_pretrained(model_path(tokenizer))
    cfg = json.loads((Path(model_path(tokenizer)) / "config.json").read_text(encoding="utf-8"))
    ids = sweep.allowed_tokens(tok, cfg["vocab_size"], SWEEP1_TOKENS).astype(np.int64)
    arrs = {"tokens": ids, "sample": np.array(sample)}
    for c in SWEEP1_CONTEXTS:
        with _Npz(sweep_files[c]) as f:
            toks = np.asarray(f["tokens"]).astype(np.int64); names = [str(x) for x in np.asarray(f["sample"])]
            pos = {int(t): j for j, t in enumerate(toks)}; rows = np.array([pos[int(t)] for t in ids])
            cols = np.array([names.index(s) for s in sample])
            for part in ("z", "u"):
                M = f[f"{SWEEP1_READ}_sample_{part}"]
                arrs[f"c{c}_{part}"] = np.asarray(M[rows][:, cols]).astype(np.float16)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs, meta=np.array(json.dumps(meta)))
    return out


def build_sweep_ref_local(model, tok, sample: list[str], out: Path, meta: dict, work: Path) -> Path:
    """Qwen3-0.6B: this package's sweep on our machine (meta records the machine, device and dtype)."""
    got = sweep_slice(model, tok, work, sample)
    arrs = {k: v for k, v in got.items() if k != "sample"}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs, sample=np.array(got["sample"]), meta=np.array(json.dumps(meta)))
    return out


def build_exam_ref(sweep_dir, sweep_tag: str, pool: list[str], judged: list[str], exam_file, tokenizer: str, out: Path,
                   meta: dict) -> Path:
    """The exam1 extract (exam.fixed_inputs) of a finished gate exam, gzipped JSON."""
    from modelmapper import exam
    fx = exam.fixed_inputs(sweep_dir, sweep_tag, pool, judged, exam_file, tokenizer=tokenizer)
    fx["meta"] = meta
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(fx, f, ensure_ascii=False, separators=(",", ":"))
    return out


def machine() -> str:
    """For a reference's meta: OS, CPU architecture and thread count (no host or user name)."""
    import torch
    return f"{platform.system()} {platform.release()}, {platform.machine()}, {torch.get_num_threads()} threads, torch {torch.__version__}"
