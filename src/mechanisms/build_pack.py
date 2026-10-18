"""Turn the MECHANISMS.md test outputs (D:/already-clean-data/mechanisms/) into ledger contribution packs (JSON lines).

Two packs, because a pack carries one model id:
  src/ledger/contrib/mechanisms/     model qwen3-0.6b - per-unit evidence from the CPU tests (H1-H4) + all claims
  src/ledger/contrib/mechanisms_8b/  model qwen3-8b   - H6 rival push (weights only) + the H1 sign-flip check on run 1
Every per-unit / per-concept measurement is one evidence row (no aggregation away); group-level results (e.g. zeroing a
set of 20 neurons) are rows without a unit, with the set in details. Then:
  python src/ledger/pack.py validate src/ledger/contrib/mechanisms && python src/ledger/pack.py from-jsonl ...
Usage: python -X utf8 src/mechanisms/build_pack.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ledger.core import concept_id  # noqa: E402

DATA = Path("D:/already-clean-data/mechanisms")
C06 = ROOT / "src" / "ledger" / "contrib" / "mechanisms"
C8 = ROOT / "src" / "ledger" / "contrib" / "mechanisms_8b"
DATE = "2026-10-18"
COMMON = {"date": DATE, "docs": ["MECHANISMS.md"]}


def load(p):
    p = DATA / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows))


def pack06():
    exps, ev, concepts = [], [], {}
    # ---- H1 inhibitors
    for sub, X, status in (("inhibitors", "mech_inhibitors_06b", "pre-registered (MECHANISMS.md H1): see claims"),
                           ("inhibitors_seed1", "mech_inhibitors_06b_retest", "re-test of a positive result: same rule, new concept order (seed 1), 20 concepts")):
        S = load(f"{sub}/summary.json")
        if not S:
            continue
        exps.append({"experiment_id": X, "title": "Inhibitor neurons on Qwen3-0.6B (knockout raises the answer)",
                     "method": "Run 1's recall prompts and answers, concepts 0.6B has in reach (log-odds > -2 on >= 6 prompts). Attribution on half the prompts; "
                               "the 20 most negative, 20 most positive, 20 random and 20 layer-matched random neurons each zeroed alone on the other half; "
                               "change in mean exact log-odds. Inhibitor = bottom neuron raising the held-out answer above the null's 99th percentile.",
                     "status": status, "script": "src/mechanisms/inhibitors.py",
                     "reproduce": "python -X utf8 src/mechanisms/inhibitors.py " + ("20 1" if sub.endswith("seed1") else "40 0"), "data_files": [f"mechanisms/{sub}/per_concept.jsonl"],
                     "params": {k: S[k] for k in S if k.startswith("P1") or k.startswith("null")}, **COMMON})
        thr = S["null_p99_rise"]
        for line in (DATA / sub / "per_concept.jsonl").open(encoding="utf-8"):
            r = json.loads(line); cid = concept_id("thing", r["concept"])
            concepts[cid] = {"concept_id": cid, "label": r["concept"], "kind": "thing", "aliases": [r["answer"]]}
            for g in ("top", "bottom", "random", "matched"):
                for half in ("B", "A"):
                    for rank, x in enumerate(r[f"{g}_{half}"], 1):
                        ev.append({"experiment": X, "evidence_type": "null_knockout_drop" if g in ("random", "matched") else "knockout_drop",
                                   "metric": "mean_logodds_drop", "unit": x["unit"], "concept": cid, "value": -x["rise"], "uom": "logodds",
                                   "n": r[f"n{half}"], "noise_floor": -thr, "threshold": -thr if g == "bottom" else None,
                                   "passes": (x["rise"] > thr) if (g == "bottom" and half == "B") else None, "rank": rank if g in ("top", "bottom") else None,
                                   "context": f"selection={g}_attr;half={half}{';control=random' if g in ('random', 'matched') else ''};positions=all;prompts=run1_recall",
                                   "details": {"attr": x["attr"], "rise_per_prompt": x["rise_per_prompt"], "d_entropy0": x["d_entropy0"],
                                               "d_rival_logprob": x["d_rival_logprob"]},
                                   "source": f"mechanisms/{sub}/per_concept.jsonl#{r['concept']}|{g}_{half}|{x['unit']}"})
    # ---- H2 confidence neurons (v1 pre-registered, v2 exploratory)
    for sub, X, status in (("entropy", "mech_entropy_06b", "pre-registered (MECHANISMS.md H2)"),
                           ("entropy_v2", "mech_entropy_06b_v2", "exploratory (rule changed after v1's candidates proved near-silent)")):
        S = load(f"{sub}/causal.json")
        if not S:
            continue
        exps.append({"experiment_id": X, "title": f"Confidence (entropy) neurons on Qwen3-0.6B ({sub})",
                     "method": "Candidates from weights (output vector's share in the unembedding's bottom 2% singular directions, x norm"
                               + (" x activation sd" if sub.endswith("v2") else "") + "), layers 25-27; mean-ablated alone on 64 FineWeb windows; "
                               "change in output entropy, total vs direct (final-norm scale frozen); nulls: 30 random neurons, 10 big ordinary-direction neurons.",
                     "status": status, "script": "src/mechanisms/entropy_neurons.py",
                     "reproduce": ("RULE=v2 " if sub.endswith("v2") else "") + "python -X utf8 src/mechanisms/entropy_neurons.py",
                     "data_files": [f"mechanisms/{sub}/causal.json", f"mechanisms/{sub}/weights.json"],
                     "params": {k: S[k] for k in S if k != "per_neuron"}, **COMMON})
        p95 = S["random_p95_abs_dH"]
        for u, v in S["per_neuron"].items():
            ev.append({"experiment": X, "evidence_type": "confidence_effect", "metric": "mean_abs_entropy_change_total", "unit": u,
                       "value": v["mean_abs_dH_total"], "uom": "nats", "n": S["n_tokens"], "noise_floor": p95, "threshold": 5 * p95,
                       "passes": u in S["passed"], "context": f"group={v['group']};ablation=mean;text=fineweb"
                       + (";control=random" if v["group"] != "candidate" else ""),
                       "details": v, "source": f"mechanisms/{sub}/causal.json#{u}"})
    # ---- H3 repeat neurons (K=20 pre-registered; K=200 exploratory follow-up)
    for sub, X, status in (("repeat", "mech_repeat_06b", "pre-registered (MECHANISMS.md H3): see claims"),
                           ("repeat_k200", "mech_repeat_06b_k200", "exploratory (zeroing 200, after K=20 barely hurt copying)")):
        S = load(f"{sub}/summary.json"); N = load(f"{sub}/neurons.json")
        if not S:
            continue
        exps.append({"experiment_id": X, "title": "Repeat / induction neurons on Qwen3-0.6B",
                     "method": "40 random tokens then either the same 40 (repeat) or 40 fresh ones (control); paired difference of mean second-half activation; "
                               "effect size d selected on seeds 0-29, confirmed on 30-59; top 20 zeroed in the second half vs 20 random same-layer neurons.",
                     "status": status, "script": "src/mechanisms/repeat_neurons.py",
                     "reproduce": ("REPEAT_K=200 " if sub.endswith("k200") else "") + "python -X utf8 src/mechanisms/repeat_neurons.py", "data_files": [f"mechanisms/{sub}/summary.json", f"mechanisms/{sub}/neurons.json"],
                     "params": {k: S[k] for k in S if k.startswith("P3") or k.startswith("n_")}, **COMMON})
        for sign in (("confirmed_up", "confirmed_down") if sub == "repeat" else ()):
            for x in N[sign]:
                ev.append({"experiment": X, "evidence_type": "repeat_selectivity", "metric": "heldout_effect_size_d", "unit": x["unit"],
                           "value": x["dB"], "threshold": 2.0 if sign.endswith("up") else -2.0, "passes": True, "n": 30,
                           "context": f"direction={sign.split('_')[1]};text=random_tokens_repeated", "details": {"d_selection": x["dA"]},
                           "source": f"mechanisms/{sub}/neurons.json#{sign}|{x['unit']}"})
        for name, c in S["causal"].items():
            ev.append({"experiment": X, "evidence_type": "group_knockout", "metric": "second_copy_loss_rise", "value": c["top_rise"], "uom": "nats_per_token",
                       "noise_floor": c["random_rise"], "n": 30, "context": f"sequences={name};set=top{len(S.get('top20') or S['top'])}_repeat",
                       "details": {"units": [t["unit"] for t in (S.get("top20") or S["top"])], "random_units": S.get("random20") or S["random"], **c},
                       "source": f"mechanisms/{sub}/summary.json#causal|{name}"})
    # ---- H4 language neurons
    S = load("language/summary.json"); N = load("language/neurons.json")
    if S:
        X = "mech_language_06b"
        exps.append({"experiment_id": X, "title": "Language-identity neurons on Qwen3-0.6B",
                     "method": "40 parallel sentences in en/fr/de/es/zh; per neuron AUC of mean activation for a language vs the rest, selected on sentences 0-19, "
                               "confirmed on 20-39 (AUC >= 0.95 both); each language's top 20 zeroed, loss per token on held-out sentences of every language vs 20 random same-layer neurons.",
                     "status": "pre-registered (MECHANISMS.md H4): see claims", "script": "src/mechanisms/language_neurons.py",
                     "reproduce": "python -X utf8 src/mechanisms/language_neurons.py", "data_files": ["mechanisms/language/summary.json", "mechanisms/language/neurons.json"],
                     "params": {k: S[k] for k in S if k.startswith("P4") or k.startswith("n_") or k == "median_layer"}, **COMMON})
        for lg, xs in N.items():
            for rank, x in enumerate(xs, 1):
                ev.append({"experiment": X, "evidence_type": "language_auc", "metric": "heldout_auc", "unit": x["unit"], "value": x["aucB"],
                           "value_text": lg, "threshold": 0.95, "passes": True, "n": 100, "rank": rank, "context": f"language={lg}",
                           "details": {"auc_selection": x["aucA"], "d_selection": x["d_sel"]}, "source": f"mechanisms/language/neurons.json#{lg}|{x['unit']}"})
        for lg, c in S["causal"].items():
            if c:
                ev.append({"experiment": X, "evidence_type": "language_necessity", "metric": "own_language_loss_rise", "value": c["rise_top"][lg],
                           "uom": "nats_per_token", "noise_floor": c["rise_random"][lg], "passes": c["passes"], "n": 20, "value_text": lg,
                           "context": f"language={lg};set=top{c['n_killed']}", "details": {"random_units": S["random_sets"][lg], **c},
                           "source": f"mechanisms/language/summary.json#causal|{lg}"})
    return exps, ev, list(concepts.values())


def pack8():
    exps, ev = [], []
    S = load("rival_push/summary.json")
    if S:
        X = "mech_rival_push_8b"
        exps.append({"experiment_id": X, "title": "Contrastive 'say' neurons: push on the top word's nearest rivals (Qwen3-8B, weights only)",
                     "method": "For run-1 'say' neurons and random late neurons: push on every token = W_U (g * w); top token t; mean z of the 20 tokens nearest "
                               "to t by unembedding cosine (spelling variants excluded). Contrastive = rivals below the vocabulary mean.",
                     "status": "pre-registered (MECHANISMS.md H6): see claims", "script": "src/mechanisms/rival_push.py",
                     "reproduce": "python -X utf8 src/mechanisms/rival_push.py 400", "data_files": ["mechanisms/rival_push/per_neuron.jsonl"],
                     "params": {k: S[k] for k in S if k not in ("say_sampled_units", "examples_contrastive_say")}, **COMMON})
        for line in (DATA / "rival_push" / "per_neuron.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            ev.append({"experiment": X, "evidence_type": "rival_push", "metric": "rival_mean_z", "unit": r["unit"], "value": r["rival_mean_z"],
                       "value_text": r["top_token"], "threshold": 0.0, "passes": r["contrastive"], "n": len(r["rival_z"]),
                       "context": f"group={r['group']}" + (";control=random" if r["group"] == "random" else ""),
                       "details": {k: r[k] for k in ("top_z", "rival_tokens", "rival_z", "bottom10")},
                       "source": f"mechanisms/rival_push/per_neuron.jsonl#{r['unit']}|{r['group']}"})
    L = load("ledger_checks.json")
    if L:
        X = "mech_signflip_8b"
        exps.append({"experiment_id": X, "title": "Sign-flip knockouts in run 1: positive-attribution candidates whose knockout raises the answer (Qwen3-8B)",
                     "method": "Read-only query over loop_v2_run1 knockout_drop vs its own null_knockout_drop; no new measurement (evidence rows already in the ledger).",
                     "status": "exploratory (ledger query)", "script": "src/mechanisms/ledger_checks.py",
                     "reproduce": "python -X utf8 src/mechanisms/ledger_checks.py", "data_files": ["mechanisms/ledger_checks.json"],
                     "params": {"beyond_null_p0.1": {k: v for k, v in L["beyond_null_p0.1"].items() if k != "by_role"},
                                "strict_lt_-0.5": {k: v for k, v in L["strict_lt_-0.5"].items() if k != "by_role"}}, **COMMON})
    return exps, ev


def main():
    e, v, c = pack06()
    src = Path(__file__).resolve().parent / "claims.jsonl"     # curated, hand-written from MECHANISMS.md section 2
    claims = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()] if src.exists() else []
    strip = lambda xs: [{k: v for k, v in x.items() if k != "model"} for x in xs]
    (C06 / "pack.json").parent.mkdir(parents=True, exist_ok=True)
    (C06 / "pack.json").write_text(json.dumps({"importer": "mechanisms", "contributor": "already-clean research (James + Claude)", "model": "qwen3-0.6b",
                                               "note": "MECHANISMS.md: new job categories tested cheaply on Qwen3-0.6B (CPU), 18 Oct 2026. Built by src/mechanisms/build_pack.py from D:/already-clean-data/mechanisms/."},
                                              indent=1), encoding="utf-8")
    jl(C06 / "experiments.jsonl", e); jl(C06 / "evidence.jsonl", v); jl(C06 / "concepts.jsonl", c)
    jl(C06 / "claims.jsonl", strip([x for x in claims if x.get("model", "qwen3-0.6b") == "qwen3-0.6b"]))
    e8, v8 = pack8()
    (C8 / "pack.json").parent.mkdir(parents=True, exist_ok=True)
    (C8 / "pack.json").write_text(json.dumps({"importer": "mechanisms_8b", "contributor": "already-clean research (James + Claude)", "model": "qwen3-8b",
                                              "note": "MECHANISMS.md: Qwen3-8B checks from weights and existing ledger rows (no forward pass), 18 Oct 2026."},
                                             indent=1), encoding="utf-8")
    jl(C8 / "experiments.jsonl", e8); jl(C8 / "evidence.jsonl", v8)
    jl(C8 / "claims.jsonl", strip([x for x in claims if x.get("model") == "qwen3-8b"]))


if __name__ == "__main__":
    main()
