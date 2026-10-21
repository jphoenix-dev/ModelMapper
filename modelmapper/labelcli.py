"""`modelmapper label`: label any part of a model in one command (LABELLING.md).

    modelmapper label L31:7508 L24:1519          these neurons
    modelmapper label --layers 20-23             every neuron in these layers
    modelmapper label --sample 500 --seed 1      a random sample, spread over the layers
    modelmapper label --all                      the whole model
      [--model ID] [--with-fakes N] [--quick] [--out DIR] [--budget USD] [--plan] [--yes] [--json]

1. The plan comes first: the scope, the controls added, whether the sweep is cached, the sweep's expected time and
   disk, and the API cost as a range (from measured runs: Qwen3-8B on an RTX 5090, Qwen3-0.6B on a 6 GB laptop GPU).
   --plan prints it and stops; --json makes it machine-readable. Without --yes it asks before spending (and refuses when
   there is no terminal to ask in). Without --budget, spend is capped at twice the plan's upper estimate.
2. Controls: the exam compares each label with other neurons' labels, and draws its decoys from other neurons'
   favourites, so it needs a pool of about 1,000 neurons spanning the layers, as in our runs. With fewer targets,
   random neurons from all layers fill the pool (seeded); they are described once, not judged (about $0.30).
3. The sweep runs once per model, over every layer, and is cached (MODEL_MAPPER_CACHE, default ~/.cache/modelmapper).
4b. Coverage, as in the research runner: a neuron that already has a label for a part (the published labels for this
   model, or your earlier runs in --out) isn't judged again for that part; its label goes into the results as it is.
   --relabel judges everything again.
4. Resumable: run the same command again and finished steps are skipped. --budget caps labeller + judge spend; past
   it the command stops with exit code 3 and keeps everything done.
5. --quick sweeps a seeded 20,000-token subset of the vocabulary: fast, but its records say "quick": true and
   "calibrated": false. --with-fakes N also labels N fake neurons (the falsifier; they need a second sweep) and gives
   a verdict per part by check --labels' rule, scaled to N: PASS at most 5% of fakes verified + generative, FAIL from
   15%, WARN in between. Exit code 4 when any part FAILs (the labels are kept; read them with caution).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from modelmapper import settings as S

# v3 run on Qwen3-8B, per judged neuron, all three parts: exam labeller $2.44 + generative words $0.69, judge $4.16, / 950
COST = {"labeller": round((2.44 + 0.69) / 950, 5), "judge": round(4.16 / 950, 5)}
# the upper end, measured: laptop trial, 21 Oct 2026, Qwen3-0.6B, 30 neurons + 20 fakes, judge $0.9004 / 50 (about 4x v3)
COST_HIGH = {"labeller": COST["labeller"], "judge": round(0.9004 / 50, 5)}
CAP_FACTOR = 2.0        # without --budget: stop at twice the plan's upper estimate
CONTROL_COST = {"labeller": 0.0003, "judge": 0.0}       # one description per control neuron
CONTROLS_MIN = 1000   # fill the exam pool to this size: decoys and controls come from the pool (v3 / v3x: ~1,000)
QUICK_TOKENS = 20000
# measured: Qwen3-8B, all 36 layers, 151,649 tokens x 12 sentences, RTX 5090 (32 GB), bfloat16: 1,178 s, ~17 GB on disk
# measured on a CPU (Windows desktop, float32): Qwen3-0.6B reads ~90 tokens / s per sentence -> scaled by size
REF_CPU = {"tokens_per_s": 90.0, "params_b": 0.38, "where": "desktop CPU, float32"}
REF_SWEEP = {"seconds": 1178.0, "params_b": 7.85, "layers": 36, "tokens": 151649, "gb": 17.0, "where": "RTX 5090, bfloat16"}
# measured: laptop trial, Qwen3-0.6B, full vocabulary, RTX 1000 Ada laptop GPU (6 GB), bfloat16: 1,427 s; then the 20 fakes'
# own sweep 870 s (0.61 of it) and 4.6 GB on disk in all against 3.3 GB for the real sweep alone (1.39x)
REF_SMALL = {"seconds": 1427.0, "params_b": 0.38, "tokens": 151649, "where": "RTX 1000 Ada laptop GPU (6 GB), bfloat16"}
FAKE_SWEEP_TIME, FAKE_SWEEP_DISK = 0.61, 0.39
# the falsifier's verdict, as check --labels (1 of 20 passes, 3 of 20 fails), as shares so it scales with --with-fakes
FAKE_PASS_SHARE, FAKE_FAIL_SHARE = 0.05, 0.15


def model_dims(name: str) -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(S.model_path(name))
    return {"layers": cfg.num_hidden_layers, "neurons": cfg.intermediate_size, "hidden": cfg.hidden_size,
            "type": getattr(cfg, "model_type", ""), "vocab": cfg.vocab_size}


def parse_layers(spec: str, n_layers: int) -> list[int]:
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b) if b else int(a)
        if not (0 <= lo <= hi < n_layers):
            raise ValueError(f"layers {part!r}: the model has layers 0-{n_layers - 1}")
        out.update(range(lo, hi + 1))
    return sorted(out)


def parse_neuron(u: str, dims: dict) -> str:
    s = u.strip().upper().lstrip("L")
    L, i = (int(x) for x in s.split(":"))
    if not (0 <= L < dims["layers"] and 0 <= i < dims["neurons"]):
        raise ValueError(f"{u}: the model has layers 0-{dims['layers'] - 1} and neurons 0-{dims['neurons'] - 1}")
    return f"L{L}:{i}"


def scope(dims: dict, neurons=(), layers=None, sample=0, all_=False, seed=0) -> list[str]:
    if sum(bool(x) for x in (neurons, layers, sample, all_)) != 1:
        raise ValueError("give exactly one scope: neurons, --layers, --sample or --all")
    NL, NI = dims["layers"], dims["neurons"]
    if neurons:
        return list(dict.fromkeys(parse_neuron(u, dims) for u in neurons))
    if layers:
        return [f"L{L}:{i}" for L in parse_layers(layers, NL) for i in range(NI)]
    if all_:
        return [f"L{L}:{i}" for L in range(NL) for i in range(NI)]
    rng = np.random.default_rng(seed)                       # spread evenly over the layers, then at random within
    per = [sample // NL + (1 if L < sample % NL else 0) for L in range(NL)]
    return [f"L{L}:{i}" for L in range(NL) for i in sorted(rng.choice(NI, per[L], replace=False).tolist())]


def controls(targets: list[str], dims: dict, seed=0) -> list[str]:
    """Random neurons from all layers that fill the exam pool up to CONTROLS_MIN (seeded; none for big scopes)."""
    if len(targets) >= CONTROLS_MIN:
        return []
    NL, NI = dims["layers"], dims["neurons"]; have = set(targets)
    rng = np.random.default_rng(1000 + seed); out = []
    for j in rng.permutation(NL * NI):                    # uniform over every neuron of every layer, as the research pools
        u = f"L{int(j) // NI}:{int(j) % NI}"
        if u not in have:
            out.append(u)
            if len(out) == CONTROLS_MIN - len(targets):
                break
    return sorted(out, key=lambda u: (int(u[1:].split(":")[0]), int(u.split(":")[1])))


def cache_dir(model: str) -> Path:
    return Path(S.CACHE_DIR) / S.short_name(model) / "sweep"


def find_sweep(cache: Path, layers: list[int], n_tokens: int) -> str | None:
    """A cached sweep (complete: all 12 sentences) of the same size that covers these layers."""
    for f in sorted(cache.glob("vocab_sweep_*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tag = f.stem[len("vocab_sweep_"):]
        if m.get("n_tokens") == n_tokens and len(m.get("contexts", [])) == 12 and m.get("real", True) \
                and m.get("layers") is None and (cache / f"hub_tokens_{tag}.npz").exists():
            return tag
    return None


def sweep_tag(layers: list[int], n_layers: int, quick: bool) -> str:
    key = "all" if len(layers) == n_layers else hashlib.sha256(",".join(map(str, layers)).encode()).hexdigest()[:8]
    return f"{'q' if quick else 'v1'}_{key}"


def estimate(dims: dict, n_layers_needed: int, deepest: int, n_tokens: int, n_targets: int, n_controls: int, n_fakes: int,
             device: str = "cuda", fakes_cached: bool = False) -> dict:
    params_b = dims["layers"] * (4 * dims["hidden"] ** 2 + 3 * dims["hidden"] * dims["neurons"]) / 1e9   # attention + MLP, roughly
    frac_run = (deepest + 1) / dims["layers"]
    secs = REF_SWEEP["seconds"] * (params_b / REF_SWEEP["params_b"]) * frac_run * (n_tokens / REF_SWEEP["tokens"])
    small = REF_SMALL["seconds"] * (params_b / REF_SMALL["params_b"]) * frac_run * (n_tokens / REF_SMALL["tokens"])
    cpu = n_tokens * 12 / REF_CPU["tokens_per_s"] * (params_b / REF_CPU["params_b"]) * frac_run
    gb = REF_SWEEP["gb"] * (n_layers_needed / REF_SWEEP["layers"]) * (dims["neurons"] / 12288)     # favourites: per neuron, not per token
    fake = n_fakes > 0 and not fakes_cached
    k_time, k_disk = (1 + FAKE_SWEEP_TIME, 1 + FAKE_SWEEP_DISK) if fake else (1.0, 1.0)
    judged = n_targets + n_fakes

    def usd(c):
        lab = judged * c["labeller"] + n_controls * CONTROL_COST["labeller"]; jud = judged * c["judge"]
        return {"labeller": round(lab, 2), "judge": round(jud, 2), "total": round(lab + jud, 2)}
    lo, hi = usd(COST), usd(COST_HIGH)
    return {"device": device, "sweep_seconds_on_reference_gpu": round(secs * k_time), "reference_gpu": REF_SWEEP["where"],
            "sweep_seconds_on_small_gpu": round(small * k_time), "small_gpu": REF_SMALL["where"],
            "sweep_seconds_on_cpu": round(cpu * k_time), "reference_cpu": REF_CPU["where"],
            "fake_sweep_included": fake, "model_gb_bf16": round(params_b * 2 * 1.1, 1),
            "sweep_disk_gb": round(gb * k_disk, 2), "api_usd": lo, "api_usd_high": hi,
            "api_prices": "measured with gpt-6-luna and Jev: low end Qwen3-8B (v3 run), high end Qwen3-0.6B (laptop trial); other endpoints differ"}


def fake_verdicts(pseudo: dict) -> dict:
    """{board: {"n", "passed", "status"}}: fakes verified + generative per part, PASS / WARN / FAIL by check --labels' rule."""
    out = {}
    for b, v in (pseudo or {}).items():
        n = int(v.get("n") or 0)
        k = round((v.get("verified_and_generative") or 0) * n)
        st = "PASS" if k <= FAKE_PASS_SHARE * n + 1e-9 else "FAIL" if k >= FAKE_FAIL_SHARE * n - 1e-9 else "WARN"
        out[b] = {"n": n, "passed": k, "status": st}
    return out


def readable(u: str, r: dict) -> list[str]:
    def part(name, p):
        if not p:
            return f"  {name:<9} (not run)"
        ok = ("✓ at the edge" if p.get("at_edge") else "✓") if p.get("verified") else "✗"
        gen = "" if p.get("gen_pass") is None else f" · generative {'✓' if p['gen_pass'] else '✗'}"
        auc = "" if p.get("auc") is None else f"  [exam {p['auc']:.2f} vs {p.get('auc_control') or 0:.2f}{gen}]"
        lab = (p.get("label") or "no description").strip().replace("\n", " ")
        return f"  {name:<9} {ok} {'verified' if p.get('verified') else 'not verified':<13} {lab[:90]}{'…' if len(lab) > 90 else ''}{auc}"
    j = r.get("jobtype") or {}
    lines = [f"{u}  (layer {r['layer']})  {r['triage']['read']}",
             part("Gate", r.get("gate")), part("Positive", r.get("apos")), part("Negative", r.get("aneg"))]
    if j:
        lines.append(f"  {'Job':<9} positive: {j['apos']} (cos {j['c_pos']:+.2f}) · negative: {j['aneg']} (cos {j['c_neg']:+.2f})")
    return lines


def coverage(model: str, out: Path, quick: bool) -> dict:
    """{board: {neuron: {"exam": record, "gen": bool | None}}}: labels that already exist for this model - the published
    ones (shipped in the package) and earlier runs' results in `out` - so they aren't judged again (the research
    runner's coverage filter). Quick runs use only earlier quick runs: their labels aren't calibrated."""
    from modelmapper import known
    cov = {b: {} for b in ("gate", "apos", "aneg")}
    part = {"gate": "gate", "apos": "pos", "aneg": "neg"}

    def add(u, b, p):
        if p and p.get("bin") is not None and u not in cov[b]:
            cov[b][u] = {"exam": {"bin": p["bin"], "list_pass": p.get("list_pass"), "auc_own": p.get("auc"),
                                  "auc_control": p.get("auc_control"), "description": p.get("label")}, "gen": p.get("gen_pass")}
    for f in sorted(Path(out).glob("*_results.json")) if Path(out).exists() else []:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if bool((d.get("provenance") or {}).get("quick")) != bool(quick):
            continue
        for u, r in (d.get("neurons") or {}).items():
            for b in cov:
                add(u, b, r.get(b))
    if not quick and S.short_name(model) == "Qwen3-8B":
        for u, r in (known.published_labels().get("neurons") or {}).items():
            for b in cov:
                add(u, b, r.get(part[b]))
    return cov


def provenance(model: str, quick: bool, tag: str) -> dict:
    """What ran: the keys export-pack reads (pack.build_provenance), plus the sweep and the tested combination."""
    import torch

    from modelmapper import __version__, judge, labeller
    from modelmapper.model import DEVICE, DTYPE
    jm = judge.MODEL if judge.BACKEND == "jev" else (S.env("JUDGE_MODEL") or "unknown")
    lm = labeller.MODEL
    tested = lm == "gpt-6-luna" and judge.BACKEND == "jev" and S.short_name(model) in ("Qwen3-8B", "Qwen3-0.6B")
    return {"package": f"modelmapper {__version__}", "model": model, "device": DEVICE, "dtype": str(DTYPE).replace("torch.", ""),
            "torch": torch.__version__, "labeller_model": lm,
            "labeller_endpoint": "OpenAI" if labeller.BASE.startswith("https://api.openai.com") else "custom",
            "judge_model": jm, "judge_backend": judge.BACKEND, "sweep": tag, "quick": quick, "calibrated": not quick,
            "tested_combination": tested,
            "tested_note": "tested: labeller gpt-6-luna + judge Jev, on Qwen3-8B and Qwen3-0.6B; anything else is untested",
            "date": time.strftime("%Y-%m-%d")}


def cli(a) -> int:
    from modelmapper import jobtype as JT, label, pseudo as PS, sweep as SW
    model_name = a.model or S.DEFAULT_MODEL
    dims = model_dims(model_name)
    try:
        targets = scope(dims, a.neurons, a.layers, a.sample, a.all, a.seed)
    except ValueError as e:
        print(e, file=sys.stderr); return 2
    ctrl = controls(targets, dims, a.seed)
    pool = targets + ctrl
    layers = list(range(dims["layers"]))                                    # the sweep covers every layer
    n_tokens = QUICK_TOKENS if a.quick else None
    cache = Path(a.cache or cache_dir(model_name)); cache.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.model_path(model_name))
    full_tokens = len(SW.allowed_tokens(tok, dims["vocab"]))
    if a.tokens:
        a.quick = True                                      # a token subset is a quick run, whatever its size
    n_tok = min(a.tokens or QUICK_TOKENS, full_tokens) if a.quick else full_tokens
    tag = find_sweep(cache, layers, n_tok) or sweep_tag(layers, dims["layers"], a.quick)
    cached = (cache / f"vocab_sweep_{tag}.json").exists() and find_sweep(cache, layers, n_tok) == tag
    out = Path(a.out or Path("labels") / S.short_name(model_name))
    prefix = a.prefix or ("run_" + hashlib.sha256(",".join(targets).encode()).hexdigest()[:8])
    from modelmapper.model import DEVICE
    cov = {b: {} for b in ("gate", "apos", "aneg")} if a.relabel else coverage(model_name, out, bool(a.quick))
    n_cov = {b: sum(1 for u in targets if u in cov[b]) for b in cov}
    n_new = len([u for u in targets if any(u not in cov[b] for b in cov)])
    fakes_cached = bool(a.with_fakes) and (cache / f"vocab_sweep_{tag}_f{a.with_fakes}.json").exists()
    est = estimate(dims, len(layers), max(layers), n_tok, n_new, len(ctrl), a.with_fakes, DEVICE, fakes_cached)
    budget = a.budget if a.budget is not None else max(1.0, round(CAP_FACTOR * est["api_usd_high"]["total"], 2))
    plan = {"model": model_name, "targets": len(targets), "targets_preview": targets[:5], "controls_added": len(ctrl),
            "already_labelled": n_cov,
            "layers": layers if len(layers) <= 12 else f"{len(layers)} layers ({layers[0]}-{layers[-1]})",
            "fakes": a.with_fakes, "quick": bool(a.quick), "sweep": {"tag": tag, "cached": cached, "tokens": n_tok, "cache_dir": str(cache)},
            "estimate": est, "budget_usd": budget, "budget_default": a.budget is None, "out": str(out / f"{prefix}_results.json"), "resumable": True}
    if a.json:
        print(json.dumps(plan, indent=1))
    else:
        print(f"Plan: label {len(targets):,} neurons of {model_name}"
              + (f" (+{len(ctrl)} control neurons, described once)" if ctrl else "") + (f" and {a.with_fakes} fake neurons" if a.with_fakes else ""))
        print(f"  layers: {plan['layers']}")
        if any(n_cov.values()):
            print(f"  already labelled (published or your earlier runs), reused, not judged again: gate {n_cov['gate']}, "
                  f"positive {n_cov['apos']}, negative {n_cov['aneg']} (--relabel to judge them anyway)")
        gb = est["sweep_disk_gb"]
        fk = " incl. the fake neurons' own sweep" if est["fake_sweep_included"] else ""
        if DEVICE == "cpu":
            hrs = est["sweep_seconds_on_cpu"] / 3600
            need = (f"needed, on this machine's CPU: ~{hrs:.1f} h{fk} (measured rate on a {REF_CPU['where'].split(',')[0]}; a GPU is far faster), "
                    f"~{gb} GB in {cache}")
        else:
            m_big, m_small = est["sweep_seconds_on_reference_gpu"] / 60, est["sweep_seconds_on_small_gpu"] / 60
            need = (f"needed, on {DEVICE}{fk}: ~{m_big:.0f} min on an RTX 5090 (32 GB), ~{m_small:.0f} min on a 6 GB laptop GPU "
                    f"(RTX 1000 Ada; measured), ~{gb} GB in {cache}")
            try:
                import torch
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram < est["model_gb_bf16"]:
                    need += (f". The model needs ~{est['model_gb_bf16']} GB but this GPU has {vram:.0f} GB: it will run partly "
                             "on the CPU, far slower than these times")
            except Exception:                   # noqa: BLE001  (the time estimate stands without it)
                pass
        print(f"  sweep: {'cached, reused' if cached and not est['fake_sweep_included'] else need}" + (f" [quick: a {n_tok:,}-token subset; results not calibrated]" if a.quick else ""))
        lo, hi = est["api_usd"], est["api_usd_high"]
        print(f"  API: ~${lo['total']}-{hi['total']} (labeller ~${lo['labeller']}, judge ~${lo['judge']}-{hi['judge']}; measured with gpt-6-luna and Jev: "
              "the low end on Qwen3-8B, the high end on Qwen3-0.6B)")
        print(f"  spend cap: ${budget}" + (" (twice the upper estimate; set your own with --budget)" if a.budget is None else " (--budget)"))
        print(f"  results: {plan['out']} (re-run the same command to resume)")
    if a.plan:
        return 0
    if not a.yes:
        if not sys.stdin.isatty():
            print("Not starting: no terminal to confirm in. Re-run with --yes once the plan's cost is agreed.", file=sys.stderr)
            return 2
        if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    from modelmapper.model import load
    model, tok = load(model_name)
    if not cached:
        SW.run_sweep(model, tok, cache, tag, max_tokens=n_tok if a.quick else 0)
        SW.hub_tokens(cache, tag, dims["vocab"], tok=tok)
    pset = ptag = focus = None
    if a.with_fakes:
        pf = cache / f"pseudo_{tag}_{a.with_fakes}.npz"
        if not pf.exists():
            from modelmapper.weights import Weights
            W = Weights(model_name)
            g = PS.make_gate_set(W, [int(u[1:].split(":")[0]) for u in targets], n=a.with_fakes)
            PS.save(PS.make_up_set(W, g), pf)
        pset = PS.load(pf); ptag = f"{tag}_f{a.with_fakes}"; focus = [str(x) for x in pset["names"]]
        if not (cache / f"vocab_sweep_{ptag}.json").exists():
            SW.run_sweep(model, tok, cache, ptag, pseudo=pset, real=False, max_tokens=n_tok if a.quick else 0)
        SW.copy_hub_tokens(cache, tag, ptag)
    jt = cache.parent / "jobtype.npz"
    if not jt.exists():
        from modelmapper.weights import Weights
        JT.save(JT.job_types(Weights(model_name), log=lambda *_: None), jt)
    try:
        res = label.label_run(pool, out, prefix, cache, tag, targets=targets, pseudo_sweep_tag=ptag, pseudo_set=pset,
                              pseudo_focus=focus, jobtype=jt, model=model, tok=tok, model_name=model_name,
                              budget=budget, provenance=provenance(model_name, bool(a.quick), tag), covered=cov)
    except label.SpendCap as e:
        print(e, file=sys.stderr); return 3
    fv = fake_verdicts(res.get("pseudo"))
    worst = "FAIL" if any(v["status"] == "FAIL" for v in fv.values()) else "WARN" if any(v["status"] == "WARN" for v in fv.values()) else "PASS"
    code = 4 if worst == "FAIL" else 0
    if a.json:
        print(json.dumps({"summary": res["summary"], "pseudo": res.get("pseudo"), "falsifier": {"parts": fv, "overall": worst} if fv else None,
                          "spend_usd": res["spend_usd"], "results": str(out / f"{prefix}_results.json")}, indent=1))
        return code
    recs = res["neurons"]
    for u in targets[:50]:
        print("\n".join(readable(u, recs[u])))
    if len(targets) > 50:
        print(f"... {len(targets) - 50:,} more in {out / f'{prefix}_results.json'}")
    s = res["summary"]
    print(f"\nLabelled {s['n']:,} neurons: triage {s['triage']}. Spend: labeller ${res['spend_usd']['labeller']}, judge ${res['spend_usd']['judge']}.")
    if fv:
        names = {"gate": "gate", "apos": "positive", "aneg": "negative"}
        print(f"Falsifier {worst}: fake neurons verified + generative: " + ", ".join(f"{names.get(b, b)} {v['passed']}/{v['n']} {v['status']}" for b, v in fv.items())
              + f" (PASS at most {FAKE_PASS_SHARE:.0%}, FAIL from {FAKE_FAIL_SHARE:.0%}, as check --labels).")
        if worst != "PASS":
            n = max(v["n"] for v in fv.values())
            print(f"  Fake neurons pass more often than they should, so treat this run's labels with caution."
                  + (f" With only {n} fakes this can be chance; --with-fakes 100 gives a firmer read." if n < 100 else ""))
    if a.quick:
        print(f"Quick run: a {n_tok:,}-token sweep. Use these labels as a first look; they are not calibrated.")
    print(f"Results: {out / f'{prefix}_results.json'}")
    return code
