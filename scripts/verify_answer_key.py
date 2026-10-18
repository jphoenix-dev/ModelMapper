"""Check this install against an answer key: does it rediscover the key's marker neurons?

    python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 1   # shipped prompts; no API calls
    python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 2   # fresh prompts from your labeller
    python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 3   # fresh labels from your atlas + labeller

Tier 1 re-runs each marker concept on the key's own recall prompts, so it tests the code and the arithmetic on this
machine: it should reproduce almost exactly. Tier 2 has your labeller write new recall questions, so it tests that the
concept -> neuron association belongs to the model, not to our wording. Tier 3 labels each marker neuron blind from
its top firings in YOUR atlas (data/atlas_<model>.pt, built from your corpus by `modelmapper atlas`; a different
FineWeb shard from ours makes the test stronger), so it tests that the neuron's job belongs to the model, not to our
text. Pass rules live in the key; the result (per marker and overall) is printed and written to --out.
Exit code 0 = pass.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def concept_tier(key, tier, work, control=False):
    from modelmapper.core import markers as mk
    mk.setup_env(work, discovery=(tier == 2))      # tier 1 = the key's prompts exactly, so no answer discovery
    from modelmapper.core.pipeline_fast import FastPipeline
    rules = key["pass_rules"][f"tier{tier}"]
    pipe = FastPipeline(key["model"])
    rows = []
    ms = key["markers"]
    # negative control: each marker is looked for in the NEXT marker's concept, where it shouldn't be
    pairs = [(m, ms[(i + 1) % len(ms)]) for i, m in enumerate(ms)] if control else [(m, m) for m in ms]
    for m, cm in pairs:
        d = work / mk.uid(m["layer"], m["neuron"])  # its own folder: tier 1 holds the key's spec, tier 2 starts empty
        mk.set_out_dir(d)
        if tier == 1:
            from modelmapper.core.concept_pipeline import slug
            (d / f"{slug(cm['concept'])}_spec.json").write_text(json.dumps(cm["spec"], ensure_ascii=False), encoding="utf-8")
        t0 = time.time()
        try:
            r = mk.run_concept(pipe, cm["concept"], pinned=cm.get("pinned") if tier == 1 else None)
        except Exception as e:  # noqa: BLE001  (reported as this marker's result; the other markers still run)
            r = {"status": "error", "detail": repr(e)[:300], "recall": 0, "recall_total": 0}
        got = next((n for n in r.get("neurons", []) if (n["layer"], n["neuron"]) == (m["layer"], m["neuron"])), None)
        ref = m["evidence"][0]
        if tier == 1:
            max_rank = rules["rank_factor"] * ref["rank"] + rules["rank_slack"]
            min_drop = rules["drop_factor"] * ref["drop"]
        else:
            max_rank, min_drop = rules["max_rank"], rules["min_drop"]
        ok = bool(got and got["rank"] <= max_rank and got["drop"] >= min_drop)
        why = (f"concept {r['status']}: {r['detail']}" if r.get("detail") else
               f"concept {r['status']} (recall {r['recall']}/{r['recall_total']})" if r["status"] != "ok" else
               f"not in the top {len(r['neurons'])}" if not got else
               f"rank {got['rank']} (needs <= {max_rank:g}), drop {got['drop']:.3f} (needs >= {min_drop:.3f})")
        rows.append({"concept": m["concept"], "checked_in": cm["concept"], "unit": m["unit"], "ok": ok, "detail": why, "key_rank": ref["rank"],
                     "key_drop": ref["drop"], "rank": got and got["rank"], "drop": got and got["drop"],
                     "recall": r["recall"], "top5": [mk.uid(n["layer"], n["neuron"]) for n in r.get("neurons", [])[:5]],
                     "seconds": round(time.time() - t0, 1)})
        where = f" in {cm['concept']}" if control else ""
        print(f"[{'ok' if ok else '--'}] {m['concept']:<24} {m['unit']:<11}{where} {why}", flush=True)
    return rows


def label_tier(key, work):
    from modelmapper.core import markers as mk
    from modelmapper.core.config import short_name
    if not (REPO / "data" / f"atlas_{short_name(key['model'])}.pt").exists():
        sys.exit(f"tier 3 needs this install's atlas: modelmapper atlas {key['model']} 2e6   (on your corpus)")
    units = [(m["layer"], m["neuron"]) for m in key["markers"]]
    labels = mk.label_units(key["model"], units, work / "dossiers_markers")
    rows = []
    for m in key["markers"]:
        lab = labels.get(m["unit"], {})
        text = (lab.get(m["sign"]) or {}).get("explanation", "")
        hits = mk.keyword_hit(text, m["keywords"])
        rows.append({"concept": m["concept"], "unit": m["unit"], "ok": bool(hits), "hits": hits, "label": text,
                     "key_label": m["job"], **({"error": lab["error"]} if "error" in lab else {})})
        shown = f"labelling failed: {lab['error']}" if "error" in lab else f"{m['sign']} pole: {text[:120]}"
        print(f"[{'ok' if hits else '--'}] {m['concept']:<24} {m['unit']:<11} {shown}", flush=True)
    # control: how often a label "matches" a DIFFERENT marker's keywords (chance level for this keyword test)
    others = [(r, o) for r in rows for o in key["markers"] if o["unit"] != r["unit"] and o["concept"] != r["concept"]]
    chance = sum(bool(mk.keyword_hit(r["label"], o["keywords"])) for r, o in others)
    print(f"control: labels match a different marker's keywords in {chance}/{len(others)} pairings", flush=True)
    for r in rows:
        r["control_pairings"] = len(others) // max(1, len(rows))
    rows and rows[0].update(control_matches=chance, control_total=len(others))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key"); ap.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--out", help="result JSON (default results/answer_key_check/<model>_tier<N>.json)")
    ap.add_argument("--control", action="store_true",
                    help="negative control (tiers 1-2): look for each marker in a different marker's concept; a "
                         "meaningful key finds few or none")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    print(f"answer key {a.key}: {key['model']}, {len(key['markers'])} markers, tier {a.tier} "
          f"({key['pass_rules'][f'tier{a.tier}']['what']})", flush=True)
    work = Path(tempfile.mkdtemp(prefix="answer_key_"))
    if a.control and a.tier == 3:
        sys.exit("--control applies to tiers 1 and 2")
    rows = concept_tier(key, a.tier, work, a.control) if a.tier in (1, 2) else label_tier(key, work)
    need = key["pass_rules"][f"tier{a.tier}"]["pass_fraction"]
    n_ok = sum(r["ok"] for r in rows)
    passed = n_ok >= need * len(rows)
    import platform   # machine name in the default file, so several machines' results never overwrite each other
    host = "".join(c if c.isalnum() or c in "-_" else "_" for c in platform.node()) or "host"
    name = f"{key['model'].split('/')[-1]}_tier{a.tier}{'_control' if a.control else ''}_{host}.json"
    out = Path(a.out or REPO / "results" / "answer_key_check" / name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"key": a.key, "model": key["model"], "tier": a.tier, "passed": passed, "found": n_ok,
                               "markers": len(rows), "pass_fraction": need, "time": time.strftime("%Y-%m-%d %H:%M"),
                               "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.control:   # the control is informative when markers are NOT found in the wrong concepts
        print(f"\nCONTROL: {n_ok}/{len(rows)} markers found in the wrong concept (a meaningful key: few or none). "
              f"Details: {out}")
        sys.exit(0)
    print(f"\n{'PASS' if passed else 'FAIL'}: tier {a.tier}, {n_ok}/{len(rows)} markers found "
          f"(needs {need:.0%}). Details: {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
