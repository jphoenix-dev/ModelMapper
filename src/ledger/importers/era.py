"""Era neurons across formats (era.py) -> ledger pack. Every raw activation, AUC, causal drop and readout goes in."""
import json
import statistics as stx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, era_century, read_jsonl, thing  # noqa: E402

D = DATA_ROOT / "era"
EXP = "era_formats"
DOC = "RESULTS.md#18 Oct 2026 - Era neurons across formats: era-concept neurons vs digit neurons (era.py)"


def main(force=False):
    files = [D / f for f in ("era_entities.json", "era_build.json", "era_acts.jsonl", "era_text.jsonl", "era_readout.json", "era_causal.jsonl", "era_rows.json")]
    with PackWriter("era_formats", inputs=files, code=[__file__], force=force) as pw:
        if pw.skip:
            print("era: unchanged, skipped"); return
        rows = json.loads((D / "era_rows.json").read_text(encoding="utf-8"))
        C = list(read_jsonl(D / "era_causal.jsonl"))
        ctl = [r["u"] for r in rows if r["is_control"]]
        nd = sorted(abs(c["drops"][u]) for c in C for u in ctl if u in c["drops"]); q99 = nd[int(0.99 * len(nd))]
        pw.experiment(EXP, title="Era neurons across formats: year / century / decade / period answers, plain text, readout, knockouts",
                      method=("26 century 'value' neurons (from the facts grid and numeric facts) and 12 random layer-28-33 controls, on 280 "
                              "entities (14th-21st century) that Qwen dates correctly. Activation at the last prompt position for 4 question "
                              "formats + 'Tell me about X'; AUC in-era vs other entities with the sign fixed in the year format. Plain "
                              "sentences mentioning a date in 6 formats (activation max over the date span). Direct readout through the "
                              "unembedding. Each neuron zeroed everywhere: drop in log-odds of Qwen's own answer per format."),
                      status="pre-registered predictions B1-B5 (B1, B3, B4, B5 held; B2 failed)", date="2026-10-18", script="src/loop_v2/era.py",
                      reproduce="python src/loop_v2/era.py entities|build|acts|text|readout|causal|summary",
                      data_files=[f"data:era/{f.name}" for f in files],
                      caveats=["Entities' years come from Luna and are kept only where Qwen's own year falls in the same century.",
                               "Text formats 'hundreds' (21 sentences) and 'named era' (36) are small: single AUCs there are noisy.",
                               "Readout tokens are an informal reading."],
                      docs=[DOC, "LAYERS_AND_JOBS.md#moderately-supported"],
                      params={"auc_threshold": 0.8, "causal_noise_floor": {"rule": "99th pct |drop| of random controls", "value": q99},
                              "predictions": {"B1": "L31:8930 AUC >= 0.8 in year, century, decade", "B2": "L31:8930 in-era drop >= 1.0 in all 3 formats, out-of-era < null p99",
                                              "B3": ">= half of facts2 century neurons AUC >= 0.8 in >= 2 of 3 formats", "B4": "text less selective than answering",
                                              "B5": "L31:8930 direct output has era/digit tokens"}})
        B = json.loads((D / "era_build.json").read_text(encoding="utf-8"))
        ent = {e: thing(pw, e, entity_kind=v["kind"], year=v["year"], century=v["century"]) for e, v in B.items()}
        for e, v in B.items():
            pw.evidence(experiment=EXP, concept=ent[e], concept2=era_century(pw, v["century"]), evidence_type="recall", metric="year_in_century",
                        value=1.0 if v["recalled"] else 0.0, passes=v["recalled"], value_text=str(v.get("qwen_year")),
                        details={"luna_year": v["year"], "answers": {fm: x["answer"] for fm, x in v["formats"].items()},
                                 "century_answer_ok": v.get("century_answer_ok"), "source": v.get("source")}, source=f"era/era_build.json#{e}")
        lab = {r["u"]: r["label"] for r in rows}
        for r in rows:
            cons = [era_century(pw, c) for c in r["label"]]
            for fm, val in r.items():
                if fm in ("u", "label", "is_control", "n_in", "sign") or val is None:
                    continue
                pw.evidence(experiment=EXP, unit=r["u"], concept=cons[0], context=f"format={fm}" + (";control=random" if r["is_control"] else ""),
                            evidence_type="activation_auc", metric="auc_in_era_vs_other", value=val, n=r["n_in"], threshold=0.8, passes=val >= 0.8,
                            details={"labels": r["label"], "sign": r["sign"], "is_control": r["is_control"], "position": "max over date span" if fm.startswith("text_") else "last prompt position"},
                            source=f"era/era_rows.json#{r['u']}")
        for i, a in enumerate(read_jsonl(D / "era_acts.jsonl")):
            for u, x in a["acts"].items():
                pw.evidence(experiment=EXP, unit=u, concept=ent[a["entity"]], context=f"format={a['format']};position=last_prompt", evidence_type="activation",
                            metric="activation", value=x, details={"century": a["century"]}, source=f"era/era_acts.jsonl#{i}")
        for i, t in enumerate(read_jsonl(D / "era_text.jsonl")):
            ce = era_century(pw, t["century"])
            for u, xs in t["acts_span_then_end"].items():
                pw.evidence(experiment=EXP, unit=u, concept=ce, context=f"text_format={t['format']};text={t['text']}", evidence_type="activation",
                            metric="activation_max_over_date_span", value=max(xs), details={"span_then_end": xs, "min": min(xs)},
                            key=f"text|{i}|{u}", source=f"era/era_text.jsonl#{i}")
        RO = json.loads((D / "era_readout.json").read_text(encoding="utf-8"))
        for u, r in RO.items():
            pw.evidence(experiment=EXP, unit=u, concept=era_century(pw, lab.get(u, [19])[0]) if u in lab else None, evidence_type="readout_tokens",
                        metric="direct_logit_top15", value_text=" | ".join(t.strip() or repr(t) for t in r["top"][:10]),
                        details={"top": r["top"], "bottom": r["bottom"], "method": "lm_head(final-norm weight * down_proj column)"}, source=f"era/era_readout.json#{u}")
        for i, c in enumerate(C):
            for u, d in c["drops"].items():
                in_era = c["century"] in lab.get(u, [])
                pw.evidence(experiment=EXP, unit=u, concept=ent[c["entity"]], concept2=era_century(pw, c["century"]),
                            context=f"format={c['format']};target=qwen_own_answer" + (";control=random" if u in ctl else ""),
                            evidence_type="knockout_drop", metric="logodds_drop", value=d, uom="logodds", n=1, noise_floor=q99, threshold=q99,
                            passes=d > q99, details={"answer": c["answer"], "base": c["base"], "in_neuron_era": in_era},
                            key=f"causal|{i}|{u}", source=f"era/era_causal.jsonl#{i}")
        from ledger.core import ordinal
        qf, tf = ("year", "century", "decade", "period"), ("text_digits", "text_decade", "text_ordinal", "text_word", "text_hundreds", "text_named era")
        for r in rows:                               # one readable summary claim per era neuron (the map shows it instead of 11 cards)
            if r["is_control"] or r["n_in"] < 5:
                continue
            qa = {f: r[f] for f in qf if r.get(f) is not None}; ta = {f: r[f] for f in tf if r.get(f) is not None}
            med = {}
            for fm in ("year", "century", "decade"):
                ci = [c["drops"][r["u"]] for c in C if c["format"] == fm and c["century"] in r["label"]]
                med[fm] = stx.median(ci) if ci else None
            era = " / ".join(f"{ordinal(c)}-century" for c in r["label"])
            flips = [f for f, v in qa.items() if v < 0.5]
            text = (f"{r['u']} ({era} neuron): AUC {min(qa.values()):.2f}-{max(qa.values()):.2f} across the 4 question formats "
                    f"({sum(v >= 0.8 for v in qa.values())} >= 0.8), {min(ta.values()):.2f}-{max(ta.values()):.2f} reading dates in text; zeroing it costs "
                    + ", ".join(f"{fm} answers {v:.2f}" for fm, v in med.items() if v is not None) + " log-odds (in-era medians)"
                    + (f"; FLIPS in the {', '.join(flips)} format (a digit / format-bound neuron, not an era concept)" if flips else ""))
            pw.claim(f"era-summary-{r['u']}", kind="finding", scope=r["u"], unit=r["u"], concept=era_century(pw, r["label"][0]), text=text,
                     grade="exploratory summary", status="pre-registered study (B1-B5)", category="unit summary",
                     metric={"auc_questions": qa, "auc_text": ta, "median_drop_in_era": med, "n_in_era": r["n_in"]},
                     evidence_sql=f"SELECT * FROM evidence WHERE experiment_id = 'era_formats' AND unit_key = '{r['u']}' AND (evidence_type = 'activation_auc' OR metric = 'median_logodds_drop_in_era')",
                     docs=[DOC], experiment=EXP)
        for r in rows:                               # per-neuron, per-format medians in / out of era (aggregate rows, named as such)
            if r["is_control"]:
                continue
            for fm in ("year", "century", "decade"):
                ci = [c["drops"][r["u"]] for c in C if c["format"] == fm and c["century"] in r["label"]]
                co = [c["drops"][r["u"]] for c in C if c["format"] == fm and c["century"] not in r["label"]]
                if ci:
                    pw.evidence(experiment=EXP, unit=r["u"], concept=era_century(pw, r["label"][0]), context=f"format={fm}", evidence_type="knockout_drop",
                                metric="median_logodds_drop_in_era", value=stx.median(ci), uom="logodds", n=len(ci), noise_floor=q99, threshold=1.0,
                                passes=stx.median(ci) >= 1.0, details={"median_out_of_era": stx.median(co) if co else None, "n_out": len(co)},
                                key=f"agg|{r['u']}|{fm}", source="era/era_causal.jsonl (aggregate)")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
