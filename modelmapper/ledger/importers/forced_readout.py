"""Forced-neuron readouts -> ledger pack: the pilot (neuron_readout.py, pilot1.json) and the fast readout
(fast_readout.py: answers / layers x4 / layers x16 / list).

fast_readout, per record (neuron, sign, mode, dose): one 'forced_readout' row, value = consistency of the top web word
(share of the 32 web backgrounds where it is in that background's top-50 by dlogp), value_text = the top-10 web words;
details = both weighted word lists (word, mean dlogp, consistency, mean forced P), the per-background top-20 (forced and
unforced P) where saved, the clamp value and sign. Extra rows per record name other metrics (strength = top word's mean
dlogp; web_vs_empty_top10_jaccard; context_dependence). Mode 'answers' adds one row per (neuron, concept): the rank of
the concept's answer token in the forced readout (threshold 10 = the headline top-10 cut), with the weights-only rank.
pilot1: one row per (target, dose, background): value = the top next-token mean dlogp, value_text = top-8 words;
details = the whole readout (next-token shifts, argmax counts, greedy-generation counts, 'specific' counts, examples).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, read_jsonl, thing  # noqa: E402

FR = DATA_ROOT / "fast_readout"
NR = DATA_ROOT / "neuron_readout"
FAST = [("answers", FR / "answers.jsonl", FR / "answers.txt", 4), ("layers", FR / "layers_x4.jsonl", FR / "layers_x4.txt", 4),
        ("layers", FR / "layers_x16.jsonl", FR / "layers_x16.txt", 16), ("list", FR / "list.jsonl", FR / "list.txt", 4)]
PILOT, PILOT_TXT = NR / "pilot1.json", NR / "pilot1.txt"
EXP_P, EXP_F = "neuron_readout_pilot1", "fast_readout"
DOC_P = "RESULTS.md#18 Oct 2026: forcing one neuron - its most common expression, read with minimal bias (pilot; research branch)"
DOC_F = "RESULTS.md#18 Oct 2026: fast forced readout - validation against known answers and across layers (research branch)"
MODE_DESC = {"answers": "the top 'say' neuron (else the top-1) of every mapped run-1 concept",
             "layers": "50 random neurons from every layer (seed 131)", "list": "hand-picked neurons from the pilot (command line)"}


def words(ws, k=10):
    return " | ".join(str(w[0]).strip() or repr(w[0]) for w in ws[:k])


def main(force=False):
    inputs = [p for _, a, b, _ in FAST for p in (a, b)] + [PILOT, PILOT_TXT]
    with PackWriter("forced_readout", inputs=inputs, code=[__file__], force=force) as pw:
        if pw.skip:
            print("forced_readout: unchanged, skipped"); return
        pw.experiment(EXP_P, title="Forced readout pilot: what one neuron says when clamped hard (10 targets + 3 random controls)",
                      method=("Each target is clamped to sign x m x its own web-text peak (m = 1, 4, 16) at the last context position "
                              "and every generated position, in three backgrounds: 200 random 24-token web snippets; a bare newline "
                              "with 50 random unrelated neurons clamped to their peaks (new set per repeat, 200); a bare newline alone. "
                              "Read: mean next-token log-prob change vs the unforced background, forced argmax counts, and 8 greedy "
                              "tokens; 'specific' generation counts = target minus the random control neurons' mean."),
                      status="exploratory", date="2026-10-18", script="src/loop_v2/neuron_readout.py",
                      reproduce="python src/loop_v2/neuron_readout.py pilot1 L31:9683 L31:10160 L26:11492 L33:5099 L23:12257 L29:6647 L29:9192 L29:4682 L34:553 L33:10933",
                      data_files=["data:neuron_readout/pilot1.json", "data:neuron_readout/pilot1.txt"],
                      caveats=["Hand-picked strong neurons; the 'low background bias' seen here did NOT hold for random neurons (fast readout correction).",
                               "x1 is often drowned by context in generation; x16 collapses to one repeated token.",
                               "Readouts are informal readings (readout tokens), never labels."],
                      docs=[DOC_P],
                      params={"doses": [1, 4, 16], "n_repeats": 200, "ctx_tokens": 24, "noise_neurons": 50, "gen_steps": 8, "seed": 121,
                              "controls": "3 random neurons from the targets' first 3 layers (sorted)", "summary_text": PILOT_TXT.read_text(encoding="utf-8")})
        pw.experiment(EXP_F, title="Fast forced readout: a weighted word list per neuron, validated on known answers and across layers",
                      method=("The neuron is clamped to m x its own web-text peak (sign = larger of max / |min| unless given) at the LAST "
                              "position only, over 32 random web snippets (24 tokens) and 16 'empty' backgrounds (a newline); next "
                              "token only, against one unforced baseline per background. Per neuron: top-50 words by mean dlogp with "
                              "their consistency across backgrounds and mean forced probability, per background set; web-vs-empty "
                              "top-10 Jaccard; strength = top word's mean dlogp; (layers runs) the per-background top-20 forced vs "
                              "unforced probabilities and context dependence = 1 - mean pairwise Jaccard of per-web-background top-10."),
                      status="exploratory", date="2026-10-18", script="src/loop_v2/fast_readout.py",
                      reproduce="python src/loop_v2/fast_readout.py answers | layers (MULT=4 / MULT=16) | list L:n[:sign] ...",
                      data_files=[f"data:fast_readout/{a.name}" for _, a, _, _ in FAST] + [f"data:fast_readout/{b.name}" for _, _, b, _ in FAST],
                      caveats=["Top word consistent across backgrounds only for late neurons (layers 28-35); early / mid neurons have no consistent expression even at x16.",
                               "Words beyond the top one shift with context (context dependence 0.59-1.0).",
                               "answers.jsonl and list.jsonl predate the per-background / context_dependence fields.",
                               "layers mode drew neurons with replacement per layer: 3 neurons appear twice (both records kept, key = line number)."],
                      docs=[DOC_F],
                      params={"n_web": 32, "n_empty": 16, "ctx_tokens": 24, "top_words": 50, "seed": 131,
                              "modes": MODE_DESC, "dose_by_file": {a.name: m for _, a, _, m in FAST},
                              "answer_rank_threshold": {"value": 10, "why": "the headline cut in answers.txt (top-10); importer's choice for passes"},
                              "summaries": {b.name: b.read_text(encoding="utf-8") for _, _, b, _ in FAST}})

        # ---- pilot
        P = json.loads(PILOT.read_text(encoding="utf-8"))
        for key, r in P["readouts"].items():
            u, m, bg = key.split("|")
            ctx = f"dose=x{m};background={bg};clamp=last_and_generated" + (";control=random" if r["kind"] == "control" else "")
            top = r["next_mean_dlogp_top"]
            pw.evidence(experiment=EXP_P, unit=u, evidence_type="forced_readout", metric="top_next_token_mean_dlogp",
                        value=top[0][1] if top else None, uom="logprob", value_text=words(top, 8), context=ctx,
                        n=200 if bg != "nothing" else 1, source=f"neuron_readout/pilot1.json#{key}", details=r)

        # ---- fast readout
        for mode, path, _, dose in FAST:
            for i, r in enumerate(read_jsonl(path)):
                u = r["neuron"]; ww = r["web_words"]
                ctx = f"mode={mode};dose=x{dose};sign={'+' if r['sign'] > 0 else '-'};clamp=last;read=next_token"
                src = f"fast_readout/{path.name}#{i}"
                kk = f"{path.name}|{i}"
                det = {k: v for k, v in r.items() if k not in ("neuron", "layer", "answers")}
                pw.evidence(experiment=EXP_F, unit=u, evidence_type="forced_readout", metric="top_word_consistency_web", value=ww[0][2] if ww else None,
                            uom="fraction", n=32, value_text=words(ww, 10), context=ctx, source=src, details=det, key=kk + "|main")
                pw.evidence(experiment=EXP_F, unit=u, evidence_type="forced_readout", metric="strength_top_word_mean_dlogp", value=r["strength"],
                            uom="logprob", n=32, context=ctx, source=src, key=kk + "|strength")
                pw.evidence(experiment=EXP_F, unit=u, evidence_type="forced_readout", metric="web_vs_empty_top10_jaccard",
                            value=r["web_vs_empty_top10_jaccard"], uom="fraction", context=ctx, source=src, key=kk + "|wve")
                if "context_dependence" in r:
                    pw.evidence(experiment=EXP_F, unit=u, evidence_type="forced_readout", metric="context_dependence", value=r["context_dependence"],
                                uom="fraction", n=32, context=ctx, source=src, key=kk + "|ctxdep")
                for j, a in enumerate(r.get("answers") or []):
                    cid = thing(pw, a["concept"])
                    pw.evidence(experiment=EXP_F, unit=u, concept=cid, evidence_type="forced_readout", metric="answer_token_rank_forced",
                                value=a["answer_rank_forced"], uom="rank", threshold=10, passes=a["answer_rank_forced"] <= 10,
                                value_text=a["role"], context=ctx, source=f"{src}.answers[{j}]", key=f"{kk}|ans|{a['concept']}",
                                details={"answer_rank_direct_weights": a["answer_rank_direct"], "role": a["role"]})


if __name__ == "__main__":
    main(force="--force" in sys.argv)
