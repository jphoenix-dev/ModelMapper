"""Lexical-confound check on recall prompt sets (ISSUES #2).

GG-1's "about to say Golden Gate" neurons turned out to be "orange" neurons because the prompts over-used "orange".
For every concept: content words that recur across its recall prompts (in >= 30% of them; template words shared by
most concepts' prompts and the answer's own words excluded), then which of its top-10 neurons have a label that
names such a word but not the concept/answer. Those neurons may be tracking the prompt wording, not the concept.

Usage: python src/lexical_check.py <concepts_dir> [...]     then: python src/lexical_check.py --split
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = set("the a an of in on to and or for with by from is are was were be this that which what who whose how when "
           "where its it his her their as at into than called name named answer just only one also most famous "
           "known called what's give word".split())


def words(s):
    s = re.sub(r"<\|.*?\|>|<think>|</think>|user|assistant", " ", s)
    return {w.lower().rstrip("s") for w in re.findall(r"[A-Za-z]{3,}", s)} - STOP


def main():
    data = {}
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(f"{d}/*__clamp_reply.json")):
            x = json.loads(Path(f).read_text(encoding="utf-8"))
            if isinstance(x, dict) and x.get("recall") and x.get("labels"):
                data[f"{Path(d).name}/{x['concept']}"] = x
    # template words: recur in the prompts of most concepts
    per_concept = {k: Counter(w for p, _ in x["recall"] for w in words(p)) for k, x in data.items()}
    spread = Counter(w for c in per_concept.values() for w in c)
    template = {w for w, n in spread.items() if n >= 0.5 * len(data)}
    out = {}
    for k, x in data.items():
        n = len(x["recall"])
        own = words(x["concept"] + " " + x["answer"])
        rec = {w: c / n for w, c in per_concept[k].items() if c / n >= 0.3 and w not in template and w not in own}
        flagged = []
        for t in x["top_neurons"][:10]:                     # the label of the pole the neuron takes on recall
            u = f"L{t['layer']}_N{t['neuron']}"
            e = ((x["labels"].get(u) or {}).get(t.get("sign_on_recall", "pos")) or {}).get("explanation", "")
            ew = words(e)
            hit = sorted(w for w in rec if w in ew)
            if hit and not (own & ew):
                flagged.append((u, hit))
        out[k] = {"recurring_words": {w: round(v, 2) for w, v in sorted(rec.items(), key=lambda t: -t[1])},
                  "flagged_top10": flagged}
        print(f"{k:<40} recurring {list(out[k]['recurring_words'].items())[:6]} | flagged {flagged}")
    (ROOT / "results/lexical_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n{sum(bool(v['flagged_top10']) for v in out.values())}/{len(out)} concepts have a top-10 neuron whose label names a "
          f"recurring prompt word but not the concept; {sum(len(v['flagged_top10']) for v in out.values())} neurons in total")


if __name__ == "__main__" and sys.argv[1:2] != ["--split"]:
    main()


def split(dirs=("results/concepts", "results/concepts_v3")):
    """For each flagged neuron: mean attribution on recall prompts that contain the word vs those that don't.
    A neuron that only matters when the word is in the prompt is reading the clue, not the concept."""
    flags = json.loads((ROOT / "results/lexical_check.json").read_text(encoding="utf-8"))
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(f"{ROOT / d}/*__clamp_reply.json")):
            x = json.loads(Path(f).read_text(encoding="utf-8"))
            k = f"{Path(d).name}/{x.get('concept') if isinstance(x, dict) else ''}"
            if k not in flags:
                continue
            for u, ws in flags[k]["flagged_top10"]:
                t = next(t for t in x["top_neurons"] if f"L{t['layer']}_N{t['neuron']}" == u)
                has = [any(w in words(p) for w in ws) for p, _ in x["recall"]]
                a = [abs(v) for v in t["per_prompt"]]
                w_ = [v for v, h in zip(a, has) if h]; wo = [v for v, h in zip(a, has) if not h]
                mw = sum(w_) / len(w_) if w_ else 0; mo = sum(wo) / len(wo) if wo else 0
                share = sum(w_) / max(sum(a), 1e-12)
                rows.append((k, u, ws, len(w_), len(wo), mw, mo, share, len(w_) / len(a)))
                print(f"{k:<34} {u:<11} {','.join(ws):<18} with-word {len(w_):>2} prompts mean {mw:.4f} | without "
                      f"{len(wo):>2} mean {mo:.4f} | share of attribution from with-word prompts {share:.0%} "
                      f"(expected if no clue-reading {len(w_)/len(a):.0%})")
    clue = [r for r in rows if r[4] and r[6] < 0.1 * r[5]]
    anti = [r for r in rows if r[3] and r[5] < 0.1 * r[6]]
    print(f"\nclue-readers (mean attribution without the word < 10% of with it): {len(clue)}/{len(rows)} "
          f"{[(r[0], r[1]) for r in clue]}")
    print(f"opposite (matter only when the word is ABSENT, i.e. the model supplies it itself): {len(anti)}/{len(rows)} "
          f"{[(r[0], r[1]) for r in anti]}")
    return rows


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--split":
    split()
