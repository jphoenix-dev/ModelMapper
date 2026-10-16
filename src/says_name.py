"""People recall through 1-2 neurons but don't steer. Are their top neurons "say the name" neurons? (no grader calls)

For each recalled concept run, take its top-10 attributed neurons (with the sign they take on recall) and read each
neuron's direct effect on the vocabulary: output weights -> final norm scale -> unembedding. A neuron "says the answer"
if any token of the answer ranks in the top 50 of the ~152k vocabulary for that neuron. Compare people vs everything
else, and relate say-fraction to steering (joint nudgeability).

Usage: python src/says_name.py
"""
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import DEFAULT_MODEL  # noqa: E402
from neurons import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PEOPLE = {"Albert Einstein", "William Shakespeare", "Napoleon Bonaparte", "Wolfgang Amadeus Mozart", "Frida Kahlo",
          "Harry Potter", "Marie Curie", "Charles Darwin"}


def main():
    model, tok = load(DEFAULT_MODEL)
    W_U = model.lm_head.weight.float()                       # [V, d]
    g = model.model.norm.weight.float()
    rows = []
    for d in ("concepts_v3", "concepts_v3b", "concepts_w2"):
        for f in sorted(glob.glob(str(ROOT / "results" / d / "*__clamp_reply.json"))):
            x = json.loads(Path(f).read_text(encoding="utf-8"))
            if not isinstance(x, dict) or x.get("status") != "ok" or not x.get("answer_tokens"):
                continue
            first = tok.encode(x["answer_tokens"][0])[0]          # stored as text pieces; re-encode the first
            says, ranks = 0, []
            for t in x["top_neurons"][:10]:
                col = model.model.layers[t["layer"]].mlp.down_proj.weight[:, t["neuron"]].float()
                sign = -1.0 if t.get("sign_on_recall") == "neg" else 1.0
                logits = sign * (W_U @ (col * g))
                # any token of the answer counts (for people the first token is often a generic first name)
                ids = [tok.encode(a)[0] for a in x["answer_tokens"]]
                rank = min(int((logits > logits[i]).sum().item()) for i in ids)
                ranks.append(rank)
                says += rank < 50
            rows.append({"run": d, "concept": x["concept"].strip(), "person": x["concept"].strip() in PEOPLE,
                         "first_token": tok.decode([first]), "says_answer_top10": says, "ranks": ranks,
                         "steer": x.get("nudgeability")})
    for grp, sel in (("people", [r for r in rows if r["person"]]), ("others", [r for r in rows if not r["person"]])):
        s = [r["says_answer_top10"] for r in sel]; st = [r["steer"] or 0 for r in sel]
        print(f"{grp:<7} runs {len(sel):>3} | 'says the answer' neurons in top 10: mean {sum(s)/len(s):.1f} | steering mean {sum(st)/len(st):.2f}")
    for r in sorted(rows, key=lambda r: -r["says_answer_top10"])[:12]:
        print(f"  {r['run']:<12} {r['concept']:<26} first token {r['first_token']!r:<10} says {r['says_answer_top10']}/10 steer {r['steer']}")
    # within non-people: does a higher say-fraction go with lower steering?
    others = [r for r in rows if not r["person"]]
    hi = [r["steer"] or 0 for r in others if r["says_answer_top10"] >= 3]; lo = [r["steer"] or 0 for r in others if r["says_answer_top10"] <= 1]
    print(f"non-people: steering with >= 3 'says' neurons {sum(hi)/max(1,len(hi)):.2f} (n={len(hi)}) vs <= 1: {sum(lo)/max(1,len(lo)):.2f} (n={len(lo)})")
    (ROOT / "results/says_name.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
