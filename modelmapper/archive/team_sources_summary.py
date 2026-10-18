"""What does each team head carry into concept neurons? Summary of route_sources_team_*.json (50 concepts).

Per head: number of concepts, median net push into the concept neuron (+ = accelerates it, - = brakes), and the
share of its absolute push coming from the question's words vs formatting tokens vs the fixed suffix; plus the words
it most often carries (counted when a word is among its top-3 question-word sources for a concept).
Usage: python src/team_sources_summary.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = {"<|im_start|>", "<|im_end|>", "user", "assistant", "<think>", "</think>", "?"}


def main():
    per = defaultdict(list); words = defaultdict(Counter)
    for p in ("results/concepts_v3/route_sources_team_v3.json", "results/concepts_w2/route_sources_team_w2.json"):
        d = json.loads((ROOT / p).read_text(encoding="utf-8"))
        for c, v in d.items():
            for h, k in v["by_kind"].items():
                tot = sum(abs(x) for x in k.values()) or 1e-9
                per[h].append({"net": sum(k.values()), "q": abs(k.get("question", 0)) / tot, "f": abs(k.get("format", 0)) / tot,
                               "s": abs(k.get("suffix", 0)) / tot})
                qw = [(w, val) for w, val in v["heads"][h] if w.strip("'\\n ") and w not in FORMAT and not w.startswith("'")]
                for w, _ in qw[:3]:
                    words[h][w] += 1
    med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
    print(f"{'head':<7} {'n':>3} {'median net push':>15} {'question':>9} {'format':>7} {'suffix':>7}  most-carried question words")
    for h, rows in sorted(per.items(), key=lambda kv: -med([r["q"] for r in kv[1]])):
        print(f"{h:<7} {len(rows):>3} {med([r['net'] for r in rows]):>+15.2f} {med([r['q'] for r in rows]):>9.2f} "
              f"{med([r['f'] for r in rows]):>7.2f} {med([r['s'] for r in rows]):>7.2f}  {[w for w, _ in words[h].most_common(8)]}")


if __name__ == "__main__":
    main()
