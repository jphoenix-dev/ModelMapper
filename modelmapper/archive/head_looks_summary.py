"""Attention study, step 3 summary: do the important heads read the question's clue words?

From head_looks_*.json: for each concept's top-3 heads, the share of attention on question content, and whether the
top content tokens are clue words (alphabetic, >= 4 letters, not a function word). Split by shared heads
(results/head_summary.json "shared") vs concept-specific heads. Usage: python src/head_looks_summary.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNC = {"what", "which", "who", "whose", "this", "that", "with", "from", "known", "name", "called", "often", "most",
        "also", "into", "their", "they", "have", "were", "been", "does", "about", "used", "many", "some", "such", "first",
        "answer", "just"}


def clue(t):
    return t.isalpha() and len(t) >= 4 and t.lower() not in FUNC


def main():
    shared = set(json.loads((ROOT / "results/head_summary.json").read_text(encoding="utf-8"))["shared"])
    rows = []
    for p in ("results/concepts_v3/head_looks_cr2.json", "results/concepts_w2/head_looks_w2.json"):
        d = json.loads((ROOT / p).read_text(encoding="utf-8"))
        for c, hs in d.items():
            for h, v in hs.items():
                top = v["top_content_tokens"]
                tot = sum(w for _, w in top) or 1e-9
                rows.append({"concept": c, "head": h, "shared": h in shared, "content": v["content"], "sink": v["sink"],
                             "clue_share_of_top": sum(w for t, w in top if clue(t)) / tot,
                             "top": [t for t, _ in top[:4]]})
    for name, sel in (("shared heads", [r for r in rows if r["shared"]]), ("concept-specific heads", [r for r in rows if not r["shared"]])):
        n = len(sel)
        if not n:
            continue
        print(f"{name:<24} n={n:>3} | attention on question content {sum(r['content'] for r in sel)/n:.2f} | on sinks "
              f"{sum(r['sink'] for r in sel)/n:.2f} | heads with >= 30% on content: {sum(r['content'] >= .3 for r in sel)} "
              f"| clue-word share of their top content tokens {sum(r['clue_share_of_top'] for r in sel)/n:.2f}")
    print("\nheads that mostly read the question (content >= 0.3):")
    for r in sorted(rows, key=lambda r: -r["content"])[:15]:
        print(f"  {r['concept']:<24} {r['head']:<7} {'shared' if r['shared'] else 'specific':<8} content {r['content']:.2f} -> {r['top']}")
    (ROOT / "results/head_looks_summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
