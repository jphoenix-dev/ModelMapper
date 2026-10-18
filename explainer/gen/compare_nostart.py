"""Before/after for no_name_start.py, scored identically on both sides (crude, symmetric, no API):
  mention   the reply contains the concept (same stems as the page's highlighting)
  stuck     under 35% of the reply's words are distinct (the page's "stuck repeating" rule)
  fluent    mention and not stuck
  loop      the name's first piece repeats 5+ times
Prints, per concept, the best fluent count over strengths (of 8 replies) and loops at the strongest settings.
Usage: .venv/Scripts/python.exe explainer/gen/compare_nostart.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "explainer" / "gen"))
from build_site_data import stuck, words_for  # noqa: E402

RAW, NEW = ROOT / "explainer/data/raw", ROOT / "explainer/data/raw_nostart"


def score(grid, stems, pieces):
    rx = re.compile("|".join(map(re.escape, stems)), re.I) if stems else None
    out = {}
    for s, reps in grid.items():
        m = [bool(rx and rx.search(r)) for r in reps]
        st = [stuck(r) for r in reps]
        lp = [any(len(re.findall(re.escape(p), r, re.I)) >= 5 for p in pieces if len(p) >= 1) for r in reps]
        out[float(s)] = {"fluent": sum(a and not b for a, b in zip(m, st)), "stuck": sum(st), "loop": sum(lp), "n": len(reps)}
    return out


def main():
    rows = []
    for f in sorted(NEW.glob("concepts_*.json")):
        y = json.loads(f.read_text(encoding="utf-8"))
        if y.get("status") != "ok" or "steer" not in y:
            continue
        x = json.loads((RAW / f.name).read_text(encoding="utf-8"))
        stems = words_for(y["run"]["concept"], y["answer"])
        b = score(x["steer"]["replies"], stems, y["pieces"])
        a = score(y["steer"]["replies"], stems, y["pieces"])
        c = score(y["steer"]["control"]["replies"], stems, y["pieces"]) if "control" in y["steer"] else None
        best = lambda g: max(v["fluent"] for v in g.values()) if g else None  # noqa: E731
        hi = [s for s in b if s >= 1.5]
        loops = lambda g: sum(g[s]["loop"] for s in hi) if g else None  # noqa: E731
        rows.append((y["run"]["concept"], best(b), best(a), best(c), loops(b), loops(a), loops(c), len(y["steer"]["removed"])))
    print(f"{'concept':<28} best fluent of 8 (original / name-start swapped / random swap)   name loops at >=1.5x (same order)   swapped")
    for r in rows:
        print(f"{r[0]:<28} {r[1]} / {r[2]} / {r[3]}{'':<46}{r[4]} / {r[5]} / {r[6]}{'':<24}{r[7]}")
    ok = [r for r in rows if r[3] is not None]
    if ok:
        n = len(ok)
        m = lambda i: sum(r[i] for r in ok) / n  # noqa: E731
        print(f"\n{n} concepts with control: mean best fluent original {m(1):.2f}, name-start swapped {m(2):.2f}, random swap {m(3):.2f}")
        print(f"swapped vs original: better {sum(r[2] > r[1] for r in ok)}, worse {sum(r[2] < r[1] for r in ok)}; "
              f"swapped vs random: better {sum(r[2] > r[3] for r in ok)}, worse {sum(r[2] < r[3] for r in ok)}")
        print(f"name loops at >=1.5x: original {sum(r[4] for r in ok)}, swapped {sum(r[5] for r in ok)}, random {sum(r[6] for r in ok)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
