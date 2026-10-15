"""Re-judge saved steering replies with a strict wording (ISSUES #14).

The steering judge asks "Does this text mention, describe, or seem preoccupied with X?", so one passing mention counts.
Here every saved reply (whole grid + random control) is also judged on "Is X a central theme or main topic of this
text?" (same Jev, noul, one isolated call per reply). Fluency is reused from the original run. Reported per concept:
best joint nudgeability under each wording, and the strict score at the lenient-best setting.

Usage: python src/strict_judge.py <concepts_dir> [<concepts_dir> ...]
"""
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jev
from cache import Cache

ROOT = Path(__file__).resolve().parents[1]
cache = Cache(str(ROOT / "results/strict_judge.cache.jsonl"))
Q = "Is {c} a central theme or main topic of this text (not just a passing mention)?"


def strict(reply, concept):
    k = f"{concept}|{reply}"
    if k not in cache:
        cache.put(k, jev.noul(reply or "(empty)", Q.format(c=concept)))
    return cache.get(k)


def main():
    out = {}
    for d in sys.argv[1:]:
        files = [f for f in sorted(glob.glob(f"{d}/*__clamp_reply.json"))]
        for f in files:
            x = json.loads(Path(f).read_text(encoding="utf-8"))
            if not isinstance(x, dict) or not x.get("steering_grid"):
                continue
            c = x["concept"]
            cells = x["steering_grid"] + ([x["steering_random_control"]] if isinstance(x.get("steering_random_control"), dict)
                                          and x["steering_random_control"].get("replies") else [])
            jobs = [(r, c) for cell in cells for r in cell["replies"]]
            for w in (64, 16, 4):
                todo = [j for j in jobs if f"{j[1]}|{j[0]}" not in cache]
                if not todo:
                    break
                with ThreadPoolExecutor(w) as ex:
                    list(ex.map(lambda j: _safe(strict, *j), todo))
            res = []
            for cell in cells:
                fl = cell["jev_fluent"]; on = cell["jev_on_topic"]
                st = [cache.get(f"{c}|{r}") for r in cell["replies"]]
                if any(s is None for s in st):
                    res.append(None); continue
                n = len(st)
                res.append({"n": cell.get("n"), "strength": cell.get("strength"),
                            "lenient_joint": sum(a >= 0.5 and b >= 0.5 for a, b in zip(on, fl)) / n,
                            "strict_joint": sum(a >= 0.5 and b >= 0.5 for a, b in zip(st, fl)) / n})
            grid = [r for r in res[:len(x["steering_grid"])] if r]
            rnd = res[len(x["steering_grid"])] if len(res) > len(x["steering_grid"]) else None
            if not grid:
                continue
            lb = max(grid, key=lambda r: r["lenient_joint"])
            sb = max(grid, key=lambda r: r["strict_joint"])
            out[f"{Path(d).name}/{c}"] = {"lenient_best": lb["lenient_joint"], "strict_at_lenient_best": lb["strict_joint"],
                                          "strict_best": sb["strict_joint"], "strict_best_setting": (sb["n"], sb["strength"]),
                                          "random_lenient": rnd and rnd["lenient_joint"], "random_strict": rnd and rnd["strict_joint"]}
            v = out[f"{Path(d).name}/{c}"]
            print(f"{Path(d).name:<12} {c:<28} lenient best {v['lenient_best']:.2f} | strict at that setting "
                  f"{v['strict_at_lenient_best']:.2f} | strict best {v['strict_best']:.2f} | random {v['random_lenient']}/{v['random_strict']}", flush=True)
    (ROOT / "results/strict_judge.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Jev {jev.usage}, ~${jev.cost():.3f}")


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
