"""Follow-up (James, explainer): does steering read better without the "say the start of the name" neuron?

Some steered sets include a neuron whose direct output is the first token of the concept's name (Serena Williams:
L34:9038 writes "Ser"; Eiffel Tower: L35:6192 writes "E"). Held up on every token, it restarts the name over and
over ("Ser Ser Ser"). Here, for every concept whose saved steering set contains such a neuron, we swap it for the
next-best neuron from the same rule (same count), and re-steer with exactly the same prompts, strengths and code as
make_data.py. Hero: the Eiffel six-neuron clamp grid likewise.

A neuron counts as "name-start" if one of its top-8 direct output tokens, lower-cased and stripped of spaces and
punctuation, equals the first token of the concept's answer (as it would start a sentence or follow a space).
Reads explainer/data/raw (from make_data.py); writes explainer/data/raw_nostart. No API calls.

Usage: .venv/Scripts/python.exe explainer/gen/no_name_start.py [--only "Serena Williams,Eiffel Tower"]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_data as md  # noqa: E402  (sets up paths, env and the no-API guard)

RAW, OUT = md.RAW, md.ROOT / "explainer" / "data" / "raw_nostart"


def norm(w):
    return re.sub(r"[\s_\-.:=,;>()\[\]\"'`\t]+", "", w).lower()


def first_pieces(tok, answer):
    out = set()
    for a in (answer, " " + answer):
        ids = tok.encode(a)
        if ids:
            out.add(norm(tok.decode([ids[0]])))
    return {p for p in out if p}


def is_name_start(writes, pieces):
    return any(norm(w) in pieces for w in writes if norm(w))


def random_swap(G, A, x, removed):
    """Control: the same set, but each name-start neuron replaced by a random neuron from the same layer (seeded by the
    concept name), so 'removing the name neuron helps' can be told apart from 'any change to the set helps'."""
    import random
    rng = random.Random("control:" + x["run"]["concept"])
    units = [tuple(u) if list(u) not in removed else (u[0], rng.randrange(G.pipe.W)) for u in x["steer"]["units"]]
    grid, _ = G.steer_grid(units, A["last_act"], md.PROMPTS, md.STRENGTHS, md.REPLY_TOKENS)
    return {"units": [list(u) for u in units], "replies": grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    G = md.Gen()
    p = G.pipe
    want = {x.strip() for x in args.only.split(",") if x.strip()}
    for f in sorted(RAW.glob("concepts_*.json")):
        x = json.loads(f.read_text(encoding="utf-8"))
        run = x["run"]
        if x.get("status") != "ok" or (want and run["concept"] not in want):
            continue
        out = OUT / f.name
        if out.exists():
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("status") != "ok" or "steer" not in prev or "control" in prev["steer"]:
                continue
            # pilot files from before the control existed: add only the random-swap control
            G.use_dir(run["dir"])
            spec, tid, recall, _ = p.make_prompts(run["concept"])
            A = p.attribute([q for q, _ in recall], tid)
            prev["steer"]["control"] = random_swap(G, A, x, prev["steer"]["removed"])
            out.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{run['dir']}/{run['concept']:<30} control added", flush=True)
            continue
        pieces = first_pieces(p.tok, x["answer_form"] or x["answer"])
        flagged = [is_name_start(w, pieces) for w in x["steer"]["writes"]]
        hero = "hero" in x
        hero_flag = [is_name_start(w, pieces) for w in x["hero"]["writes"]] if hero else []
        if not any(flagged) and not any(hero_flag):
            out.write_text(json.dumps({"run": run, "status": "no_name_start", "pieces": sorted(pieces)}), encoding="utf-8")
            continue
        t0 = time.time()
        G.use_dir(run["dir"])
        spec, tid, recall, _ = p.make_prompts(run["concept"])
        A = p.attribute([q for q, _ in recall], tid)
        row = {"run": run, "status": "ok", "pieces": sorted(pieces), "answer": x["answer_form"] or x["answer"]}
        if any(flagged):
            pool = G.candidates(A, tid, k=40)[run["rule"]]
            assert [list(u) for u in pool[:run["n"]]] == x["steer"]["units"], "re-derived set differs from saved"
            keep = []
            for u in pool:
                if len(keep) == run["n"]:
                    break
                if not is_name_start(G.writes(A, u), pieces):
                    keep.append(u)
            grid, peaks = G.steer_grid(keep, A["last_act"], md.PROMPTS, md.STRENGTHS, md.REPLY_TOKENS)
            removed = [u for u, fl in zip(x["steer"]["units"], flagged) if fl]
            row["steer"] = {"removed": removed,
                            "units": [list(u) for u in keep], "writes": [G.writes(A, u) for u in keep],
                            "peaks": peaks, "replies": grid, "control": random_swap(G, A, x, removed)}
        if hero and any(hero_flag):
            late = [u for u in A["top"][:200] if u[0] >= p.L // 2]
            assert [list(u) for u in late[:6]] == x["hero"]["units"], "re-derived hero set differs from saved"
            keep = [u for u in late if not is_name_start(G.writes(A, u), pieces)][:6]
            mode, ro = p.mode, p.reply_only
            try:
                p.mode, p.reply_only = "clamp", True
                hgrid, _ = G.steer_grid(keep, A["last_act"], md.ORIGINAL_NEUTRAL, x["hero"]["strengths"], 120)
            finally:
                p.mode, p.reply_only = mode, ro
            row["hero"] = {"removed": [u for u, fl in zip(x["hero"]["units"], hero_flag) if fl],
                           "units": [list(u) for u in keep], "writes": [G.writes(A, u) for u in keep],
                           "strengths": x["hero"]["strengths"], "reply_only": hgrid}
        row["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(row, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{run['dir']}/{run['concept']:<30} removed {row.get('steer', {}).get('removed')} "
              f"hero removed {row.get('hero', {}).get('removed')} {row['seconds']}s", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
