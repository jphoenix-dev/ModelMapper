"""The `modelmapper` command. `modelmapper <command> --help` for any command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HELP = """Model Mapper: find which neurons of a language model a concept depends on, and what they do.

New here? Three steps:  pip install modelmapper   then   modelmapper check   then   modelmapper map-concept "Coffee"
(map-concept needs a labeller key: SETUP.md. On a small GPU or a CPU: map-concept "Bitcoin" --model Qwen/Qwen3-0.6B)

Start here
  check         check this install; writes a report to send back (--apis adds the labeller checks)
  map-concept   map ONE concept to the neurons it depends on (questions -> score -> switch off -> random controls)
  map-batch     the same for a list of concepts (one per line); resumable, one record per concept
Explore
  steer         hold chosen neurons at a value and see what the model says (optionally judged by a judge model)
  lookup        what the published map (or your own records) already knows about a neuron or a concept
Set-up
  ranges        build a neuron range table for another model (for steer's "x peak"; Qwen3-0.6B / 8B ship with one)
  settings      where every setting points, and where it came from

What the numbers mean: README.md. The procedure and its controls: METHOD.md. The record format: FORMAT.md."""


def _print_result(r, n=15):
    if r.status != "ok":
        print(f"\n{r.concept}: {r.status}" + (f" - {r.detail}" if r.detail else
              f" - the model answered {r.recall} of the questions correctly (5 needed); nothing to map") + f". Record: {r.path}")
        return
    print(f"\n{r.concept}: answer {r.answer_form!r}, {r.recall} questions answered correctly (mean log-odds "
          f"{r.baseline_logodds}). Switching off neurons one at a time:")
    print(f"  {'neuron':<12} {'log-odds drop':>13}  {'role':<11} pushes the model towards (6 of its 30 words)")
    for t in r.top[:n]:
        words = " ".join(w.strip() for w in t.pushes[:6] if w.strip())
        print(f"  {t.neuron:<12} {t.drop:>13.2f}  {t.role or '':<11} {words}" + ("   [uncertain]" if t.call == "uncertain" else ""))
    if len(r.top) > n:
        print(f"  ... {len(r.top) - n} more in the top 30 (the record has all 200 candidates)")
    g = r.guards
    if g.get("n_in") is not None:
        print(f"  {g['n_in']} of the 30 are clearly in; {g['n_uncertain']} are within {g['band']['value']:g} of the 30th "
              f"neuron's drop, so could swap places on other hardware ([uncertain]), as could {g['n_out_but_close']} just below")
    elif g.get("band"):
        print(f"  calls: uncalibrated ({g['band']['source']}); order and drops are as measured")
    print(f"  random same-layer neurons switched off (control): {', '.join(f'{v:.2f}' for v in r.nulls.values())}")
    def ks(c):
        return " / ".join("-" if c.get(k) is None else f"{c[k] + 0.0:.2f}" for k in (1, 5, 10))
    print(f"  top 1 / 5 / 10 switched off together: mean drop {ks(r.k_curve)} (as many random neurons: {ks(r.random_k_curve)})")
    for note in r.notes:
        print(f"  note: {note}")
    print(f"\nRecord: {r.path}. What else is known about a neuron: modelmapper lookup <neuron>")


def cmd_map_concept(a):
    from modelmapper.api import LabellerError, map_concept
    spec = json.loads(Path(a.questions).read_text(encoding="utf-8")) if a.questions else None
    try:
        r = map_concept(a.concept, model=a.model, out=a.out, budget=a.budget, spec=spec, band=a.band)
    except LabellerError as e:
        print(f"The labeller step failed: {e}", file=sys.stderr)
        return 1
    _print_result(r)
    return 0 if r.status == "ok" else 3


def cmd_map_batch(a):
    from modelmapper.api import map_batch
    text = Path(a.concepts).read_text(encoding="utf-8")
    concepts = json.loads(text) if a.concepts.endswith(".json") else [l.split("#")[0] for l in text.splitlines()]
    rs = map_batch(concepts, model=a.model, out=a.out, budget=a.budget, resume=not a.no_resume, band=a.band)
    by = {}
    for r in rs:
        by[r.status] = by.get(r.status, 0) + 1
    print(f"\n{len(rs)} concepts: " + ", ".join(f"{v} {k}" for k, v in sorted(by.items())))
    for r in rs:
        top = ", ".join(t.neuron for t in r.top[:3])
        print(f"  {r.concept:<32} {r.status:<13} {top}")
    return 0 if by.get("failed", 0) == 0 else 1


def cmd_steer(a):
    from modelmapper.api import steer
    r = steer(a.units, model=a.model, prompts=a.prompt or None, max_new=a.tokens, judge=a.judge, baseline=not a.no_baseline,
              reply_only=not a.whole)
    print("holding: " + ", ".join(f"L{l}:{n} = {v:g}" for l, n, v in r["units"]) + (" (reply only)" if r["reply_only"] else " (question and reply)"))
    j = r["judge"]
    for i, p in enumerate(r["prompts"]):
        print(f"\nQ: {p}")
        if r["baseline"]:
            print(f"  without: {r['baseline'][i]!r}" + (f"   [judge {j['baseline'][i]:.2f}, fluent {j['fluent_baseline'][i]:.2f}]" if j else ""))
        print(f"  steered: {r['steered'][i]!r}" + (f"   [judge {j['steered'][i]:.2f}, fluent {j['fluent_steered'][i]:.2f}]" if j else ""))
    if j:
        print(f"\njudge ({j['backend']}): P(yes) to {j['question']!r}; the published results used Jev")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8"); print(f"saved {a.out}")
    return 0


def cmd_lookup(a):
    from modelmapper import known as L
    try:
        res = L.lookup(a.query, records=a.records, refresh=a.refresh)
    except (KeyError, ConnectionError) as e:
        print(e.args[0] if e.args else e, file=sys.stderr)
        return 2
    print(json.dumps(res, indent=1, ensure_ascii=False) if a.json else L.show(res))
    return 0


def cmd_ranges(a):
    from modelmapper.ranges import build_ranges
    out = build_ranges(a.model, int(float(a.tokens)), a.corpus, a.out)
    print(f"saved {out}")
    return 0


def cmd_settings(a):
    from modelmapper import settings as S
    print(f"working folder: {S.WORK_DIR}   (.env {'found' if S.DOTENV.exists() else 'not found'}: {S.DOTENV.name}; "
          "values of keys are never printed)\n")
    for var, value, src in S.describe():
        print(f"  {var:<22} {value:<60} [{src}]")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    import modelmapper
    p = argparse.ArgumentParser(prog="modelmapper", usage="modelmapper <command> [options]", description=HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"modelmapper {modelmapper.__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="", prog="modelmapper")
    sub.add_parser("check", help="check this install", add_help=False)
    m = sub.add_parser("map-concept", help="map one concept", description="Map ONE concept to the neurons it depends on.")
    m.add_argument("concept", help='e.g. "Coffee" or "Eiffel Tower"')
    m.add_argument("--model", help="Hugging Face id (default MODEL_MAPPER_MODEL, else Qwen/Qwen3-8B)")
    m.add_argument("--out", help="output folder (default ./maps); each model gets a subfolder")
    m.add_argument("--budget", type=float, default=0.5, help="labeller spend cap for the concept, USD at Luna's prices (default 0.5)")
    m.add_argument("--questions", help="a saved question set (spec JSON) to use instead of asking the labeller")
    m.add_argument("--band", type=float, help="hardware band for the in / uncertain / out calls, in log-odds (default: the model's calibrated band, 0.13 "
                        "for Qwen3-8B; other models: calls are 'uncalibrated'). Recorded in the record.")
    b = sub.add_parser("map-batch", help="map a list of concepts", description="map-concept over a list (one per line, or a JSON list).")
    b.add_argument("concepts", help="a text file, one concept per line (# starts a comment), or a .json list")
    b.add_argument("--model"); b.add_argument("--out"); b.add_argument("--budget", type=float, default=0.5)
    b.add_argument("--no-resume", action="store_true", help="re-run concepts that already have a record")
    b.add_argument("--band", type=float, help="hardware band for the calls (default: the model's calibrated one; see map-concept --help)")
    s = sub.add_parser("steer", help="hold neurons at a value; see (and optionally judge) the replies",
                       description='Hold neurons at a value while the model answers. Units: L23:2392=x8 (8 x its own peak, '
                                   'from its range table), L23:2392=x-4 (4 x its minimum), L23:2392=12.5 (an absolute value).')
    s.add_argument("units", nargs="+", help="e.g. L23:2392=x8 L4:2581=-3")
    s.add_argument("--model"); s.add_argument("--prompt", action="append", help="a prompt (repeat for more; default 5 neutral prompts)")
    s.add_argument("--tokens", type=int, default=60, help="new tokens per reply (default 60)")
    s.add_argument("--judge", help='a yes/no question for the judge model, e.g. "Does the text mention cryptocurrency?"')
    s.add_argument("--whole", action="store_true", help="also steer while the question is read (default: the reply only)")
    s.add_argument("--no-baseline", action="store_true", help="skip the unsteered replies")
    s.add_argument("--out", help="save the replies (and judge scores) as JSON")
    k = sub.add_parser("lookup", help="what is known about a neuron or concept", description="Look up a neuron (L31:8930) or a concept.")
    k.add_argument("query"); k.add_argument("--records", help="a folder of your own records instead of the published map")
    k.add_argument("--json", action="store_true"); k.add_argument("--refresh", action="store_true", help="download the map again")
    r = sub.add_parser("ranges", help="build a range table for a model", description="Each neuron's max / min on web text.")
    r.add_argument("model"); r.add_argument("--tokens", default="2e6"); r.add_argument("--corpus"); r.add_argument("--out")
    sub.add_parser("settings", help="where every setting points")
    if not argv or argv[0] in ("-h", "--help"):
        p.print_help(); return 0
    if argv[0] == "check":
        from modelmapper.selfcheck import main as check_main
        return check_main(argv[1:])
    a = p.parse_args(argv)
    return {"map-concept": cmd_map_concept, "map-batch": cmd_map_batch, "steer": cmd_steer, "lookup": cmd_lookup,
            "ranges": cmd_ranges, "settings": cmd_settings}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
