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
  check         check this install; writes a report to send back (--apis: labeller checks; --labels: labelling checks)
  map-concept   map ONE concept to the neurons it depends on (questions -> score -> switch off -> random controls)
  map-batch     the same for a list of concepts (one per line); resumable, one record per concept
  label         label what each part of a neuron does, checked: neurons, --layers, --sample N or --all (plan first)
Explore
  lookup        what is already known about a neuron or concept: the published map and labels, or your own records
  steer         hold chosen neurons at a value and see what the model says (optionally judged by a judge model)
Contribute
  export-pack   your labelling runs and map records as a contribution pack for the research ledger (LEDGER.md format)
Advanced: the labelling steps that `label` runs (LABELLING.md)
  sweep         every token in 12 sentences, read for every neuron: favourites, profiles, hub tokens (GPU for big models)
  pseudo        make the fake-neuron sets that labelling must fail on (the falsifier)
  exam          the label exam: describe each neuron's favourites, judge held-out ones against decoys (labeller + judge)
  gencheck      generative check: words written from a verified label must switch the neuron on (AUC >= 0.75)
  jobtype       each neuron's job type per side (amplifier / eraser / redirector), from the weights alone
Experimental
  realtext      check labels on real text: the corpus exam and graded simulation (the research run's code; LABELLING.md)
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
    print(f"  {'neuron':<12} {'log-odds drop':>13}  {'role':<11} pushes the model towards (6 of its 30 words)"
          + ("   [roles approximate: post-MLP norm]" if any(t.role_approximate for t in r.top) else ""))
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
    from modelmapper.measure import PromptFormatError
    spec = json.loads(Path(a.questions).read_text(encoding="utf-8")) if a.questions else None
    try:
        r = map_concept(a.concept, model=a.model, out=a.out, budget=a.budget, spec=spec, band=a.band)
    except LabellerError as e:
        print(f"The labeller step failed: {e}", file=sys.stderr)
        return 1
    except PromptFormatError as e:
        print(f"Cannot ask this model the questions: {e}", file=sys.stderr)
        return 2
    _print_result(r)
    return 0 if r.status == "ok" else 3


def cmd_map_batch(a):
    from modelmapper.api import map_batch
    from modelmapper.measure import PromptFormatError
    text = Path(a.concepts).read_text(encoding="utf-8")
    concepts = json.loads(text) if a.concepts.endswith(".json") else [l.split("#")[0] for l in text.splitlines()]
    try:
        rs = map_batch(concepts, model=a.model, out=a.out, budget=a.budget, resume=not a.no_resume, band=a.band)
    except PromptFormatError as e:
        print(f"Cannot ask this model the questions: {e}", file=sys.stderr)
        return 2
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
    if hasattr(sys.stdout, "reconfigure") and not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")       # labels have non-ASCII: a pipe gets UTF-8
    print(json.dumps(res, indent=1, ensure_ascii=False) if a.json else L.show(res))
    return 0


def cmd_ranges(a):
    from modelmapper.ranges import build_ranges
    out = build_ranges(a.model, int(float(a.tokens)), a.corpus, a.out)
    print(f"saved {out}")
    return 0


def cmd_jobtype(a):
    from modelmapper import jobtype
    from modelmapper.weights import Weights
    jt = jobtype.job_types(Weights(a.model))
    out = jobtype.save(jt, a.out)
    print(jobtype.summary(jt, a.neurons))
    print(f"saved {out}")
    return 0


def cmd_pseudo(a):
    from modelmapper import pseudo
    from modelmapper.weights import Weights
    data = json.loads(Path(a.sample).read_text(encoding="utf-8"))
    sample = data["neurons"] if isinstance(data, dict) else data
    W = Weights(a.model)
    p1 = pseudo.make_gate_set(W, [int(s[1:].split(":")[0]) for s in sample], n=a.n)
    out = Path(a.out)
    print(f"saved {pseudo.save(p1, out / 'pseudo_gate.npz')} and {pseudo.save(pseudo.make_up_set(W, p1), out / 'pseudo_up.npz')}")
    return 0


def cmd_sweep(a):
    from modelmapper import pseudo, sweep
    from modelmapper.model import load
    from modelmapper.settings import DEFAULT_MODEL
    sample = None
    if a.sample:
        data = json.loads(Path(a.sample).read_text(encoding="utf-8"))
        sample = list(data["neurons"] if isinstance(data, dict) else data)
    pset = pseudo.load(a.pseudo) if a.pseudo else None
    model, tok = load(a.model or DEFAULT_MODEL)
    sweep.run_sweep(model, tok, a.out, a.tag, sample=sample, pseudo=pset, check=sweep.default_check(sample) if pset is not None else (),
                    k=a.k, max_tokens=a.max_tokens)
    sweep.hub_tokens(a.out, a.tag, model.get_input_embeddings().weight.shape[0], tok=tok)
    print(f"saved {Path(a.out) / f'vocab_sweep_{a.tag}.json'} and the files beside it")
    return 0


def cmd_gencheck(a):
    from modelmapper import gencheck
    return gencheck.cli(a)




def cmd_exam(a):
    from modelmapper import exam
    pool, focus = None, None
    if a.neurons:
        d = json.loads(Path(a.neurons).read_text(encoding="utf-8"))
        pool = d["neurons"] if isinstance(d, dict) and "neurons" in d else d
        pool = list(pool.keys()) if isinstance(pool, dict) else list(pool)
        focus = d.get("focus") if isinstance(d, dict) else None
    if a.pseudo:
        d = json.loads(Path(a.pseudo).read_text(encoding="utf-8"))
        focus = list(d["neurons"] if isinstance(d, dict) else d)
    elif not pool:
        print("exam: give --neurons (or --pseudo)"); return 2
    try:
        exam.run_exam(a.sweep_dir, a.sweep_tag, pool, a.out, focus=focus, pseudo=bool(a.pseudo), reuse_desc=a.reuse_desc,
                      describe_only=a.describe_only, readings_dir=a.readings, tokenizer=a.model or "Qwen/Qwen3-8B",
                      config=exam.ExamConfig(board=a.board, budget_usd=a.budget))
    except exam.BudgetExceeded as e:
        print(e); return 3
    print(f"saved {a.out}")
    return 0


def cmd_label(a):
    from modelmapper import labelcli
    return labelcli.cli(a)


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
    j = sub.add_parser("jobtype", help="each neuron's job type per side, from the weights",
                       description="Amplifier / eraser / redirector for each neuron's + and - side, from the weights alone (CPU).")
    j.add_argument("neurons", nargs="*", help="neurons to print, e.g. L31:10454")
    j.add_argument("--model"); j.add_argument("--out", default="jobtype.npz")
    q = sub.add_parser("pseudo", help="make the pseudo-neuron falsifier sets",
                       description="Fake neurons (fixed once written) that go through labelling like real ones: the falsifier.")
    q.add_argument("sample", help="JSON list of the real neurons being labelled (or {\"neurons\": [...]}); the fakes copy its layers")
    q.add_argument("--model"); q.add_argument("--n", type=int, default=300); q.add_argument("--out", default="pseudo")
    w = sub.add_parser("sweep", help="the vocabulary sweep (favourites of every neuron), then hub tokens",
                       description="Every token in the slot of each panel context, read for every neuron (sweep.py). Needs a GPU for a full run.")
    w.add_argument("tag", help="names the output files: vocab_sweep_<tag>_c<k>.npz, vocab_sweep_<tag>.json, hub_tokens_<tag>.*")
    w.add_argument("--model"); w.add_argument("--out", default="sweep")
    w.add_argument("--sample", help="JSON list of neurons to keep full profiles for (or {\"neurons\": [...]})")
    w.add_argument("--pseudo", help="a pseudo-neuron set (.npz from `modelmapper pseudo`) to score")
    w.add_argument("--max-tokens", type=int, default=0, help="a random subset of the vocabulary (smoke run; default all)")
    w.add_argument("--k", type=int, default=100, help="favourites kept per neuron, list and reading (default 100)")
    g = sub.add_parser("gencheck", help="generative check of exam-verified labels",
                       description="Luna writes 30 words from each exam-verified label and 30 from a surface-only description of "
                                   "the same neuron; the neuron must fire more on the first set (AUC >= 0.75). Writes "
                                   "gen_check_<tag>_surf.json (and the steps' files) next to each exam, or in --out.")
    g.add_argument("exams", nargs="+", help="exam outputs intelligibility_<tag>.json (words for all first, then one model load)")
    g.add_argument("--sweep", help="the vocabulary sweep's vocab_sweep_<tag>.json (its _c<k>.npz files beside it)")
    g.add_argument("--board", choices=("gate", "apos", "aneg"), help="default: the board the exam ran on")
    g.add_argument("--pseudo", help="the pseudo-neuron set (.npz) when the exams labelled pseudo-neurons")
    g.add_argument("--model"); g.add_argument("--out", help="output folder (default: each exam's folder)")
    g.add_argument("--words-only", action="store_true", help="stop after the Luna steps (no model; run before taking the GPU)")
    g.add_argument("--words-from", help="score this gen_words_<tag>.json instead of generating words (one exam)")
    e = sub.add_parser("exam", help="the label exam (describe, then judge held-out favourites against decoys)",
                       description="The v3 label exam on a vocabulary sweep (modelmapper/exam.py has the method). Needs the "
                                   "labeller and judge keys (SETUP.md); CPU only.")
    e.add_argument("board", choices=("gate", "apos", "aneg"))
    e.add_argument("--sweep-dir", required=True, help="folder with vocab_sweep_<tag>.json, _c*.npz and hub_tokens_<tag>.npz")
    e.add_argument("--sweep-tag", required=True)
    e.add_argument("--neurons", help='JSON {"neurons": [pool...], optional "focus": [judge only these]}')
    e.add_argument("--out", required=True, help="output JSON (intelligibility_<tag>.json; a .txt is written beside it)")
    e.add_argument("--pseudo", help="JSON list of the sweep's pseudo-neurons to judge (the pool is all of them)")
    e.add_argument("--reuse-desc", help="an earlier exam or describe-only file: reuse its descriptions")
    e.add_argument("--describe-only", action="store_true", help="describe only (for later judged slices with --reuse-desc)")
    e.add_argument("--readings", help="folder with readings_{web,questions,wiki}_full.pt for reach (default: the sweep folder)")
    e.add_argument("--budget", type=float, default=5.0, help="labeller spend cap, USD at Luna's prices (default 5)")
    e.add_argument("--model", help="tokenizer of the swept model (default Qwen/Qwen3-8B)")
    lb = sub.add_parser("label", help="label neurons: these, some layers, a sample, or the whole model (plan first)",
                        description="Label what each part of a neuron does, with every label checked (LABELLING.md). Shows the plan "
                                    "(scope, sweep, time, disk, API cost) and asks before spending; re-run to resume.")
    lb.add_argument("neurons", nargs="*", help="neurons, e.g. L31:7508 L24:1519")
    lb.add_argument("--layers", help="every neuron in these layers, e.g. 20-23 or 5,9,30")
    lb.add_argument("--sample", type=int, default=0, help="a random sample of this many neurons, spread over the layers")
    lb.add_argument("--all", action="store_true", help="every neuron in the model")
    lb.add_argument("--seed", type=int, default=0, help="seed for --sample and the control neurons (default 0)")
    lb.add_argument("--model", help="Hugging Face id (default MODEL_MAPPER_MODEL, else Qwen/Qwen3-8B)")
    lb.add_argument("--with-fakes", type=int, default=0, metavar="N", help="also label N fake neurons (the falsifier); they should almost never pass")
    lb.add_argument("--quick", action="store_true", help="sweep a 20,000-token subset: a fast first look, not calibrated")
    lb.add_argument("--tokens", type=int, default=0, help="sweep this many tokens (a quick run of your size; not calibrated)")
    lb.add_argument("--budget", type=float, help="cap on labeller + judge spend in USD; past it, stop (exit 3) and keep what is done")
    lb.add_argument("--relabel", action="store_true", help="judge neurons again even if they already have labels (published or earlier runs)")
    lb.add_argument("--plan", action="store_true", help="print the plan and stop (no model, no spend)")
    lb.add_argument("--yes", action="store_true", help="don't ask before starting (agents: only once a human has agreed the plan)")
    lb.add_argument("--json", action="store_true", help="machine-readable plan and results")
    lb.add_argument("--out", help="results folder (default labels/<model>)"); lb.add_argument("--prefix", help="run name (default: from the targets)")
    lb.add_argument("--cache", help="sweep cache folder (default MODEL_MAPPER_CACHE/<model>/sweep)")
    from modelmapper import pack
    pack.add_parser(sub)
    from modelmapper.experimental import cli as realtext
    realtext.add_parser(sub)
    sub.add_parser("settings", help="where every setting points")
    if not argv or argv[0] in ("-h", "--help"):
        p.print_help(); return 0
    if argv[0] == "check":
        from modelmapper.selfcheck import main as check_main
        return check_main(argv[1:])
    a = p.parse_args(argv)
    return {"map-concept": cmd_map_concept, "map-batch": cmd_map_batch, "steer": cmd_steer, "lookup": cmd_lookup,
            "ranges": cmd_ranges, "jobtype": cmd_jobtype, "pseudo": cmd_pseudo, "exam": cmd_exam, "gencheck": cmd_gencheck, "sweep": cmd_sweep,
            "label": cmd_label, "settings": cmd_settings, "export-pack": pack.cli, "realtext": realtext.cli}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
