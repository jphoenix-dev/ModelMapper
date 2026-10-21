"""`modelmapper realtext ...`: the EXPERIMENTAL real-text checks (modelmapper/experimental; LABELLING.md "Experimental").

    modelmapper realtext readings --corpus web --source fineweb.parquet --max-tokens 5000000
    modelmapper realtext corpus-exam --corpus web --tag t1 --labels labels/Qwen3-8B --yes
    modelmapper realtext simulate --corpus web --tag s1 --labels labels/Qwen3-8B --yes

Each step writes into --out (default realtext/<model>) and is skipped when its output exists, so a stopped run resumes.
The judge steps (corpus-exam's exam, simulate's run) spend judge calls and need --yes.
"""
from __future__ import annotations

import sys
from pathlib import Path

from modelmapper import settings as S


def add_parser(sub):
    p = sub.add_parser("realtext", help="EXPERIMENTAL: check labels on real text (corpus exam, graded simulation)",
                       description="EXPERIMENTAL real-text checks of labels: the research run's code, not calibrated in this package "
                                   "(LABELLING.md, 'Experimental').")
    s = p.add_subparsers(dest="rt", required=True)

    def common(q, judge=False):
        q.add_argument("--model", help="Hugging Face id (default MODEL_MAPPER_MODEL, else Qwen/Qwen3-8B)")
        q.add_argument("--out", help="working folder (default realtext/<model>)")
        if judge:
            q.add_argument("--labels", nargs="*", default=[], help="`label` results: *_results.json files or folders of them")
            q.add_argument("--published", action="store_true", help="also the published Qwen3-8B labels (11,000 neurons)")
            q.add_argument("--pseudo-set", help="the fake-neuron set of a label run (pseudo_*.npz in the sweep cache)")
            q.add_argument("--pseudo-exams", help="the folder with that run's intelligibility_*_pq_<part>.json")
            q.add_argument("--workers", type=int, default=128, help="parallel judge calls (default 128, as in the research run)")
            q.add_argument("--yes", action="store_true", help="spend judge calls without asking (agents: only once a human has agreed)")
    r = s.add_parser("readings", help="each neuron's strongest moments in a corpus (one forward pass over it)")
    r.add_argument("--corpus", required=True, help="a name for this corpus, e.g. web")
    r.add_argument("--source", required=True, help="the documents: .parquet (FineWeb, 'text' column), .jsonl ('text') or .txt (one per line)")
    r.add_argument("--max-tokens", type=float, default=5e6, help="stop after this many tokens (default 5,000,000, as the research web corpus)")
    r.add_argument("--offset", type=int, default=0, help="skip this many documents first")
    r.add_argument("--k", type=int, default=100, help="positions kept per neuron and board (default 100)")
    r.add_argument("--batch", type=int, default=8, help="documents per forward pass (default 8)")
    common(r)
    c = s.add_parser("corpus-exam", help="the corpus exam: positives against same-token, same-topic and random decoys")
    c.add_argument("--corpus", required=True, help="readings name(s), comma-separated, e.g. web,questions,wiki")
    c.add_argument("--tag", required=True, help="a name for this exam")
    c.add_argument("--smoke", type=int, default=0, metavar="N", help="judge only N real units (stratified) and N/3 fakes")
    common(c, judge=True)
    m = s.add_parser("simulate", help="graded simulation: the judge predicts activation strength from the label")
    m.add_argument("--corpus", default="web", help="readings name (default web)")
    m.add_argument("--tag", required=True, help="a name for this simulation")
    m.add_argument("--corpus-tag", help="a corpus-exam tag on the same corpus, for the fakes' full profiles")
    m.add_argument("--n", type=int, default=0, help="a seeded subset of N labels per part (default all)")
    m.add_argument("--positions", type=int, default=4000, help="shared positions in the pass (default 4,000)")
    common(m, judge=True)


def _pseudo(a):
    from modelmapper.experimental import realtext as RT
    if not (a.pseudo_set and a.pseudo_exams):
        return {b: {} for b in RT.BOARDS}
    ex = {}
    for b in RT.BOARDS:
        fs = sorted(f for f in Path(a.pseudo_exams).glob(f"intelligibility_*_pq_{b}.json"))
        if fs:
            ex[b] = fs[-1]
    return RT.load_pseudo(a.pseudo_set, ex)


def _confirm(a, what) -> bool:
    if a.yes:
        return True
    if not sys.stdin.isatty():
        print(f"Not starting {what}: it spends judge calls. Re-run with --yes once the cost is agreed.", file=sys.stderr)
        return False
    return input(f"{what} spends judge calls. Continue? [y/N] ").strip().lower() in ("y", "yes")


def cli(a) -> int:
    from modelmapper.experimental import corpus_exam as CE, realtext as RT, simulate as SM
    name = a.model or S.DEFAULT_MODEL
    out = Path(a.out or Path("realtext") / S.short_name(name)); out.mkdir(parents=True, exist_ok=True)
    print("EXPERIMENTAL: the research run's real-text checks, not calibrated in this package (LABELLING.md).", file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(S.model_path(name))
    loaded = {}

    def model():
        if "m" not in loaded:
            from modelmapper.model import load
            loaded["m"] = load(name)[0]
        return loaded["m"]
    if a.rt == "readings":
        if (out / f"readings_{a.corpus}.pt").exists():
            print(f"readings_{a.corpus} exists in {out}, skipped"); return 0
        RT.readings(model(), tok, RT.docs(a.source, tok, a.offset), out, a.corpus, int(a.max_tokens), K=a.k, B=a.batch)
        return 0
    labels = RT.load_labels(a.labels, a.published)
    if not any(labels.values()):
        print("no labels: give --labels (label results) and/or --published", file=sys.stderr); return 2
    pseudo = _pseudo(a)
    if a.rt == "corpus-exam":
        corpora = [c.strip() for c in a.corpus.split(",") if c.strip()]
        for c in corpora:
            if not (out / f"corpus_plan_{c}_{a.tag}.npz").exists():
                CE.plan(out, c, a.tag, labels["gate"], pseudo["gate"], tok, out)
            if not (out / f"corpus_vals_{c}_{a.tag}.npz").exists():
                CE.gather(model(), tok, out, c, a.tag, a.pseudo_set if pseudo["gate"] else None, out)
        if not _confirm(a, "the corpus exam"):
            return 1
        CE.exam(tok, out, a.tag, a.tag, corpora, out, which="smoke" if a.smoke else "all", n_real=a.smoke or 100,
                n_pseudo=max(1, a.smoke // 3) if a.smoke else 30, workers=a.workers)
        print(f"Results: {out / f'corpus_exam_{a.tag}.json'}")
        return 0
    if a.rt == "simulate":
        if any(pseudo.values()) and not a.corpus_tag:
            print("fakes need --corpus-tag: a corpus exam on the same corpus (it holds their full profiles)", file=sys.stderr); return 2
        if not (out / f"sim_pass_{a.tag}.npz").exists():
            SM.sim_pass(model(), tok, out, a.corpus, a.tag, a.pseudo_set if pseudo["gate"] else None, pseudo["gate"], out, n_pos=a.positions)
        if not _confirm(a, "the simulation"):
            return 1
        SM.sim_run(tok, out, a.tag, a.tag, labels, pseudo if any(pseudo.values()) else None, a.corpus_tag, out, n=a.n, workers=a.workers)
        print(f"Results: {out}/sim_run_{a.tag}_<part>.json")
        return 0
    return 2
