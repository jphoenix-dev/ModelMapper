"""Label a list of neurons end to end: the exam on every board, the generative check of the verified labels, the
pseudo-neuron falsifier and the scorecard (the research run's v3x_runner, in one process).

    label_run(neurons, out_dir="labels", prefix="run", sweep_dir="sweep", sweep_tag="v2", chunk=100,
              pseudo_sweep_tag="v2q", pseudo_set="pseudo/pseudo_up.npz", pseudo_focus=[...], jobtype="jobtype.npz")

- POOLS: the neurons are split into interleaved pools of about 1,000 (POOL), as in the research runs; each pool is one
  exam per board (all three boards in parallel), and its controls and decoys come from that pool.
- PRODUCTION FLOW (the research v3x_runner's defaults): per board, describe once (describe_only), judge the focus list
  in `batches` (2) disjoint slices that reuse those descriptions, then merge the slices into one file (exam.merge_parts);
  describe requests unanswered after `hedge_s` (45 s) get a duplicate and the first answer is used (exam.ExamConfig).
- COVERAGE: `covered` ({board: {neuron: {"exam": record, "gen": bool | None}}}) holds labels that already exist (the
  published ones, or your earlier runs); those neurons are not judged again on that board, and their records go into
  the scorecard, as the research runner skips neurons the ledger already covers.
- RESUMABLE: every step writes its file once, at its end, and is skipped when that file is already there and complete.
  A stopped run, re-run with the same arguments, redoes only the step it was in.
- SPEND CAPS: before each chunk, the labeller (Luna) and judge (Jev) spend so far is summed from the steps' summaries;
  past a cap no new chunk starts and label_run raises SpendCap (re-run with a higher cap to continue).
- THE MODEL is needed only to score the generative check's words: words are written for every board first (labeller
  only), then the model is loaded once and scores them all.
Files, in out_dir: intelligibility_<prefix>_<chunk>_<board>.json (exam), the generative check's files beside each
(modelmapper.gencheck), the same with <prefix>_pq_<board> for the pseudo-neurons, and <prefix>_results.json (the
scorecard). Throughput differs from the research runner (it overlapped describing the next chunk with judging this
one, and split each board into two judge batches); the outputs are the same method and format.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from modelmapper import scorecard
from modelmapper.exam import ExamConfig, merge_parts, run_exam
from modelmapper.gencheck import run_gencheck
from modelmapper.settings import DEFAULT_MODEL

BOARDS = ("gate", "apos", "aneg")


class SpendCap(RuntimeError):
    pass


def _done(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if "_summary" in d else None
    except (OSError, ValueError):
        return None


def spend(out_dir: Path, prefix: str) -> tuple[float, float]:
    """(labeller USD, judge USD) so far, from the exam and generative-check summaries of this run."""
    lu = jv = 0.0
    files = {f.stem: f for f in out_dir.glob(f"intelligibility_{prefix}_*.json") if not f.stem.endswith(("_ver", "_hedge"))}
    for stem, f in files.items():
        base = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[-1] in ("desc", "b0", "b1", "b2", "b3") else None
        if base and base in files and _done(files[base]):
            continue                                       # the merged file already counts this part
        s = (_done(f) or {}).get("_summary", {})
        lu += s.get("cost_usd") or 0; jv += s.get("jev_usd") or 0
    for f in out_dir.glob(f"graded_surface_{prefix}_*.json"):          # surface labels (the generative check's control)
        lu += (_done(f) or {}).get("_summary", {}).get("luna_usd") or 0
    for f in out_dir.glob(f"gen_words_{prefix}_*.json"):               # the words (gen_check repeats this figure: not added)
        try:
            lu += json.loads(f.read_text(encoding="utf-8")).get("luna_usd") or 0
        except (OSError, ValueError):
            pass
    return round(lu, 4), round(jv, 4)


POOL = 1000          # neurons per exam pool: controls and decoys come from the pool (the v3 / v3x runs used ~1,000)


def pools(neurons: list[str], size: int = POOL) -> list[list[str]]:
    """Interleaved pools of about `size` (neurons[k::n]), so each pool spans all the layers, as the research runs did."""
    n = max(1, -(-len(neurons) // size))
    return [neurons[k::n] for k in range(n)]


def label_run(neurons: list[str], out_dir, prefix: str, sweep_dir, sweep_tag: str, *, chunk: int = POOL,
              boards=BOARDS, pseudo_sweep_tag: str | None = None, pseudo_set=None, pseudo_focus=None, jobtype=None,
              model=None, tok=None, model_name: str | None = None, luna_cap: float = 45.0, jev_cap: float = 65.0,
              config: dict | None = None, targets: list[str] | None = None, budget: float | None = None,
              provenance: dict | None = None, batches: int = 2, hedge_s: float = 45.0, covered: dict | None = None,
              log=print) -> dict:
    """Run (or resume) a labelling run; returns the scorecard dict and writes <prefix>_results.json.

    neurons: every neuron in the run, split into interleaved pools of about `chunk` (POOL); each pool is one exam per
    board, and its controls and decoys come from that pool. targets: the neurons to judge (default: all of them); the
    rest are controls, described but not judged.
    budget: a cap on labeller + judge spend together, in USD (on top of the per-service caps)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    targets = list(targets or neurons); tset = set(targets)
    chunks = [(pool, [u for u in pool if u in tset]) for pool in pools(neurons, chunk)]
    chunks = [c for c in chunks if c[1]]
    exams = {b: [] for b in boards}
    cfg = lambda b: ExamConfig(board=b, **{"hedge_s": hedge_s, **(config or {})})
    cov = covered or {}

    def exam(path, **kw):
        if _done(path):
            log(f"{path.name}: done, skipped"); return
        run_exam(sweep_dir=kw.pop("sweep_dir", sweep_dir), out=path, log=log, tokenizer=model_name or DEFAULT_MODEL, **kw)

    def guard():
        lu, jv = spend(out, prefix)
        log(f"spend so far: labeller ${lu:.2f} (cap {luna_cap}), judge ${jv:.2f} (cap {jev_cap})")
        if lu > luna_cap or jv > jev_cap or (budget is not None and lu + jv > budget):
            raise SpendCap(f"spend cap reached: labeller ${lu:.2f} / {luna_cap}, judge ${jv:.2f} / {jev_cap}; "
                           "re-run with higher caps to continue (finished steps are kept)")

    def boards_parallel(jobs):
        errs = []

        def one(fn):
            try:
                fn()
            except Exception as e:          # noqa: BLE001  (reported below; the other boards finish)
                errs.append(e)
        ts = [threading.Thread(target=one, args=(fn,)) for fn in jobs]
        [t.start() for t in ts]; [t.join() for t in ts]
        if errs:
            raise errs[0]

    def board_exam(k, pool, part, b):
        """One board of one pool, the production way: describe once, judge `batches` slices, merge. -> path or None."""
        path = out / f"intelligibility_{prefix}_c{k}_{b}.json"
        if _done(path):
            log(f"{path.name}: done, skipped"); return path
        focus = [u for u in part if u not in cov.get(b, {})]
        if not focus:
            log(f"{path.stem}: every neuron already has a {b} label, skipped"); return None
        if batches <= 1:
            exam(path, sweep_tag=sweep_tag, neurons=pool, focus=focus, config=cfg(b)); return path
        desc = out / f"intelligibility_{prefix}_c{k}_{b}_desc.json"
        exam(desc, sweep_tag=sweep_tag, neurons=pool, focus=focus, describe_only=True, config=cfg(b))
        parts = [out / f"intelligibility_{prefix}_c{k}_{b}_b{j}.json" for j in range(batches)]
        boards_parallel([lambda j=j: exam(parts[j], sweep_tag=sweep_tag, neurons=pool, focus=focus[j::batches], reuse_desc=desc,
                                          config=cfg(b)) for j in range(batches) if focus[j::batches]])
        merge_parts(path, [desc] + [q for q in parts if q.exists()])
        return path

    # 1. exams, pool by pool (three boards at a time), then the words for each board's verified labels
    for k, (pool, part) in enumerate(chunks):
        guard()
        paths = {}
        boards_parallel([lambda b=b: paths.__setitem__(b, board_exam(k, pool, part, b)) for b in boards])
        boards_parallel([lambda b=b: run_gencheck(paths[b], out, board=b, words_only=True, log=log) for b in boards if paths.get(b)])
        for b in boards:
            if paths.get(b):
                exams[b].append(paths[b])
    # 2. the falsifier: the same exams on pseudo-neurons
    pq = {}
    if pseudo_sweep_tag and pseudo_focus:
        guard()
        pq = {b: out / f"intelligibility_{prefix}_pq_{b}.json" for b in boards}
        boards_parallel([lambda b=b: exam(pq[b], sweep_tag=pseudo_sweep_tag, neurons=None, focus=list(pseudo_focus), pseudo=True,
                                          config=cfg(b)) for b in boards])
        boards_parallel([lambda b=b: run_gencheck(pq[b], out, board=b, words_only=True, log=log) for b in boards])
    # 3. one model load scores every board's words
    jobs = [(p, b, sweep_tag, None) for b in boards for p in exams[b]] + ([(pq[b], b, pseudo_sweep_tag, pseudo_set) for b in boards] if pq else [])
    pending = [j for j in jobs if not _done(out / f"gen_check_{j[0].stem[len('intelligibility_'):]}_surf.json")]
    if pending and model is None:
        from modelmapper.model import load
        model, tok = load(model_name or DEFAULT_MODEL)
    for p, b, tag, ps in pending:
        run_gencheck(p, out, sweep_dir=sweep_dir, sweep_tag=tag, board=b, model=model, tok=tok, pseudo=ps, log=log)
    # 4. the scorecard
    def merged(pattern):
        """{board: all chunks' records} from the files that exist (pattern takes the file's tag)."""
        got = {}
        for b in boards:
            got[b] = {}
            for p in exams[b]:
                f = out / pattern.format(p.stem[len("intelligibility_"):])
                if f.exists():
                    got[b].update({k: v for k, v in json.loads(f.read_text(encoding="utf-8")).items() if not k.startswith("_")})
        return got
    ex_all, gen_all = merged("intelligibility_{}.json"), merged("gen_check_{}_surf.json")
    for b, recs in cov.items():                          # labels that already existed (coverage): reused as they are
        for u, r in recs.items():
            if u in tset and u not in ex_all.get(b, {}):
                ex_all.setdefault(b, {})[u] = r["exam"]
                if r.get("gen") is not None:
                    gen_all.setdefault(b, {})[u] = {"pass": r["gen"]}
    res = scorecard.combine(targets, ex_all, gen_all, jobtype)
    if pq:
        res["pseudo"] = scorecard.pseudo_summary(pq, {b: out / f"gen_check_{prefix}_pq_{b}_surf.json" for b in boards})
    res["spend_usd"] = dict(zip(("labeller", "judge"), spend(out, prefix)))
    if provenance:
        res["provenance"] = provenance
    scorecard.save(res, out / f"{prefix}_results.json")
    log(f"saved {out / f'{prefix}_results.json'}: triage {res['summary']['triage']}")
    return res
