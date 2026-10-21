"""The v3 label exam: can a one-line label pick out a neuron's favourite tokens that its writer never saw? (CPU + labeller
+ judge; no GPU.) Ported from the research run (intelligibility.py with the v3 production settings, 21 Oct 2026); the same
inputs and seeds give the same favourites, held-out tokens, decoys and draws, and the output file has the same format.

Per neuron (or pseudo-neuron), from the vocabulary sweep (every vocabulary token placed in a panel of contexts; the neuron
read at the slot and at the end of each context):
  favourites  per reading, the neuron's top-100 tokens as z-scores over the whole vocabulary. The readings split by
              CONTEXT into half A (even k) and half B (odd k); in each half a token ranks by how many readings it is a
              favourite in, then by its mean z. Hub (glitch) tokens are dropped per layer: tokens that are a favourite of
              more than hub_x times the fair share of the layer's neurons (hub_tokens_<tag>.npz).
  describe    the labeller (Luna) sees ONLY half A's top 40, with z-scores, and writes n_desc descriptions (one request,
              n samples), or says there is no pattern (kind 'none').
  test list   n_test of the neuron's half-B favourites it never showed (case or space variants of shown tokens excluded),
              plus one decoy per true token with the same surface (script, leading space, length; relaxed step by step if
              none), drawn from other neurons' held-out favourites: at most 2 per neuron, never the neuron itself or its
              controls, never one of its own favourites in either half. The list is shuffled.
  label exam  the judge (Jev) rates every candidate 0-100 from a description alone (ratings under `floor` count as 0);
              AUC of true tokens over decoys, averaged over the descriptions. Control: the same list judged with n_ctrl
              OTHER neurons' descriptions (median AUC).
  list exam   the same with the shown token list in place of a description (no labeller involved); control: other
              neurons' shown lists.
Verdict per draw: an exam passes if AUC >= 0.75 and AUC - control >= 0.2 (the label exam also needs a kind other than
'none'). Bins: intelligible (both pass), list-only, label-only (check), else no-generalisation (or panel-misses, see
reach). ' (judge failed)' is added if a judge call failed; 'decoys-too-easy (<bin>)' if either control AUC is > 0.65.
Draws: draw 1 for every neuron; draw 2 (fresh decoys) only where draw 1 is within `adaptive_delta` of a threshold; draw 3
only where draws 1 and 2 disagree. Final bin: the agreed bin, else a clear majority of 3, else 'uncertain (a / b / c)'.

Reach (optional): the neuron's best sweep value / its corpus peak (readings_{web,questions,wiki}_full.pt, from the
research corpus readings; not shipped). It only NAMES a failing neuron: panel-misses (reach < 0.3) instead of
no-generalisation. Pass / fail, the draws and the 'stable' flag never depend on it. If the files are missing, reach is
None, that bin is never given and everything else is unchanged. Stability (descriptive only; no bin uses it): the share
of half A's top 50 found among half B's favourites, with a null from other neurons' halves.

Spend: `budget_usd` caps the LABELLER (Luna prices): no new description request starts once it is passed, the
descriptions so far are saved to <out>.partial.json (reuse them with reuse_desc) and BudgetExceeded is raised before any
judging. The judge is cheap (~$1.2 for 950 neurons on Jev) and not capped here.

Production flow (as the research runner): describe once (describe_only=True), judge slices of the focus list in parallel
processes reusing those descriptions (reuse_desc=..., focus=slice; every slice uses the whole pool, so decoys and controls
are those of one run), then merge_parts(). Or run_exam() in one go.

    from modelmapper.exam import run_exam
    run_exam("sweeps", "v2", neurons, "intelligibility_v3_gate.json", focus=None, config=ExamConfig(board="gate"))

Output (FORMAT of the research files, read by the scorecard and the ledger importer): {"_summary": {...}, "<neuron>":
{layer, reach, corpus_peak, n_hub_dropped, stability, test {candidates, label}, descriptions, auc_own, auc_control(s),
list_auc_own, list_auc_control(s), ratings_*, control_neurons, decoy_sources, label_pass, list_pass, bin, draws, stable,
...}} plus a one-line-per-neuron .txt beside it.
"""
from __future__ import annotations

import functools
import json
import re
import struct
import threading
import time
import unicodedata
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOARDS = ("gate", "apos", "aneg")      # gate: z_g favourites; apos / aneg: one SIGN of a = SiLU(z_g) z_u (sign-split features)
N_SHOW, N_STAB, N_POOL = 40, 50, 60    # tokens shown to the labeller; stability top-n; held-out favourites per neuron in the decoy pool
AUC_MIN, AUC_GAP, EASY, REACH_MIN = 0.75, 0.2, 0.65, 0.3
CTRL_OFFSETS = (2, 3, 5, 17, 19, 23)   # control neurons: fixed rotations of the pool (never the neuron itself)
DISTRACTOR_OFFSETS = (7, 11, 13)       # recorded as distractor_neurons (format only; the decoys are surface-matched)
CORPORA = ("web", "questions", "wiki")
LAYER_WIDTH = 12288                    # Qwen3-8B's MLP width: hub shares are per layer (sweep meta "intermediate_size" overrides)
LUNA_USD_IN, LUNA_USD_OUT = 0.10e-6, 0.50e-6
SYS_DESCRIBE = ("You see the tokens (words or word pieces) that most strongly switch on one unit inside a language model, "
                "strongest first, each with a z-score (how exceptional it is across the whole vocabulary). Whitespace is "
                "shown as ⏎ newline, ⇥ tab, ␍ carriage return, ␣ leading space. The pattern may be about MEANING (a topic, "
                "a category), FORM (spelling, letters, word pieces, a script or language) or FORMATTING (code, punctuation, "
                "whitespace), or a mix; it may also be no recognisable pattern at all - say so if so, do not invent one. "
                'Reply as JSON: {"description": "<one or two sentences concrete enough to judge new tokens>", '
                '"kind": "meaning|form|formatting|mixed|none", "confidence": <0-1>}')


@dataclass
class ExamConfig:
    """The v3 production settings (research v3x_runner BASE + V3 + BOARDS)."""
    board: str = "gate"
    n_test: int = 30               # held-out true tokens per exam (and as many decoys)
    n_ctrl: int = 5                # control descriptions / lists (other neurons), at most 6
    n_desc: int = 3                # descriptions per judged neuron (one labeller request with n = 3)
    ctrl_ndesc: int = 1            # descriptions for a neuron used ONLY as a control (controls read the first)
    floor: float = 10.0            # label-exam ratings below this count as 0
    hub_x: float = 10.0            # hub-token cut (0 = keep hub tokens)
    draws: int = 3                 # at most this many decoy draws
    adaptive_delta: float = 0.07   # draw 2 only within this of a threshold (0 = every draw for every neuron)
    merge: int = 1000              # judge questions per call (one call per judge state and round, split at this)
    reasoning: str | None = "auto"  # labeller reasoning effort; auto = 'none' on the gate board, the API default on apos / aneg
    budget_usd: float = 5.0        # labeller spend cap
    workers: int = 128             # parallel labeller / judge requests
    hedge_s: float = 0.0           # > 0: a describe request (n = n_desc) unanswered after this many s gets a duplicate; the
                                   # first good answer is used; <out>_hedge.json records them; cost includes finished
                                   # duplicates (research default 0; the production runner passes 45)
    hedge_keep: bool = False       # wait for the losing requests too (their times and answers in the side file)

    def __post_init__(self):
        if self.board not in BOARDS:
            raise ValueError(f"board must be one of {BOARDS}, not {self.board!r}")
        if not 1 <= self.n_ctrl <= len(CTRL_OFFSETS):
            raise ValueError(f"n_ctrl must be 1-{len(CTRL_OFFSETS)}")

    def reasoning_kw(self) -> dict:
        r = ("none" if self.board == "gate" else None) if self.reasoning == "auto" else self.reasoning
        return {"reasoning_effort": r} if r else {}


class BudgetExceeded(RuntimeError):
    pass


# ---- small pure helpers -----------------------------------------------------------------------------------
def visible(t: str) -> str:
    return (("␣" + t[1:]) if t.startswith(" ") else t).replace("\n", "⏎").replace("\t", "⇥").replace("\r", "␍")


def norm(t: str) -> str:
    return t.strip().lower()


def auc(pos, neg) -> float:
    p, n = np.array(pos, float)[:, None], np.array(neg, float)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def surface_class(t: str) -> str:
    """byte / space / cjk / code / latin / other-script / punct-digit"""
    st = t.strip()
    if "�" in t:
        return "byte"
    if not st:
        return "space"
    if any("一" <= c <= "鿿" or "぀" <= c <= "ヿ" or "가" <= c <= "힯" for c in st):
        return "cjk"
    if re.search(r"[_./(){}\[\]<>=;:#$@|]", st) or re.search(r"[a-z][A-Z]", st):
        return "code"
    letters = [c for c in st if c.isalpha()]
    if letters and all("LATIN" in unicodedata.name(c, "") for c in letters):
        return "latin"
    return "other-script" if letters else "punct-digit"


def surface_key(t: str) -> tuple:
    """(class, leading space, length band): what a decoy must share with its true token"""
    n = len(t.strip())
    return surface_class(t), t.startswith(" "), 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 6 else 3


def verdict(auc_own, auc_control, list_auc_own, list_auc_control, kind, reach=None) -> tuple[bool, bool, str]:
    """(list_pass, label_pass, bin) for one draw. reach only renames a failing bin (panel-misses)."""
    ok = auc_own is not None and auc_control is not None
    label_pass = ok and kind != "none" and auc_own >= AUC_MIN and auc_own - auc_control >= AUC_GAP
    lok = list_auc_own is not None and list_auc_control is not None
    list_pass = lok and list_auc_own >= AUC_MIN and list_auc_own - list_auc_control >= AUC_GAP
    b = ("intelligible" if list_pass and label_pass else "list-only" if list_pass else "label-only (check)" if label_pass
         else "panel-misses" if reach is not None and reach < REACH_MIN else "no-generalisation")
    if not (ok and lok):
        b += " (judge failed)"
    elif max(auc_control, list_auc_control) > EASY:          # unrelated labels / lists separate the decoys: exam too easy
        b = "decoys-too-easy (" + b + ")"
    return bool(list_pass), bool(label_pass), b


def margin(r: dict) -> float:
    """distance of a draw's AUCs from the nearest threshold (0 if any is missing)"""
    v = [r.get("auc_own"), r.get("auc_control"), r.get("list_auc_own"), r.get("list_auc_control")]
    if any(x is None for x in v):
        return 0.0
    own, ctl, lown, lctl = v
    return min(abs(own - AUC_MIN), abs(own - ctl - AUC_GAP), abs(lown - AUC_MIN), abs(lown - lctl - AUC_GAP),
               abs(ctl - EASY), abs(lctl - EASY))


def combine_draws(base: dict, drs: list[dict], adaptive: bool) -> dict:
    """draw 1's fields, every draw kept; stable only if the bins agree; best of 3 by clear majority, else uncertain"""
    base.update({k: v for k, v in drs[0].items() if k != "draw"})
    if adaptive and len(drs) == 1:
        base["draws"] = drs; base["stable"] = True; base["second_draw"] = False
    if len(drs) > 1:
        base["second_draw"] = True
        base["draws"] = drs; bs = [x["bin"] for x in drs]
        base["stable"] = len(set(bs)) == 1
        top = max(set(bs), key=bs.count); base["votes"] = {b: bs.count(b) for b in set(bs)}
        if base["stable"]:
            base["bin"] = bs[0]
        elif len(bs) >= 3 and bs.count(top) * 2 > len(bs):
            base["bin"] = top; base["majority"] = f"{bs.count(top)} of {len(bs)}"
        else:
            base["bin"] = "uncertain (" + " / ".join(bs) + ")"
    return base


# ---- reading the sweep -------------------------------------------------------------------------------------
class _Npz:
    """One array of an .npz at a time. Members stored uncompressed (the sweep's are) are memory-mapped, so taking a few
    neurons' rows reads only those rows; compressed members are read whole."""

    def __init__(self, path):
        self.path = Path(path); self.z = zipfile.ZipFile(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.z.close()

    def __getitem__(self, key):
        info = self.z.getinfo(key + ".npy")
        if info.compress_type == zipfile.ZIP_STORED:
            with open(self.path, "rb") as fh:
                fh.seek(info.header_offset); n, m = struct.unpack("<HH", fh.read(30)[26:30])
                fh.seek(info.header_offset + 30 + n + m)
                version = np.lib.format.read_magic(fh)
                read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = read(fh); offset = fh.tell()
            if not dtype.hasobject and int(np.prod(shape)) > 0:
                return np.memmap(self.path, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C", offset=offset)
        with self.z.open(info) as f:
            return np.lib.format.read_array(f, allow_pickle=False)


def _rows(sweep_dir: Path, tag: str, meta: dict, units, board: str, pseudo_cols=None) -> dict:
    """(k, read) -> {top_i, top_v, mean, std} per unit (each array read once, only these units' rows)"""
    rows = {}
    if pseudo_cols is not None:                            # pseudo-neurons: favourites and stats from their full profiles
        for c in meta["contexts"]:
            with _Npz(sweep_dir / f"vocab_sweep_{tag}_c{c['k']}.npz") as f:
                toks_ = np.asarray(f["tokens"])
                for r in c["read"]:
                    G = np.asarray(f[f"{r}_pseudo_g"])[:, pseudo_cols].astype(np.float32)        # [tokens, pseudo]
                    if board != "gate":                                                          # a = SiLU(g) u, one sign
                        G = G / (1 + np.exp(-G)) * np.asarray(f[f"{r}_pseudo_u"])[:, pseudo_cols].astype(np.float32)
                        G = G if board == "apos" else -G
                    top = np.argsort(-G, axis=0)[:meta["K"]]; cols = np.arange(G.shape[1])[None]
                    rows[(c["k"], r)] = {"top_i": toks_[top].T, "top_v": G[top, cols].T, "mean": G.mean(0), "std": G.std(0)}
        return rows
    from modelmapper.sweep import layer_rows
    row = layer_rows(meta)                                 # layer -> array row (a layer-scoped sweep stores some layers)
    ul, ui = np.array([row(l) for l, _ in units]), np.array([i for _, i in units])
    for c in meta["contexts"]:
        with _Npz(sweep_dir / f"vocab_sweep_{tag}_c{c['k']}.npz") as f:
            for r in c["read"]:
                if board == "gate":
                    rows[(c["k"], r)] = {q: np.asarray(f[f"{r}_{q}"][ul, ui]) for q in ("top_i", "top_v", "mean", "std")}
                else:                                      # top-100 by +a (apos) or -a (aneg); values a or -a; stats of a
                    g_ = np.asarray(f[f"{r}_{board}_g"][ul, ui]).astype(np.float32)
                    u_ = np.asarray(f[f"{r}_{board}_u"][ul, ui]).astype(np.float32)
                    sg = 1.0 if board == "apos" else -1.0
                    rows[(c["k"], r)] = {"top_i": np.asarray(f[f"{r}_{board}_i"][ul, ui]), "top_v": sg * (g_ / (1 + np.exp(-g_)) * u_),
                                         "mean": sg * np.asarray(f[f"{r}_amean"][ul, ui]), "std": np.asarray(f[f"{r}_astd"][ul, ui])}
    return rows


def hub_tokens(sweep_dir: Path, tag: str, meta: dict, layers, hub_x: float) -> dict:
    """layer -> token ids that are a favourite of > hub_x x the fair share of that layer's neurons ({} if hub_x is 0)"""
    if hub_x <= 0:
        return {}
    H = np.load(sweep_dir / f"hub_tokens_{tag}.npz"); n_read = H["per_read"].shape[0]; fair = meta["K"] / meta["n_tokens"]
    width = meta.get("intermediate_size", LAYER_WIDTH)
    rows = {int(L): j for j, L in enumerate(H["layers"])} if "layers" in H.files else None
    return {l: set(np.where(H["per_layer"][rows[l] if rows else l] / (width * n_read) > hub_x * fair)[0].tolist()) for l in sorted(set(layers))}


def favourites(sweep_dir, tag: str, neurons=None, board: str = "gate", hub_x: float = 10.0, pseudo: bool = False):
    """-> (specs, units, fav, meta). fav[spec] = {layer, best, rank {A, B}, half {A, B}, n_hub}; half[h][token] = z per reading.
    pseudo=True: every pseudo-neuron of the sweep (names from the c0 file, CHECK rows skipped); neurons is ignored."""
    sweep_dir = Path(sweep_dir)
    meta = json.loads((sweep_dir / f"vocab_sweep_{tag}.json").read_text(encoding="utf-8"))
    if pseudo:
        with _Npz(sweep_dir / f"vocab_sweep_{tag}_c{meta['contexts'][0]['k']}.npz") as f0:
            pn, pl = [str(x) for x in np.asarray(f0["pseudo"])], np.asarray(f0["pseudo_layer"])
        cols = [j for j, nm in enumerate(pn) if not nm.startswith("CHECK")]
        specs = [pn[j] for j in cols]; units = [(int(pl[j]), j) for j in cols]
        rows = _rows(sweep_dir, tag, meta, units, board, pseudo_cols=cols)
    else:
        specs = list(neurons); units = [tuple(map(int, s[1:].split(":"))) for s in specs]
        rows = _rows(sweep_dir, tag, meta, units, board)
    hub = hub_tokens(sweep_dir, tag, meta, [l for l, _ in units], hub_x)
    fav = {}
    for j, ((l, i), spec) in enumerate(zip(units, specs)):
        half = {"A": {}, "B": {}}; best = -1e9
        for c in meta["contexts"]:
            h = "A" if c["k"] % 2 == 0 else "B"
            for r in c["read"]:
                q = rows[(c["k"], r)]; ti, tv = q["top_i"][j], q["top_v"][j].astype(np.float32)
                mu, sd = float(q["mean"][j]), max(float(q["std"][j]), 1e-6); best = max(best, float(tv[0]))
                for a, v in zip(ti, (tv - mu) / sd):
                    if int(a) not in hub.get(l, ()):
                        half[h].setdefault(int(a), []).append(float(v))
        rank = {h: sorted(d, key=lambda a: (-len(d[a]), -np.mean(d[a]))) for h, d in half.items()}
        fav[spec] = {"layer": l, "best": best, "rank": rank, "half": half, "n_hub": len(hub.get(l, ()))}
    return specs, units, fav, meta


def corpus_peaks(readings_dir, units) -> dict | None:
    """(layer, i) -> the neuron's peak gate value on the research corpora; None if any readings file is missing"""
    files = [Path(readings_dir) / f"readings_{c}_full.pt" for c in CORPORA]
    if not all(f.exists() for f in files):
        return None
    import torch
    peak = {}
    for f in files:
        R = torch.load(f, weights_only=False, map_location="cpu")      # the user's own research files
        for l, i in units:
            peak[(l, i)] = max(peak.get((l, i), -1e9), float(R["gate_v"][l, i, 0]))
        del R
    return peak


def stability(rec: dict, sp: list, seed: int = 2026) -> tuple[float, list]:
    """sets rec[s]['stability']; returns (null p95, null) - other neurons' B halves, same layer band (+-3) if any"""
    for s in sp:
        r = rec[s]; r["stability"] = round(len(set(r["_rank"]["A"][:N_STAB]) & set(r["_half"]["B"])) / N_STAB, 3)
    rng = np.random.default_rng(seed); null = []
    for s in sp:
        others = [x for x in sp if x != s and abs(rec[x]["layer"] - rec[s]["layer"]) <= 3] or [x for x in sp if x != s]
        for o in (rng.choice(others, 3) if others else []):
            null.append(len(set(rec[s]["_rank"]["A"][:N_STAB]) & set(rec[o]["_half"]["B"])) / N_STAB)
    null = null or [0.0]
    return float(np.percentile(null, 95)), null


# ---- the test list -----------------------------------------------------------------------------------------
def held_out(r: dict, dec, n_test: int) -> list:
    """the neuron's half-B favourites whose text (case / space aside) was never shown, strongest first"""
    seen = {norm(dec(a)) for a in r["_rank"]["A"][:N_SHOW]}
    return [a for a in r["_rank"]["B"] if norm(dec(a)) not in seen][:n_test]


def decoy_index(rec: dict, sp: list, dec) -> list[dict]:
    """the pool of possible decoys (every pool neuron's top-60 held-out favourites), indexed by the full surface key,
    (class, space), (class,) and () for the relaxed matches"""
    pool = {}
    for o in sp:
        for a in rec[o]["_rank"]["B"][:N_POOL]:
            pool.setdefault(surface_key(dec(a)), []).append((a, o))
    idx = [{}, {}, {}, {}]
    for kk, lst in pool.items():
        for lv in range(4):
            idx[lv].setdefault(kk[:3 - lv] if lv < 3 else (), []).extend(lst)
    return idx


def surface_decoys(r: dict, pos: list, idx: list[dict], banned: set, rng_key, dec) -> tuple[list, dict, list]:
    """one decoy per true token, same surface key (relaxed step by step if none), at most 2 per source neuron.
    -> (decoys, {decoy text: source neuron}, match level per decoy)"""
    seen = {norm(dec(a)) for a in r["_rank"]["A"][:N_SHOW]}; A, B = r["_half"]["A"], r["_half"]["B"]
    g = np.random.default_rng(rng_key); used, per, neg, src, lv = set(), {}, [], {}, []
    for t in pos:
        k = surface_key(dec(t))
        for level in range(4):
            cand = [(a, o) for a, o in idx[level].get(k[:3 - level] if level < 3 else (), [])
                    if o not in banned and a not in used and per.get(o, 0) < 2 and a not in A and a not in B
                    and norm(dec(a)) not in seen]
            if cand:
                a, o = cand[g.integers(len(cand))]; used.add(a); per[o] = per.get(o, 0) + 1; neg.append(a); src[dec(a)] = o
                lv.append(level)
                break
    return neg, src, lv


def test_list(r: dict, pos: list, idx: list[dict], banned: set, si: int, d: int, dec) -> tuple[list, list, dict, list]:
    """draw d's shuffled test list for the neuron at pool position si: (candidates, labels 1/0, decoy sources, levels)"""
    neg, src, lv = surface_decoys(r, pos, idx, banned, [si, d], dec)
    cands = pos + neg
    order = np.random.default_rng([si, 99, d]).permutation(len(cands))
    return [cands[j] for j in order], [1 if j < len(pos) else 0 for j in order], src, lv


def rotations(sp: list, n_ctrl: int) -> tuple[dict, dict]:
    """(controls, distractor neurons) per neuron: fixed rotations of the pool"""
    n = len(sp)
    ctrl = {s: [sp[(k + o) % n] for o in CTRL_OFFSETS[:n_ctrl]] for k, s in enumerate(sp)}
    other = {s: [sp[(k + o) % n] for o in DISTRACTOR_OFFSETS] for k, s in enumerate(sp)}
    return ctrl, other


# ---- scoring one draw --------------------------------------------------------------------------------------
def _floor(passes, floor):
    m = np.mean(passes, 0); return np.where(m < floor, 0.0, m)


def score_draw(cand_text: list[str], lab, own, ctl, lown, lctl, descs: list[dict], floor: float) -> dict:
    """AUCs and descriptive stats of one draw from the judge's ratings. own: per description a list of passes (or None);
    ctl: per control; lown / lctl: the list exam's (description-free) ratings, same shapes."""
    lab = np.array(lab); r = {}; both = bool(lab.sum() and (1 - lab).sum())
    each = [round(auc(_floor(x, floor)[lab == 1], _floor(x, floor)[lab == 0]), 3) if x else None for x in own]
    good = [k for k, a in enumerate(each) if a is not None]
    r["descriptions"] = [{"description": descs[k].get("description"), "kind": descs[k].get("kind"),
                          "confidence": descs[k].get("confidence"), "auc": each[k] if k < len(each) else None} for k in range(len(descs))]
    mid = sorted(good, key=lambda k: each[k])[len(good) // 2] if good else 0         # the median-scoring description is shown
    d_ = descs[mid] if descs else {}; kinds = [x.get("kind") for x in descs]
    r["description"], r["confidence"] = d_.get("description"), d_.get("confidence")
    r["kind"] = ("none" if kinds.count("none") * 2 > len(kinds) else
                 (d_.get("kind") if d_.get("kind") != "none" else next(k for k in kinds if k != "none")))
    ro = [p for k in good for p in own[k]] if good else None
    so = _floor(ro, floor) if ro else None
    r["auc_own"] = round(float(np.mean([each[k] for k in good])), 3) if good and both else None
    cs = [auc(_floor(x, floor)[lab == 1], _floor(x, floor)[lab == 0]) for x in ctl if x] if both else []
    if so is not None and r["auc_own"] is not None:          # permutation p: random relabellings scoring at least this well
        pr = np.random.default_rng(7); perm_auc = [auc(so[l_ == 1], so[l_ == 0]) for l_ in (pr.permutation(lab) for _ in range(2000))]
        r["p_perm"] = round(float((np.sum(np.array(perm_auc) >= r["auc_own"]) + 1) / 2001), 4)
        # coverage (descriptive): how much of the held-out halo the label reaches
        r["coverage"] = round(float(np.mean(so[lab == 1] > np.median(so[lab == 0]))), 3)
        r["coverage_50"] = round(float(np.mean(so[lab == 1] >= 50)), 3)
        r["uncovered"] = [cand_text[j] for j in np.where((lab == 1) & (so <= np.median(so[lab == 0])))[0]]
    r["auc_controls"] = [round(x, 3) for x in cs]; r["auc_control"] = round(float(np.median(cs)), 3) if cs else None
    r["n_test"] = [int(lab.sum()), int((1 - lab).sum())]
    r["ratings_own"], r["ratings_controls"] = own, ctl
    r["list_auc_own"] = round(auc(np.array(lown[0])[lab == 1], np.array(lown[0])[lab == 0]), 3) if lown else None
    lcs = [auc(np.array(x[0])[lab == 1], np.array(x[0])[lab == 0]) for x in lctl if x]
    r["list_auc_controls"] = [round(x, 3) for x in lcs]
    r["list_auc_control"] = round(float(np.median(lcs)), 3) if lcs else None
    r["ratings_list_own"], r["ratings_list_controls"] = lown, lctl
    return r


# ---- the run -----------------------------------------------------------------------------------------------
def load_decoder(model: str = "Qwen/Qwen3-8B"):
    """token id -> text, memoised (the model's tokenizer only; local copy in MODEL_MAPPER_MODELS if there)"""
    from transformers import AutoTokenizer

    from modelmapper.settings import model_path
    tok = AutoTokenizer.from_pretrained(model_path(model))
    cached = functools.lru_cache(maxsize=None)(lambda j: tok.decode([j]))
    return lambda j: cached(int(j))


def _daemon(fn) -> Future:
    """run fn on a DAEMON thread (an executor's threads would hold up the process exit for a losing duplicate request)"""
    f = Future()

    def go():
        try:
            f.set_result(fn())
        except Exception as e:  # noqa: BLE001
            f.set_exception(e)
    threading.Thread(target=go, daemon=True).start()
    return f


def _tag(path) -> str:
    stem = Path(path).stem
    return stem[len("intelligibility_"):] if stem.startswith("intelligibility_") else str(path)


def _reused(path, need) -> dict:
    old = json.loads(Path(path).read_text(encoding="utf-8")); t = _tag(path)
    return {s: ([dict(x, reused_from=t) for x in old[s]["descriptions"]] if old[s].get("descriptions") else
                [{"description": old[s]["description"], "kind": old[s]["kind"], "confidence": old[s]["confidence"], "reused_from": t}])
            for s in need if s in old and old[s].get("description")}


def _describe_record(ds: list[dict]) -> dict:
    d0 = next((x for x in ds if x.get("description")), ds[0] if ds else {})
    return {"descriptions": ds, "description": d0.get("description"), "kind": d0.get("kind"), "confidence": d0.get("confidence")}


def run_exam(sweep_dir, sweep_tag: str, neurons, out, *, focus=None, pseudo: bool = False, reuse_desc=None,
             describe_only: bool = False, config: ExamConfig | None = None, readings_dir=None, decode=None,
             tokenizer: str = "Qwen/Qwen3-8B", labeller=None, judge=None, log=print) -> dict:
    """Run the exam and write `out` (intelligibility_<tag>.json) and its .txt. Returns the written dict.

    neurons      the pool, e.g. ["L0:771", ...] (controls and decoys come from it); ignored if pseudo
    focus        judge only these (the rest of the pool supplies controls and decoys); default: the whole pool
    pseudo       the sweep's pseudo-neurons are the pool (focus: which of them to judge)
    reuse_desc   an earlier output or describe-only file whose descriptions are reused (no labeller call for those)
    describe_only  describe the focus neurons and their controls, write a describe-only file, no judging
    readings_dir   folder with readings_{web,questions,wiki}_full.pt for reach (default: the sweep folder; skipped if absent)
    decode / labeller / judge   injectable (tests); defaults: the tokenizer, modelmapper.labeller, modelmapper.judge
    """
    cfg = config or ExamConfig(); sweep_dir, out = Path(sweep_dir), Path(out)
    if labeller is None:
        from modelmapper import labeller
    if judge is None:
        from modelmapper import judge
    dec = decode or load_decoder(tokenizer)
    u0 = dict(labeller.usage); j0 = judge.cost(); t0 = time.time()
    luna = lambda: ((labeller.usage["input_tokens"] - u0.get("input_tokens", 0)) * LUNA_USD_IN
                    + (labeller.usage["output_tokens"] - u0.get("output_tokens", 0)) * LUNA_USD_OUT)
    luna_calls = lambda: labeller.usage.get("calls", 0) - u0.get("calls", 0)
    sp, units, fav, meta = favourites(sweep_dir, sweep_tag, neurons, cfg.board, cfg.hub_x, pseudo)
    peak = {u: 0.0 for u in units} if pseudo else (None if describe_only else corpus_peaks(readings_dir or sweep_dir, units))
    if peak is None and not describe_only:
        log("reach: readings_{web,questions,wiki}_full.pt not found; reach is None (no panel-misses bin)")
    rec = {}
    for u, s in zip(units, sp):
        f = fav[s]; pk = peak.get(u) if peak else None
        rec[s] = {"layer": f["layer"], "reach": round(f["best"] / pk, 3) if pk is not None and pk > 0 else None, "corpus_peak": pk,
                  "_rank": f["rank"], "_half": f["half"], "n_hub_dropped": f["n_hub"]}
    stab_thr, null = stability(rec, sp)
    sidx = {s: k for k, s in enumerate(sp)}
    focus = list(focus or []); fset = set(focus)
    if fset - set(sp):
        log(f"focus: {len(fset - set(sp))} not in the pool, ignored (e.g. {sorted(fset - set(sp))[:3]})")
    judged = [s for s in sp if not fset or s in fset]
    ctrl, other = rotations(sp, cfg.n_ctrl)
    need = sorted(set(judged) | {c for s in judged for c in ctrl[s]}, key=sidx.get)
    shown_of = lambda x: [visible(dec(a)) for a in rec[x]["_rank"]["A"][:N_SHOW]]

    # ---- describe (labeller) ----
    rkw = cfg.reasoning_kw()

    def prompt(s):
        return "\n".join(f"{visible(dec(a))}  (z {np.mean(rec[s]['_half']['A'][a]):.1f})" for a in rec[s]["_rank"]["A"][:N_SHOW])

    def describe_one(s):
        try:
            return labeller.chat_json(prompt(s), SYS_DESCRIBE, **rkw)
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def describe(s):
        if luna() > cfg.budget_usd:                       # budget stop: no new request
            return None
        k = cfg.ctrl_ndesc if cfg.ctrl_ndesc and fset and s not in fset else cfg.n_desc
        if k <= 1:
            return [describe_one(s) for _ in range(k)]
        try:
            if cfg.hedge_s > 0:
                data = hedged_request(s, prompt(s), k); labeller.tally(data); got = labeller.parse_n(data, k)
            else:
                got = labeller.chat_json_n(prompt(s), SYS_DESCRIBE, n=k, **rkw)
        except Exception:  # noqa: BLE001
            got = []
        return got + [describe_one(s) for _ in range(k - len(got))]          # top up if the reply had fewer choices

    hedged = []                                            # duplicated describe requests (hedge_s > 0)

    def hedged_request(s, p, k):
        """the request; if it has not answered after hedge_s, a duplicate, and the first good answer is used. Both run on
        daemon threads, so a losing request still running never holds up the process exit."""
        req = lambda: labeller.chat_n_raw(p, SYS_DESCRIBE, n=k, **rkw)
        t_ = time.time(); fp = _daemon(lambda: (req(), time.time() - t_))
        done_, _ = futures_wait([fp], timeout=cfg.hedge_s)
        if done_:
            return fp.result()[0]
        th = time.time(); fh = _daemon(lambda: (req(), time.time() - th))
        done_, _ = futures_wait([fp, fh], return_when=FIRST_COMPLETED)
        win = fp if fp in done_ and fp.exception() is None else fh if fh in done_ and fh.exception() is None else None
        if win is None:                                    # the first one failed: take the other
            futures_wait([fp, fh]); win = fp if fp.exception() is None else fh
        data = win.result()[0]
        hedged.append({"s": s, "fp": fp, "fh": fh, "hedge_won": win is fh, "hedge_after": cfg.hedge_s, "k": k})
        return data

    descs = _reused(reuse_desc, need) if reuse_desc else {}
    todo = [s for s in need if s not in descs]
    with ThreadPoolExecutor(cfg.workers) as ex:
        descs.update(dict(zip(todo, ex.map(describe, todo))))
    missing = [s for s in need if descs.get(s) is None]
    log(f"described {len(todo) - len(missing)} neurons (reused {len(need) - len(todo)}), ${luna():.3f}, {time.time() - t0:.0f} s")
    if cfg.hedge_s > 0:                                    # duplicated requests: winners always; with hedge_keep also wait for
        hrec = {}                                          # the losers (times + both answers)
        for h in hedged:
            if cfg.hedge_keep:
                futures_wait([h["fp"], h["fh"]])
            r = {"hedge_after": h["hedge_after"], "hedge_won": h["hedge_won"]}
            for nm in ("fp", "fh"):
                f = h[nm]
                if not f.done():
                    r[nm + "_unfinished"] = True           # loser still running: its tokens are not counted
                elif f.exception() is None:
                    d_, sec = f.result(); r[nm + "_s"] = round(sec, 2); r[nm + "_desc"] = labeller.parse_n(d_, h["k"])
                    if (nm == "fh") != h["hedge_won"]:
                        labeller.tally(d_)                 # the losing request was paid for too
                else:
                    r[nm + "_error"] = repr(f.exception())
            hrec[h["s"]] = r
        hfile = out.with_name(out.stem + "_hedge.json")
        hfile.parent.mkdir(parents=True, exist_ok=True)
        hfile.write_text(json.dumps(hrec, ensure_ascii=False), encoding="utf-8")
        log(f"hedge: {len(hrec)} of {len(todo)} describe requests duplicated after {cfg.hedge_s} s; the duplicate answered first "
            f"for {sum(r['hedge_won'] for r in hrec.values())}; cost incl. duplicates ${luna():.3f}, {time.time() - t0:.0f} s")
    if missing or luna() > cfg.budget_usd:
        part = out.with_name(out.stem + ".partial.json")
        got = {s: _describe_record(descs[s]) for s in need if descs.get(s) is not None}
        _write(part, {"_summary": {"describe_only": True, "budget_stopped": True, "board": cfg.board, "n": len(got),
                                   "cost_usd": round(luna(), 3), "luna_calls": luna_calls(), "seconds": round(time.time() - t0)}, **got},
               indent=None)
        raise BudgetExceeded(f"labeller budget ${cfg.budget_usd} reached (${luna():.2f}) before judging; {len(missing)} neurons "
                             f"not described. Descriptions so far: {part} (pass it as reuse_desc to continue)")
    if describe_only:
        res = {"_summary": {"describe_only": True, "board": cfg.board, "n": len(need), "cost_usd": round(luna(), 3),
                            "luna_calls": luna_calls(), "seconds": round(time.time() - t0)}}
        res.update({s: _describe_record(descs[s]) for s in need})
        _write(out, res, indent=None)
        log(json.dumps(res["_summary"]))
        return res

    # ---- judge (merged calls, each (judge state, token) asked once per run) ----
    cache, qcount, lock, errors = {}, [0], threading.Lock(), [0]

    def judge_desc(text, cands):
        try:
            return [judge.rate(text, [dec(a) for a in cands])]
        except Exception:  # noqa: BLE001
            errors[0] += 1; return None

    def judge_list(shown, cands):
        try:
            return [list(judge.rate_examples(shown, [dec(a) for a in cands]))]
        except Exception:  # noqa: BLE001
            errors[0] += 1; return None

    def states(s):
        """(key, fn) per judge state of neuron s: own descriptions, control descriptions, own list, control lists"""
        txt = lambda d: d.get("description", "")
        own = [(("d", txt(d)), functools.partial(judge_desc, txt(d))) for d in descs[s][:cfg.n_desc]]
        ctl = [(("d", txt((descs[c] or [{}])[0])), functools.partial(judge_desc, txt((descs[c] or [{}])[0]))) for c in ctrl[s]]
        lst = [(("l", tuple(shown_of(x))), functools.partial(judge_list, shown_of(x))) for x in [s] + ctrl[s]]
        return own, ctl, lst

    def cached(key, fn, cands):
        with lock:
            miss = [a for a in dict.fromkeys(cands) if (key, a) not in cache]
        if miss:
            res = fn(miss)
            if not res:
                return None
            with lock:
                for a, v in zip(miss, res[0]):
                    cache[(key, a)] = v
                qcount[0] += len(miss)
        return [[cache[(key, a)] for a in cands]]

    def prefetch(jobs):
        need_ = {}
        for s, cands, _, _ in jobs:
            own, ctl, lst = states(s)
            for key, fn in own + ctl + lst:
                need_.setdefault(key, (fn, []))[1].extend(cands)
        calls = []
        for key, (fn, toks) in need_.items():
            miss = [a for a in dict.fromkeys(toks) if (key, a) not in cache]
            calls += [(key, fn, miss[i:i + cfg.merge]) for i in range(0, len(miss), cfg.merge)]

        def one(c):
            key, fn, miss = c
            res = fn(miss)
            if res:
                with lock:
                    for a, v in zip(miss, res[0]):
                        cache[(key, a)] = v
                    qcount[0] += len(miss)
        with ThreadPoolExecutor(cfg.workers) as ex:
            list(ex.map(one, calls))
        log(f"prefetch: {len(calls)} merged calls for {len(need_)} judge states")

    pos_of = {s: held_out(rec[s], dec, cfg.n_test) for s in judged}
    idx = decoy_index(rec, sp, dec); dsrc = {}

    def make_job(s, d):
        cands, lab, src, lv = test_list(rec[s], pos_of[s], idx, {s, *ctrl[s]}, sidx[s], d, dec); dsrc[(s, d)] = (src, lv)
        return s, cands, lab, d

    lines, per_draw = [], {}

    def run_jobs(jobs):
        if not jobs:
            return
        prefetch(jobs)

        def ratings(job):
            own, ctl, lst = states(job[0])
            o = [cached(k, fn, job[1]) for k, fn in own] + [None] * (cfg.n_desc - len(own))
            return o, [cached(k, fn, job[1]) for k, fn in ctl], cached(*lst[0], job[1]), [cached(k, fn, job[1]) for k, fn in lst[1:]]
        with ThreadPoolExecutor(cfg.workers) as ex:
            got = list(ex.map(ratings, jobs))
        for (s, cands, lab, dn), (ro, rc, lo, lc) in zip(jobs, got):
            ctext = [dec(a) for a in cands]
            r = {"layer": rec[s]["layer"], "reach": rec[s]["reach"], "stability": rec[s]["stability"], "draw": dn,
                 "test": {"candidates": ctext, "label": list(lab)}}
            per_draw.setdefault(s, []).append(r)
            r.update(score_draw(ctext, lab, ro, rc, lo, lc, descs[s], cfg.floor))
            r["control_neurons"], r["distractor_neurons"] = ctrl[s], other[s]
            r["control_descriptions"] = {c: (descs[c] or [{}])[0].get("description") for c in ctrl[s]}
            r["decoy_sources"], lv = dsrc[(s, dn)]; r["decoy_match_levels"] = {str(k): lv.count(k) for k in range(4)}
            r["list_pass"], r["label_pass"], r["bin"] = verdict(r["auc_own"], r["auc_control"], r["list_auc_own"],
                                                               r["list_auc_control"], r["kind"], r["reach"])
            r["shown"] = shown_of(s)
            lines.append(f"{s:>10} L{r['layer']:<2} {r['bin']:<26} AUC {r['auc_own']} (ctrl {r['auc_control']}) list {r.get('list_auc_own')} "
                         f"(ctrl {r.get('list_auc_control')}) stab {r['stability']} reach {r['reach']} "
                         f"[{r['kind']}] {str(r['description'])[:110]} | shown: {' '.join(r['shown'][:8])}")

    adaptive = cfg.adaptive_delta > 0
    run_jobs([make_job(s, d) for s in judged for d in range(1 if adaptive else cfg.draws)])
    n_second, n_third = 0, None
    if adaptive and cfg.draws > 1:                         # draw 2 only near a threshold; draw 3+ only where draws 1 and 2 disagree
        near = [s for s in judged if margin(per_draw[s][0]) <= cfg.adaptive_delta]; n_second = len(near)
        for s in judged:
            per_draw[s][0]["draw1_margin"] = round(margin(per_draw[s][0]), 3)
        log(f"second draw for {len(near)} of {len(judged)} neurons (within {cfg.adaptive_delta} of a threshold)")
        run_jobs([make_job(s, 1) for s in near])
        if cfg.draws > 2:
            third = [s for s in near if per_draw[s][0]["bin"] != per_draw[s][1]["bin"]]; n_third = len(third)
            log(f"third draw for {len(third)} of {len(near)} (draws 1 and 2 disagree)")
            run_jobs([make_job(s, d) for s in third for d in range(2, cfg.draws)])
    for s, drs in per_draw.items():
        combine_draws(rec[s], drs, adaptive)
    if errors[0]:
        log(f"judge: {errors[0]} failed calls (those neurons are marked '(judge failed)')")
    for s in judged:                                       # working data dropped only now
        for k in ("_rank", "_half"):
            rec[s].pop(k)
    rec = {s: rec[s] for s in judged}
    bins, by_layer = {}, {}
    for s in judged:
        bins.setdefault(rec[s]["bin"], []).append(s)
        b = "early 0-11" if rec[s]["layer"] < 12 else ("middle 12-23" if rec[s]["layer"] < 24 else "late 24-35")
        by_layer.setdefault(b, {}).setdefault(rec[s]["bin"], 0); by_layer[b][rec[s]["bin"]] += 1
    rk = rkw.get("reasoning_effort")
    exam = {"board": cfg.board, "n_test": cfg.n_test, "n_ctrl": cfg.n_ctrl, "n_judge": 1, "floor": cfg.floor,
            "reuse_desc": _tag(reuse_desc) if reuse_desc else "", "focus": focus, "hub_x": cfg.hub_x, "match_kind": "",
            "ctrl_stat": "median", "judge": "jev", "paired": True, "n_desc": cfg.n_desc, "decoys": "surface", "draws": cfg.draws,
            "adaptive_delta": cfg.adaptive_delta, "pseudo": pseudo, **({"ctrl_ndesc": cfg.ctrl_ndesc} if cfg.ctrl_ndesc else {}),
            "seq3": True, "desc_n": True, **({"reasoning": rk} if rk else {}), **({"merge": cfg.merge} if cfg.merge else {})}
    ctl_aucs = [rec[s]["auc_control"] for s in judged if rec[s]["auc_control"] is not None]
    summary = {"exam": exam, "n": len(judged), "stability_null_p95": round(stab_thr, 3), "stability_null_median": round(float(np.median(null)), 3),
               "bins": {b: len(v) for b, v in bins.items()}, "by_layer": by_layer,
               "auc_control_median": round(float(np.median(ctl_aucs)), 3) if ctl_aucs else None,
               "cost_usd": round(luna(), 3), "jev_usd": round(judge.cost() - j0, 4), "questions": qcount[0], "second_draw_neurons": n_second,
               **({"third_draw_neurons": n_third} if n_third is not None else {}), "luna_calls": luna_calls(),
               "seconds": round(time.time() - t0)}
    res = {"_summary": summary, **rec}
    _write(out, res, indent=1)
    out.with_suffix(".txt").write_text(json.dumps(summary, ensure_ascii=False) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    log(json.dumps({k: v for k, v in summary.items() if k != "exam"}, ensure_ascii=False))
    return res


def _write(path: Path, obj: dict, indent):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def merge_parts(out, parts) -> dict:
    """Merge the outputs of a batched exam (disjoint focus slices of the same pool, and optionally the describe-only part)
    into one file, as if one process had judged them all: records unioned; summary = the first batch's settings, summed
    questions / cost, max seconds; a describe-only part adds its cost, labeller calls and seconds."""
    recs, summ, desc = {}, None, None
    for t in parts:
        d = json.loads(Path(t).read_text(encoding="utf-8")); s = d.pop("_summary")
        if s.get("describe_only"):
            desc = (_tag(t), s); continue
        dup = set(recs) & set(d)
        if dup:
            raise ValueError(f"batches overlap: {sorted(dup)[:5]}")
        recs.update(d)
        if summ is None:
            summ = dict(s); summ["batches"] = [_tag(t)]
        else:
            summ["batches"].append(_tag(t))
            for k in ("questions", "jev_usd", "cost_usd"):
                summ[k] = round(summ.get(k, 0) + s.get(k, 0), 4)
            summ["seconds"] = max(summ.get("seconds", 0), s.get("seconds", 0))
    if summ is None:
        raise ValueError("no judged batch among the parts")
    summ["n"] = len(recs)
    if desc:
        summ["describe_part"] = desc[0]; summ["luna_calls"] = (summ.get("luna_calls") or 0) + (desc[1].get("luna_calls") or 0)
        summ["cost_usd"] = round(summ.get("cost_usd", 0) + desc[1].get("cost_usd", 0), 4); summ["seconds"] += desc[1].get("seconds", 0)
    res = {"_summary": summ, **recs}
    _write(Path(out), res, indent=1)
    return res


# ---- fixed exams: judged again without the sweep (`modelmapper check --labels`, step exam1) ----------------
def fixed_inputs(sweep_dir, sweep_tag: str, neurons, judged, exam_file, *, config: ExamConfig | None = None, draws: int = 3,
                 decode=None, tokenizer: str = "Qwen/Qwen3-8B") -> dict:
    """What re-judging `judged` needs, taken from a finished exam (exam_file, run on this sweep with this pool and
    config): per neuron its descriptions, its controls and their descriptions, every draw's test list (draws 1 to
    `draws`, whether or not the exam ran them) and the exam's verdict, plus the shown lists of these neurons and of
    their controls. Small (no sweep arrays); rejudge() runs from it. Raises if a test list or control differs from
    the exam's own, so the extract is the exam's inputs exactly."""
    cfg = config or ExamConfig(); dec = decode or load_decoder(tokenizer)
    ex = json.loads(Path(exam_file).read_text(encoding="utf-8"))
    sp, units, fav, meta = favourites(sweep_dir, sweep_tag, neurons, cfg.board, cfg.hub_x)
    rec = {s: {"layer": fav[s]["layer"], "_rank": fav[s]["rank"], "_half": fav[s]["half"]} for s in sp}
    sidx = {s: k for k, s in enumerate(sp)}; ctrl, _ = rotations(sp, cfg.n_ctrl); idx = decoy_index(rec, sp, dec)
    out = {"board": cfg.board, "settings": {k: getattr(cfg, k) for k in ("n_test", "n_ctrl", "n_desc", "floor", "hub_x", "draws",
                                                                          "adaptive_delta")},
           "pool_size": len(sp), "neurons": {}, "shown": {}}
    for s in judged:
        r = ex[s]
        if r["control_neurons"] != ctrl[s]:
            raise ValueError(f"{s}: controls {ctrl[s]} differ from the exam's {r['control_neurons']} (another pool?)")
        pos = held_out(rec[s], dec, cfg.n_test); dl = []
        for d in range(draws):
            cands, lab, src, _ = test_list(rec[s], pos, idx, {s, *ctrl[s]}, sidx[s], d, dec)
            dl.append({"candidates": [dec(a) for a in cands], "label": lab})
            if d < len(r["draws"]) and (r["draws"][d]["test"] != dl[-1] or r["draws"][d]["decoy_sources"] != src):
                raise ValueError(f"{s}: draw {d + 1}'s test list differs from the exam's")
        out["neurons"][s] = {"layer": r["layer"], "reach": r.get("reach"), "controls": ctrl[s],
                             "descriptions": [{k: x.get(k) for k in ("description", "kind", "confidence")} for x in r["descriptions"]],
                             "control_descriptions": r["control_descriptions"], "draws": dl,
                             "ours": {k: r.get(k) for k in ("bin", "label_pass", "list_pass", "auc_own", "auc_control", "list_auc_own",
                                                            "list_auc_control", "kind")} | {"n_draws": len(r["draws"])}}
        for x in [s] + ctrl[s]:
            out["shown"][x] = [visible(dec(a)) for a in rec[x]["_rank"]["A"][:N_SHOW]]
    return out


def rejudge(fixed: dict, *, judge=None, config: ExamConfig | None = None, log=print) -> dict:
    """Judge fixed exams again (fixed_inputs' extract): the same judge states, merged calls, scoring, verdicts and
    adaptive draws as run_exam, with every draw's test list taken from the extract instead of the sweep. The labeller
    is never called. Returns {"_summary": {...}, neuron: record} with run_exam's per-neuron fields."""
    cfg = config or ExamConfig(board=fixed.get("board", "gate"), **{k: v for k, v in fixed.get("settings", {}).items()
                                                                     if k in ("n_test", "n_ctrl", "n_desc", "floor", "draws", "adaptive_delta")})
    if judge is None:
        from modelmapper import judge
    N, shown = fixed["neurons"], fixed["shown"]; judged = list(N)
    j0, t0 = judge.cost(), time.time()
    cache, qcount, lock, errors = {}, [0], threading.Lock(), [0]

    def judge_desc(text, cands):
        try:
            return [judge.rate(text, list(cands))]
        except Exception:  # noqa: BLE001
            errors[0] += 1; return None

    def judge_list(sh, cands):
        try:
            return [list(judge.rate_examples(sh, list(cands)))]
        except Exception:  # noqa: BLE001
            errors[0] += 1; return None

    def states(s):
        txt = lambda d: d.get("description") or ""
        own = [(("d", txt(d)), functools.partial(judge_desc, txt(d))) for d in N[s]["descriptions"][:cfg.n_desc]]
        cd = [N[s]["control_descriptions"].get(c) or "" for c in N[s]["controls"]]
        ctl = [(("d", t), functools.partial(judge_desc, t)) for t in cd]
        lst = [(("l", tuple(shown[x])), functools.partial(judge_list, shown[x])) for x in [s] + N[s]["controls"]]
        return own, ctl, lst

    def cached(key, fn, cands):
        with lock:
            miss = [a for a in dict.fromkeys(cands) if (key, a) not in cache]
        if miss:
            res = fn(miss)
            if not res:
                return None
            with lock:
                for a, v in zip(miss, res[0]):
                    cache[(key, a)] = v
                qcount[0] += len(miss)
        return [[cache[(key, a)] for a in cands]]

    def prefetch(jobs):
        need = {}
        for s, cands, _, _ in jobs:
            own, ctl, lst = states(s)
            for key, fn in own + ctl + lst:
                need.setdefault(key, (fn, []))[1].extend(cands)
        calls = []
        for key, (fn, toks) in need.items():
            miss = [a for a in dict.fromkeys(toks) if (key, a) not in cache]
            calls += [(key, fn, miss[i:i + cfg.merge]) for i in range(0, len(miss), cfg.merge)]

        def one(c):
            key, fn, miss = c
            res = fn(miss)
            if res:
                with lock:
                    for a, v in zip(miss, res[0]):
                        cache[(key, a)] = v
                    qcount[0] += len(miss)
        with ThreadPoolExecutor(cfg.workers) as ex:
            list(ex.map(one, calls))

    def job(s, d):
        if d >= len(N[s]["draws"]):
            raise ValueError(f"{s}: the extract has no test list for draw {d + 1}")
        t = N[s]["draws"][d]
        return s, t["candidates"], t["label"], d

    per_draw = {}

    def run_jobs(jobs):
        if not jobs:
            return
        prefetch(jobs)

        def ratings(jb):
            own, ctl, lst = states(jb[0])
            o = [cached(k, fn, jb[1]) for k, fn in own] + [None] * (cfg.n_desc - len(own))
            return o, [cached(k, fn, jb[1]) for k, fn in ctl], cached(*lst[0], jb[1]), [cached(k, fn, jb[1]) for k, fn in lst[1:]]
        with ThreadPoolExecutor(cfg.workers) as ex:
            got = list(ex.map(ratings, jobs))
        for (s, cands, lab, dn), (ro, rc, lo, lc) in zip(jobs, got):
            r = {"layer": N[s]["layer"], "reach": N[s].get("reach"), "draw": dn, "test": {"candidates": list(cands), "label": list(lab)}}
            r.update(score_draw(list(cands), lab, ro, rc, lo, lc, N[s]["descriptions"], cfg.floor))
            r["control_neurons"] = N[s]["controls"]
            r["list_pass"], r["label_pass"], r["bin"] = verdict(r["auc_own"], r["auc_control"], r["list_auc_own"],
                                                               r["list_auc_control"], r["kind"], r["reach"])
            per_draw.setdefault(s, []).append(r)

    adaptive = cfg.adaptive_delta > 0
    run_jobs([job(s, d) for s in judged for d in range(1 if adaptive else cfg.draws)])
    if adaptive and cfg.draws > 1:
        near = [s for s in judged if margin(per_draw[s][0]) <= cfg.adaptive_delta]
        run_jobs([job(s, 1) for s in near])
        if cfg.draws > 2:
            run_jobs([job(s, d) for s in near if per_draw[s][0]["bin"] != per_draw[s][1]["bin"] for d in range(2, cfg.draws)])
    rec = {s: combine_draws({"layer": N[s]["layer"], "reach": N[s].get("reach")}, per_draw[s], adaptive) for s in judged}
    bins = {}
    for s in judged:
        bins[rec[s]["bin"]] = bins.get(rec[s]["bin"], 0) + 1
    if errors[0]:
        log(f"judge: {errors[0]} failed calls (those neurons are marked '(judge failed)')")
    summary = {"n": len(judged), "bins": bins, "jev_usd": round(judge.cost() - j0, 4), "questions": qcount[0],
               "judge_errors": errors[0], "seconds": round(time.time() - t0)}
    return {"_summary": summary, **rec}
