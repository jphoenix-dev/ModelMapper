"""The generative check (v3): does a verified label predict what really switches the neuron on?

Judge-free on the neuron side. For each exam-verified label (exam bin "intelligible"):
  1. surface_labels: Luna sees the tokens the describer saw (the exam record's "shown", top 40) and describes ONLY their
     surface form (script, leading space, case, length, digits, punctuation), not their meaning. This is the control.
  2. generate_words: Luna writes 30 new words from the label alone, and 30 from the surface-only description of the
     SAME neuron. No model needed: run it before taking the GPU (the research runner did, with 128 requests in flight).
  3. score_words: each word becomes one token (its leading-space form if that is one token, else the bare form if that
     is, else the first piece; the single-token share is recorded), is put into the slot of the 12 panel sentences
     (PANEL_V1) and read at the slot and, where the sentence goes on, at the end (22 readings). The unit's value there
     becomes a z-score against the vocabulary sweep's statistics for that reading; a token's score is its mean z.
     Score = AUC of the label's tokens against the surface tokens. Pass = AUC >= 0.75.
Boards: gate (the gate_proj output; z over the sweep's {r}_mean / {r}_std), and apos / aneg, one sign of the activation
a = SiLU(gate) * up: z = sign * (a - mean a) / sd a over the sweep's {r}_amean / {r}_astd (sign -1 for aneg), the
statistics the side exam ranked its favourites by. Pseudo-neurons (pseudo.py): the unit is v . gate row (and v . up
row), v = the MLP input (post_attention_layernorm output); their statistics come from the sweep's full pseudo profiles.

    ver = verified_subset("intelligibility_v3_gate.json", "intelligibility_v3_gate_ver.json")
    surface_labels(ver, "graded_surface_v3_gate_ver.json")
    generate_words(ver, "graded_surface_v3_gate_ver.json", "gen_words_v3_gate_surf.json")        # Luna, no model
    score_words(model, tok, "gen_words_v3_gate_surf.json", sweep_dir, "v2", "gen_check_v3_gate_surf.json", board="gate")
or run_gencheck(...), which chains them and skips any step whose output file is already there.

Output gen_check_<tag>.json (+ a .txt digest) in the research run's format: "_summary" (pass_by_bin, median_auc_by_bin,
...) and per label bin, layer, label, words, tokens, single_token_share, z_own, z_ctrl, control_tokens, auc, pass,
median_z_own, share_z_over_2. Sweep files: vocab_sweep_<tag>.json (its "contexts": k and readings) and
vocab_sweep_<tag>_c<k>.npz per context.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from modelmapper import labeller as oai

N_GEN, AUC_PASS = 30, 0.75
BOARDS = ("gate", "apos", "aneg")
WORD_WORKERS, SURFACE_WORKERS = 128, 16           # Luna requests in flight (the research runner's settings)
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6            # Luna's price per token (= questions.PRICE_IN / PRICE_OUT)
SURFACE_CONTROL = "surface-only label (same neuron)"

from modelmapper.sweep import PANEL_V1, SLOT  # noqa: E402  (the vocabulary sweep's panel: one definition)

SYS_GEN = ("A unit (neuron) inside a language model is described below: the description says which tokens (words or word pieces) "
           "switch it on. Write 30 NEW single words or short word pieces, in any language, that should switch it on most strongly "
           "according to the description. Prefer single common words. Reply as JSON: {\"tokens\": [...]}")
SYS_SURF = ("Below are tokens (words or word pieces) that switch on a unit in a language model; whitespace is shown as ␣ ⏎ ⇥. "
            "Describe ONLY their surface form in one or two sentences: writing system / script, leading space, capitalisation, "
            "length, digits, punctuation or code symbols. Do NOT mention meaning, topic, word class or what the words refer to. "
            "Reply as JSON: {\"description\": \"...\"}")


# ---------------------------------------------------------------- files
def _load(x) -> tuple[dict, dict, str]:
    """A json file path or the dict itself -> (records without "_summary", the summary, the file's stem or "")."""
    if isinstance(x, (str, Path)):
        p = Path(x)
        d, name = json.loads(p.read_text(encoding="utf-8")), p.stem
    else:
        d, name = x, ""
    d = dict(d)
    return d, d.pop("_summary", None) or {}, name


def _tag(name: str, prefix: str) -> str:
    return name[len(prefix):] if name.startswith(prefix) else name


def _write(path, obj, indent=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    return path


def _done(path, key="_summary"):
    """The parsed file if it exists and is complete (every output is written once, at the end of its step)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if key in d else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- Luna
class _Spend:
    def __init__(self):
        self.lock = threading.Lock()
        self.tokens = [0, 0]

    def add(self, usage: dict):
        with self.lock:
            self.tokens[0] += usage.get("prompt_tokens", 0)
            self.tokens[1] += usage.get("completion_tokens", 0)

    def usd(self) -> float:
        return round(self.tokens[0] * PRICE_IN + self.tokens[1] * PRICE_OUT, 4)


def _ask(prompt: str, system: str) -> tuple[dict, dict]:
    """One Luna call in JSON mode -> (reply, usage). Usage per call, so parallel calls don't race on a shared counter."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    data = oai._request("POST", "/chat/completions", {"model": oai.MODEL, "messages": msgs, "response_format": {"type": "json_object"}})
    return json.loads(data["choices"][0]["message"]["content"]), data.get("usage") or {}


def _ask_many(prompts: dict, system: str, workers: int, spend: _Spend) -> dict:
    """{key: prompt} -> {key: reply dict, or None if the call failed}."""
    def one(k):
        try:
            reply, usage = _ask(prompts[k], system)
        except Exception:  # noqa: BLE001  (a failed call gives no words for that neuron, as in the research run)
            return None
        spend.add(usage)
        return reply if isinstance(reply, dict) else None
    keys = list(prompts)
    if not keys:
        return {}
    with ThreadPoolExecutor(max(1, min(workers, len(keys)))) as ex:
        return dict(zip(keys, ex.map(one, keys)))


def parse_words(reply, n: int = N_GEN) -> list[str]:
    """Luna's {"tokens": [...]} -> up to n words (strings); anything else -> []."""
    toks = reply.get("tokens") if isinstance(reply, dict) else None
    return [str(x) for x in toks][:n] if isinstance(toks, list) else []


# ---------------------------------------------------------------- steps 1-2: verified subset, surface labels, words
def verified_subset(exam, out_path=None) -> dict:
    """Keep the exam-verified labels (bin exactly "intelligible", as in the v3 / v3x runs: labels verified at the edge,
    "decoys-too-easy (intelligible)", get no generative check; an "uncertain (...)" bin is not verified). The joint
    v3 verdict needs exam AND generative check, so the check runs only where the exam passed. Writes out_path if given."""
    recs, _, name = _load(exam)
    ver = {k: r for k, r in recs.items() if r.get("bin") == "intelligible"}
    out = {"_summary": {"subset_of": _tag(name, "intelligibility_"), "n": len(ver)}, **ver}
    if out_path:
        _write(out_path, out)
    return out


def surface_labels(exam, out_path=None, *, workers: int = SURFACE_WORKERS, log=print) -> dict:
    """Surface-only descriptions of the tokens each label was written from (the exam record's "shown", top 40), in the
    shape of research graded_surface_<tag>.json: {neuron: {"surface_label", "bin"}} (no graded ratings)."""
    recs, _, name = _load(exam)
    keys = [k for k, r in recs.items() if r.get("description") and r.get("shown")]
    spend = _Spend()
    replies = _ask_many({k: "Tokens:\n" + "\n".join(recs[k]["shown"][:40]) for k in keys}, SYS_SURF, workers, spend)
    sl = {k: str((replies.get(k) or {}).get("description") or "") for k in keys}
    out = {"_summary": {"tag": _tag(name, "intelligibility_"), "n": len(keys), "empty": sum(not v for v in sl.values()),
                        "note": "surface-only labels only (no graded exam)", "luna_usd": spend.usd()}}
    out.update({k: {"surface_label": v, "bin": recs[k].get("bin")} for k, v in sl.items()})
    if out_path:
        _write(out_path, out, indent=1)
    log(json.dumps(out["_summary"]))
    return out


def generate_words(labels, surface, out_path=None, *, n: int = N_GEN, workers: int = WORD_WORKERS, log=print) -> dict:
    """The Luna step (no model): n words from each label, and n from the same neuron's surface-only label.

    labels: verified exam records (path or dict); surface: surface_labels output (path or dict). Neurons without a
    surface label are left out (they have no control). Saved as gen_words_<tag>.json (the research file, plus
    "records": each neuron's bin, layer and label, so the scoring step needs no other file)."""
    recs, s, name = _load(labels)
    sl, ss, sname = _load(surface)
    ltag = _tag(name, "intelligibility_") if name else (f"{s['subset_of']}_ver" if s.get("subset_of") else "")
    stag = _tag(sname, "graded_surface_") if sname else ss.get("tag", "")
    keys =[k for k, r in recs.items() if r.get("description") and (sl.get(k) or {}).get("surface_label")]
    t0, spend = time.time(), _Spend()
    prompts = {("own", k): f"Description: {recs[k]['description']}" for k in keys}
    prompts.update({("surf", k): f"Description: {sl[k]['surface_label']}" for k in keys})
    replies = _ask_many(prompts, SYS_GEN, workers, spend)
    out = {"labels": ltag, "control": "surface:" + stag,
           "words": {k: parse_words(replies[("own", k)], n) for k in keys},
           "words_s": {k: parse_words(replies[("surf", k)], n) for k in keys},
           "records": {k: {"bin": recs[k].get("bin"), "layer": recs[k].get("layer"), "label": recs[k]["description"]} for k in keys},
           "luna_usd": spend.usd(), "seconds": round(time.time() - t0)}
    log(f"generated words for {len(keys)} neurons in {out['seconds']} s (Luna ${out['luna_usd']})")
    if out_path:
        _write(out_path, out)
    return out


# ---------------------------------------------------------------- step 3: the model
def to_token(tok, word: str):
    """A word -> (token id, single token?): its leading-space form if that is one token, else the bare form if that is,
    else the first piece of the leading-space form. (None, False) if it encodes to nothing."""
    a, b = tok.encode(" " + word.strip()), tok.encode(word.strip())
    if len(a) == 1:
        return a[0], True
    if len(b) == 1:
        return b[0], True
    return (a[0] if a else None), False


def auc(pos, neg) -> float:
    """P(a positive scores above a negative), ties count half."""
    p, n = np.array(pos, float)[:, None], np.array(neg, float)[None]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _units(keys, pseudo):
    """neuron -> (layer, column): 'L5:123' -> (5, 123); a pseudo-neuron name -> (its layer, its row in the set)."""
    if pseudo is None:
        return {k: tuple(map(int, k[1:].split(":"))) for k in keys}
    names = [str(x) for x in pseudo["names"]]
    return {k: (int(pseudo["layer"][names.index(k)]), names.index(k)) for k in keys}


def sweep_stats(sweep_dir, sweep_tag, keys, units, board="gate", pseudo=None) -> tuple[list, dict]:
    """(contexts, {(context k, reading): (mean [n_keys], sd [n_keys])}) from the sweep files. Gate board: gate mean / sd
    over the vocabulary; apos / aneg: activation mean / sd. Pseudo-neurons: from their full profiles in the sweep."""
    d = Path(sweep_dir)
    meta = json.loads((d / f"vocab_sweep_{sweep_tag}.json").read_text(encoding="utf-8"))
    if meta.get("panel", "v1") != "v1":
        raise ValueError(f"sweep {sweep_tag} used panel {meta.get('panel')!r}; the check reads PANEL_V1")
    stats = {}
    for c in meta["contexts"]:
        f = np.load(d / f"vocab_sweep_{sweep_tag}_c{c['k']}.npz")
        for r in c["read"]:
            if pseudo is not None:
                sn = [str(x) for x in f["pseudo"]] if "pseudo" in f.files else None
                cols = [sn.index(k) for k in keys] if sn else [units[k][1] for k in keys]
                G = f[f"{r}_pseudo_g"][:, cols].astype(np.float32)
                if board != "gate":
                    U = f[f"{r}_pseudo_u"][:, cols].astype(np.float32)
                    with np.errstate(over="ignore"):
                        G = G / (1 + np.exp(-G)) * U
                stats[(c["k"], r)] = (G.mean(0), G.std(0))
            else:
                from modelmapper.sweep import layer_rows
                row = layer_rows(meta)
                ul, ui = np.array([row(units[k][0]) for k in keys]), np.array([units[k][1] for k in keys])
                m_, s_ = ("amean", "astd") if board != "gate" else ("mean", "std")
                stats[(c["k"], r)] = (f[f"{r}_{m_}"][ul, ui], f[f"{r}_{s_}"][ul, ui])
    return meta["contexts"], stats


class _Stop(Exception):
    pass


def unit_values(model, tok, units, token_ids, contexts, *, board="gate", pseudo=None, batch=512, log=print) -> dict:
    """Each unit's value with each token in the panel slot: {(context k, reading): float32 [n_units, n_tokens]}.

    units: list of (layer, column), real neurons or rows of the pseudo set. Gate board: the gate value; apos / aneg: the
    activation SiLU(gate) * up (unsigned). Runs on the model's own device, in batches (batching.run_batched), and stops
    each forward pass after the deepest layer needed."""
    import torch
    from modelmapper.batching import run_batched
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    lay = model.model.layers
    by_layer = {}
    for j, (L, _) in enumerate(units):
        by_layer.setdefault(L, []).append(j)
    cols = {L: torch.tensor([units[j][1] for j in js], device=dev) for L, js in by_layer.items()}
    if pseudo is not None:
        rows = {L: (torch.tensor(pseudo["gate"][[units[j][1] for j in js]], device=dev, dtype=dtype),
                    torch.tensor(pseudo["up"][[units[j][1] for j in js]], device=dev, dtype=dtype)) for L, js in by_layer.items()}
    side = board != "gate"
    token_ids = [int(t) for t in token_ids]
    out = {}
    with torch.inference_mode():
        for c in contexts:
            a_, b_ = PANEL_V1[c["k"]].split(SLOT)
            P, S = tok.encode(a_), tok.encode(b_)
            pos = {"slot": len(P), "end": len(P) + len(S)}
            RP = [pos[r] for r in c["read"]]
            cap = {}

            def on_gate(m, a, o, L):
                cap[("g", L)] = o[:, RP][..., cols[L]].float()

            def on_up(m, a, o, L):
                cap[("u", L)] = o[:, RP][..., cols[L]].float()

            def on_norm(m, a, o, L):                     # pseudo-neurons: v . row, v = the MLP input
                v = o[:, RP]
                cap[("g", L)] = (v @ rows[L][0].T).float()
                if side:
                    cap[("u", L)] = (v @ rows[L][1].T).float()

            def stop(m, a, o):
                raise _Stop
            hs = []
            for L in by_layer:
                if pseudo is not None:
                    hs.append(lay[L].post_attention_layernorm.register_forward_hook(lambda m, a, o, L=L: on_norm(m, a, o, L)))
                else:
                    hs.append(lay[L].mlp.gate_proj.register_forward_hook(lambda m, a, o, L=L: on_gate(m, a, o, L)))
                    if side:
                        hs.append(lay[L].mlp.up_proj.register_forward_hook(lambda m, a, o, L=L: on_up(m, a, o, L)))
            hs.append(lay[max(by_layer)].mlp.register_forward_hook(stop))
            vals = {r: np.zeros((len(units), len(token_ids)), np.float32) for r in c["read"]}

            def step(a0, a1):
                ids = torch.tensor([P + [t] + S for t in token_ids[a0:a1]], device=dev)
                cap.clear()
                try:
                    model.model(input_ids=ids)
                except _Stop:
                    pass
                for L, js in by_layer.items():
                    v = cap[("g", L)]                                          # [batch, readings, units in layer L]
                    if side:
                        v = torch.nn.functional.silu(v) * cap[("u", L)]        # activation a = SiLU(g) * u
                    for q, r in enumerate(c["read"]):
                        vals[r][js, a0:a1] = v[:, q].T.cpu().numpy()
            try:
                run_batched(len(token_ids), batch, step, label=f"context {c['k']}", device=str(dev), log=log)
            finally:
                for h in hs:
                    h.remove()
            out.update({(c["k"], r): vals[r] for r in c["read"]})
    return out


def zscores(vals: dict, stats: dict, contexts, board="gate") -> np.ndarray:
    """Mean z over all readings: [n_units, n_tokens]. z = sign * (value - mean) / max(sd, 1e-6); sign -1 for aneg."""
    sg = -1.0 if board == "aneg" else 1.0
    zs = []
    for c in contexts:
        for r in c["read"]:
            mu, sd = stats[(c["k"], r)]
            mu = np.asarray(mu, np.float32)[:, None]
            sd = np.maximum(np.asarray(sd, np.float32), np.float32(1e-6))[:, None]
            zs.append(sg * (vals[(c["k"], r)] - mu) / sd)
    return np.stack(zs).mean(0)


def _result(rec, words, toks, toks_s, z, uidx, decode) -> dict:
    """One neuron's row, in the research format. toks / toks_s: [(token id, single?)]; z: its mean z per unique token."""
    own = [float(z[uidx[t]]) for t, _ in toks]
    cc = [float(z[uidx[t]]) for t, _ in toks_s]
    a = auc(own, cc) if own and cc else None
    return {"bin": rec.get("bin"), "layer": rec.get("layer"), "label": rec.get("label"), "words": words,
            "tokens": [decode([t]) for t, _ in toks],
            "single_token_share": round(float(np.mean([s for _, s in toks])), 3) if toks else None,
            "z_own": [round(x, 2) for x in own], "z_ctrl": [round(x, 2) for x in cc], "control_neuron": SURFACE_CONTROL,
            "control_tokens": [decode([t]) for t, _ in toks_s],
            "auc": None if a is None else round(a, 3), "pass": bool(a is not None and a >= AUC_PASS),
            "median_z_own": None if not own else round(float(np.median(own)), 2),
            "share_z_over_2": None if not own else round(float(np.mean(np.array(own) > 2)), 3)}


def _summary(out: dict, **head) -> dict:
    groups = {}
    for r in out.values():
        b = "uncertain" if str(r["bin"]).startswith("uncertain") else str(r["bin"]).split(" (")[0]
        groups.setdefault(b, []).append(r)
    med = lambda v: round(float(np.median(v)), 3) if v else None   # noqa: E731
    sts = [r["single_token_share"] for r in out.values() if r["single_token_share"] is not None]
    return {**head, "pass_by_bin": {b: f"{sum(x['pass'] for x in v)}/{len(v)} ({100 * np.mean([x['pass'] for x in v]):.0f}%)" for b, v in groups.items()},
            "median_auc_by_bin": {b: med([x["auc"] for x in v if x["auc"] is not None]) for b, v in groups.items()},
            "single_token_share": round(float(np.mean(sts)), 3) if sts else None}


def score_words(model, tok, words_file, sweep_dir, sweep_tag, out_path=None, *, board: str, pseudo=None, labels=None,
                batch: int = 512, log=print) -> dict:
    """The model step: z-score every word's token against the sweep's per-unit statistics, and the AUC of the label's
    tokens against the surface tokens. Writes out_path (gen_check_<tag>.json) and a .txt digest next to it.

    words_file: generate_words output (path or dict). labels: the verified exam records, only needed for a research
    gen_words file (which has no "records"). pseudo: a pseudo-neuron set (path, or pseudo.load's dict) when the
    labelled units are pseudo-neurons; real neurons are named L<layer>:<index>."""
    if board not in BOARDS:
        raise ValueError(f"board must be one of {BOARDS}, not {board!r}")
    t0 = time.time()
    wd, _, _ = _load(words_file)
    words, words_s = wd["words"], wd["words_s"]
    recs = wd.get("records")
    if recs is None:
        lab, _, _ = _load(labels)
        recs = {k: {"bin": r.get("bin"), "layer": r.get("layer"), "label": r.get("description")} for k, r in lab.items() if r.get("description")}
    keys = [k for k in recs if k in words and k in words_s]
    pseudo_name = ""
    if isinstance(pseudo, (str, Path)):
        from modelmapper import pseudo as pseudo_sets
        pseudo_name, pseudo = Path(pseudo).name, pseudo_sets.load(pseudo)
    elif pseudo is not None:
        pseudo_name = "(given)"
    enc = lambda ws: [t for t in (to_token(tok, w) for w in ws if w.strip()) if t[0] is not None]   # noqa: E731
    toks = {k: enc(words[k]) for k in keys}
    toks_s = {k: enc(words_s[k]) for k in keys}
    uniq = sorted({t for v in (*toks.values(), *toks_s.values()) for t, _ in v})
    uidx = {t: j for j, t in enumerate(uniq)}
    out = {}
    if keys and uniq:
        unit = _units(keys, pseudo)
        contexts, stats = sweep_stats(sweep_dir, sweep_tag, keys, unit, board, pseudo)
        vals = unit_values(model, tok, [unit[k] for k in keys], uniq, contexts, board=board, pseudo=pseudo, batch=batch, log=log)
        Z = zscores(vals, stats, contexts, board)
        out = {k: _result(recs[k], words[k], toks[k], toks_s[k], Z[i], uidx, tok.decode) for i, k in enumerate(keys)}
    tag = _tag(Path(out_path).stem, "gen_check_") if out_path else ""
    summ = _summary(out, tag=tag, labels=wd.get("labels", ""), pseudo=pseudo_name, control=wd.get("control", ""), n=len(out),
                    **({"board": board} if board != "gate" else {}))
    summ.update({"luna_usd": wd.get("luna_usd", 0.0), "seconds": round(time.time() - t0)})
    res = {"_summary": summ, **out}
    if out_path:
        _write(out_path, res, indent=1)
        lines = [f"{k:>14} {str(r['bin'])[:22]:<22} AUC {r['auc']} median z {r['median_z_own']} | {str(r['label'])[:70]} | "
                 f"{' '.join(r['tokens'][:8])}" for k, r in out.items()]
        Path(out_path).with_suffix(".txt").write_text(json.dumps(summ, ensure_ascii=False) + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    log(json.dumps(summ, ensure_ascii=False))
    return res


# ---------------------------------------------------------------- the whole check
def exam_board(exam) -> str:
    """The board an exam file was run on (its summary's exam settings), default gate."""
    _, s, _ = _load(exam)
    ex = s.get("exam") if isinstance(s.get("exam"), dict) else {}
    return ex.get("board") or s.get("board") or "gate"


def run_gencheck(exam, out_dir=None, *, sweep_dir=None, sweep_tag=None, board=None, model=None, tok=None, model_name=None,
                 pseudo=None, words_only=False, words_from=None, workers: int = WORD_WORKERS, log=print):
    """Verified subset -> surface labels -> words -> score, as the research runner does for one exam board. Files (in
    out_dir, default the exam's folder; tag = the exam file's name after "intelligibility_"):
    intelligibility_<tag>_ver.json, graded_surface_<tag>_ver.json, gen_words_<tag>_surf.json, gen_check_<tag>_surf.json.
    A step whose output is already there is skipped (resumable). Returns the gen_check dict; the words dict with
    words_only (no model is touched); None when the exam verified no label. words_from: a gen_words file to score
    instead of generating. The model (model + tok, else loaded from model_name) is loaded only for the last step."""
    exam = Path(exam)
    tag = _tag(exam.stem, "intelligibility_")
    d = Path(out_dir) if out_dir else exam.parent
    board = board or exam_board(exam)
    f_ver, f_surf = d / f"intelligibility_{tag}_ver.json", d / f"graded_surface_{tag}_ver.json"
    f_words, f_out = d / f"gen_words_{tag}_surf.json", d / f"gen_check_{tag}_surf.json"
    if not words_only and (done := _done(f_out)):
        log(f"{f_out.name} exists: skipped")
        return done
    ver = _done(f_ver) or verified_subset(exam, f_ver)
    n = ver["_summary"]["n"]
    log(f"{tag}: {n} exam-verified labels")
    if n == 0:
        return None
    if words_from:
        words = _load(words_from)[0]
    else:
        words = _done(f_words, key="words")
        if words is None:
            if not _done(f_surf):
                surface_labels(f_ver, f_surf, log=log)
            words = generate_words(f_ver, f_surf, f_words, workers=workers, log=log)
    if words_only:
        return words
    if sweep_dir is None or sweep_tag is None:
        raise ValueError("scoring needs the vocabulary sweep (sweep_dir and sweep_tag)")
    if model is None:
        from modelmapper.model import load
        from modelmapper.settings import DEFAULT_MODEL
        model, tok = load(model_name or DEFAULT_MODEL)
    return score_words(model, tok, words, sweep_dir, sweep_tag, f_out, board=board, pseudo=pseudo, labels=f_ver, log=log)


def cli(a) -> int:
    """`modelmapper gencheck`: words for every exam first (Luna, no model), then one model load scores them all."""
    if a.words_from and len(a.exams) != 1:
        print("--words-from takes one exam file", file=sys.stderr)
        return 2
    sweep = Path(a.sweep) if a.sweep else None
    if not a.words_only and (sweep is None or not sweep.name.startswith("vocab_sweep_")):
        print("--sweep must name the sweep's vocab_sweep_<tag>.json (needed unless --words-only)", file=sys.stderr)
        return 2
    kw = {"out_dir": a.out, "board": a.board, "pseudo": a.pseudo}
    todo = []
    for e in a.exams:
        w = run_gencheck(e, words_only=True, words_from=a.words_from, **kw)
        if w is not None:
            todo.append(e)
    if a.words_only or not todo:
        return 0
    from modelmapper.model import load
    from modelmapper.settings import DEFAULT_MODEL
    model, tok = load(a.model or DEFAULT_MODEL)
    for e in todo:
        r = run_gencheck(e, sweep_dir=sweep.parent, sweep_tag=_tag(sweep.stem, "vocab_sweep_"), model=model, tok=tok,
                         words_from=a.words_from, **kw)
        print(f"{Path(e).name}: {r['_summary']['n']} labels, pass {r['_summary']['pass_by_bin']}")
    return 0
