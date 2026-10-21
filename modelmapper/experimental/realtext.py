"""EXPERIMENTAL. Corpora, the corpus readings pass, and the labels the real-text checks examine.

Ported from the research run (readings_atlas.py / readings_atlas2.py, 20 Oct 2026; the label loaders of corpus2_pass.py
and sim_run.py), with its fixed paths and model made arguments. Not re-run or calibrated in this package.

Corpora: documents from
  a FineWeb parquet shard (`.parquet`, its "text" column; the research run read fineweb_000_00000 from row 0),
  a `.jsonl` file (a "text" field per line; the research run's other corpora were its own chat-format recall questions
  and Wikipedia intros), or a `.txt` file (one document per line).
Each document is tokenised and cut at CTX = 1024 tokens; parquet documents shorter than 32 tokens are skipped (as in
the research run), others shorter than MIN_TOKENS.

The readings pass (GPU for big models): one forward pass per batch of documents reads every MLP neuron's gate value z_g
and up value z_u at every position (position 0 and padding skipped) and keeps, per neuron, three leaderboards of K
positions with z_g and z_u: gate (top z_g), act_pos (top a = SiLU(z_g) z_u) and act_neg (top -a), plus running sums.
Position code = document index * CTX + position. Output: readings_<name>.pt and readings_<name>_docs.npz (the documents'
token ids). The research pass also stored full profiles of a 294-neuron sample; that part is left out here.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

CTX = 1024
MIN_TOKENS = 16
NEG = -3e4
BOARDS = ("gate", "apos", "aneg")
EDGE_BIN = "decoys-too-easy (intelligible)"
VERIFIED_BINS = ("intelligible", EDGE_BIN)
band = lambda L, n_layers=36: "early" if L < n_layers / 3 else "middle" if L < 2 * n_layers / 3 else "late"   # noqa: E731


# ------------------------------------------------------------------------------------------------ corpora
def docs(source, tok, offset: int = 0, min_tokens: int = MIN_TOKENS):
    """Token-id lists, one per document, from a .parquet ("text" column), .jsonl ("text" field) or .txt (one per line)."""
    p = Path(source)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        row = 0
        for b in pq.ParquetFile(str(p)).iter_batches(batch_size=256, columns=["text"]):
            for t in b.column("text").to_pylist():
                row += 1
                if row > offset:
                    ids = tok.encode(t)[:CTX]
                    if len(ids) >= 32:
                        yield ids
        return
    with open(p, encoding="utf-8") as f:
        for k, line in enumerate(f):
            if k < offset or not line.strip():
                continue
            t = json.loads(line)["text"] if p.suffix == ".jsonl" else line.rstrip("\n")
            ids = tok.encode(t)[:CTX]
            if len(ids) >= min_tokens:
                yield ids


def special_ids(tok) -> set:
    return set(tok.all_special_ids) | {j for t, j in tok.get_vocab().items() if t.startswith("<|")}


class Docs:
    """A readings pass's documents: flat token ids, lengths, starts; snippets that end AT a marked token."""

    def __init__(self, path):
        z = np.load(path)
        self.toks, self.ln = z["tokens"], z["lengths"]
        self.st = np.concatenate([[0], np.cumsum(self.ln)[:-1]]).astype(np.int64)

    def doc(self, f):
        return int(np.searchsorted(self.st, f, side="right") - 1)

    def snip(self, tok, f, left: int = 32) -> str:
        clean = lambda ids: re.sub(r"\s+", " ", re.sub(r"<\|[^|]*\|>", " ", tok.decode(ids))).strip()   # noqa: E731
        d = self.doc(f)
        return clean(self.toks[max(self.st[d], f - left):f].tolist()) + "[[" + tok.decode(self.toks[f:f + 1].tolist()) + "]]"


# ------------------------------------------------------------------------------------------------ readings pass
def readings(model, tok, documents, out_dir, name: str, max_tokens: int, K: int = 100, B: int = 8, log=print) -> Path:
    """The corpus readings pass (see the module docstring). documents: an iterable of token-id lists."""
    import torch
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    layers = model.model.layers
    L, W = len(layers), layers[0].mlp.gate_proj.out_features
    dev = next(model.parameters()).device
    with torch.inference_mode():
        boards = {k: {"v": torch.full((L, W, K), NEG, device=dev), "c": torch.zeros((L, W, K), dtype=torch.int32, device=dev),
                      "g": torch.zeros((L, W, K), dtype=torch.float16, device=dev), "u": torch.zeros((L, W, K), dtype=torch.float16, device=dev)}
                  for k in ("gate", "act_pos", "act_neg")}
        stats = {k: torch.zeros((L, W), dtype=torch.float64, device=dev) for k in ("open", "apos", "g", "g2", "u", "u2", "a", "a2")}
        state, job = {}, {}

        def merge(key, l, score, codes, g, u):
            """keep each neuron's top K by score; only candidates beating the current K-th place are gathered"""
            b = boards[key]; v = b["v"]; st = score.T                         # [W, N]
            mask = st > v[l][:, -1:]; cnt = mask.sum(1); maxc = int(cnt.max())
            if maxc == 0:
                return
            nz = mask.nonzero(); w_i, n_i = nz[:, 0], nz[:, 1]
            slot = torch.arange(len(nz), device=dev) - (torch.cumsum(cnt, 0) - cnt)[w_i]
            pv = torch.full((W, maxc), NEG, device=dev); pc = torch.zeros((W, maxc), dtype=torch.int32, device=dev)
            pg = torch.zeros((W, maxc), dtype=torch.float16, device=dev); pu = torch.zeros((W, maxc), dtype=torch.float16, device=dev)
            pv[w_i, slot] = st[w_i, n_i]; pc[w_i, slot] = codes[n_i]; pg[w_i, slot] = g.T[w_i, n_i].half(); pu[w_i, slot] = u.T[w_i, n_i].half()
            allv = torch.cat([v[l], pv], 1); top = allv.topk(K, dim=1).indices
            b["v"][l] = allv.gather(1, top)
            for f_, p_ in (("c", pc), ("g", pg), ("u", pu)):
                b[f_][l] = torch.cat([b[f_][l], p_], 1).gather(1, top)

        def on_gate(mod, inp, out, l):
            job["g"] = out[state["mask"]].float()                             # [N, W]

        def on_up(mod, inp, out, l):
            zg = job.pop("g"); zu = out[state["mask"]].float(); a = torch.nn.functional.silu(zg) * zu; codes = state["codes"]
            merge("gate", l, zg, codes, zg, zu); merge("act_pos", l, a, codes, zg, zu); merge("act_neg", l, -a, codes, zg, zu)
            for k_, x in (("open", (zg > 0).double()), ("apos", (a > 0).double()), ("g", zg), ("g2", zg * zg), ("u", zu), ("u2", zu * zu), ("a", a), ("a2", a * a)):
                stats[k_][l] += x.sum(0).double()
        hs = [layers[l].mlp.gate_proj.register_forward_hook(lambda mod, i, o, l=l: on_gate(mod, i, o, l)) for l in range(L)]
        hs += [layers[l].mlp.up_proj.register_forward_hook(lambda mod, i, o, l=l: on_up(mod, i, o, l)) for l in range(L)]
        all_docs, seen, t0 = [], 0, time.time()
        gen = iter(documents)
        try:
            pool = []
            while seen < max_tokens:
                if not pool:
                    pool = sorted([d for _, d in zip(range(B * 32), gen)], key=len)   # length bucketing, as in the research pass
                    if not pool:
                        break
                buf, pool = pool[:B], pool[B:]
                T = max(map(len, buf)); base = len(all_docs)
                ids = torch.full((len(buf), T), tok.pad_token_id or 0, device=dev); att = torch.zeros_like(ids)
                for j, d in enumerate(buf):
                    ids[j, :len(d)] = torch.tensor(d, device=dev); att[j, :len(d)] = 1
                mask = att.bool().clone(); mask[:, 0] = False
                pos = torch.arange(T, device=dev)[None].expand(len(buf), -1)
                state["mask"] = mask; state["codes"] = ((torch.arange(len(buf), device=dev)[:, None] + base) * CTX + pos)[mask].int()
                model.model(input_ids=ids, attention_mask=att)
                all_docs += buf; seen += int(mask.sum())
                if dev.type == "cuda" and len(all_docs) % (B * 25) == 0:      # variable-length batches fragment the allocator's cache
                    torch.cuda.empty_cache()
                if len(all_docs) % (B * 50) == 0:
                    log(f"{seen:,} tokens, {time.time() - t0:.0f} s")
        finally:
            for h in hs:
                h.remove()
        torch.save({"name": name, "tokens": seen, "docs": len(all_docs), "K": K, "ctx": CTX, "version": 2, "seconds": time.time() - t0,
                    "torch": str(torch.__version__), **{f"{k}_{f}": (b[f].half() if f == "v" else b[f]).cpu() for k, b in boards.items() for f in ("v", "c", "g", "u")},
                    **{f"sum_{k}": v.cpu() for k, v in stats.items()}}, out_dir / f"readings_{name}.pt")
    lens = np.array([len(d) for d in all_docs]); flat = np.concatenate([np.array(d, np.int32) for d in all_docs]) if all_docs else np.zeros(0, np.int32)
    np.savez_compressed(out_dir / f"readings_{name}_docs.npz", tokens=flat, lengths=lens)
    log(f"saved readings_{name}: {seen:,} tokens, {len(all_docs):,} docs, {time.time() - t0:.0f} s")
    return out_dir / f"readings_{name}.pt"


# ------------------------------------------------------------------------------------------------ labels to examine
def load_labels(results=(), published: bool = False) -> dict:
    """{board: {neuron: (label, verified, source, verified_strict)}}: from `label` results (<prefix>_results.json files or
    folders of them; later files win) and, with published=True, the packaged Qwen3-8B labels (yours win over them).
    verified counts labels at the edge; verified_strict is the bin "intelligible" only."""
    lab = {b: {} for b in BOARDS}

    def add(src, neurons):
        for u, r in neurons.items():
            for b in BOARDS:
                p = r.get(b) or {}
                if p.get("label"):
                    lab[b][u] = (p["label"], p.get("bin") in VERIFIED_BINS, src, p.get("bin") == "intelligible")
    if published:
        from modelmapper import known
        add("published", known.published_labels().get("neurons") or {})
    for s in results:
        s = Path(s)
        for f in sorted(s.glob("*_results.json")) if s.is_dir() else [s]:
            add(f.stem, json.loads(f.read_text(encoding="utf-8")).get("neurons") or {})
    return lab


def load_pseudo(pseudo_set, exams: dict) -> dict:
    """The labelled fake neurons: {board: {name: (label, verified, source, verified_strict, layer, row)}} from a pseudo set
    (modelmapper.pseudo's .npz: names, layer, gate, up) and the label run's pseudo exams {board: intelligibility_*_pq_<board>.json}."""
    z = np.load(pseudo_set); names = [str(x) for x in z["names"]]; lay = z["layer"]
    out = {b: {} for b in BOARDS}
    for b, f in (exams or {}).items():
        d = json.loads(Path(f).read_text(encoding="utf-8")); d.pop("_summary", None)
        for k, e in d.items():
            if e.get("description") and k in names:
                j = names.index(k)
                out[b][k] = (e["description"], e.get("bin") in VERIFIED_BINS, Path(f).stem, e.get("bin") == "intelligible", int(lay[j]), j)
    return out
