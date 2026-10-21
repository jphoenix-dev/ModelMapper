"""The generative check (gencheck.py). Fast tests: mocked Luna, a tiny random Qwen3 and a toy tokenizer, synthetic sweep
statistics. Then Qwen3-0.6B on the CPU (mark model): the scored values match a direct forward pass. Then one real
end-to-end smoke (marks model + repro): real Luna on 2 exam-verified research labels, scored on Qwen3-0.6B against a
small sweep built here on the CPU; skipped without a labeller key or the research files (MODEL_MAPPER_V3_REF)."""
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from modelmapper import gencheck as gc

REF = Path(os.environ.get("MODEL_MAPPER_V3_REF", ""))
CTX = [{"k": 0, "read": ["slot"]}, {"k": 1, "read": ["slot", "end"]}, {"k": 9, "read": ["slot", "end"]}]


# ---------------------------------------------------------------- toy tokenizer, tiny model, synthetic sweep
class ToyTok:
    """Whole words from a short list are one token (with and without a leading space); anything else is characters."""
    WORDS = ["cat", "dog", "fish", "bird", "horse", "red", "blue", "green", "one", "two", "three", "The", "new", "word"]
    BARE_ONLY = ["xyz"]

    def __init__(self):
        self.vocab = {}
        for w in self.WORDS:
            self.vocab.setdefault(" " + w, len(self.vocab)); self.vocab.setdefault(w, len(self.vocab))
        for w in self.BARE_ONLY:
            self.vocab.setdefault(w, len(self.vocab))
        for ch in [chr(c) for c in range(32, 127)] + ["\n"]:
            self.vocab.setdefault(ch, len(self.vocab))
        self.inv = {v: k for k, v in self.vocab.items()}

    def encode(self, text, add_special_tokens=True):
        ids = []
        for piece in re.findall(r" ?[A-Za-z]+|.", text, flags=re.S):
            ids += [self.vocab[piece]] if piece in self.vocab else [self.vocab.get(ch, 0) for ch in piece]
        return ids

    def decode(self, ids):
        return "".join(self.inv[i] for i in ids)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    m = Qwen3ForCausalLM(cfg).eval()
    for p in m.parameters():                       # bigger weights than the init, so values are not all ~0
        p.data.normal_(0, 0.3)
    return m, ToyTok()


def write_sweep(d, tag, n_layers, inter, contexts=CTX, pseudo_names=(), n_tok=50, seed=0):
    """A stats-only sweep in the research format: per context and reading, gate / activation mean and sd per neuron,
    and full pseudo profiles (n_tok tokens). Returns the stats it wrote."""
    rng = np.random.default_rng(seed); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    meta = {"panel": "v1", "version": 2, "contexts": [{"k": c["k"], "text": gc.PANEL_V1[c["k"]].replace(gc.SLOT, "[X]"), "read": c["read"]}
                                                      for c in contexts]}
    (d / f"vocab_sweep_{tag}.json").write_text(json.dumps(meta), encoding="utf-8")
    wrote = {}
    for c in contexts:
        out = {}
        for r in c["read"]:
            for k in ("mean", "amean"):
                out[f"{r}_{k}"] = rng.normal(0, 0.5, (n_layers, inter)).astype(np.float32)
            for k in ("std", "astd"):
                out[f"{r}_{k}"] = rng.uniform(0.5, 2, (n_layers, inter)).astype(np.float32)
            out[f"{r}_pseudo_g"] = rng.normal(0, 1, (n_tok, len(pseudo_names))).astype(np.float16)
            out[f"{r}_pseudo_u"] = rng.normal(0, 1, (n_tok, len(pseudo_names))).astype(np.float16)
            wrote[(c["k"], r)] = out
        np.savez(d / f"vocab_sweep_{tag}_c{c['k']}.npz", **out, pseudo=np.array(list(pseudo_names)))
    return wrote


def direct(model, tok, k, read, token_ids, L, i, board):
    """The unit's value straight from a full forward pass with modelmapper.model.capture."""
    from modelmapper.model import capture
    a_, b_ = gc.PANEL_V1[k].split(gc.SLOT)
    P, S = tok.encode(a_), tok.encode(b_)
    pos = {"slot": len(P), "end": len(P) + len(S)}[read]
    with torch.no_grad(), capture(model, [L], gate_value=True) as st:
        model(input_ids=torch.tensor([P + [t] + S for t in token_ids], device=model.device))
    g, u = st["gate"][L][:, pos, i], st["value"][L][:, pos, i]
    return (g if board == "gate" else torch.nn.functional.silu(g) * u).cpu().numpy()


# ---------------------------------------------------------------- mocked Luna
SURF_TEXT = "Short lowercase Latin words with a leading space."


def fake_ask(prompt, system):
    if system == gc.SYS_SURF:
        return {"description": SURF_TEXT}, {"prompt_tokens": 100, "completion_tokens": 20}
    assert system == gc.SYS_GEN and prompt.startswith("Description: ")
    if "FAIL" in prompt:
        raise RuntimeError("Luna is down")
    if SURF_TEXT in prompt:
        return {"tokens": ["red", "blue", "one", "two", "xyz"]}, {"prompt_tokens": 80, "completion_tokens": 30}
    return {"tokens": ["cat", "dog", "fish", " bird", "", "zebra"] + ["horse"] * 40}, {"prompt_tokens": 80, "completion_tokens": 30}


def exam(d, bins, name="intelligibility_toy_gate.json", board="gate", fail=()):
    """An exam file: {neuron: bin}; a neuron in `fail` gets a label that makes the mocked Luna fail."""
    d.mkdir(parents=True, exist_ok=True)
    bins = {k: "intelligible" for k in bins} if isinstance(bins, list) else bins
    recs = {k: {"layer": int(k[1:].split(":")[0]), "description": "FAIL" if k in fail else f"animals {k}",
                "shown": [f"tok{x}" for x in range(60)], "bin": b} for k, b in bins.items()}
    p = d / name
    p.write_text(json.dumps({"_summary": {"exam": {"board": board}, "n": len(recs)}, **recs}), encoding="utf-8")
    return p


# ---------------------------------------------------------------- pure parts
def test_parse_words():
    assert gc.parse_words({"tokens": ["a", 2, "c"]}) == ["a", "2", "c"]
    assert gc.parse_words({"tokens": [str(x) for x in range(50)]}) == [str(x) for x in range(30)]
    assert gc.parse_words({"tokens": "not a list"}) == [] and gc.parse_words({}) == [] and gc.parse_words(None) == []


def test_to_token():
    t = ToyTok()
    assert gc.to_token(t, "cat") == (t.vocab[" cat"], True)          # leading-space form first
    assert gc.to_token(t, "  cat ") == (t.vocab[" cat"], True)
    assert gc.to_token(t, "xyz") == (t.vocab["xyz"], True)           # else the bare form
    assert gc.to_token(t, "zebra") == (t.vocab[" "], False)          # else the first piece of the leading-space form


def test_auc():
    assert gc.auc([3, 2], [1, 0]) == 1.0 and gc.auc([0], [1, 2]) == 0.0
    assert gc.auc([1], [1]) == 0.5 and gc.auc([0, 2], [1]) == 0.5
    assert gc.auc([2, 1, 1], [1, 0]) == pytest.approx((2 + 1.5 + 1.5) / 6)


def test_zscores_and_result():
    ctx = [{"k": 0, "read": ["slot"]}, {"k": 1, "read": ["slot", "end"]}]
    vals = {(0, "slot"): np.array([[1, 2, 3.0]], np.float32), (1, "slot"): np.array([[3, 2, 1.0]], np.float32),
            (1, "end"): np.array([[0, 0, 0.0]], np.float32)}
    stats = {(0, "slot"): ([1.0], [1.0]), (1, "slot"): ([0.0], [2.0]), (1, "end"): ([0.0], [0.0])}   # sd 0 -> floor 1e-6
    z = gc.zscores(vals, stats, ctx)
    assert np.allclose(z, [[(0 + 1.5 + 0) / 3, (1 + 1 + 0) / 3, (2 + 0.5 + 0) / 3]])
    assert np.allclose(gc.zscores(vals, stats, ctx, "aneg"), -z) and np.allclose(gc.zscores(vals, stats, ctx, "apos"), z)
    zz = np.array([3.0, 2.5, 0.0, -1.0])
    uidx = {10: 0, 11: 1, 12: 2, 13: 3}
    r = gc._result({"bin": "intelligible", "layer": 4, "label": "L"}, ["a", "b"], [(10, True), (11, False)],
                   [(12, True), (13, True), (10, True)], zz, uidx, lambda ids: f"<{ids[0]}>")
    assert r["z_own"] == [3.0, 2.5] and r["z_ctrl"] == [0.0, -1.0, 3.0]
    assert r["auc"] == 0.75 and r["pass"] is True                       # (1 + 1 + 0.5 + 1 + 1 + 0) / 6: the pass line itself
    assert r["median_z_own"] == 2.75 and r["share_z_over_2"] == 1.0 and r["single_token_share"] == 0.5
    assert r["tokens"] == ["<10>", "<11>"] and r["control_tokens"] == ["<12>", "<13>", "<10>"]
    r2 = gc._result({"bin": "intelligible"}, [], [(10, True)], [(13, True)], zz, uidx, str)
    assert r2["auc"] == 1.0 and r2["pass"] is True
    s = gc._summary({"a": r, "b": r2, "c": {**r2, "bin": "uncertain (x / intelligible)"}}, tag="t")
    assert s["pass_by_bin"] == {"intelligible": "2/2 (100%)", "uncertain": "1/1 (100%)"}
    assert gc._summary({"a": {**r, "auc": 0.749, "pass": False}, "b": r2})["pass_by_bin"] == {"intelligible": "1/2 (50%)"}
    assert s["median_auc_by_bin"]["intelligible"] == round(float(np.median([r["auc"], 1.0])), 3)


def test_verified_subset(tmp_path):
    p = exam(tmp_path, {"L0:1": "intelligible", "L0:2": "list-only", "L0:3": "uncertain (list-only / intelligible)", "L1:4": "intelligible"})
    ver = gc.verified_subset(p, tmp_path / "intelligibility_toy_gate_ver.json")
    assert set(ver) - {"_summary"} == {"L0:1", "L1:4"}                  # not list-only, not "uncertain (... / intelligible)"
    assert ver["_summary"] == {"subset_of": "toy_gate", "n": 2}
    assert json.loads((tmp_path / "intelligibility_toy_gate_ver.json").read_text(encoding="utf-8"))["_summary"]["n"] == 2
    assert gc.exam_board(p) == "gate" and gc.exam_board(exam(tmp_path, ["L0:1"], "intelligibility_b.json", "aneg")) == "aneg"


def test_surface_labels_and_words(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gc, "_ask", lambda p, s: (seen.append((p, s)), fake_ask(p, s))[1])
    p = exam(tmp_path, ["L0:1", "L1:2", "L2:7"], fail=("L2:7",))
    ver = gc.verified_subset(p, tmp_path / "intelligibility_toy_gate_ver.json")
    sl = gc.surface_labels(tmp_path / "intelligibility_toy_gate_ver.json", tmp_path / "graded_surface_toy_gate_ver.json", log=lambda *_: None)
    assert sl["_summary"]["n"] == 3 and sl["_summary"]["empty"] == 0 and sl["_summary"]["luna_usd"] == round(3 * (100e-7 + 20 * 0.5e-6), 4)
    assert sl["L0:1"] == {"surface_label": SURF_TEXT, "bin": "intelligible"}
    surf_prompts = [q for q, s in seen if s == gc.SYS_SURF]
    assert all(q == "Tokens:\n" + "\n".join(f"tok{x}" for x in range(40)) for q in surf_prompts)   # the top 40 shown
    sl["L1:2"]["surface_label"] = ""                                    # no surface label -> no control -> left out
    w = gc.generate_words(ver, sl, tmp_path / "gen_words_toy_gate_surf.json", log=lambda *_: None)
    assert set(w["words"]) == set(w["words_s"]) == set(w["records"]) == {"L0:1", "L2:7"}
    assert w["words"]["L0:1"][:6] == ["cat", "dog", "fish", " bird", "", "zebra"] and len(w["words"]["L0:1"]) == 30
    assert w["words"]["L2:7"] == [] and w["words_s"]["L2:7"] == ["red", "blue", "one", "two", "xyz"]   # a failed call: no words
    assert w["labels"] == "toy_gate_ver" and w["control"] == "surface:toy_gate_ver"
    assert w["records"]["L0:1"] == {"bin": "intelligible", "layer": 0, "label": "animals L0:1"}
    assert w["luna_usd"] == round(3 * (80e-7 + 30 * 0.5e-6), 4)        # 3 successful calls of 4
    assert json.loads((tmp_path / "gen_words_toy_gate_surf.json").read_text(encoding="utf-8"))["words"] == w["words"]


# ---------------------------------------------------------------- the model step on a tiny model
def words_for(neurons):
    return {"labels": "toy_gate_ver", "control": "surface:toy_gate_ver", "luna_usd": 0.01,
            "words": {k: ["cat", "dog", "fish", "bird", "zebra", "horse"] for k in neurons},
            "words_s": {k: ["red", "blue", "one", "two", "xyz"] for k in neurons},
            "records": {k: {"bin": "intelligible", "layer": int(k[1:].split(":")[0]) if k[0] == "L" else 1, "label": f"animals {k}"} for k in neurons}}


@pytest.mark.parametrize("board", gc.BOARDS)
def test_score_words_matches_direct_forward(tiny, tmp_path, board):
    m, tok = tiny
    neurons = ["L0:3", "L2:40", "L2:7"]
    st = write_sweep(tmp_path, "toy", 3, 48)
    res = gc.score_words(m, tok, words_for(neurons), tmp_path, "toy", tmp_path / f"gen_check_toy_{board}_surf.json", board=board,
                         batch=4, log=lambda *_: None)
    sg = -1.0 if board == "aneg" else 1.0
    mk, sk = ("amean", "astd") if board != "gate" else ("mean", "std")
    for k in neurons:
        L, i = map(int, k[1:].split(":"))
        r = res[k]
        own_ids = [gc.to_token(tok, w)[0] for w in words_for(neurons)["words"][k]]
        ctl_ids = [gc.to_token(tok, w)[0] for w in words_for(neurons)["words_s"][k]]
        exp = lambda ids: np.mean([sg * (direct(m, tok, c["k"], rd, ids, L, i, board) - st[(c["k"], rd)][f"{rd}_{mk}"][L, i])
                                   / st[(c["k"], rd)][f"{rd}_{sk}"][L, i] for c in CTX for rd in c["read"]], 0)
        e_own, e_ctl = exp(own_ids), exp(ctl_ids)
        assert np.allclose(r["z_own"], e_own, atol=0.006) and np.allclose(r["z_ctrl"], e_ctl, atol=0.006)
        assert r["auc"] == round(gc.auc(e_own, e_ctl), 3) and r["pass"] == (gc.auc(e_own, e_ctl) >= 0.75)
        assert r["tokens"] == [" cat", " dog", " fish", " bird", " ", " horse"] and r["single_token_share"] == round(5 / 6, 3)
        assert r["control_tokens"] == [" red", " blue", " one", " two", "xyz"]
    s = res["_summary"]
    assert s["n"] == 3 and s["tag"] == f"toy_{board}_surf" and s["labels"] == "toy_gate_ver" and s["luna_usd"] == 0.01
    assert ("board" in s) == (board != "gate") and s["pseudo"] == ""
    assert (tmp_path / f"gen_check_toy_{board}_surf.txt").read_text(encoding="utf-8").count("\n") == 4


@pytest.mark.parametrize("board", gc.BOARDS)
def test_pseudo_rows_copying_real_rows_give_real_values(tiny, tmp_path, board):
    """As the research sweep's CHECK rows: a pseudo-neuron whose gate / up rows are a real neuron's must read the same."""
    m, tok = tiny
    real = [(2, 40), (0, 3)]
    ps = {"names": np.array(["P0:L2:mix", "P1:L0:gauss"]), "layer": np.array([2, 0]),
          "gate": np.stack([m.model.layers[L].mlp.gate_proj.weight[i].detach().numpy() for L, i in real]),
          "up": np.stack([m.model.layers[L].mlp.up_proj.weight[i].detach().numpy() for L, i in real])}
    ids = list(range(0, 60, 3))
    a = gc.unit_values(m, tok, real, ids, CTX, board=board, batch=7, log=lambda *_: None)
    b = gc.unit_values(m, tok, [(2, 0), (0, 1)], ids, CTX, board=board, pseudo=ps, batch=64, log=lambda *_: None)
    for key in a:
        assert np.allclose(a[key], b[key], atol=1e-5), key
    st = write_sweep(tmp_path, "pq", 3, 48, pseudo_names=["X", "P1:L0:gauss", "P0:L2:mix"])
    w = words_for(["P0:L2:mix", "P1:L0:gauss"])
    res = gc.score_words(m, tok, w, tmp_path, "pq", tmp_path / "gen_check_pq_surf.json", board=board, pseudo=ps, log=lambda *_: None)
    assert res["_summary"]["n"] == 2 and res["_summary"]["pseudo"] == "(given)"
    G = st[(0, "slot")]["slot_pseudo_g"][:, 2].astype(np.float32)        # stats come from the profile column named like the unit
    U = st[(0, "slot")]["slot_pseudo_u"][:, 2].astype(np.float32)
    v = G if board == "gate" else G / (1 + np.exp(-G)) * U
    _, stats = gc.sweep_stats(tmp_path, "pq", ["P0:L2:mix"], gc._units(["P0:L2:mix"], ps), board, ps)
    assert np.allclose(stats[(0, "slot")][0], [v.mean()]) and np.allclose(stats[(0, "slot")][1], [v.std()])


def test_research_words_file_needs_labels(tiny, tmp_path):
    m, tok = tiny
    write_sweep(tmp_path, "toy", 3, 48)
    w = words_for(["L1:5"]); recs = w.pop("records")
    lab = {"_summary": {}, "L1:5": {"bin": "intelligible", "layer": 1, "description": recs["L1:5"]["label"]}}
    res = gc.score_words(m, tok, w, tmp_path, "toy", None, board="gate", labels=lab, log=lambda *_: None)
    assert res["L1:5"]["label"] == "animals L1:5" and res["L1:5"]["auc"] is not None


def test_run_gencheck_chain_and_resume(tiny, tmp_path, monkeypatch):
    m, tok = tiny
    monkeypatch.setattr(gc, "_ask", fake_ask)
    p = exam(tmp_path, {"L0:1": "intelligible", "L2:9": "intelligible", "L1:3": "list-only"})   # L1:3 is not checked
    write_sweep(tmp_path / "sweep", "toy", 3, 48)
    res = gc.run_gencheck(p, sweep_dir=tmp_path / "sweep", sweep_tag="toy", model=m, tok=tok, log=lambda *_: None)
    assert set(res) - {"_summary"} == {"L0:1", "L2:9"}
    for f in ("intelligibility_toy_gate_ver.json", "graded_surface_toy_gate_ver.json", "gen_words_toy_gate_surf.json",
              "gen_check_toy_gate_surf.json", "gen_check_toy_gate_surf.txt"):
        assert (tmp_path / f).exists(), f
    s = res["_summary"]
    assert s["control"] == "surface:toy_gate_ver" and s["labels"] == "toy_gate_ver" and s["tag"] == "toy_gate_surf"

    def no_luna(*a):
        raise AssertionError("a finished step ran again")
    monkeypatch.setattr(gc, "_ask", no_luna)
    assert gc.run_gencheck(p, sweep_dir=tmp_path / "sweep", sweep_tag="toy", model=m, tok=tok, log=lambda *_: None)["_summary"] == s
    (tmp_path / "gen_check_toy_gate_surf.json").unlink()                 # only the model step is redone
    again = gc.run_gencheck(p, sweep_dir=tmp_path / "sweep", sweep_tag="toy", model=m, tok=tok, log=lambda *_: None)
    assert {k: v for k, v in again.items() if k != "_summary"} == {k: v for k, v in res.items() if k != "_summary"}
    none = exam(tmp_path, {"L0:2": "list-only"}, "intelligibility_none.json")   # nothing verified: no check
    assert gc.run_gencheck(none, sweep_dir=tmp_path / "sweep", sweep_tag="toy", model=m, tok=tok, log=lambda *_: None) is None


def test_cli_gencheck(tiny, tmp_path, monkeypatch, capsys):
    from modelmapper import cli, model as mm
    m, tok = tiny
    monkeypatch.setattr(gc, "_ask", fake_ask)
    a = exam(tmp_path / "a", ["L0:1"]); b = exam(tmp_path / "b", ["L2:4"], board="aneg")
    assert cli.main(["gencheck", str(a), str(b), "--words-only"]) == 0
    assert (tmp_path / "a" / "gen_words_toy_gate_surf.json").exists() and not (tmp_path / "a" / "gen_check_toy_gate_surf.json").exists()
    assert cli.main(["gencheck", str(a)]) == 2                            # scoring needs --sweep
    write_sweep(tmp_path / "s", "toy", 3, 48)
    loads = []
    monkeypatch.setattr(mm, "load", lambda name: (loads.append(name), (m, tok))[1])
    assert cli.main(["gencheck", str(a), str(b), "--sweep", str(tmp_path / "s" / "vocab_sweep_toy.json"), "--model", "toy"]) == 0
    assert loads == ["toy"]                                               # one model load for both boards
    rb = json.loads((tmp_path / "b" / "gen_check_toy_gate_surf.json").read_text(encoding="utf-8"))
    assert rb["_summary"]["board"] == "aneg"                              # the board the exam ran on
    out = tmp_path / "o"
    wf = tmp_path / "a" / "gen_words_toy_gate_surf.json"
    assert cli.main(["gencheck", str(a), "--sweep", str(tmp_path / "s" / "vocab_sweep_toy.json"), "--words-from", str(wf),
                     "--out", str(out), "--board", "apos"]) == 0
    assert json.loads((out / "gen_check_toy_gate_surf.json").read_text(encoding="utf-8"))["_summary"]["board"] == "apos"


@pytest.mark.parametrize("board", gc.BOARDS)
def test_schema_matches_research(tiny, tmp_path, board):
    """Same fields as the research run's file (the ledger importer and the v3 scorecard read them)."""
    f = REF / f"gen_check_v3_{board}_surf.json"
    if not f.exists():
        pytest.skip(f"no research file {f}")
    ref = json.loads(f.read_text(encoding="utf-8"))
    m, tok = tiny
    write_sweep(tmp_path, "toy", 3, 48)
    res = gc.score_words(m, tok, words_for(["L1:2"]), tmp_path, "toy", tmp_path / f"gen_check_v3_{board}_surf.json", board=board,
                         log=lambda *_: None)
    assert list(res["_summary"]) == list(ref["_summary"])
    rk = next(k for k in ref if k != "_summary")
    assert list(res["L1:2"]) == list(ref[rk])
    for field, value in ref[rk].items():
        assert type(res["L1:2"][field]) is type(value) or value is None, field
    assert res["L1:2"]["control_neuron"] == ref[rk]["control_neuron"]


# ---------------------------------------------------------------- Qwen3-0.6B on the CPU
@pytest.fixture(scope="module")
def qwen():
    import torch
    from modelmapper.model import load
    # float32: in bf16 a token's value moves by a rounding step (~0.01) with the other tokens in its batch, which is
    # noise, not the arithmetic under test
    return load("Qwen/Qwen3-0.6B", dtype=torch.float32)


@pytest.mark.model
@pytest.mark.parametrize("board", gc.BOARDS)
def test_qwen06_values_match_hooks(qwen, tmp_path, board):
    m, tok = qwen
    neurons = ["L3:100", "L20:2000"]
    ctx = [{"k": 1, "read": ["slot", "end"]}, {"k": 4, "read": ["slot", "end"]}]
    st = write_sweep(tmp_path, "q", 28, 3072, contexts=ctx)
    w = {"labels": "q", "control": "surface:q", "words": {k: ["coffee", "Paris", "running", "the", "xylophone"] for k in neurons},
         "words_s": {k: ["A", "b", "123", "!!"] for k in neurons},
         "records": {k: {"bin": "intelligible", "layer": int(k[1:].split(":")[0]), "label": "x"} for k in neurons}}
    res = gc.score_words(m, tok, w, tmp_path, "q", tmp_path / "gen_check_q_surf.json", board=board, batch=3, log=lambda *_: None)
    sg = -1.0 if board == "aneg" else 1.0
    mk, sk = ("amean", "astd") if board != "gate" else ("mean", "std")
    for k in neurons:
        L, i = map(int, k[1:].split(":"))
        ids = [gc.to_token(tok, x)[0] for x in w["words"][k]]
        raw = gc.unit_values(m, tok, [(L, i)], ids, ctx, board=board, batch=2, log=lambda *_: None)
        for c in ctx:
            for rd in c["read"]:
                assert np.allclose(raw[(c["k"], rd)][0], direct(m, tok, c["k"], rd, ids, L, i, board), rtol=1e-4, atol=1e-4)
        exp = np.mean([sg * (raw[(c["k"], rd)][0] - st[(c["k"], rd)][f"{rd}_{mk}"][L, i]) / st[(c["k"], rd)][f"{rd}_{sk}"][L, i]
                       for c in ctx for rd in c["read"]], 0)
        assert np.allclose(res[k]["z_own"], exp, atol=0.006)
        assert res[k]["tokens"] == [tok.decode([t]) for t in ids]


# ---------------------------------------------------------------- real end-to-end smoke (Luna + Qwen3-0.6B)
def mini_sweep(model, tok, d, tag, n_tokens=128, batch=64, seed=2026):
    """A small stats-only sweep on the CPU: every neuron's gate and activation mean / sd per panel reading over a random
    subset of n_tokens vocabulary tokens (the real sweep uses all ~150k)."""
    lay = model.model.layers; NL, NI = len(lay), model.config.intermediate_size
    special = set(tok.all_special_ids) | {j for t, j in tok.get_vocab().items() if t.startswith("<|")}
    allowed = np.array([j for j in range(len(tok)) if j not in special])
    ids_all = np.sort(np.random.default_rng(seed).choice(allowed, n_tokens, replace=False)).tolist()
    d = Path(d); d.mkdir(parents=True, exist_ok=True); contexts = []
    for k, text in enumerate(gc.PANEL_V1):
        a_, b_ = text.split(gc.SLOT); P, S = tok.encode(a_), tok.encode(b_)
        reads = ["slot", "end"] if b_ else ["slot"]; RP = [{"slot": len(P), "end": len(P) + len(S)}[r] for r in reads]
        sums = np.zeros((len(reads), 4, NL, NI)); cap = {}
        hs = [lay[L].mlp.gate_proj.register_forward_hook(lambda m_, a, o, L=L: cap.__setitem__(("g", L), o[:, RP].float())) for L in range(NL)]
        hs += [lay[L].mlp.up_proj.register_forward_hook(lambda m_, a, o, L=L: cap.__setitem__(("u", L), o[:, RP].float())) for L in range(NL)]
        try:
            with torch.inference_mode():
                for a0 in range(0, n_tokens, batch):
                    model.model(input_ids=torch.tensor([P + [t] + S for t in ids_all[a0:a0 + batch]], device=model.device))
                    for L in range(NL):
                        G = cap[("g", L)]; A = torch.nn.functional.silu(G) * cap[("u", L)]
                        for q in range(len(reads)):
                            sums[q, :, L] += torch.stack([G[:, q].sum(0), (G[:, q] ** 2).sum(0), A[:, q].sum(0), (A[:, q] ** 2).sum(0)]).double().cpu().numpy()
        finally:
            for h in hs:
                h.remove()
        out = {}
        for q, r in enumerate(reads):
            s = sums[q] / n_tokens
            out[f"{r}_mean"], out[f"{r}_amean"] = s[0].astype(np.float32), s[2].astype(np.float32)
            out[f"{r}_std"] = np.sqrt(np.maximum(s[1] - s[0] ** 2, 0)).astype(np.float32)
            out[f"{r}_astd"] = np.sqrt(np.maximum(s[3] - s[2] ** 2, 0)).astype(np.float32)
        np.savez(d / f"vocab_sweep_{tag}_c{k}.npz", **out, tokens=np.array(ids_all))
        contexts.append({"k": k, "text": text.replace(gc.SLOT, "[X]"), "read": reads})
    (d / f"vocab_sweep_{tag}.json").write_text(json.dumps({"panel": "v1", "version": 2, "n_tokens": n_tokens, "contexts": contexts}), encoding="utf-8")


def _have_key():
    from modelmapper import labeller
    try:
        return bool(labeller._load_key())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.model
@pytest.mark.repro
def test_real_smoke_two_labels(qwen, tmp_path):
    """Real Luna on 2 exam-verified Qwen3-8B gate labels (whose L:i also exists in Qwen3-0.6B), scored on Qwen3-0.6B.
    Runs end to end; the scores mean nothing for 0.6B (the labels describe 8B neurons). Spend: about US$0.001."""
    src = REF / "intelligibility_v3_gate.json"
    if not src.exists():
        pytest.skip(f"no research exam file {src}")
    if not _have_key():
        pytest.skip("no labeller key (OPENAI_API_KEY or OAI_KEY_FILE)")
    m, tok = qwen
    ver = gc.verified_subset(src)
    pick = [k for k in ver if k != "_summary" and int(k[1:].split(":")[0]) < 28 and int(k.split(":")[1]) < 3072][:2]
    assert len(pick) == 2
    ex = tmp_path / "intelligibility_smoke_gate.json"
    ex.write_text(json.dumps({"_summary": {"exam": {"board": "gate"}}, **{k: ver[k] for k in pick}}, ensure_ascii=False), encoding="utf-8")
    mini_sweep(m, tok, tmp_path / "sweep", "mini")
    res = gc.run_gencheck(ex, sweep_dir=tmp_path / "sweep", sweep_tag="mini", model=m, tok=tok)
    sl = json.loads((tmp_path / "graded_surface_smoke_gate_ver.json").read_text(encoding="utf-8"))
    spend = sl["_summary"]["luna_usd"] + res["_summary"]["luna_usd"]
    print(f"\nsmoke: {json.dumps(res['_summary'])}\nLuna spend ${spend:.4f}")
    for k in pick:
        print(f"  {k}: AUC {res[k]['auc']}, median z {res[k]['median_z_own']}, tokens {res[k]['tokens'][:6]}, control {res[k]['control_tokens'][:6]}")
    assert set(res) - {"_summary"} == set(pick) and all(res[k]["auc"] is not None for k in pick)
    assert spend < 0.15
