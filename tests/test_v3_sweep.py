"""The vocabulary sweep (modelmapper/sweep.py) against a brute-force forward pass of each whole sentence, without the
prefix cache. Fast tests: a tiny randomly initialised Qwen3 with the Qwen3-0.6B tokenizer (a local copy in
MODEL_MAPPER_MODELS; skipped otherwise, no download). Model tests: Qwen3-0.6B on the CPU (marked `model`)."""
import json
import os

import numpy as np
import pytest
import torch

from modelmapper import sweep
from modelmapper.settings import model_path

SMALL = "Qwen/Qwen3-0.6B"
TOL = dict(rtol=2e-3, atol=2e-3)       # stored values are float16


def local(name):
    p = model_path(name)
    if not os.path.isdir(p):
        pytest.skip(f"no local copy of {name} (set MODEL_MAPPER_MODELS)")
    return p


def brute(model, tok, ctx, tokens, chunk=64):
    """Each whole sentence (prefix + token + suffix) in one forward pass, no cache: z_g, z_u [N, layers, neurons] and
    the MLP input x [N, layers, hidden] at each read position."""
    lay = model.model.layers; P, S = ctx["prefix"], ctx["suffix"]; T = len(P) + 1 + len(S)
    rp = [{"slot": len(P), "end": T - 1}[r] for r in ctx["read"]]
    store, out = {}, {r: {"g": [], "u": [], "x": []} for r in ctx["read"]}
    hs = []
    for l in range(len(lay)):
        for nm, mod in (("g", lay[l].mlp.gate_proj), ("u", lay[l].mlp.up_proj), ("x", lay[l].post_attention_layernorm)):
            hs.append(mod.register_forward_hook(lambda m, a, o, nm=nm, l=l: store.__setitem__((nm, l), o[:, rp].float())))
    try:
        with torch.inference_mode():
            for a in range(0, len(tokens), chunk):
                model.model(input_ids=torch.tensor([P + [int(t)] + S for t in tokens[a:a + chunk]]), use_cache=False)
                for q, r in enumerate(ctx["read"]):
                    for nm in ("g", "u", "x"):
                        out[r][nm].append(torch.stack([store[(nm, l)][:, q] for l in range(len(lay))], 1))
    finally:
        for h in hs:
            h.remove()
    return {r: {nm: torch.cat(v).numpy() for nm, v in d.items()} for r, d in out.items()}


def at(Z, rows):
    """Z [N, L, I] at stored token rows [L, I, K] -> [L, I, K]."""
    L, I, _ = rows.shape
    return Z[rows, np.arange(L)[:, None, None], np.arange(I)[None, :, None]]


def silu(x):
    with np.errstate(over="ignore"):        # exp(-x) = inf for very negative x gives -0, as it should
        return x / (1 + np.exp(-x))


def verify(model, tok, out_dir, tag, contexts, sample, pset, check, K):
    """Every stored number against the brute-force pass."""
    meta = json.loads((out_dir / f"vocab_sweep_{tag}.json").read_text(encoding="utf-8"))
    assert [c["k"] for c in meta["contexts"]] == sorted(contexts) and meta["K"] == K
    ctxs = sweep.panel_contexts(tok)
    lay = model.model.layers
    for ck in contexts:
        f = np.load(out_dir / f"vocab_sweep_{tag}_c{ck}.npz")
        tokens = f["tokens"]; N = len(tokens)
        assert list(f["sample"]) == sample and list(f["pseudo"]) == [str(x) for x in pset["names"]] + [f"CHECK:{c}" for c in check]
        B = brute(model, tok, ctxs[ck], tokens)
        for r in ctxs[ck]["read"]:
            Zg, Zu, X = B[r]["g"], B[r]["u"], B[r]["x"]; A = silu(Zg) * Zu
            # favourites: every stored (token, z_g, z_u) is what the full sentence gives, lists are sorted, and the
            # top k values are the brute-force top k (index sets agree up to near-ties)
            for ls, score, sign in (("top", Zg, 1), ("apos", A, 1), ("aneg", A, -1)):
                gk = "v" if ls == "top" else "g"
                ids = f[f"{r}_{ls}_i"]; rows = np.searchsorted(tokens, ids)
                assert ids.shape == (len(lay), lay[0].mlp.gate_proj.weight.shape[0], K)
                assert np.array_equal(tokens[rows], ids)
                np.testing.assert_allclose(f[f"{r}_{ls}_{gk}"].astype(np.float32), at(Zg, rows), **TOL)
                np.testing.assert_allclose(f[f"{r}_{ls}_u"].astype(np.float32), at(Zu, rows), **TOL)
                s = sign * at(score, rows)
                assert (np.diff(s, axis=-1) <= 1e-4 * (1 + np.abs(s[..., 1:]))).all(), f"{ls} not sorted"
                want = -np.sort(-sign * score.transpose(1, 2, 0), axis=-1)[..., :K]
                np.testing.assert_allclose(s, want, rtol=1e-4, atol=1e-4)
                got_set = np.sort(ids, -1)
                want_set = np.sort(tokens[np.argsort(-sign * score.transpose(1, 2, 0), axis=-1, kind="stable")[..., :K]], -1)
                assert (got_set == want_set).all(-1).mean() > 0.999
            # stored activations agree with SiLU(z_g) * z_u
            for ls in ("apos", "aneg"):
                a = silu(f[f"{r}_{ls}_g"].astype(np.float32)) * f[f"{r}_{ls}_u"].astype(np.float32)
                np.testing.assert_allclose(a, at(A, np.searchsorted(tokens, f[f"{r}_{ls}_i"])), rtol=1e-2, atol=1e-2)
            # stats over all tokens
            np.testing.assert_allclose(f[f"{r}_mean"], Zg.mean(0), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(f[f"{r}_std"], Zg.astype(np.float64).std(0), rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(f[f"{r}_amean"], A.mean(0), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(f[f"{r}_astd"], A.astype(np.float64).std(0), rtol=1e-3, atol=1e-4)
            # full profiles: sample neurons, pseudo-neurons (MLP input . row), and check rows = their sample profiles
            for col, u in enumerate(sample):
                l, i = sweep.parse_neuron(u)
                np.testing.assert_allclose(f[f"{r}_sample_z"][:, col].astype(np.float32), Zg[:, l, i], **TOL)
                np.testing.assert_allclose(f[f"{r}_sample_u"][:, col].astype(np.float32), Zu[:, l, i], **TOL)
            for j, L in enumerate(pset["layer"]):
                np.testing.assert_allclose(f[f"{r}_pseudo_g"][:, j].astype(np.float32), X[:, L] @ pset["gate"][j], **TOL)
                np.testing.assert_allclose(f[f"{r}_pseudo_u"][:, j].astype(np.float32), X[:, L] @ pset["up"][j], **TOL)
            for c in check:
                j = list(f["pseudo"]).index(f"CHECK:{c}"); col = sample.index(c)
                np.testing.assert_allclose(f[f"{r}_pseudo_g"][:, j].astype(np.float32), f[f"{r}_sample_z"][:, col].astype(np.float32), **TOL)
                np.testing.assert_allclose(f[f"{r}_pseudo_u"][:, j].astype(np.float32), f[f"{r}_sample_u"][:, col].astype(np.float32), **TOL)
        f.close()


def verify_hubs(out_dir, tag, contexts, ctxs, NL, NI, K, V):
    h = sweep.hub_tokens(out_dir, tag, V, log=lambda *_: None)
    names = [f"c{ck}:{r}" for ck in sorted(contexts) for r in ctxs[ck]["read"]]
    assert list(h["readings"]) == names and h["per_read"].shape == (len(names), V) and h["per_layer"].shape == (NL, V)
    assert (h["per_read"].sum(1) == NL * NI * K).all()                       # every favourite slot counted once
    assert (h["per_layer"].sum(1) == len(names) * NI * K).all()
    assert np.array_equal(h["per_read"].sum(0), h["per_layer"].sum(0))
    saved = np.load(out_dir / f"hub_tokens_{tag}.npz")
    assert np.array_equal(saved["per_read"], h["per_read"]) and (out_dir / f"hub_tokens_{tag}.txt").exists()
    saved.close()


# ---- fast: a tiny random Qwen3 ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(local(SMALL))


@pytest.fixture(scope="module")
def tiny(tok):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64, initializer_range=0.3)
    return Qwen3ForCausalLM(cfg).to(torch.float32).eval()


def tiny_pseudo(n=4, hidden=32, seed=1):
    rng = np.random.default_rng(seed)
    return {"layer": np.array([0, 1, 1, 0][:n], dtype=np.int64), "gate": rng.standard_normal((n, hidden)).astype(np.float32),
            "up": rng.standard_normal((n, hidden)).astype(np.float32), "names": np.array([f"P{j}:L{[0, 1, 1, 0][j]}:mix" for j in range(n)])}


def test_helpers(tok):
    assert sweep.parse_neuron("L31:10454") == (31, 10454) == sweep.parse_neuron("31:10454")
    assert sweep.default_check(["L1:2", "L3:4"]) == ["L1:2", "L3:4"]
    assert sweep.default_check(sweep.CHECK_V3[::-1] + ["L5:6"]) == sweep.CHECK_V3
    a = sweep.allowed_tokens(tok, 151936)
    assert len(a) == 151649 and not set(a.tolist()) & set(tok.all_special_ids)      # v3's 151,649 tokens
    b = sweep.allowed_tokens(tok, 151936, 300)
    assert len(b) == 300 and (np.diff(b) > 0).all() and np.array_equal(b, sweep.allowed_tokens(tok, 151936, 300))
    c = sweep.panel_contexts(tok)
    assert len(c) == 12 and c[0]["read"] == ["slot"] and c[1]["read"] == ["slot", "end"] and c[1]["text"].endswith("[X] and I love it.")


def test_sweep_tiny_matches_brute_force(tiny, tok, tmp_path):
    sample, pset = ["L0:5", "L1:47", "L1:3"], tiny_pseudo()
    check = sweep.default_check(sample)
    meta = sweep.run_sweep(tiny, tok, tmp_path, "t", sample=sample, pseudo=pset, check=check, k=20, max_tokens=300,
                           contexts=[0, 1, 3], batch=64, log=lambda *_: None)
    assert meta["n_tokens"] == 300 and meta["gpu"] == "cpu" and len(meta["pseudo"]) == 4 + len(check)
    verify(tiny, tok, tmp_path, "t", [0, 1, 3], sample, pset, check, 20)
    verify_hubs(tmp_path, "t", [0, 1, 3], sweep.panel_contexts(tok), 2, 48, 20, 151936)
    # a pseudo-only run gives the same pseudo profiles and nothing else
    sweep.run_sweep(tiny, tok, tmp_path, "p", pseudo=pset, check=check, real=False, k=20, max_tokens=300, contexts=[1],
                    batch=64, log=lambda *_: None)
    with np.load(tmp_path / "vocab_sweep_t_c1.npz") as a, np.load(tmp_path / "vocab_sweep_p_c1.npz") as b:
        assert set(b.files) == {"slot_pseudo_g", "slot_pseudo_u", "end_pseudo_g", "end_pseudo_u", "tokens", "sample", "pseudo", "pseudo_layer"}
        for key in ("slot_pseudo_g", "end_pseudo_u", "pseudo", "pseudo_layer", "tokens"):
            assert np.array_equal(a[key], b[key]), key
    with pytest.raises(ValueError):
        sweep.hub_tokens(tmp_path, "p", log=lambda *_: None)


def test_sweep_tiny_rerun_keeps_other_contexts(tiny, tok, tmp_path):
    kw = dict(k=5, max_tokens=40, batch=16, log=lambda *_: None)
    sweep.run_sweep(tiny, tok, tmp_path, "r", contexts=[0, 2], **kw)
    meta = sweep.run_sweep(tiny, tok, tmp_path, "r", contexts=[2], **kw)
    assert [c["k"] for c in meta["contexts"]] == [0, 2]                       # context 0 kept, 2 replaced
    meta = sweep.run_sweep(tiny, tok, tmp_path, "r", contexts=[2], **dict(kw, max_tokens=50))
    assert [c["k"] for c in meta["contexts"]] == [2]                          # a different token set: not merged
    with np.load(tmp_path / "vocab_sweep_r_c2.npz") as f:                     # no sample or pseudo: empty columns
        assert f["slot_sample_z"].shape == (50, 0) and f["slot_pseudo_g"].shape == (50, 0) and f["pseudo_layer"].dtype == np.int64


def test_cli_sweep(tiny, tok, tmp_path, monkeypatch):
    from modelmapper import cli, model as M
    monkeypatch.setattr(M, "load", lambda name, dtype=None: (tiny, tok))
    (tmp_path / "s.json").write_text(json.dumps({"neurons": {"L1:2": "x", "L0:7": "y"}}))
    np.savez(tmp_path / "p.npz", **tiny_pseudo(), kind=np.array(["mix"] * 4), up_src=np.zeros(4, np.int64))
    out = tmp_path / "o"
    assert cli.main(["sweep", "c", "--out", str(out), "--sample", str(tmp_path / "s.json"), "--pseudo", str(tmp_path / "p.npz"),
                     "--max-tokens", "30", "--k", "5"]) == 0
    meta = json.loads((out / "vocab_sweep_c.json").read_text(encoding="utf-8"))
    assert len(meta["contexts"]) == 12 and meta["pseudo"][-2:] == ["CHECK:L1:2", "CHECK:L0:7"]
    with np.load(out / "hub_tokens_c.npz") as h:
        assert h["per_read"].shape == (22, 151936)                             # 12 contexts, 10 with an end reading


# ---- Qwen3-0.6B on the CPU ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = local(SMALL)
    return AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, device_map="cpu").eval(), AutoTokenizer.from_pretrained(p)


@pytest.mark.model
def test_sweep_small_matches_brute_force(small, tmp_path):
    from modelmapper import pseudo
    from modelmapper.weights import Weights
    m, tk = small
    sample = ["L3:17", "L14:2000", "L27:3071"]
    pset = pseudo.make_gate_set(Weights(folder=local(SMALL)), [3, 14, 27], n=4, log=lambda *_: None)
    check = sweep.default_check(sample)
    sweep.run_sweep(m, tk, tmp_path, "s", sample=sample, pseudo=pset, check=check, max_tokens=200, contexts=[0, 1],
                    batch=64, log=lambda *_: None)
    verify(m, tk, tmp_path, "s", [0, 1], sample, pset, check, 100)
    verify_hubs(tmp_path, "s", [0, 1], sweep.panel_contexts(tk), 28, 3072, 100, m.get_input_embeddings().weight.shape[0])


def test_layer_scoped_sweep_matches_full(tiny, tok, tmp_path):
    """layers=[0] keeps layer 0 only and never runs layer 1; its arrays equal the full sweep's layer-0 rows, and the
    exam reads the same favourites from either."""
    from modelmapper import exam
    kw = dict(k=20, max_tokens=200, contexts=[0, 1], batch=64, log=lambda *_: None)
    sweep.run_sweep(tiny, tok, tmp_path, "full", **kw)
    calls = []
    h = tiny.model.layers[1].register_forward_hook(lambda *a: calls.append(1))
    try:
        meta = sweep.run_sweep(tiny, tok, tmp_path, "l0", layers=[0], **kw)
    finally:
        h.remove()
    assert meta["layers"] == [0] and not calls and len(tiny.model.layers) == 2       # layer 1 skipped, model restored
    for c in (0, 1):
        with np.load(tmp_path / f"vocab_sweep_full_c{c}.npz") as a, np.load(tmp_path / f"vocab_sweep_l0_c{c}.npz") as b:
            for key in a.files:
                if key.endswith(("_i", "_v", "_u", "_g", "_mean", "_std", "_amean", "_astd")) and "sample" not in key and "pseudo" not in key:
                    assert b[key].shape[0] == 1 and np.array_equal(a[key][0], b[key][0]), key
    for t in ("full", "l0"):
        sweep.hub_tokens(tmp_path, t, 151936, tok=tok, log=lambda *_: None)
    with np.load(tmp_path / "hub_tokens_l0.npz") as h0:
        assert h0["layers"].tolist() == [0]
    for board in ("gate", "apos"):
        fa = exam.favourites(tmp_path, "full", ["L0:5", "L0:7"], board, hub_x=10.0)
        fb = exam.favourites(tmp_path, "l0", ["L0:5", "L0:7"], board, hub_x=10.0)
        assert fa[2] == fb[2], board
