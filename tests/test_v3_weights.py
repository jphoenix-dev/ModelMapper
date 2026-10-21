"""Weights-only v3 stages (pseudo-neurons, job type) on a tiny synthetic model written to a temp folder: no download."""
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

LAYERS, HIDDEN, INTER = 3, 16, 40


@pytest.fixture
def tiny(tmp_path):
    def make(model_type="qwen3", amplify_layer=None):
        g = torch.Generator().manual_seed(0)
        t = {}
        for L in range(LAYERS):
            p = f"model.layers.{L}."
            t[p + "mlp.gate_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
            t[p + "mlp.up_proj.weight"] = torch.randn(INTER, HIDDEN, generator=g)
            t[p + "mlp.down_proj.weight"] = torch.randn(HIDDEN, INTER, generator=g)
            t[p + "post_attention_layernorm.weight"] = torch.rand(HIDDEN, generator=g) + 0.5
            if L == amplify_layer:   # every neuron writes along its + side read direction: all + sides are amplifiers
                G, U = t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]
                gam = t[p + "post_attention_layernorm.weight"]
                n = lambda x: x / x.norm(dim=1, keepdim=True)
                t[p + "mlp.down_proj.weight"] = n(n(G * gam) + n(U * gam)).T.contiguous()
        d = tmp_path / model_type
        d.mkdir()
        save_file(t, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps({"model_type": model_type, "num_hidden_layers": LAYERS,
                                                   "hidden_size": HIDDEN, "intermediate_size": INTER}))
        from modelmapper.weights import Weights
        return Weights(folder=d)
    return make


def test_weights_reader(tiny):
    W = tiny()
    assert (W.n_layers, W.hidden, W.inter) == (LAYERS, HIDDEN, INTER)
    m = W.mlp(1)
    assert m["gate"].shape == (INTER, HIDDEN) and m["down"].shape == (HIDDEN, INTER) and not m["approximate"]
    assert not torch.equal(m["norm"], torch.ones(HIDDEN))
    o = tiny("olmo2").mlp(1)     # post-norm model: the MLP reads the residual directly
    assert torch.equal(o["norm"], torch.ones(HIDDEN)) and o["approximate"]


def test_pseudo_sets(tiny, tmp_path):
    from modelmapper import pseudo
    W = tiny()
    p1 = pseudo.make_gate_set(W, [0, 1, 1, 2], n=10, log=lambda *_: None)
    p2 = pseudo.make_up_set(W, p1)
    assert p1["gate"].shape == (10, HIDDEN) and list(p1["kind"]).count("mix") == 5
    for j in range(10):          # gate rows scaled to the layer's median norm; up rows are real rows of the layer
        L = int(p1["layer"][j])
        G = W.get(f"model.layers.{L}.mlp.gate_proj.weight"); U = W.get(f"model.layers.{L}.mlp.up_proj.weight")
        assert np.isclose(np.linalg.norm(p1["gate"][j]), float(G.norm(dim=1).median()), rtol=1e-5)
        assert np.array_equal(p1["up"][j], U[p1["up_src"][j]].numpy())
        assert np.isclose(np.linalg.norm(p2["up"][j]), float(U.norm(dim=1).median()), rtol=1e-5)
    assert np.array_equal(p2["gate"], p1["gate"]) and (p2["up_src"] == -1).all()
    assert p1["names"][0].startswith("P0:L") and p2["names"][0].startswith("Q0:L")
    again = pseudo.make_gate_set(W, [0, 1, 1, 2], n=10, log=lambda *_: None)
    assert np.array_equal(again["gate"], p1["gate"])                  # seeded: the same set every time
    path = pseudo.save(p1, tmp_path / "p.npz")
    assert np.array_equal(pseudo.load(path)["gate"], p1["gate"])
    with pytest.raises(FileExistsError):                              # fixed once written
        pseudo.save(p1, path)


def test_job_types(tiny):
    from modelmapper import jobtype
    jt = jobtype.job_types(tiny(amplify_layer=1), log=lambda *_: None)
    assert jt["c_pos"].shape == (LAYERS, INTER) and jt["t"].shape == (LAYERS,)
    assert np.allclose(jt["c_pos"][1], 1, atol=1e-5) and (jt["type_pos"][1] == 1).all()
    assert set(np.unique(jt["type_neg"])) <= {-1, 0, 1}
    text = jobtype.summary(jt, ["L1:3"])
    assert "L1:3" in text and "amplifier" in text
