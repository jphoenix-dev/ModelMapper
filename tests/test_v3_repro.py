"""The ported v3 stages give the research run's outputs. Needs Qwen3-8B's weights (MODEL_MAPPER_MODELS) and the
research run's files in MODEL_MAPPER_V3_REF (the folder of the v3 run's outputs); skipped otherwise."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

REF = Path(os.environ.get("MODEL_MAPPER_V3_REF", ""))
pytestmark = [pytest.mark.repro,
              pytest.mark.skipif(not (REF.name and (REF / "pseudo_v1.npz").exists()), reason="set MODEL_MAPPER_V3_REF to the research run's folder")]


@pytest.fixture(scope="module")
def W():
    from modelmapper.weights import Weights
    w = Weights("Qwen/Qwen3-8B")
    if not (w.dir / "config.json").exists():
        pytest.skip("no Qwen3-8B weights")
    return w


def test_pseudo_sets_match_research(W):
    from modelmapper import pseudo
    sample = json.loads((REF / "sample_v1.json").read_text())["neurons"]
    p1 = pseudo.make_gate_set(W, [int(s[1:].split(":")[0]) for s in sample], log=lambda *_: None)
    p2 = pseudo.make_up_set(W, p1)
    for got, ref in ((p1, np.load(REF / "pseudo_v1.npz")), (p2, np.load(REF / "pseudo_v2.npz"))):
        for k in pseudo.FIELDS:
            assert np.array_equal(got[k], ref[k]), k


def _headers(npz):
    """{key: (dtype, shape)} of an open np.load(...) archive, read from the array headers only (no data loaded)."""
    out = {}
    for name in npz.files:
        with npz.zip.open(name + ".npy") as f:
            v = np.lib.format.read_magic(f)
            shape, _, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        out[name] = (dt, shape)
    return out


@pytest.mark.model
def test_sweep_format_matches_research(tmp_path):
    """The sweep's files have the research v2 sweep's keys, with the same dtype per key and shapes equal up to model
    size (layers, neurons, tokens, sample and pseudo-neuron counts); the .json and hub-token files too. The research
    files are Qwen3-8B's; this side runs a small Qwen3-0.6B sweep on the CPU (8B is never loaded)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from modelmapper import pseudo, sweep
    from modelmapper.settings import model_path
    from modelmapper.weights import Weights
    files = sorted(REF.glob("vocab_sweep_v2_c*.npz"))
    if not files or not (REF / "vocab_sweep_v2.json").exists():
        pytest.skip("no research v2 sweep in MODEL_MAPPER_V3_REF")
    p = model_path("Qwen/Qwen3-0.6B")
    if not os.path.isdir(p):
        pytest.skip("no local Qwen3-0.6B (MODEL_MAPPER_MODELS)")
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, device_map="cpu").eval(); tok = AutoTokenizer.from_pretrained(p)
    sample = ["L3:17", "L14:2000", "L27:3071"]
    pset = pseudo.make_gate_set(Weights(folder=p), [3, 14, 27], n=4, log=lambda *_: None)
    sweep.run_sweep(m, tok, tmp_path, "f", sample=sample, pseudo=pset, check=sweep.default_check(sample), max_tokens=120,
                    contexts=[0, 1], batch=64, log=lambda *_: None)
    sweep.hub_tokens(tmp_path, "f", m.get_input_embeddings().weight.shape[0], tok=tok, log=lambda *_: None)
    for ck in (0, 1):                                    # c0 reads the slot only, c1 the slot and the end
        with np.load(REF / f"vocab_sweep_v2_c{ck}.npz") as ref, np.load(tmp_path / f"vocab_sweep_f_c{ck}.npz") as got:
            hr, hg = _headers(ref), _headers(got)
        assert set(hg) == set(hr), set(hg) ^ set(hr)
        for key, (dt, shape) in hr.items():
            gdt, gshape = hg[key]
            assert (gdt.kind, gdt.itemsize) == (dt.kind, dt.itemsize) or (dt.kind == gdt.kind == "U"), (key, dt, gdt)
            assert len(gshape) == len(shape), key
            if key.endswith(("_i", "_v", "_u", "_g")) and len(shape) == 3:
                assert gshape[2] == shape[2] == 100, key              # k
    ref_meta = json.loads((REF / "vocab_sweep_v2.json").read_text(encoding="utf-8"))
    meta = json.loads((tmp_path / "vocab_sweep_f.json").read_text(encoding="utf-8"))
    assert set(meta) == set(ref_meta) and set(meta["contexts"][1]) == set(ref_meta["contexts"][1])
    assert [c["positions"] for c in meta["contexts"]] == [c["positions"] for c in ref_meta["contexts"][:2]]   # same tokenizer
    assert [c["text"] for c in meta["contexts"]] == [c["text"] for c in ref_meta["contexts"][:2]]
    hub_ref = REF / "hub_tokens_v2.npz"
    if hub_ref.exists():
        with np.load(hub_ref) as ref, np.load(tmp_path / "hub_tokens_f.npz") as got:
            hr, hg = _headers(ref), _headers(got)
        assert set(hg) == set(hr)
        for key in ("per_read", "per_layer"):
            assert hg[key][0] == hr[key][0] and hg[key][1][1] == hr[key][1][1] == 151936, key   # int32, full vocabulary


def test_job_types_match_research(W):
    from modelmapper import jobtype
    jt, ref = jobtype.job_types(W, log=lambda *_: None), np.load(REF / "jobtype_v3.npz")
    assert not jt["approximate"]
    for k in ("c_pos", "c_neg", "uw", "null_pos", "null_neg", "t", "type_pos", "type_neg"):
        assert np.array_equal(jt[k], ref[k]), k
