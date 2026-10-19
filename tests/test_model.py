"""Tests on the real Qwen3-0.6B (downloaded once from the Hugging Face hub, ~1.5 GB). About 10 minutes on a CPU.
Skip them with  pytest -m "not model".  No API calls: the question sets are the answer key's pinned ones."""
import json
import math

import pytest

MODEL = "Qwen/Qwen3-0.6B"
pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def m():
    from modelmapper import load_model
    return load_model(MODEL)


@pytest.fixture(scope="module")
def key():
    from modelmapper import answerkey
    return answerkey.load_key(MODEL)


def test_measurer_shape(m):
    assert (m.L, m.W) == (28, 3072) and m.model_name == MODEL


def test_logodds_exact_and_repeatable(m):
    """Exact log-odds (METHOD.md step 5): float64, the same on a repeat, and knocking a neuron out changes it."""
    from modelmapper.model import steer
    q = m.tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? Answer with just the name."}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pair = [(q, m.tok.encode("Paris"))]
    a, b = m.batch_logodds(pair)[0], m.batch_logodds(pair)[0]
    assert isinstance(a, float) and math.isfinite(a) and a == b
    with steer(m.model, [(l, n, 0.0) for l in range(20, 28) for n in range(0, 3072, 3)]):
        c = m.batch_logodds(pair)[0]
    assert c != a


def test_answer_key_tier1(m, key):
    """Every marker found again near the key's rank (the key's own pass rule)."""
    from modelmapper import answerkey
    r = answerkey.tier1(m, key, log=lambda s: None)
    assert r["passed"], json.dumps(r["rows"], indent=1)
    assert r["found"] == r["markers"] == 3


def test_map_concept_pinned(m, key, tmp_path):
    """The whole map_concept on the key's pinned Bitcoin questions, offline: the marker near the top, the controls
    near 0, and a record in the published format with provenance and the guards block."""
    from modelmapper import map_concept
    mk = next(x for x in key["markers"] if x["concept"] == "Bitcoin")
    r = map_concept("Bitcoin", spec=json.loads(json.dumps(mk["spec"])), measurer=m, out=tmp_path, log=lambda s: None)
    assert r.status == "ok" and r.recall >= 5
    top = [t.neuron for t in r.top]
    assert "L23:2392" in top[:5], top[:10]
    assert len(r.top) == 30 and len(r.candidates) == 200
    assert all(abs(v) < 0.2 for v in r.nulls.values()), r.nulls
    assert r.k_curve[50] > r.random_k_curve[50]
    p = r.record["v2"]["provenance"]
    for f in ("schema", "model", "dtype", "torch", "transformers", "gpu", "package", "device", "cuda", "numpy", "os"):
        assert f in p
    assert p["schema"] == "loop_v2.0" and p["package"] == "modelmapper 0.1.0"
    assert r.record["guards"]["bar"] == r.candidates[29].drop
    assert (tmp_path / "Qwen3-0.6B" / "bitcoin.json").exists() and (tmp_path / "Qwen3-0.6B" / "arrays" / "bitcoin.npz").exists()


def test_steer_deterministic(m):
    from modelmapper import steer
    kw = dict(prompts=["Tell me a fun fact."], max_new=12, measurer=m)
    a = steer(["L23:2392=x8"], **kw)
    b = steer(["L23:2392=x8"], baseline=False, **kw)
    assert a["steered"] == b["steered"] and a["steered"] != a["baseline"]
    assert a["units"][0][:2] == (23, 2392) and a["units"][0][2] > 0
    c = steer(["L23:2392=0"], **kw)                              # holding a neuron at 0 only during the reply
    assert isinstance(c["steered"][0], str)
