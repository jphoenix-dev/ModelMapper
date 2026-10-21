"""The EXPERIMENTAL real-text checks end to end on Qwen3-0.6B (CPU) with a stand-in judge: the readings pass, the corpus
exam (plan, gather, exam) and graded simulation (pass, run). Also the research run's exactness check: the gather step
reads the same z_g at every positive as the readings boards stored. No API calls."""
import json

import numpy as np
import pytest

from modelmapper.experimental import corpus_exam as CE, realtext as RT, simulate as SM

TEXTS = ["The cat sat on the mat while the dog slept by the fire in the old farmhouse kitchen.",
         "Bitcoin and other cryptocurrencies trade on exchanges around the world every day and night.",
         "She played the violin, the cello and the piano at the concert hall last Sunday evening.",
         "Paris is the capital of France, and Berlin is the capital of Germany in central Europe.",
         "The football match ended in a draw after both teams scored twice in the second half.",
         "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen."] * 6


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    import torch

    from modelmapper.model import load
    d = tmp_path_factory.mktemp("rt")
    src = d / "docs.txt"; src.write_text("\n".join(f"{t} ({k})" for k, t in enumerate(TEXTS)), encoding="utf-8")
    m, tok = load("Qwen/Qwen3-0.6B", dtype=torch.float32)
    RT.readings(m, tok, RT.docs(src, tok, min_tokens=8), d, "toy", max_tokens=10**6, K=20, B=4, log=lambda *_: None)
    # a small fake-neuron set and its "exams", in the formats modelmapper.pseudo and the exam write
    L, W, H = len(m.model.layers), m.model.layers[0].mlp.gate_proj.out_features, m.config.hidden_size
    rng = np.random.default_rng(0)
    names = [f"P{k}:L{3 + 5 * k}:gate" for k in range(4)]
    np.savez(d / "pseudo.npz", names=np.array(names), layer=np.array([3 + 5 * k for k in range(4)]),
             gate=rng.normal(0, 0.02, (4, H)).astype(np.float32), up=rng.normal(0, 0.02, (4, H)).astype(np.float32))
    for b in RT.BOARDS:
        (d / f"intelligibility_toy_pq_{b}.json").write_text(json.dumps(
            {"_summary": {}, **{n: {"description": f"fake label {k}", "bin": "no-generalisation"} for k, n in enumerate(names)}}), encoding="utf-8")
    units = [f"L{l}:{i}" for l in (2, 9, 20) for i in (5, 77, 300)]
    res = {"neurons": {u: {b: {"label": f"words about topic {k} {b}", "bin": "intelligible" if k % 2 else "no-generalisation"}
                           for b in RT.BOARDS} for k, u in enumerate(units)}}
    (d / "run_toy_results.json").write_text(json.dumps(res), encoding="utf-8")
    return d, m, tok, L, W


@pytest.fixture
def fake_judge(monkeypatch):
    from modelmapper import judge

    def ask(state, qs):
        return {k: {"noul": (hash((state, q["instructions"])) % 1000) / 1000} for k, q in qs.items()}
    monkeypatch.setattr(judge, "ask", ask)
    monkeypatch.setattr(CE, "embed", lambda texts, model_name=None: np.random.default_rng(1).normal(size=(len(texts), 8)))


@pytest.mark.model
def test_readings_boards(setup):
    import torch
    d, m, tok, L, W = setup
    R = torch.load(d / "readings_toy.pt", weights_only=False)
    assert R["gate_v"].shape == (L, W, 20) and R["docs"] == len(TEXTS)
    z = np.load(d / "readings_toy_docs.npz"); assert z["lengths"].sum() == len(z["tokens"])
    v = R["gate_v"].float()
    assert bool((v[:, :, :-1] >= v[:, :, 1:]).all())                      # each board sorted, strongest first


@pytest.mark.model
def test_corpus_exam_end_to_end(setup, fake_judge):
    import torch
    d, m, tok, L, W = setup
    labels = RT.load_labels([d / "run_toy_results.json"]); pseudo = RT.load_pseudo(d / "pseudo.npz", {b: d / f"intelligibility_toy_pq_{b}.json" for b in RT.BOARDS})
    assert len(labels["gate"]) == 9 and len(pseudo["gate"]) == 4
    CE.plan(d, "toy", "t", labels["gate"], pseudo["gate"], tok, d, log=lambda *_: None)
    CE.gather(m, tok, d, "toy", "t", d / "pseudo.npz", d, B=4, log=lambda *_: None)
    # exactness, as in the research run: the gathered gate value at each positive is the readings board's value
    P, V = np.load(d / "corpus_plan_toy_t.npz"), np.load(d / "corpus_vals_toy_t.npz")
    R = torch.load(d / "readings_toy.pt", weights_only=False); gv, gc = R["gate_v"].float().numpy(), R["gate_c"].numpy()
    D = RT.Docs(d / "readings_toy_docs.npz"); code = {}
    for (l, i, r) in np.ndindex(*gv.shape):
        if gv[l, i, r] > -1e4:
            dd, pp = divmod(int(gc[l, i, r]), RT.CTX); code[(l, i, int(D.st[dd] + pp))] = gv[l, i, r]
    pos = P["role"] == 0
    got = [(V["zg"][k], code[(int(P["layer"][k]), int(P["neuron"][k]), int(P["flat"][k]))]) for k in np.where(pos)[0]]
    assert got and max(abs(a - b) for a, b in got) < 0.02                 # float16 boards
    summ = CE.exam(tok, d, "t", "t", ["toy"], d, workers=4, log=lambda *_: None)
    e = json.loads((d / "corpus_exam_t.json").read_text(encoding="utf-8"))
    real = [v for k, v in e.items() if k.startswith("L")]
    assert summ["units_judged"] == len(real) + summ["pseudo"]["n"] and real
    assert all(set(v) >= {"A", "B", "C", "token_locked", "keyword_auc_A", "items", "ratings"} for v in real)
    assert all(v["ratings"]["own"] and len(v["control_units"]) == 5 for v in real)


@pytest.mark.model
def test_simulation_end_to_end(setup, fake_judge):
    d, m, tok, L, W = setup
    labels = RT.load_labels([d]); pseudo = RT.load_pseudo(d / "pseudo.npz", {b: d / f"intelligibility_toy_pq_{b}.json" for b in RT.BOARDS})
    if not (d / "corpus_units_t.json").exists():
        CE.plan(d, "toy", "t", labels["gate"], pseudo["gate"], tok, d, log=lambda *_: None)
        CE.gather(m, tok, d, "toy", "t", d / "pseudo.npz", d, B=4, log=lambda *_: None)
    SM.sim_pass(m, tok, d, "toy", "s", d / "pseudo.npz", pseudo["gate"], d, n_docs=10, n_pos=200, B=4, min_len=16, log=lambda *_: None)
    g = np.load(d / "sim_pass_s_g.npy", mmap_mode="r"); n = len(np.load(d / "sim_pass_s.npz")["flat"])
    assert g.shape == (n, L, W) and 0 < n <= 200                              # as many as the short documents hold
    out = SM.sim_run(tok, d, "s", "s", labels, pseudo, "t", d, workers=4, log=lambda *_: None)
    assert set(out) == set(RT.BOARDS)
    r = json.loads((d / "sim_run_s_gate.json").read_text(encoding="utf-8"))
    real = [v for k, v in r.items() if k.startswith("L")]
    assert len(real) == 9 and all(v["rho_own"] is not None and len(v["value"]) == len(v["text"]) for v in real)
    assert r["_summary"]["pseudo"]["n"] == 4
