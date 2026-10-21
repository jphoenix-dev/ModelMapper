"""`modelmapper check --labels` (modelmapper/labelcheck.py, the label steps in selfcheck.py and exam.rejudge): the pass
rules, the shipped references, re-judging fixed exams (against run_exam on the synthetic sweep of test_v3_exam), the
falsifier's shrunk quick sweep (against the full sweep, on the tiny Qwen3 of test_v3_sweep). Fast unless marked:
`model` runs sweep1 on the real Qwen3-0.6B; `repro` calls the judge (exam1) or the labeller and judge (falsifier)."""
import json
import os

import numpy as np
import pytest

from modelmapper import exam as E
from modelmapper import labelcheck as LC
from modelmapper import selfcheck

from test_v3_exam import POOL, FakeJudge, FakeLabeller, decode, sweep  # noqa: F401  (sweep: a fixture)
from test_v3_sweep import tiny, tiny_pseudo, tok  # noqa: F401  (tiny, tok: fixtures)

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B")


# ---- the shipped references ----
@pytest.mark.parametrize("model", MODELS)
def test_sweep_reference_loads(model):
    ref = LC.load_sweep_ref(model)
    assert ref["sample"] == LC.MARKERS[model] and len(ref["tokens"]) == LC.SWEEP1_TOKENS
    assert (np.diff(ref["tokens"]) > 0).all()
    for c in LC.SWEEP1_CONTEXTS:
        for p in ("z", "u"):
            a = ref[f"c{c}_{p}"]
            assert a.shape == (LC.SWEEP1_TOKENS, 6) and a.dtype == np.float16 and np.isfinite(a).all() and (a.std(0) > 0).all()
    assert {"model", "device", "dtype"} <= set(ref["meta"]) and ref["meta"]["model"] == model
    assert LC.ref_path(model, "sweep1").stat().st_size < 100_000


def test_both_slices_are_the_same_tokens():
    """Qwen3-0.6B and Qwen3-8B share a tokenizer, so the seeded slice is the same 512 tokens"""
    assert np.array_equal(LC.load_sweep_ref(MODELS[0])["tokens"], LC.load_sweep_ref(MODELS[1])["tokens"])


@pytest.mark.parametrize("model", MODELS)
def test_exam_reference_loads(model):
    ref = LC.load_exam_ref(model)
    assert len(ref["neurons"]) == 20 and ref["board"] == "gate" and ref["meta"]["model"] == model
    for s, r in ref["neurons"].items():
        assert len(r["draws"]) == 3 and r["ours"]["bin"] and 1 <= r["ours"]["n_draws"] <= 3
        for d in r["draws"]:
            assert len(d["candidates"]) == len(d["label"]) and 0 < sum(d["label"]) <= 30 and sum(d["label"]) < len(d["label"])
        assert len(r["controls"]) == 5 and set(r["control_descriptions"]) == set(r["controls"])
        assert all(x in ref["shown"] and 0 < len(ref["shown"][x]) <= E.N_SHOW for x in [s] + r["controls"])   # fewer after hub cuts
        assert len(r["descriptions"]) == 3
    assert LC.ref_path(model, "exam1").stat().st_size < 200_000


def test_missing_reference_is_none(tmp_path):
    assert LC.load_sweep_ref("Org/Other-1B") is None and LC.load_exam_ref("Org/Other-1B") is None
    assert LC.full_name("Qwen3-8B") == "Qwen/Qwen3-8B" and LC.full_name("Org/Other-1B") == "Org/Other-1B"


# ---- sweep1's rule ----
def _ref(rng, n=LC.SWEEP1_TOKENS):
    r = {"tokens": np.arange(n), "sample": ["L0:1", "L1:2"]}
    for c in LC.SWEEP1_CONTEXTS:
        for p in ("z", "u"):
            r[f"c{c}_{p}"] = rng.standard_normal((n, 2)).astype(np.float16)
    return r


def _noisy(ref, rng, sd):
    g = dict(ref)
    for c in LC.SWEEP1_CONTEXTS:
        for p in ("z", "u"):
            a = ref[f"c{c}_{p}"].astype(np.float32)
            g[f"c{c}_{p}"] = (a + sd * a.std(0) * rng.standard_normal(a.shape)).astype(np.float16)
    return g


def test_sweep1_rule():
    rng = np.random.default_rng(0); ref = _ref(rng)
    same = LC.compare_profiles(ref, ref)
    assert same["max_abs"] == 0 and same["corr_min"] == pytest.approx(1.0) and same["n"] == 8 and LC.sweep1_status(same) == "PASS"
    small = LC.compare_profiles(ref, _noisy(ref, rng, 0.03))          # bf16-sized noise: inside the band
    assert LC.sweep1_status(small) == "PASS" and small["max_sd"] < LC.SWEEP1_SD
    mid = LC.compare_profiles(ref, _noisy(ref, rng, 0.12))            # correlation ~0.99: outside the band, same neurons
    assert LC.SWEEP1_WARN_CORR <= mid["corr_min"] < LC.SWEEP1_CORR and LC.sweep1_status(mid) == "WARN"
    other = LC.compare_profiles(ref, _ref(np.random.default_rng(1)))   # unrelated profiles
    assert LC.sweep1_status(other) == "FAIL"
    shifted = dict(ref, tokens=ref["tokens"] + 1)
    assert LC.sweep1_status(LC.compare_profiles(ref, shifted)) == "FAIL" and "token slice" in LC.compare_profiles(ref, shifted)["error"]
    assert "marker" in LC.compare_profiles(ref, dict(ref, sample=["L0:1", "L9:9"]))["error"]
    assert "max abs difference 0.000" in LC.sweep1_value(same)


# ---- exam1's rule and re-judging ----
def test_exam1_rule():
    assert LC.exam1_status(18, 20) == "PASS" and LC.exam1_status(17, 20) == "WARN" and LC.exam1_status(20, 20) == "PASS"
    ref = {"neurons": {"L0:1": {"ours": {"bin": "intelligible", "auc_own": 0.9}}, "L0:2": {"ours": {"bin": "list-only", "auc_own": 0.6}}}}
    res = {"L0:1": {"bin": "intelligible", "auc_own": 0.91}, "L0:2": {"bin": "no-generalisation", "auc_own": 0.5}}
    same, n, rows = LC.exam1_compare(ref, res)
    assert (same, n) == (1, 2) and rows[0].startswith("L0:2: no-generalisation (ours list-only")


def test_rejudge_matches_run_exam(sweep, tmp_path):  # noqa: F811
    """fixed_inputs + rejudge give run_exam's verdicts, AUCs and draws, with the same judge and no labeller or sweep"""
    cfg = E.ExamConfig(workers=4)
    res = E.run_exam(sweep, "t", POOL, tmp_path / "intelligibility_x.json", decode=decode, readings_dir=tmp_path, config=cfg,
                     labeller=FakeLabeller(), judge=FakeJudge(), log=lambda *_: None)
    judged = POOL[2:24]
    fx = E.fixed_inputs(sweep, "t", POOL, judged, tmp_path / "intelligibility_x.json", config=cfg, decode=decode)
    json.dumps(fx)                                                     # plain data: ships as JSON
    again = E.rejudge(json.loads(json.dumps(fx)), judge=FakeJudge(), config=cfg)
    assert list(fx["neurons"]) == judged and again["_summary"]["n"] == len(judged)
    for s in judged:
        a, b = again[s], res[s]
        for k in ("bin", "auc_own", "auc_control", "list_auc_own", "list_auc_control", "kind", "stable"):
            assert a[k] == b[k], (s, k)
        assert [d["test"] for d in a["draws"]] == [d["test"] for d in b["draws"]]
        assert fx["neurons"][s]["ours"]["bin"] == b["bin"]
    assert any(len(res[s]["draws"]) > 1 for s in judged)                # the adaptive draws were exercised
    same, n, _ = LC.exam1_compare(fx, again)
    assert same == n


def test_fixed_inputs_refuse_another_pool(sweep, tmp_path):  # noqa: F811
    cfg = E.ExamConfig(workers=4)
    E.run_exam(sweep, "t", POOL, tmp_path / "intelligibility_y.json", decode=decode, readings_dir=tmp_path, config=cfg,
               labeller=FakeLabeller(), judge=FakeJudge(), log=lambda *_: None)
    with pytest.raises(ValueError):
        E.fixed_inputs(sweep, "t", POOL[::-1], POOL[:2], tmp_path / "intelligibility_y.json", config=cfg, decode=decode)


def test_rejudge_needs_every_draw():
    fx = {"neurons": {"L0:1": {"layer": 0, "reach": None, "controls": ["L0:2"], "descriptions": [{"description": "x", "kind": "form"}],
                               "control_descriptions": {"L0:2": "y"}, "draws": []}}, "shown": {"L0:1": ["a"], "L0:2": ["b"]}}
    with pytest.raises(ValueError):
        E.rejudge(fx, judge=FakeJudge(), config=E.ExamConfig(workers=1))


# ---- the falsifier ----
def _summ(fakes=(0, 1, 0), ctrl=0.5):
    return {"n": 20, "n_tokens": 3000, "ctrl_auc_median": ctrl, "spend_usd": {"labeller": 0.2, "judge": 0.05},
            "fakes": {b: {"n": 20, "verified": f + 1, "verified_and_generative": f} for b, f in zip(("gate", "apos", "aneg"), fakes)},
            "real": {"gate": {"n": 20, "verified": 0.4, "verified_and_generative": 0.3}}}


def test_falsifier_rule():
    assert LC.falsifier_status(_summ()) == "PASS"
    assert LC.falsifier_status(_summ(ctrl=0.40)) == LC.falsifier_status(_summ(ctrl=0.60)) == "PASS"
    assert LC.falsifier_status(_summ((2, 0, 0))) == "WARN" and LC.falsifier_status(_summ((0, 3, 0))) == "FAIL"
    assert LC.falsifier_status(_summ(ctrl=0.65)) == "WARN" and LC.falsifier_status(_summ(ctrl=0.72)) == "FAIL"
    assert LC.falsifier_status(_summ(ctrl=None)) == "FAIL" and LC.falsifier_status(dict(_summ(), fakes={})) == "FAIL"
    v = LC.falsifier_value(_summ())
    assert "gate 0/20, apos 1/20, aneg 0/20" in v and "control AUC median 0.5" in v and "reported only" in v


def test_summarise_falsifier(tmp_path):
    for b, vals in (("gate", [0.4, 0.5]), ("apos", [0.6]), ("aneg", [0.45, None])):
        recs = {f"L0:{j}": {"auc_control": v} for j, v in enumerate(vals)}
        (tmp_path / f"intelligibility_fz_c0_{b}.json").write_text(json.dumps({"_summary": {}, **recs}))
    (tmp_path / "intelligibility_fz_pq_gate.json").write_text(json.dumps({"_summary": {}, "Q0": {"auc_control": 0.99}}))
    res = {"pseudo": {"gate": {"n": 20, "verified": 0.1, "verified_and_generative": 0.05}},
           "summary": {"gate": {"n": 20, "verified": 0.4, "verified_and_generative": 0.25}}, "spend_usd": {"labeller": 0.1, "judge": 0.02}}
    s = LC.summarise_falsifier(res, tmp_path, 20)
    assert s["fakes"]["gate"] == {"n": 20, "verified": 2, "verified_and_generative": 1}
    assert s["ctrl_auc_median"] == 0.475 and s["ctrl_auc_median_by_part"] == {"gate": 0.45, "apos": 0.6, "aneg": 0.45}
    assert s["real"]["gate"]["verified"] == 0.4


def test_falsifier_neurons_are_seeded():
    a = LC.falsifier_neurons(28, 3072)
    assert a == LC.falsifier_neurons(28, 3072) and len(set(a)) == 20 and a != LC.falsifier_neurons(28, 3072, seed=1)
    assert all(0 <= int(s[1:].split(":")[0]) < 28 and 0 <= int(s.split(":")[1]) < 3072 for s in a)


def test_quick_sweep_shrinks_but_keeps_what_the_exam_reads(tiny, tok, tmp_path):  # noqa: F811
    """the shrunk files give the kept neurons' favourites, the statistics, the pseudo profiles and the hub tokens of the
    full sweep"""
    from modelmapper import sweep as SW
    keep, pset = [(0, 5), (1, 3), (1, 40)], tiny_pseudo()
    LC.quick_sweep(tiny, tok, tmp_path / "q", keep, pset, n_tokens=60, log=lambda *_: None)
    SW.run_sweep(tiny, tok, tmp_path / "f", LC.TAG, pseudo=pset, max_tokens=60, log=lambda *_: None)
    SW.hub_tokens(tmp_path / "f", LC.TAG, vocab_size=tiny.get_input_embeddings().weight.shape[0], log=lambda *_: None)
    with np.load(tmp_path / "q" / f"hub_tokens_{LC.TAG}.npz") as a, np.load(tmp_path / "f" / f"hub_tokens_{LC.TAG}.npz") as b:
        for k in ("per_read", "per_layer", "readings"):
            assert np.array_equal(a[k], b[k]), k
    mq = json.loads((tmp_path / "q" / f"vocab_sweep_{LC.TAG}.json").read_text(encoding="utf-8"))
    assert len(mq["contexts"]) == 12 and mq["intermediate_size"] == 48
    for c in mq["contexts"]:
        with np.load(tmp_path / "q" / f"vocab_sweep_{LC.TAG}_c{c['k']}.npz") as a, np.load(tmp_path / "f" / f"vocab_sweep_{LC.TAG}_c{c['k']}.npz") as b:
            assert set(a.files) == set(b.files)
            for k in a.files:
                if a[k].ndim == 3:
                    ul, ui = zip(*keep)
                    assert np.array_equal(a[k][list(ul), list(ui)], b[k][list(ul), list(ui)]), k
                    assert not a[k][0, 0].any()                          # not kept: zeros
                else:
                    assert np.array_equal(a[k], b[k]), k
    names = [f"L{l}:{i}" for l, i in keep]
    fa, fb = E.favourites(tmp_path / "q", LC.TAG, names)[2], E.favourites(tmp_path / "f", LC.TAG, names)[2]
    assert all(fa[s]["rank"] == fb[s]["rank"] for s in names)


# ---- the label steps in the check ----
class _Rep(selfcheck.Report):
    def __init__(self, out):
        super().__init__(out); out.mkdir(parents=True, exist_ok=True)


def test_label_steps_without_keys_or_reference(tmp_path):
    rep = _Rep(tmp_path)
    selfcheck.label_steps(rep, list(selfcheck.LABEL_STEPS), "Org/Other-1B", lambda: pytest.fail("no model needed"), False, False)
    got = {i["id"]: i for i in rep.items}
    assert got["labelsetup"]["status"] == "WARN" and "untested" in got["labelsetup"]["value"]
    assert got["labelsetup"]["detail"] == selfcheck.UNCALIBRATED
    assert got["sweep1"]["status"] == "SKIPPED" and got["sweep1"]["value"] == "no reference for Org/Other-1B"
    assert got["sweep1"]["detail"] == "labelling uncalibrated for this model: internal controls only"
    assert got["exam1"]["status"] == got["falsifier"]["status"] == "SKIPPED"
    assert got["exam1"]["value"] == "no reference for Org/Other-1B"                 # the reference first, then the key
    assert "no labeller key or judge set up" in got["falsifier"]["value"]


def test_label_steps_tested_combination(tmp_path, monkeypatch):
    from modelmapper import judge, labeller
    monkeypatch.setattr(labeller, "MODEL", "gpt-6-luna"); monkeypatch.setattr(judge, "BACKEND", "jev")
    rep = _Rep(tmp_path)
    selfcheck.label_steps(rep, ["exam1"], "Qwen/Qwen3-8B", None, True, False)
    got = {i["id"]: i for i in rep.items}
    assert got["labelsetup"]["status"] == "PASS" and got["labelsetup"]["value"].endswith("the tested combination")
    assert got["labelsetup"]["detail"] == "" and got["exam1"]["status"] == "SKIPPED" and "no judge" in got["exam1"]["value"]
    monkeypatch.setattr(labeller, "MODEL", "gpt-other")
    assert not LC.combination("Qwen/Qwen3-8B", labeller.MODEL, "jev")[0]
    assert not LC.combination("Qwen/Qwen3-8B", "gpt-6-luna", "llm")[0] and not LC.combination("Org/X", "gpt-6-luna", "jev")[0]


def test_exam1_step_with_a_fake_judge(tmp_path, monkeypatch):
    """exam1 on the shipped Qwen3-8B extract with a judge that answers our stored verdicts' way is a PASS or WARN, never
    a crash; with a judge that fails every call the bins say so and the step WARNs"""
    class Broken:
        def cost(self):
            return 0.0

        def rate(self, *a):
            raise RuntimeError("down")
        rate_examples = rate
    import modelmapper.judge as J
    monkeypatch.setattr(J, "rate", Broken().rate); monkeypatch.setattr(J, "rate_examples", Broken().rate)
    rep = _Rep(tmp_path)
    selfcheck.step_exam1(rep, "Qwen/Qwen3-8B")
    it = rep.items[-1]
    assert it["id"] == "exam1" and it["status"] == "WARN" and it["value"].startswith("0/20") and "failed judge calls" in it["value"]


def test_run_selects_the_label_steps(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(selfcheck, "environment", lambda rep, apis: [rep.add("labeller", "labeller key", "WARN", "not found"),
                                                                     rep.add("judge", "judge", "SKIPPED", "no key")])
    monkeypatch.setattr(selfcheck, "label_steps", lambda rep, steps, model, get_model, lab_ok, judge_ok: seen.update(
        steps=steps, model=model, lab_ok=lab_ok, judge_ok=judge_ok))
    selfcheck.run(labels=True, out=tmp_path / "a")
    assert seen == {"steps": ["sweep1", "exam1", "falsifier"], "model": "Qwen/Qwen3-0.6B", "lab_ok": False, "judge_ok": False}
    seen.clear(); selfcheck.run(only=["exam1"], model="Qwen3-8B", out=tmp_path / "b")
    assert seen["steps"] == ["exam1"] and seen["model"] == "Qwen/Qwen3-8B"
    seen.clear(); selfcheck.run(labels=True, quick=True, out=tmp_path / "c")
    assert not seen
    with pytest.raises(SystemExit):
        selfcheck.main(["--only", "sweep2"])


# ---- Qwen3-0.6B on the CPU ----
@pytest.mark.model
def test_sweep1_on_qwen3_06b(tmp_path):
    from modelmapper.model import load
    rep = _Rep(tmp_path)
    selfcheck.step_sweep1(rep, "Qwen/Qwen3-0.6B", lambda: load("Qwen/Qwen3-0.6B"))
    it = rep.items[-1]
    assert it["id"] == "sweep1" and it["status"] == "PASS", it
    cmp = json.loads((tmp_path / "sweep1.json").read_text(encoding="utf-8"))
    assert cmp["corr_min"] >= LC.SWEEP1_CORR and cmp["n"] == 24


# ---- live: the judge (exam1, ~$0.03 each) and the labeller + judge (falsifier, ~$0.30) ----
def _jev_key():
    from modelmapper import judge
    return bool(os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or judge.KEY_FILE.exists())


@pytest.mark.repro
@pytest.mark.skipif(not _jev_key(), reason="no judge (Jev) key")
@pytest.mark.parametrize("model", MODELS)
def test_exam1_live(model, tmp_path):
    rep = _Rep(tmp_path)
    selfcheck.step_exam1(rep, model)
    it = rep.items[-1]
    print(it)
    assert it["status"] == "PASS", it


@pytest.mark.repro
@pytest.mark.model
@pytest.mark.skipif(not (_jev_key() and os.environ.get("MODEL_MAPPER_LIVE_FALSIFIER")),
                    reason="set MODEL_MAPPER_LIVE_FALSIFIER=1 (about $0.30 of labeller and judge calls) and a Jev key")
def test_falsifier_live_qwen3_06b(tmp_path):
    from modelmapper.model import load
    rep = _Rep(tmp_path)
    selfcheck.step_falsifier(rep, "Qwen/Qwen3-0.6B", lambda: load("Qwen/Qwen3-0.6B"))
    it = rep.items[-1]
    print(it)
    assert it["status"] == "PASS", it
