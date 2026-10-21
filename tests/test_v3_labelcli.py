"""`modelmapper label`, the front door: scope, controls, cached-sweep matching, the plan, and refusing to start
without confirmation. No model and no API calls."""
import json

import pytest

from modelmapper import labelcli as LC

DIMS = {"layers": 4, "neurons": 10, "hidden": 8, "type": "qwen3", "vocab": 100}


def test_scope_forms():
    assert LC.scope(DIMS, neurons=["l1:3", "L1:3", "2:9"]) == ["L1:3", "L2:9"]
    assert LC.scope(DIMS, layers="1-2") == [f"L{L}:{i}" for L in (1, 2) for i in range(10)]
    assert len(LC.scope(DIMS, all_=True)) == 40
    s = LC.scope(DIMS, sample=6, seed=1)
    assert len(s) == len(set(s)) == 6 and {int(u[1:].split(":")[0]) for u in s} == {0, 1, 2, 3}
    assert s == LC.scope(DIMS, sample=6, seed=1)
    for bad in (dict(neurons=["L9:1"]), dict(layers="3-5"), dict(), dict(neurons=["L1:1"], all_=True)):
        with pytest.raises(ValueError):
            LC.scope(DIMS, **bad)


def test_controls_only_for_small_scopes():
    dims = dict(DIMS, neurons=1000)
    c = LC.controls(["L1:3", "L2:9"], dims)
    assert len(c) == LC.CONTROLS_MIN - 2 and not {"L1:3", "L2:9"} & set(c)
    assert len({int(u[1:].split(":")[0]) for u in c}) == 4 and c == LC.controls(["L1:3", "L2:9"], dims)   # all layers
    assert len(LC.controls([f"L0:{i}" for i in range(60)], dims)) == 940              # the pool is filled to 1,000
    assert LC.controls([f"L0:{i}" for i in range(1000)], dims) == []


def test_find_sweep_needs_a_complete_covering_real_sweep(tmp_path):
    def sweep(tag, layers=None, n=12, tokens=500, real=True, hubs=True):
        meta = {"n_tokens": tokens, "contexts": [{"k": k} for k in range(n)], "real": real, **({"layers": layers} if layers else {})}
        (tmp_path / f"vocab_sweep_{tag}.json").write_text(json.dumps(meta))
        if hubs:
            (tmp_path / f"hub_tokens_{tag}.npz").write_bytes(b"")
    sweep("partial", n=5); sweep("pseudo", real=False); sweep("nohubs", hubs=False); sweep("few", layers=[1])
    assert LC.find_sweep(tmp_path, [1, 2], 500) is None
    sweep("cover", layers=[1, 2, 3])                       # layer-scoped: not used by label (it sweeps every layer)
    assert LC.find_sweep(tmp_path, [1, 2], 500) is None
    sweep("cover")
    assert LC.find_sweep(tmp_path, [1, 2], 500) == "cover" and LC.find_sweep(tmp_path, [1, 2], 400) is None


def test_estimate_scales():
    big = LC.estimate({"layers": 36, "neurons": 12288, "hidden": 4096}, 36, 35, 151649, 1000, 0, 0, "cuda")
    assert abs(big["sweep_seconds_on_reference_gpu"] - LC.REF_SWEEP["seconds"]) < 60 and abs(big["sweep_disk_gb"] - 17) < 0.5
    assert abs(big["api_usd"]["total"] - 7.7) < 0.2                            # v3: ~$7.29 + the pilot for 1,000 neurons
    small = LC.estimate({"layers": 36, "neurons": 12288, "hidden": 4096}, 2, 5, 151649, 10, 50, 0, "cpu")
    assert small["sweep_seconds_on_reference_gpu"] < big["sweep_seconds_on_reference_gpu"] / 5 and small["sweep_disk_gb"] < 1
    # the range: the high end (Qwen3-0.6B's judge cost, measured on the laptop trial) is above the low end
    assert big["api_usd_high"]["judge"] > 3 * big["api_usd"]["judge"] and big["api_usd_high"]["labeller"] == big["api_usd"]["labeller"]
    # the laptop trial's run: Qwen3-0.6B, 30 targets + 20 fakes + 970 controls -> $1.32 actual, inside the range
    q06 = {"layers": 28, "neurons": 3072, "hidden": 1024}
    e = LC.estimate(q06, 28, 27, 151649, 30, 970, 20, "cuda")
    assert e["api_usd"]["total"] < 1.32 <= e["api_usd_high"]["total"] * 1.05
    # its sweeps: 1,427 s + 870 s on the 6 GB GPU, and the fakes' sweep is in the plan unless it is cached
    assert abs(e["sweep_seconds_on_small_gpu"] - (1427 + 870)) < 150 and e["fake_sweep_included"]
    cached = LC.estimate(q06, 28, 27, 151649, 30, 970, 20, "cuda", fakes_cached=True)
    assert not cached["fake_sweep_included"] and abs(cached["sweep_seconds_on_small_gpu"] - 1427) < 100
    assert e["sweep_disk_gb"] > cached["sweep_disk_gb"]


def test_fake_verdicts_follow_check_labels():
    v = lambda k, n=20: LC.fake_verdicts({"aneg": {"n": n, "verified_and_generative": k / n}})["aneg"]["status"]   # noqa: E731
    assert [v(k) for k in (0, 1, 2, 3, 5)] == ["PASS", "PASS", "WARN", "FAIL", "FAIL"]       # check --labels: 1 passes, 3 fail
    assert [v(k, 100) for k in (5, 6, 14, 15)] == ["PASS", "WARN", "WARN", "FAIL"]            # scaled to the number of fakes
    assert LC.fake_verdicts(None) == {}
    # the laptop trial: gate 2/20, apos 0/20, aneg 3/20
    out = LC.fake_verdicts({"gate": {"n": 20, "verified_and_generative": 0.1}, "apos": {"n": 20, "verified_and_generative": 0.0},
                            "aneg": {"n": 20, "verified_and_generative": 0.15}})
    assert {b: x["status"] for b, x in out.items()} == {"gate": "WARN", "apos": "PASS", "aneg": "FAIL"}


class Args:
    neurons, layers, sample, all, seed, model, with_fakes, quick, tokens, budget = ["L1:3"], None, 0, False, 0, "tiny", 0, False, 0, None
    plan, yes, json, out, prefix, cache, relabel = False, False, False, None, None, None, False


def test_refuses_to_start_without_a_terminal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(LC, "model_dims", lambda name: DIMS)
    monkeypatch.setattr(LC.S, "model_path", lambda name: name)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *_: type("T", (), {"all_special_ids": [], "get_vocab": lambda self: {}, "__len__": lambda self: 100})()))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    a = Args(); a.cache = str(tmp_path / "c"); a.out = str(tmp_path / "o")
    assert LC.cli(a) == 2 and "--yes" in capsys.readouterr().err
    a.plan = True; a.json = True
    assert LC.cli(a) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["targets"] == 1 and plan["controls_added"] == 39 and plan["sweep"]["cached"] is False   # all 40 neurons of the tiny model, minus the target
    assert plan["budget_default"] and plan["budget_usd"] == max(1.0, round(2 * plan["estimate"]["api_usd_high"]["total"], 2))
    a.budget = 0.25
    assert LC.cli(a) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["budget_usd"] == 0.25 and not plan["budget_default"]
