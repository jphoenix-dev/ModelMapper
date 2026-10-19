"""Model-free unit tests: settings, records, guards, steering units, lookup, privacy, the command line."""
import json

import pytest

from conftest import make_record
from modelmapper import guards, known as lookup, settings
from modelmapper.records import MapResult


# ---------------------------------------------------------------- settings
@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    def write(text):
        p = tmp_path / ".env"
        p.write_text(text, encoding="utf-8")
        monkeypatch.setattr(settings, "DOTENV", p)
        monkeypatch.setattr(settings, "_dotenv", None)
        return p
    return write


def test_dotenv_parsing(dotenv, monkeypatch):
    dotenv('A=plain\nexport B=exported\nC="quoted # not a comment"\nD=value  # a comment\nE=\'single\'\n# F=no\nG=\n')
    for k in "ABCDEFG":
        monkeypatch.delenv(k, raising=False)
    assert settings.env("A") == "plain"
    assert settings.env("B") == "exported"
    assert settings.env("C") == "quoted # not a comment"
    assert settings.env("D") == "value"
    assert settings.env("E") == "single"
    assert settings.env("F") is None
    assert settings.env("G", "default") == "default"


def test_environment_beats_dotenv(dotenv, monkeypatch):
    dotenv("A=from-file\n")
    monkeypatch.setenv("A", "from-env")
    assert settings.env("A") == "from-env" and settings.source("A") == "environment"
    monkeypatch.delenv("A")
    assert settings.source("A") == ".env"
    assert settings.source("NOT_SET_ANYWHERE_123") == "default"


def test_model_names(tmp_path, monkeypatch):
    assert settings.short_name("Qwen/Qwen3-8B") == "Qwen3-8B"
    monkeypatch.setattr(settings, "MODELS_DIR", "")
    assert settings.model_path("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"
    (tmp_path / "Qwen3-0.6B").mkdir()
    (tmp_path / "Qwen3-0.6B" / "config.json").write_text("{}")
    monkeypatch.setattr(settings, "MODELS_DIR", str(tmp_path))
    assert settings.model_path("Qwen/Qwen3-0.6B") == str(tmp_path / "Qwen3-0.6B")
    assert settings.model_path("Qwen/Qwen3-8B") == "Qwen/Qwen3-8B"      # no local copy: the hub id


def test_describe_never_shows_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-123456")
    assert "sk-test" not in json.dumps(settings.describe())


# ---------------------------------------------------------------- records
def test_record_to_result(record):
    r = MapResult.from_record(record)
    assert r.status == "ok" and r.concept == "Bitcoin" and r.model == "Qwen/Qwen3-0.6B" and r.recall == 9
    assert len(r.top) == 30 and len(r.candidates) == 40
    assert r.top[0].neuron == "L0:100" and r.top[0].layer == 0 and r.top[0].index == 100 and r.top[0].drop == 4.0
    assert r.top[0].role == "say" and r.top[0].pushes == [" Bitcoin", " BTC"] and r.top[1].role is None
    assert r.nulls == {"L23:198": 0.01, "L27:1271": 0.0}
    assert r.k_curve[1] == 1.4 and r.k_curve[50] == 11.7 and r.random_k_curve[50] == -0.12
    assert [h.head for h in r.heads] == ["L3H1", "L19H29"]           # by drop


def test_record_round_trip(record, tmp_path):
    r = MapResult.from_record(record)
    p = r.save(tmp_path / "x" / "bitcoin.json")
    back = MapResult.load(p)
    assert back.record == record and back.path == str(p)
    d = back.to_dict()
    assert "record" not in d and d["top"][0]["neuron"] == "L0:100"
    json.dumps(d)                                                     # plain JSON


def test_not_recalled_and_failed_records():
    r = MapResult.from_record(make_record(status="not_recalled"))
    assert r.status == "not_recalled" and r.top == [] and r.baseline_logodds is None
    r = MapResult.from_record({"concept": "X", "row": {"status": "failed", "detail": "boom"}, "rank": None, "v2": {}})
    assert r.status == "failed" and r.detail == "boom"


# ---------------------------------------------------------------- guards
def test_call():
    assert guards.call(0.2, 0.13) == "in" and guards.call(-0.2, 0.13) == "out"
    assert guards.call(0.0, 0.13) == "uncertain"                     # the 30th neuron itself
    assert guards.call(0.13, 0.13) == "uncertain" and guards.call(-0.13, 0.13) == "uncertain"
    assert guards.call(5.0, None) == "uncalibrated"                 # no band for this model


def test_band(monkeypatch):
    monkeypatch.delenv("MODEL_MAPPER_BAND", raising=False)
    assert guards.band("Qwen/Qwen3-8B") == {"value": 0.13, "source": "Qwen3-8B, 3090 vs 5090, 10 concepts, p99, 19 Oct 2026"}
    b = guards.band("Qwen/Qwen3-0.6B")                                 # no calibrated band: never a silent default
    assert b["value"] is None and "Qwen3-0.6B" in b["source"]
    assert guards.band("Qwen/Qwen3-0.6B", 0.2) == {"value": 0.2, "source": "override (argument / --band)"}
    monkeypatch.setenv("MODEL_MAPPER_BAND", "0.05")
    assert guards.band("Qwen/Qwen3-8B") == {"value": 0.05, "source": "override (MODEL_MAPPER_BAND)"}


def test_annotate(monkeypatch):
    import copy
    monkeypatch.delenv("MODEL_MAPPER_BAND", raising=False)
    record = make_record(model="Qwen/Qwen3-8B")
    before = copy.deepcopy(record)
    guards.annotate(record)                                             # drops 4.0, 3.9, ...: the 30th is 1.1
    g = record["guards"]
    cands = record["rank"]["candidates_all200"]
    assert g["bar"] == 1.1 and g["band"]["value"] == 0.13 and g["bar_rank"] == 30
    calls = [c["call"] for c in cands]
    # within 0.13 of 1.1: 1.2 (29th), 1.1 (30th), 1.0 (31st)
    assert calls[:28] == ["in"] * 28 and calls[28:31] == ["uncertain"] * 3 and calls[31:] == ["out"] * 9
    assert (g["n_in"], g["n_uncertain"], g["n_out_but_close"]) == (28, 2, 1)
    assert cands[29]["margin"] == 0.0 and cands[0]["margin"] == 2.9
    assert record["rank"]["top"][29]["call"] == "uncertain"
    for b, c in zip(before["rank"]["candidates_all200"], cands):        # order and drops untouched
        assert (b["neuron"], b["single_drop"]) == (c["neuron"], c["single_drop"])
    r = MapResult.from_record(record)
    assert r.top[29].call == "uncertain" and r.top[29].margin == 0.0 and r.guards["n_in"] == 28


def test_annotate_uncalibrated_model(record, monkeypatch):
    monkeypatch.delenv("MODEL_MAPPER_BAND", raising=False)
    guards.annotate(record)                                             # Qwen3-0.6B: no calibrated band
    g = record["guards"]
    assert g["band"]["value"] is None and g["bar"] == 1.1
    assert (g["n_in"], g["n_uncertain"], g["n_out_but_close"]) == (None, None, None)
    assert {c["call"] for c in record["rank"]["candidates_all200"]} == {"uncalibrated"}
    assert record["rank"]["candidates_all200"][0]["margin"] == 2.9      # margins are still recorded


def test_annotate_override_is_recorded(record):
    guards.annotate(record, 0.35)
    g = record["guards"]
    assert g["band"] == {"value": 0.35, "source": "override (argument / --band)"}
    assert (g["n_in"], g["n_uncertain"], g["n_out_but_close"]) == (26, 4, 3)


def test_annotate_short_record():
    rec = make_record(n=10)
    guards.annotate(rec)
    assert rec["guards"]["bar"] is None and rec["guards"]["n_in"] is None
    assert "call" not in rec["rank"]["candidates_all200"][0]


def test_provenance_extra_never_omits():
    p = guards.provenance_extra(None)
    for f in ("package", "device", "cuda", "cudnn", "numpy", "os", "machine", "gpu_memory_gb", "model_revision"):
        assert p[f] not in (None, ""), f


def test_gpu_name_never_fails():
    assert isinstance(guards.gpu_name(), str)


# ---------------------------------------------------------------- steering units
def test_parse_unit():
    from modelmapper.steering import parse_unit
    assert parse_unit("L23:2392=x8") == (23, 2392, "x", 8.0)
    assert parse_unit("L23:2392=x-4") == (23, 2392, "x", -4.0)
    assert parse_unit("L4:2581=-3") == (4, 2581, "abs", -3.0)
    assert parse_unit(" 23_2392 = 12.5 ") == (23, 2392, "abs", 12.5)
    for bad in ("L23:2392", "L23=5", "L23:2392=fast", "x"):
        with pytest.raises(ValueError, match="L<layer>:<neuron>"):
            parse_unit(bad)


def test_resolve(monkeypatch):
    import torch
    from modelmapper import ranges, steering
    table = {"max": torch.full((28, 3072), 2.0), "min": torch.full((28, 3072), -0.5)}
    monkeypatch.setattr(ranges, "load_ranges", lambda name: table)
    got = steering.resolve([(23, 2392, "x", 8.0), (23, 1, "x", -4.0), (0, 0, "abs", 1.5)], "Qwen/Qwen3-0.6B", 28, 3072)
    assert got == [(23, 2392, 16.0), (23, 1, -2.0), (0, 0, 1.5)]
    with pytest.raises(ValueError, match="does not exist"):
        steering.resolve([(28, 0, "abs", 1.0)], "Qwen/Qwen3-0.6B", 28, 3072)


def test_shipped_ranges():
    from modelmapper import ranges
    for name, shape in (("Qwen/Qwen3-0.6B", (28, 3072)), ("Qwen/Qwen3-8B", (36, 12288))):
        t = ranges.load_ranges(name)
        assert tuple(t["max"].shape) == shape and tuple(t["min"].shape) == shape
        assert bool((t["max"] >= t["min"]).all())
    with pytest.raises(FileNotFoundError, match="modelmapper ranges"):
        ranges.load_ranges("nobody/NoModel-1B")


# ---------------------------------------------------------------- lookup
@pytest.fixture
def published(tmp_path, monkeypatch):
    """A tiny published map in the cache (FORMAT.md, 'The published map summary')."""
    root = tmp_path / "map"
    (root / "n").mkdir(parents=True)
    cols = ["slug", "name", "answer", "section", "tier", "kind", "type", "status", "k_mean", "k_flip", "top1",
            "sources", "noise", "baseline"]
    rows = [["eiffel_tower", "Eiffel Tower", "Eiffel Tower", "Arts", 3, "place", "", "ok", 3, 2, "L31:8930", 1, 0.01, 2.1],
            ["bitcoin", "Bitcoin", "Bitcoin", "Tech", 4, "thing", "", "ok", 5, 1, "L23:2392", 1, 0.0, 0.3]]
    (root / "concepts.json").write_text(json.dumps({"cols": cols, "rows": rows}))
    for l in range(36):
        layer = {}
        if l == 31:
            layer = {"8930": {"uses": [[0, 0, 1, 2.5], [1, 1, 12, 0.3]], "n_uses": 2, "role": "say",
                              "roles": {"say": 2}, "jobs": [" Eiffel", " Paris"], "az": 3.1, "hub": False, "top10": 1, "findings": []}}
        if l == 4:
            layer = {"7": {"uses": [[0, 0, 4, 0.9]], "n_uses": 1, "role": "early"}}
        (root / "n" / f"L{l}.json").write_text(json.dumps(layer))
    monkeypatch.setattr(lookup, "CACHE_DIR", tmp_path)

    def no_network(*a, **k):
        raise AssertionError("lookup went to the network")
    monkeypatch.setattr(lookup.urllib.request, "urlopen", no_network)
    return root


def test_lookup_neuron(published):
    r = lookup.lookup("L31:8930")
    assert r["found"] and r["n_uses"] == 2 and r["role"] == "say"
    assert [u["concept"] for u in r["uses"]] == ["Eiffel Tower", "Bitcoin"]
    assert r["uses"][1]["from"] == "earlier maps"
    assert "Eiffel Tower" in lookup.show(r)
    assert lookup.lookup("31_8930")["found"]
    assert not lookup.lookup("L2:5")["found"]


def test_lookup_concept(published):
    r = lookup.lookup("eiffel tower")
    assert r["kind"] == "concept" and [x["neuron"] for x in r["neurons"]] == ["L31:8930", "L4:7"]
    with pytest.raises(KeyError, match="did you mean: Eiffel Tower"):
        lookup.lookup("Eiffel")


def test_lookup_records(records_dir):
    r = lookup.lookup("L0:100", records=records_dir)
    assert r["found"] and r["n_uses"] == 2 and {u["concept"] for u in r["uses"]} == {"Bitcoin", "YouTube"}
    r = lookup.lookup("bitcoin", records=records_dir)
    assert r["kind"] == "concept" and len(r["neurons"]) == 30 and r["neurons"][0]["neuron"] == "L0:100"
    with pytest.raises(KeyError):
        lookup.lookup("Coffee", records=records_dir)
    r = lookup.lookup("L5:5", records=records_dir)
    assert not r["found"] and "no record" in lookup.show(r) and "different runs" not in lookup.show(r)


def test_lookup_download_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(lookup, "MAP_URL", "http://127.0.0.1:9/none")
    with pytest.raises(ConnectionError, match="could not download"):
        lookup.lookup("L1:1")


# ---------------------------------------------------------------- privacy of check reports
def test_private(monkeypatch):
    import getpass
    import platform
    from pathlib import Path
    from modelmapper.selfcheck import private
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")
    monkeypatch.setattr(platform, "node", lambda: "ALICE-PC")
    home = str(Path.home())
    s = private(f"{home}\\x.py by alice on ALICE-PC: HTTP 401: {{'error': 'bad key sk-abcdef123456'}}")
    assert home not in s and "alice" not in s.lower() and "{'error'" not in s
    assert "HTTP 401 (response body left out)" in s
    assert "sk-abcdef" not in private("key sk-abcdef123456 here")
    assert private("malice") == "malice"                              # a user name only as a whole word


# ---------------------------------------------------------------- command line
def run_cli(args, capsys):
    from modelmapper.cli import main
    try:
        code = main(args)
    except SystemExit as e:
        code = e.code
    out = capsys.readouterr()
    return code, out.out + out.err


def test_cli_help_and_version(capsys):
    import modelmapper
    code, out = run_cli([], capsys)
    assert code == 0 and "map-concept" in out and "lookup" in out
    code, out = run_cli(["--version"], capsys)
    assert code == 0 and modelmapper.__version__ == "0.1.0" and "modelmapper 0.1.0" in out
    code, out = run_cli(["steer", "--help"], capsys)
    assert code == 0 and out.startswith("usage: modelmapper steer")


def test_cli_errors(capsys):
    code, out = run_cli(["map-concept"], capsys)
    assert code == 2 and "concept" in out
    code, out = run_cli(["no-such-command"], capsys)
    assert code == 2
    code, out = run_cli(["check", "--only", "nope"], capsys)
    assert code == 2 and "unknown step" in out
    code, out = run_cli(["check", "--only", "live"], capsys)
    assert code == 2 and "--apis" in out


def test_cli_lookup(records_dir, capsys):
    code, out = run_cli(["lookup", "L0:100", "--records", str(records_dir)], capsys)
    assert code == 0 and "Bitcoin" in out
    code, out = run_cli(["lookup", "Coffee", "--records", str(records_dir)], capsys)
    assert code == 2 and "no record" in out


def test_cli_settings(capsys):
    code, out = run_cli(["settings"], capsys)
    assert code == 0 and "MODEL_MAPPER_MODEL" in out


def test_import_is_light():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", "import sys, modelmapper; print('torch' in sys.modules)"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stderr


def test_submodules_do_not_shadow_the_api():
    import importlib
    import modelmapper
    for mod in ("known", "selfcheck", "steering", "mapping", "measure", "records", "guards", "ranges", "answerkey"):
        importlib.import_module("modelmapper." + mod)
    for name in ("map_concept", "map_batch", "steer", "lookup", "load_model", "check"):
        assert getattr(modelmapper, name).__module__ == "modelmapper.api", name


def test_os_name_windows_11(monkeypatch):
    import platform
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(platform, "release", lambda: "10")             # what Windows 11 reports
    monkeypatch.setattr(sys, "getwindowsversion", lambda: SimpleNamespace(build=26100), raising=False)
    assert guards.os_name() == "Windows 11 (build 26100)"
    monkeypatch.setattr(sys, "getwindowsversion", lambda: SimpleNamespace(build=19045), raising=False)
    assert guards.os_name() == "Windows 10 (build 19045)"
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "release", lambda: "6.8.0")
    assert guards.os_name() == "Linux 6.8.0"


def test_check_report_hides_endpoint_hosts(tmp_path):
    """check --quick with custom labeller and judge endpoints: the report says 'custom endpoint', never the host."""
    import os
    import subprocess
    import sys
    env = {**os.environ, "OPENAI_BASE_URL": "https://secret-region.example.net/openai/v1", "OAI_MODEL": "my-deploy",
           "JUDGE_BACKEND": "llm", "JUDGE_BASE_URL": "https://judge-host.example.org/v1", "JUDGE_MODEL": "j",
           "MODEL_MAPPER_DOTENV": str(tmp_path / "none.env")}
    import modelmapper
    from pathlib import Path
    env["PYTHONPATH"] = os.pathsep.join([str(Path(modelmapper.__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, "-m", "modelmapper", "check", "--quick", "--out", str(tmp_path / "rep")],
                       capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path, timeout=300)
    assert r.returncode in (0, 1), r.stderr[-2000:]
    txt = (tmp_path / "rep" / "report.txt").read_text(encoding="utf-8") + (tmp_path / "rep" / "report.json").read_text(encoding="utf-8")
    assert "custom endpoint" in txt and "example.net" not in txt and "example.org" not in txt and "secret-region" not in txt
