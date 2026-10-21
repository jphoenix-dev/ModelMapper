"""`lookup` with labels: the packaged v3 labels, word search over labels, your own `label` output. Offline."""
import gzip
import json

import pytest

from modelmapper import cli, known


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """No network and no cached map: the concept map can't be read, so only the packaged labels answer."""
    monkeypatch.setattr(known, "CACHE_DIR", tmp_path)

    def no_network(*a, **k):
        raise OSError("offline (test)")
    monkeypatch.setattr(known.urllib.request, "urlopen", no_network)


@pytest.fixture
def tiny_map(tmp_path, monkeypatch):
    """A tiny published map in the cache: L31:7508 in one concept's map, and a concept called Music."""
    root = tmp_path / "map"
    (root / "n").mkdir(parents=True)
    cols = ["slug", "name", "answer", "section", "tier", "kind", "type", "status", "k_mean", "k_flip", "top1",
            "sources", "noise", "baseline"]
    rows = [["singing", "Singing", "Singing", "Arts", 3, "thing", "", "ok", 3, 2, "L31:7508", 1, 0.0, 1.0],
            ["music", "Music", "Music", "Arts", 3, "thing", "", "ok", 3, 2, "L32:7581", 1, 0.0, 1.0]]
    (root / "concepts.json").write_text(json.dumps({"cols": cols, "rows": rows}))
    for l in range(36):
        layer = {"7508": {"uses": [[0, 0, 4, 1.738]], "n_uses": 1, "role": "late-other", "jobs": [" sing"]}} if l == 31 else {}
        if l == 32:
            layer = {"7581": {"uses": [[1, 0, 1, 1.26]], "n_uses": 1, "role": "attribute"}}
        (root / "n" / f"L{l}.json").write_text(json.dumps(layer))
    monkeypatch.setattr(known, "CACHE_DIR", tmp_path)

    def no_network(*a, **k):
        raise AssertionError("lookup went to the network")
    monkeypatch.setattr(known.urllib.request, "urlopen", no_network)


def test_packaged_file():
    assert known.LABELS_FILE.stat().st_size < 6_000_000                    # 11,000 neurons, gzipped
    d = json.loads(gzip.decompress(known.LABELS_FILE.read_bytes()))
    m = d["_meta"]
    assert m["labeller"] == "gpt-6-luna" and m["judge"] == "Jev" and "21 Oct 2026" in m["source"]
    assert m["rules"]["exam_auc_min"] == 0.75 and m["rules"]["exam_margin_min"] == 0.2 and m["rules"]["job_cut"] == 0.2
    assert len(d["neurons"]) == 11000 and {e["set"] for e in d["neurons"].values()} == {"random", "bridge", "knockout-linked"}
    edge = [p for e in d["neurons"].values() for p in (e["gate"], e["apos"], e["aneg"]) if p and p.get("at_edge")]
    assert m["rules"]["at_edge"] and edge and all(p["verified"] and p["gen_pass"] is None for p in edge)


def test_neuron_from_packaged_labels(offline):
    r = known.lookup("L31:7508")
    assert not r["found"] and "could not download" in r["note"]           # no map, but the labels still answer
    (e,) = r["labels"]
    assert e["gate"]["verified"] and e["gate"]["auc"] == 0.984 and e["gate"]["gen_pass"] is True
    assert e["jobtype"]["apos"] == "redirector" and e["pushes"]["apos"][:2] == ["bows", "bow"]
    assert e["knockout"]["concepts"][0] == ["Singing", 1.74] and e["triage"]["read"] == "clear"
    t = " ".join(known.show(r).split())          # wrapping aside
    assert "Gate: Music, especially classical music" in t and "✓ exam 0.98 vs 0.48, generative ✓" in t
    assert "Positive: Musical instruments" in t and "Negative: Related to singing" in t
    assert "Job: positive redirector" in t and "negative redirector" in t
    assert "Pushes: + bows, bow" in t and "/ − sing, Sing" in t
    assert "Knockout: breaks Singing (1.74), Janis Joplin (0.66)" in t
    assert "labeller gpt-6-luna, judge Jev" in t


def test_neuron_with_map_and_labels(tiny_map):
    r = known.lookup("L31:7508")
    assert r["found"] and r["uses"][0]["concept"] == "Singing" and r["labels"][0]["gate"]["verified"]
    t = " ".join(known.show(r).split())          # wrapping aside
    assert t.index("Singing") < t.index("Published labels")


def test_unverified_parts_say_so(offline):
    t = " ".join(known.show(known.lookup("L0:404")).split())
    assert "Gate: (unchecked guess)" in t and "NOT VERIFIED: no-generalisation; exam 0.65 vs 0.53" in t
    assert "generative ✗ (AUC 0.69, needs 0.75)" in t            # the - side: verified, generative check failed


def test_unknown_neuron(tiny_map):
    r = known.lookup("L2:5")
    assert not r["found"] and r["labels"] == []
    assert "not found" in known.show(r) and "labels" not in known.show(r).lower()


def test_unknown_neuron_offline(offline):
    with pytest.raises(ConnectionError):                    # no map and no labels: the download error is the answer
        known.lookup("L1:1")


def test_word_search(tiny_map):
    r = known.lookup("music")
    assert r["found"] and r["concept"]["name"] == "Music"
    hits = r["label_hits"]
    assert hits[0]["verified"] and any(h["neuron"] == "L31:7508" and h["part"] == "gate" and h["verified"] for h in hits)
    assert all(h["verified"] for h in hits[:3]) and not hits[-1]["verified"]       # verified first
    t = known.show(r)                            # one line per hit: no wrapping
    assert "Labelled neurons whose labels mention 'music'" in t and hits[0]["neuron"] in t
    lines = [x for x in t.split("Labelled neurons", 1)[1].splitlines()[1:] if x.strip()]
    assert len(lines) == min(len(hits), known.MAX_HITS) + (len(hits) > known.MAX_HITS)
    if len(hits) > known.MAX_HITS:
        assert lines[-1].strip().startswith(f"… {len(hits) - known.MAX_HITS} more")


def test_word_search_labels_only(offline):
    r = known.lookup("tropical")                              # no map: the label hits still answer
    assert not r["found"] and any(h["neuron"] == "L29:4707" for h in r["label_hits"])
    with pytest.raises(ConnectionError):
        known.lookup("zzqx")                                  # nothing anywhere: the map's error stands


def _own(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    part = lambda label, ok, auc, gen: {"bin": "intelligible" if ok else "no-generalisation", "verified": ok, "label": label,
                                        "list_pass": ok, "auc": auc, "auc_control": 0.5, "gen_pass": gen}
    res = {"neurons": {"L3:77": {"layer": 3, "gate": part("Bicycles and cycling: bikes, pedals, saddles.", True, 0.91, True),
                                 "apos": part("Wheels and tyres", False, 0.6, None), "aneg": None,
                                 "jobtype": {"apos": "amplifier", "c_pos": 0.31, "aneg": "eraser", "c_neg": -0.22},
                                 "knockout": {"profile": "untested", "n_concepts": None, "max_drop": None},
                                 "triage": {"read": "clear", "verified_sides": [], "with_knockout": None}}},
           "summary": {"n": 1}}
    (d / "run_results.json").write_text(json.dumps(res), encoding="utf-8")
    (d / "intelligibility_run_0_gate.json").write_text(json.dumps({"L3:77": {"bin": "intelligible"}}), encoding="utf-8")
    return d


def test_own_records(tmp_path, records_dir):
    lab = _own(records_dir)                                   # label output inside a folder of concept records
    r = known.lookup("L3:77", records=records_dir)
    assert not r["found"] and r["labels"][0]["source"]["file"].endswith("run_results.json")
    t = " ".join(known.show(r).split())          # wrapping aside
    assert "Your labels:" in t and "Gate: Bicycles and cycling" in t and "✓ exam 0.91 vs 0.50, generative ✓" in t
    assert "Positive: (unchecked guess) Wheels and tyres" in t and "Negative: not labelled" in t
    assert "Job: positive amplifier (cos 0.31), negative eraser (cos -0.22)" in t and "Knockout: not tested" in t
    assert f"Source: your labels, {lab / 'run_results.json'}" in t
    r = known.lookup("bicycle", records=records_dir)
    assert [h["neuron"] for h in r["label_hits"]] == ["L3:77"] and r["label_hits"][0]["short"] == "Bicycles and cycling"
    assert known.lookup("L0:100", records=records_dir)["found"]   # the concept records still answer
    with pytest.raises(KeyError):
        known.lookup("Coffee", records=records_dir)


def test_cli_json(offline, capsys):
    assert cli.main(["lookup", "L31:7508", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["kind"] == "neuron" and d["labels"][0]["gate"]["gen_words"][:2] == ["music", "musique"]
    assert d["labels"][0]["source"]["labeller"] == "gpt-6-luna"
    assert cli.main(["lookup", "L31:7508"]) == 0
    assert "Gate: Music" in capsys.readouterr().out
