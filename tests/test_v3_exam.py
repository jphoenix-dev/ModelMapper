"""The v3 label exam (modelmapper/exam.py) on a tiny synthetic sweep with deterministic fake labeller and judge (no
network), plus a schema check against a real research file and one small real reproduction run (repro: Jev only)."""
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from modelmapper import exam as E

REF = Path(os.environ.get("MODEL_MAPPER_V3_REF", ""))
REF_GATE = REF / "intelligibility_v3_gate.json"

# ---- a synthetic sweep: 2 layers x 16 neurons; neuron n's favourites come from its own block of 120 tokens ----
N_LAYERS, WIDTH, K, BLOCK = 2, 16, 100, 120
N_NEURONS = N_LAYERS * WIDTH
V = N_NEURONS * BLOCK
CONTEXTS = [{"k": k, "read": ["slot", "end"] if k else ["slot"]} for k in range(4)]   # A: k 0, 2; B: k 1, 3


def _letters(t):
    s = ""
    t += 26
    while t:
        t, r = divmod(t, 26); s = chr(97 + r) + s
    return s


def _text(t):
    """four surface classes: ' abc' (latin, space), 'Abc' (latin), '123' (punct-digit), '_abc' (code)"""
    return [" " + _letters(t), _letters(t).capitalize(), str(1000 + t), "_" + _letters(t)][t % 4]


TEXT = [_text(t) for t in range(V)]
FAMILY = {E.visible(x): t // BLOCK for t, x in enumerate(TEXT)}           # visible text -> family (= neuron index)
FAMILY.update({x: t // BLOCK for t, x in enumerate(TEXT)})
decode = lambda j: TEXT[int(j)]
NONE_FAMILIES, LISTFAIL_FAMILIES, HALF_FAMILY = {5}, {7}, 9               # label kind 'none'; list exam blind; near threshold


def neuron_index(spec):
    l, i = map(int, spec[1:].split(":")); return l * WIDTH + i


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    d = tmp_path_factory.mktemp("sweep")
    meta = {"K": K, "n_tokens": V, "intermediate_size": WIDTH, "contexts": CONTEXTS}
    (d / "vocab_sweep_t.json").write_text(json.dumps(meta))
    for c in CONTEXTS:
        arrs = {}
        for r in c["read"]:
            rng = np.random.default_rng([c["k"], len(r)])
            top_i = np.zeros((N_LAYERS, WIDTH, K), np.int32); top_v = np.zeros((N_LAYERS, WIDTH, K), np.float16)
            for n in range(N_NEURONS):
                lo = BLOCK * n + (0 if c["k"] % 2 == 0 else 20)                 # half A: block[0:100], half B: block[20:120]
                top_i[n // WIDTH, n % WIDTH] = lo + rng.permutation(K)
                top_v[n // WIDTH, n % WIDTH] = np.linspace(10, 5, K)
            arrs.update({f"{r}_top_i": top_i, f"{r}_top_v": top_v, f"{r}_mean": np.ones((N_LAYERS, WIDTH), np.float32),
                         f"{r}_std": np.full((N_LAYERS, WIDTH), 2.0, np.float32)})
        np.savez(d / f"vocab_sweep_t_c{c['k']}.npz", **arrs)                    # stored, so _Npz memory-maps it
    hub = np.zeros((N_LAYERS, V), np.int32); hub[:, 3] = 10 ** 6               # token 3 is a hub token in every layer
    np.savez(d / "hub_tokens_t.npz", per_read=np.zeros((7, V), np.int32), per_layer=hub)
    return d


POOL = [f"L{n // WIDTH}:{n % WIDTH}" for n in range(N_NEURONS)]


class FakeLabeller:
    """describes a shown list by its majority family; kind 'none' for NONE_FAMILIES. $1 of 'usage' per request if costly"""

    def __init__(self, costly=False):
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}; self.costly = costly

    def _one(self, prompt):
        fam = Counter(FAMILY[line.split("  (z")[0]] for line in prompt.splitlines()).most_common(1)[0][0]
        if fam in NONE_FAMILIES:
            return {"description": "no clear pattern", "kind": "none", "confidence": 0.2}
        return {"description": f"tokens of family {fam}", "kind": "form", "confidence": 0.9}

    def _bill(self):
        self.usage["calls"] += 1; self.usage["input_tokens"] += 10 ** 7 if self.costly else 1000

    def chat_json(self, prompt, system=None, **kw):
        self._bill(); return self._one(prompt)

    def chat_json_n(self, prompt, system=None, n=1, **kw):
        self._bill(); return [self._one(prompt) for _ in range(n)]


TOKEN_ID = {x: t for t, x in enumerate(TEXT)}


class FakeJudge:
    """rates 90 for tokens of the family a description names (or a shown list is made of), else 5. Family HALF_FAMILY:
    every other one of its tokens in a call is missed, so its AUC is exactly 0.75 (on a threshold: the adaptive draws run)"""

    def __init__(self):
        self.calls = 0; self.usage = {"input_tokens": 0}

    def cost(self):
        return self.usage["input_tokens"] * 0.042 / 1e6

    def _rate(self, fam, tokens):
        self.calls += 1; self.usage["input_tokens"] += 400 + 12 * len(tokens)
        out, seen = [], 0
        for t in tokens:
            hit = fam is not None and TOKEN_ID[t] // BLOCK == fam
            if hit and fam == HALF_FAMILY:
                seen += 1; hit = seen % 2 == 1
            out.append(90.0 if hit else 5.0)
        return out

    def rate(self, description, tokens):
        m = re.search(r"family (\d+)", description); return self._rate(int(m.group(1)) if m else None, tokens)

    def rate_examples(self, shown, tokens):
        fam = Counter(FAMILY[x] for x in shown).most_common(1)[0][0]
        return self._rate(None if fam in LISTFAIL_FAMILIES else fam, tokens)


def _run(sweep, tmp_path, name="out", **kw):
    kw.setdefault("labeller", FakeLabeller()); kw.setdefault("judge", FakeJudge())
    return E.run_exam(sweep, "t", POOL, tmp_path / f"intelligibility_{name}.json", decode=decode, readings_dir=tmp_path,
                      config=kw.pop("config", E.ExamConfig(workers=4)), log=lambda *_: None, **kw)


@pytest.fixture(scope="module")
def result(sweep, tmp_path_factory):
    return _run(sweep, tmp_path_factory.mktemp("run"))


# ---- favourites and the test list ----
def test_halves_split_by_context(sweep):
    specs, units, fav, meta = E.favourites(sweep, "t", POOL, hub_x=10.0)
    for s in specs:
        n = neuron_index(s); blk = range(BLOCK * n, BLOCK * n + BLOCK)
        assert set(fav[s]["half"]["A"]) == set(blk[:100]) - {3}                # even contexts; hub token 3 dropped
        assert set(fav[s]["half"]["B"]) == set(blk[20:]) - {3}                 # odd contexts
        A = fav[s]["half"]["A"]; ranks = [len(A[a]) for a in fav[s]["rank"]["A"]]
        assert ranks == sorted(ranks, reverse=True) and ranks[0] == 3           # in all three half-A readings
    assert fav["L0:0"]["n_hub"] == 1 and E.favourites(sweep, "t", POOL[:2], hub_x=0)[2]["L0:0"]["n_hub"] == 0


def test_npz_reader_matches_numpy(sweep, tmp_path):
    a = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    np.savez_compressed(tmp_path / "c.npz", x=a)
    with E._Npz(tmp_path / "c.npz") as f:
        assert np.array_equal(f["x"], a)
    with E._Npz(sweep / "vocab_sweep_t_c1.npz") as f:
        assert isinstance(f["slot_top_i"], np.memmap)
        assert np.array_equal(f["slot_top_i"][[0, 1], [2, 3]], np.load(sweep / "vocab_sweep_t_c1.npz")["slot_top_i"][[0, 1], [2, 3]])


def test_held_out_and_decoys(result):
    for s, r in result.items():
        if s == "_summary":
            continue
        n = neuron_index(s); shown = {E.norm(x.replace("␣", " ")) for x in r["shown"]}
        for dr in r["draws"]:
            cands, lab = dr["test"]["candidates"], np.array(dr["test"]["label"])
            true = [c for c, y in zip(cands, lab) if y]; neg = [c for c, y in zip(cands, lab) if not y]
            assert len(true) == len(neg) == 30
            assert all(TEXT.index(c) // BLOCK == n and TEXT.index(c) % BLOCK >= 20 for c in true)   # held-out half B
            assert not {E.norm(c) for c in true} & shown                                           # never shown
            assert not {TEXT.index(c) for c in neg} & set(range(BLOCK * n, BLOCK * n + BLOCK))      # not its own favourites
            lv = dr["decoy_match_levels"]                       # 0: same class, space and length band; 1: length relaxed
            assert sum(lv.values()) == 30 and lv["0"] >= 20 and lv["2"] == lv["3"] == 0
            assert Counter(E.surface_key(c)[:2] for c in neg) == Counter(E.surface_key(c)[:2] for c in true)   # surface-matched
            src = dr["decoy_sources"]
            assert set(src) == set(neg) and not set(src.values()) & {s, *r["control_neurons"]}
            assert max(Counter(src.values()).values()) <= 2
            assert all(POOL[TEXT.index(c) // BLOCK] == o for c, o in src.items())                 # from that neuron's favourites


def test_decoys_change_between_draws_only(sweep):
    specs, units, fav, meta = E.favourites(sweep, "t", POOL)
    rec = {s: {"layer": fav[s]["layer"], "_rank": fav[s]["rank"], "_half": fav[s]["half"]} for s in specs}
    ctrl, _ = E.rotations(specs, 5); idx = E.decoy_index(rec, specs, decode); pos = E.held_out(rec["L0:4"], decode, 30)
    d = [E.surface_decoys(rec["L0:4"], pos, idx, {"L0:4", *ctrl["L0:4"]}, [4, k], decode)[0] for k in (0, 0, 1)]
    assert d[0] == d[1] and d[0] != d[2]


# ---- scoring, verdicts, draws ----
def test_control_scoring_and_floor():
    lab = [1, 1, 0, 0]
    own = [[[90, 80, 9, 5]], [[9.5, 90, 90, 9]], None]          # description 2: 9.5 vs 9 is a tie under the floor (0.5, not 0.75)
    ctl = [[[50, 50, 50, 50]], [[90, 90, 5, 5]], [[5, 5, 90, 90]], [[90, 5, 90, 5]], None]
    r = E.score_draw(list("abcd"), lab, own, ctl, [[90, 90, 5, 5]], [[[5, 5, 90, 90]], None], [{"kind": "form"}, {"kind": "none"}], 10.0)
    assert r["descriptions"][0]["auc"] == 1.0 and r["descriptions"][1]["auc"] == 0.5
    assert r["auc_own"] == 0.75 and r["kind"] == "form"
    assert r["auc_controls"] == [0.5, 1.0, 0.0, 0.5] and r["auc_control"] == 0.5     # median of the controls that answered
    assert r["list_auc_own"] == 1.0 and r["list_auc_control"] == 0.0
    assert r["n_test"] == [2, 2] and 0 < r["p_perm"] <= 1


@pytest.mark.parametrize("aucs, kind, reach, want", [
    ((0.80, 0.50, 0.80, 0.50), "form", None, (True, True, "intelligible")),
    ((0.75, 0.50, 0.75, 0.50), "form", None, (True, True, "intelligible")),           # thresholds are inclusive
    ((0.749, 0.50, 0.80, 0.50), "form", None, (True, False, "list-only")),
    ((0.80, 0.61, 0.80, 0.50), "form", None, (True, False, "list-only")),             # gap 0.19 < 0.2
    ((0.90, 0.50, 0.70, 0.50), "meaning", None, (False, True, "label-only (check)")),
    ((0.90, 0.50, 0.90, 0.50), "none", None, (True, False, "list-only")),             # kind 'none' never passes the label exam
    ((0.60, 0.50, 0.60, 0.50), "form", None, (False, False, "no-generalisation")),
    ((0.60, 0.50, 0.60, 0.50), "form", 0.2, (False, False, "panel-misses")),          # reach only renames a failing bin
    ((0.80, 0.50, 0.80, 0.50), "form", 0.2, (True, True, "intelligible")),
    ((0.95, 0.66, 0.95, 0.50), "form", None, (True, True, "decoys-too-easy (intelligible)")),
    ((None, 0.50, 0.80, 0.50), "form", None, (True, False, "list-only (judge failed)")),
])
def test_verdict_rule(aucs, kind, reach, want):
    assert E.verdict(*aucs, kind, reach) == want


def test_verdicts_do_not_depend_on_reach():
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = [round(float(x), 3) for x in rng.uniform(0.3, 1.0, 4)]; k = rng.choice(["form", "none"])
        v0, v1 = E.verdict(*a, k, None), E.verdict(*a, k, 0.1)
        assert v0[:2] == v1[:2] and (v0[2] == v1[2]) == (v1[2].replace("panel-misses", "no-generalisation") == v1[2])
        assert E.margin(dict(zip(("auc_own", "auc_control", "list_auc_own", "list_auc_control"), a))) >= 0


def test_combine_draws():
    d = lambda b, n=0: {"bin": b, "draw": n, "auc_own": 0.8}
    one = E.combine_draws({}, [d("intelligible")], True)
    assert one["stable"] and not one["second_draw"] and "draw" not in one
    two = E.combine_draws({}, [d("list-only"), d("list-only", 1)], True)
    assert two["stable"] and two["bin"] == "list-only" and two["votes"] == {"list-only": 2}
    maj = E.combine_draws({}, [d("a"), d("b", 1), d("b", 2)], True)
    assert not maj["stable"] and maj["bin"] == "b" and maj["majority"] == "2 of 3"
    unc = E.combine_draws({}, [d("a"), d("b", 1), d("c", 2)], True)
    assert unc["bin"] == "uncertain (a / b / c)"


# ---- the whole run ----
def test_run_bins_and_draws(result):
    recs = {s: r for s, r in result.items() if s != "_summary"}
    assert len(recs) == N_NEURONS
    for s, r in recs.items():
        fam = neuron_index(s)
        if fam in NONE_FAMILIES:
            assert r["bin"] == "list-only" and r["kind"] == "none"
        elif fam in LISTFAIL_FAMILIES:
            assert r["bin"] == "label-only (check)"
        elif fam == HALF_FAMILY:
            assert r["second_draw"] and len(r["draws"]) >= 2 and r["draws"][0]["draw1_margin"] <= 0.07
        else:
            assert r["bin"] == "intelligible" and r["auc_own"] == 1.0 and r["auc_control"] == 0.5 and not r["second_draw"]
        assert r["reach"] is None and r["corpus_peak"] is None                        # no readings files: reach skipped
        assert r["kind"] == ("none" if fam in NONE_FAMILIES else "form") and len(r["descriptions"]) == 3
    s = result["_summary"]
    assert s["second_draw_neurons"] == 1 and s["n"] == N_NEURONS and s["luna_calls"] == N_NEURONS
    assert s["exam"]["n_test"] == 30 and s["exam"]["reasoning"] == "none" and s["exam"]["merge"] == 1000


def test_merged_calls_ask_each_question_once(sweep, tmp_path):
    j = FakeJudge(); res = _run(sweep, tmp_path, judge=j)
    recs = {s: r for s, r in res.items() if s != "_summary"}
    pairs, keys1 = set(), set()
    for s, r in recs.items():
        keys = ({("d", d["description"]) for d in r["descriptions"]} | {("d", t) for t in r["control_descriptions"].values()}
                | {("l", tuple(recs[x]["shown"])) for x in [s] + r["control_neurons"]})
        keys1 |= keys
        pairs |= {(k, c) for dr in r["draws"] for c in dr["test"]["candidates"] for k in keys}
    assert res["_summary"]["questions"] == len(pairs)          # every (judge state, token) asked exactly once
    assert j.calls <= len(keys1) + 12                           # one call per state in round 1; <= 12 states in round 2


def test_focus_slices_and_merge_match_one_run(sweep, tmp_path, result):
    focus = POOL[3:11]
    desc = _run(sweep, tmp_path, "d", focus=focus, describe_only=True)
    assert desc["_summary"]["describe_only"] and set(desc) - {"_summary"} >= set(focus)
    ctrl_only = [s for s in desc if s not in focus and s != "_summary"]
    assert ctrl_only and all(len(desc[s]["descriptions"]) == 1 for s in ctrl_only)    # control-only neurons: 1 description
    lab = FakeLabeller()
    parts = [_run(sweep, tmp_path, f"b{k}", focus=focus[k::2], reuse_desc=tmp_path / "intelligibility_d.json", labeller=lab)
             for k in range(2)]
    assert lab.usage["calls"] == 0                                                     # every description reused
    m = E.merge_parts(tmp_path / "intelligibility_m.json", [tmp_path / f"intelligibility_{t}.json" for t in ("d", "b0", "b1")])
    assert m["_summary"]["n"] == len(focus) and m["_summary"]["describe_part"] == "d" and parts[0]["_summary"]["exam"]["reuse_desc"] == "d"
    for s in focus:                                                                    # same decoys, controls and verdicts
        a, b = m[s], result[s]
        for k in ("bin", "auc_own", "auc_control", "list_auc_own", "control_neurons", "decoy_sources", "test", "draws"):
            if k != "draws":
                assert a[k] == b[k], (s, k)
        assert [x["test"] for x in a["draws"]] == [x["test"] for x in b["draws"]]


def test_budget_stop(sweep, tmp_path):
    lab, j = FakeLabeller(costly=True), FakeJudge()
    cfg = E.ExamConfig(workers=1, budget_usd=2.5)
    with pytest.raises(E.BudgetExceeded):
        _run(sweep, tmp_path, "budget", labeller=lab, judge=j, config=cfg)
    assert lab.usage["calls"] == 3 and j.calls == 0                                    # stopped once past $2.5; nothing judged
    part = json.loads((tmp_path / "intelligibility_budget.partial.json").read_text(encoding="utf-8"))
    assert part["_summary"]["budget_stopped"] and len(part) - 1 == 3
    assert not (tmp_path / "intelligibility_budget.json").exists()
    lab2 = FakeLabeller()                                                              # resume: only the rest are described
    _run(sweep, tmp_path, "budget", labeller=lab2, reuse_desc=tmp_path / "intelligibility_budget.partial.json")
    assert lab2.usage["calls"] == N_NEURONS - 3


class SlowLabeller(FakeLabeller):
    """raw-request interface (as labeller.chat_n_raw / tally / parse_n); the FIRST request for families in `slow` takes
    `delay` s, any later one answers at once. Records when each request was sent."""

    def __init__(self, slow, delay):
        super().__init__(); self.slow, self.delay, self.sent, self.raw = slow, delay, {}, 0
        import threading
        self.lock = threading.Lock()

    def chat_json_n(self, *a, **k):
        raise AssertionError("hedged runs use the raw request")

    def chat_n_raw(self, prompt, system=None, n=1, **kw):
        import time
        fam = Counter(FAMILY[line.split("  (z")[0]] for line in prompt.splitlines()).most_common(1)[0][0]
        with self.lock:
            self.raw += 1; first = fam not in self.sent; self.sent.setdefault(fam, []).append(time.time())
        if first and fam in self.slow:
            time.sleep(self.delay)
        d = self._one(prompt)
        return {"choices": [{"message": {"content": json.dumps(d)}}] * n, "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}

    def tally(self, data):
        with self.lock:
            self.usage["calls"] += 1; self.usage["input_tokens"] += data["usage"]["prompt_tokens"]
            self.usage["output_tokens"] += data["usage"]["completion_tokens"]

    def parse_n(self, data, n):
        return [json.loads(c["message"]["content"]) for c in data["choices"][:n]]


@pytest.mark.parametrize("keep", [True, False])
def test_hedged_describe_requests(sweep, tmp_path, result, keep):
    slow = {2, 6}; lab = SlowLabeller(slow, 1.5 if keep else 4.0)
    res = _run(sweep, tmp_path, "h", labeller=lab, config=E.ExamConfig(workers=8, hedge_s=0.3, hedge_keep=keep))
    for fam in range(N_NEURONS):                                    # a duplicate only for the slow ones, sent after hedge_s
        t = lab.sent[fam]
        assert len(t) == (2 if fam in slow else 1) and (len(t) == 1 or t[1] - t[0] >= 0.3)
    side = json.loads((tmp_path / "intelligibility_h_hedge.json").read_text(encoding="utf-8"))
    assert set(side) == {POOL[f] for f in slow}
    for s, r in side.items():
        assert r["hedge_after"] == 0.3 and r["hedge_won"] is True and r["fh_s"] < 1.0
        assert r["fh_desc"] == [{"description": f"tokens of family {neuron_index(s)}", "kind": "form", "confidence": 0.9}] * 3
        if keep:                                                    # the loser finished too: its time, answer and cost
            assert r["fp_s"] >= 1.5 and r["fp_desc"] == r["fh_desc"] and "fp_unfinished" not in r
        else:
            assert r["fp_unfinished"] is True and "fp_s" not in r
    paid = N_NEURONS + (len(slow) if keep else 0)                   # winners, plus losers that finished
    assert res["_summary"]["luna_calls"] == paid and lab.usage["input_tokens"] == 1000 * paid
    assert res["_summary"]["cost_usd"] == round(paid * (1000 * E.LUNA_USD_IN + 100 * E.LUNA_USD_OUT), 3)
    for s in POOL:                                                  # the main output is that of an unhedged run
        assert {k: v for k, v in res[s].items()} == result[s], s


def test_no_hedging_by_default(sweep, tmp_path):
    assert E.ExamConfig().hedge_s == 0 and E.ExamConfig().hedge_keep is False
    lab = FakeLabeller()                                            # has no raw-request interface: the old path is used
    _run(sweep, tmp_path, "nh", labeller=lab)
    assert lab.usage["calls"] == N_NEURONS and not (tmp_path / "intelligibility_nh_hedge.json").exists()


def test_cli_exam(sweep, tmp_path, monkeypatch):
    from modelmapper import cli
    seen = {}
    monkeypatch.setattr(E, "run_exam", lambda *a, **kw: seen.update(a=a, kw=kw))
    (tmp_path / "n.json").write_text(json.dumps({"neurons": POOL, "focus": POOL[:2]}))
    assert cli.main(["exam", "apos", "--sweep-dir", str(sweep), "--sweep-tag", "t", "--neurons", str(tmp_path / "n.json"),
                     "--out", str(tmp_path / "o.json"), "--budget", "1.5"]) == 0
    assert seen["a"][2] == POOL and seen["kw"]["focus"] == POOL[:2] and seen["kw"]["config"].board == "apos"
    assert seen["kw"]["config"].budget_usd == 1.5 and seen["kw"]["config"].reasoning_kw() == {}


# ---- the research file ----
@pytest.mark.skipif(not REF_GATE.exists(), reason=f"no research file {REF_GATE}")
def test_output_schema_matches_research(result):
    ref = json.loads(REF_GATE.read_text(encoding="utf-8"))
    rs, ms = ref["_summary"], result["_summary"]
    assert set(ms) == set(rs) and set(ms["exam"]) == set(rs["exam"])
    assert {k: v for k, v in ms["exam"].items() if k not in ("focus", "reuse_desc")} == \
           {k: v for k, v in rs["exam"].items() if k not in ("focus", "reuse_desc")}          # same settings as production
    drop = {"votes", "majority"}
    for n_draws in (1, 2):
        mine = next(r for s, r in result.items() if s != "_summary" and len(r["draws"]) == n_draws)
        theirs = next(r for s, r in ref.items() if s != "_summary" and len(r["draws"]) == n_draws)
        assert set(mine) - drop == set(theirs) - drop
        assert [set(x) for x in mine["draws"]] == [set(x) for x in theirs["draws"]]
        assert set(mine["descriptions"][0]) == set(theirs["descriptions"][0])
        for k in ("ratings_own", "ratings_controls", "ratings_list_own", "ratings_list_controls"):
            assert np.array(mine[k], dtype=object).shape[:2] == np.array(theirs[k], dtype=object).shape[:2], k


@pytest.mark.skipif(not REF_GATE.exists(), reason=f"no research file {REF_GATE}")
def test_rescoring_research_ratings_gives_research_verdicts():
    """the stored judge ratings of 100 research neurons, scored by the port: same AUCs, bins, margins and final bins"""
    ref = json.loads(REF_GATE.read_text(encoding="utf-8"))
    for s in [k for k in ref if k != "_summary"][::9]:
        r = ref[s]; drs = []
        for dr in r["draws"]:
            got = E.score_draw(dr["test"]["candidates"], dr["test"]["label"], dr["ratings_own"], dr["ratings_controls"],
                               dr["ratings_list_own"], dr["ratings_list_controls"], dr["descriptions"], 10.0)
            for k in ("auc_own", "auc_controls", "auc_control", "list_auc_own", "list_auc_control", "p_perm", "coverage", "kind"):
                assert got[k] == dr[k], (s, dr["draw"], k)
            assert E.verdict(got["auc_own"], got["auc_control"], got["list_auc_own"], got["list_auc_control"], got["kind"],
                             dr["reach"]) == (dr["list_pass"], dr["label_pass"], dr["bin"])
            drs.append(dr)
        assert round(E.margin(r["draws"][0]), 3) == r["draws"][0]["draw1_margin"]
        assert E.combine_draws({}, drs, True)["bin"] == r["bin"]


# ---- reproduction: a few real v3 gate neurons, research descriptions reused, Jev judging (~$0.01) ----
def _jev_key():
    from modelmapper import judge
    return bool(os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or judge.KEY_FILE.exists())


@pytest.mark.repro
@pytest.mark.skipif(not (os.environ.get("MODEL_MAPPER_V3_REF") and REF_GATE.exists() and (REF / "vocab_sweep_v2.json").exists()),
                    reason="set MODEL_MAPPER_V3_REF to the research run's folder")
@pytest.mark.skipif(not _jev_key(), reason="no judge (Jev) key")
def test_reproduces_research_gate_exam(tmp_path):
    from modelmapper import settings
    tok = settings.model_path("Qwen/Qwen3-8B")
    if not (Path(tok) / "tokenizer.json").exists():
        pytest.skip("no local Qwen3-8B tokenizer (MODEL_MAPPER_MODELS)")
    ref = json.loads(REF_GATE.read_text(encoding="utf-8")); recs = {k: v for k, v in ref.items() if k != "_summary"}
    pick = lambda f: next(s for s, r in recs.items() if f(r))
    chosen = list(dict.fromkeys([pick(lambda r: r["bin"] == "intelligible" and len(r["draws"]) == 1),
                                 pick(lambda r: r["bin"] == "list-only" and len(r["draws"]) == 2),
                                 pick(lambda r: r["bin"] == "no-generalisation" and len(r["draws"]) == 1),
                                 pick(lambda r: len(r["draws"]) == 3),
                                 pick(lambda r: r["bin"].startswith("decoys-too-easy"))]))
    reuse = {}
    for s in chosen:
        ds = [{k: d[k] for k in ("description", "kind", "confidence")} for d in recs[s]["descriptions"]]
        reuse[s] = {"descriptions": ds, "description": ds[0]["description"], "kind": ds[0]["kind"], "confidence": ds[0]["confidence"]}
    for s in chosen:
        for c, text in recs[s]["control_descriptions"].items():
            reuse.setdefault(c, {"description": text, "kind": None, "confidence": None})
    (tmp_path / "intelligibility_reuse.json").write_text(json.dumps(reuse, ensure_ascii=False), encoding="utf-8")

    class NoLabeller:
        usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}; called = []

        def chat_json(self, *a, **k):
            self.called.append(a); raise RuntimeError("the labeller must not be called")
        chat_json_n = chat_json
    pool = json.loads((REF / "v3_neurons.json").read_text())["neurons"]
    from modelmapper import judge
    j0, lab = judge.cost(), NoLabeller()
    res = E.run_exam(REF, "v2", pool, tmp_path / "intelligibility_repro.json", focus=chosen, labeller=lab,
                     reuse_desc=tmp_path / "intelligibility_reuse.json", decode=E.load_decoder(tok), readings_dir=REF)
    spent = judge.cost() - j0
    assert not lab.called
    rows, same = [], 0
    for s in chosen:
        a, b = res[s], recs[s]
        for da, db in zip(a["draws"], b["draws"]):
            assert da["test"] == db["test"] and da["decoy_sources"] == db["decoy_sources"], (s, da["draw"])   # decoys bit-identical
        same += a["bin"] == b["bin"]
        rows.append(f"{s:>9} draws {len(a['draws'])}/{len(b['draws'])} bin {a['bin']!r} / {b['bin']!r}  "
                    f"AUC {a['auc_own']}/{b['auc_own']} ctrl {a['auc_control']}/{b['auc_control']} "
                    f"list {a['list_auc_own']}/{b['list_auc_own']} ctrl {a['list_auc_control']}/{b['list_auc_control']}")
    print("\n".join(rows) + f"\nverdicts {same}/{len(chosen)} identical; Jev ${spent:.4f}")
    assert spent < 0.30
    assert same == len(chosen), "\n".join(rows)
