"""Typed results. A MapResult is a view of one concept's record; the record itself (FORMAT.md) is what is saved and
what the Qwen Neuron Map is built from, so a result from `map_concept` can sit next to the published map."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class NeuronEffect:
    neuron: str              # "L23:2392"
    layer: int
    index: int
    drop: float              # mean drop in the answer's log-odds over the questions, this neuron switched off alone
    attr_rank: int           # its rank by attribution among the 200 candidates (0 = largest)
    activation: float        # its mean activation at the answer position
    role: str | None = None  # read from its weights: 'say', 'attribute', 'late-other' or 'early' (METHOD.md); top 30 only
    pushes: list[str] = field(default_factory=list)   # the 30 words it pushes most, from its weights (top 30 only)
    margin: float | None = None  # its drop minus the 30th neuron's drop, in this run (guards.py)
    call: str | None = None      # 'in' / 'uncertain' / 'out' of the top 30, allowing for the hardware band (guards.py)


@dataclass
class HeadEffect:
    head: str                # "L19H29"
    drop: float              # mean log-odds drop with this attention head switched off


@dataclass
class MapResult:
    concept: str
    model: str
    status: str              # 'ok', 'not_recalled' or 'failed'
    answer: str | None = None
    answer_form: str | None = None
    recall: int = 0                       # questions the model answers correctly (P(answer) >= 0.2); 5 needed
    baseline_logodds: float | None = None
    top: list[NeuronEffect] = field(default_factory=list)       # the 30 largest single-knockout drops, largest first
    candidates: list[NeuronEffect] = field(default_factory=list)  # all 200 attributed neurons, by drop
    nulls: dict[str, float] = field(default_factory=dict)       # random same-layer neurons switched off: their mean drop
    k_mean: int | None = None             # how many top neurons together cut the mean log-odds by >= 3 (None: more than 50)
    k_flip: int | None = None             # how many flip at least half of the correct answers
    k_curve: dict[int, float] = field(default_factory=dict)     # k -> mean drop with the top k switched off together
    random_k_curve: dict[int, float] = field(default_factory=dict)
    heads: list[HeadEffect] = field(default_factory=list)       # the 50 most-attributed heads, by drop
    guards: dict = field(default_factory=dict)              # the band used, the bar and the n_in / n_uncertain / n_out_but_close counts
    provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    detail: str | None = None             # why, when status is not 'ok'
    path: str | None = None               # where the record was saved
    record: dict = field(default_factory=dict, repr=False)      # the full record (FORMAT.md)

    @classmethod
    def from_record(cls, rec: dict, path=None) -> "MapResult":
        row, rank, v2 = rec.get("row") or {}, rec.get("rank") or {}, rec.get("v2") or {}
        prov = v2.get("provenance") or {}
        roles = {r["neuron"]: r for r in v2.get("roles_top30", [])}

        def eff(c):
            l, n = c["neuron"][1:].split(":")
            r = roles.get(c["neuron"], {})
            return NeuronEffect(c["neuron"], int(l), int(n), c["single_drop"], c["attr_rank"], c["act_answer_pos"],
                                r.get("role"), list(r.get("job_tokens", [])), c.get("margin"), c.get("call"))
        cands = [eff(c) for c in rank.get("candidates_all200", [])]
        from modelmapper.mapping import KS
        heads = sorted((HeadEffect(h, d) for h, d in (rank.get("head_knockout_mean") or {}).items()), key=lambda h: -h.drop)
        return cls(concept=rec["concept"], model=prov.get("model", ""), status=row.get("status", "failed"),
                   answer=row.get("answer"), answer_form=row.get("answer_form"), recall=row.get("recall", 0),
                   baseline_logodds=rank.get("baseline_mean_logodds"), top=cands[:30], candidates=cands,
                   nulls={k: sum(v) / len(v) for k, v in (v2.get("null_single_drops") or {}).items()},
                   k_mean=row.get("k_mean"), k_flip=row.get("k_flip"),
                   k_curve=dict(zip(KS, row.get("drops") or [])), random_k_curve=dict(zip(KS, row.get("random_drops") or [])),
                   heads=heads, guards=dict(rec.get("guards") or {}), provenance=prov, notes=list(v2.get("map_one_notes", [])), detail=row.get("detail"),
                   path=str(path) if path else None, record=rec)

    @classmethod
    def load(cls, path) -> "MapResult":
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")), path)

    def save(self, path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.record, indent=1, ensure_ascii=False), encoding="utf-8")
        self.path = str(p)
        return p

    def to_dict(self) -> dict:
        d = asdict(self); d.pop("record"); return d
