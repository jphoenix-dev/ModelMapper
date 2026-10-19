# Caps and thresholds

Every number in the package that limits what is measured, or decides a call. None is applied silently: each is in
the code as a named constant, and the ones a user can change are recorded in every record that used them.

| # | What | Value | Where | Can change | Effect |
|---|---|---|---|---|---|
| 1 | A question counts as answered correctly | P(answer) ≥ 0.2 | `measure.MIN_P` | `CONCEPT_MIN_P` | Fewer or more kept questions |
| 2 | Questions needed to map a concept | 5 | `mapping.map_concept` | no | Below it: `not_recalled`, nothing mapped |
| 3 | Candidates knocked out one at a time | 200 (by attribution) | `mapping.NCAND` | no | A neuron outside the 200 is never tested |
| 4 | The concept's map | top 30 by drop | `mapping.TOPN` | no | The rest stay in `candidates_all200` |
| 5 | Random same-layer controls | 2 | `mapping.map_concept` | no | The per-concept noise floor |
| 6 | k curve sizes | 1, 2, 3, 5, 10, 20, 50 | `mapping.KS` | no | k_mean / k_flip are null above 50 |
| 7 | k_mean / k_flip bars | mean drop ≥ 3; half the right answers below P = 0.5 | `mapping.map_concept` | no | |
| 8 | Attention heads knocked out | 50 (by attribution) | `mapping.NHEADS` | no | |
| 9 | `early` role | first 20/36 of the layers | `mapping.LATE_FRAC` | no | A reading, not a measurement |
| 10 | **Hardware band for the calls, per model** | **Qwen3-8B: 0.13 log-odds** (3090 vs 5090, 10 concepts, p99, 19 Oct 2026). Other models: none, calls `uncalibrated` | `guards.BANDS` | `--band`, `MODEL_MAPPER_BAND` (recorded in `guards.band`) | Decides in / uncertain / out; changes no drop or rank |
| 11 | Labeller spend per concept | $0.50 at Luna's prices | `map-concept --budget` | yes | The labeller step stops with an error past it |
| 12 | Error text kept in check reports | 300 characters | `selfcheck.TAIL` | no | Endpoint error bodies are left out entirely |
| 13 | `check` map step | marker L23:2392 in the top 5 | `selfcheck.MAP_PASS_RANK` | no | METHOD.md, "How check is calibrated" |
| 14 | `check` live step | 2 of 3 markers in their top 20 | `selfcheck.LIVE_PASS_*` | no | Else WARN: fresh questions vary |
