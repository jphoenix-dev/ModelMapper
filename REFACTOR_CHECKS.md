# Refactor checks (18 Oct 2026)

The package refactor moved the 232 files under `src/` into `modelmapper/` and added the `modelmapper` command
(SCRIPTS.md has the map). This file lists every check run before the release, with its result. Everything ran on CPU
(`CUDA_VISIBLE_DEVICES=""`, `MODEL_MAPPER_DEVICE=cpu`, `MODEL_MAPPER_DTYPE=float32`) with Qwen3-0.6B where a model
was needed. No paid API was called.

- **Branch:** `claude/refactor-cli-b5646f`, cut from `release-2026-10-18` at 94f7a4c. Merged in afterwards:
  `release-2026-10-18` at 8e5de09 and at 71cfe9c (explainer files only), and `research/attention` at 0a4288b. The
  last merge brought the mechanisms study, the E7 stage and the loop v3 snapshot importer; see "Second round" at the
  end. Checks 1-8 below are from the first round, before that merge.
- **"Old"** is the unrefactored tree: `git archive release-2026-10-18` (94f7a4c), unpacked to a scratch folder on
  D: and run as `python src/...`.
- **"New"** is this branch, run as `python -m modelmapper ...`.
- Both used the same Python environment and the same settings.

**Verdict: nothing broke.**
- Every old-versus-new comparison came out identical, except where the old code differs from itself between two
  runs (DuckDB tie order; details below).
- All smoke tests pass, in this worktree and in a fresh clone installed with `pip install -e .`.

## Summary

| # | check | result |
|---|---|---|
| 1 | Static import check (`scripts/check_imports.py`) | PASS: 243 files compile; 487 `modelmapper.*` imports resolve to a real module and name; no bare old-style imports left in package modules |
| 2 | Unchanged files are byte-identical | PASS: 73 archive scripts, 24 ledger files whose bytes feed pack ids (`core.py`, `pack.py`, `importers/`), 5 ledger data files |
| 3 | CLI smoke test (`scripts/smoke_cli.py`) | PASS: 423 invocations, 227 old paths resolve, registry 64 = 64 modules, 4/4 argument checks |
| 4 | A real run per command group (CPU, Qwen3-0.6B) | PASS: core, loop, experiment, ledger, dist and archive (table below) |
| 5 | Old vs new equivalence | PASS: 10 comparisons (table below); the loop map task could only be compared up to its GPU-only provenance step |
| 6 | Pack ids | PASS: old and new code mint the same id for every importer. All 50 code files behind the real ledger's current packs match the moved files' content |
| 7 | Fresh clone on D:, throwaway venv, `pip install -e . --no-deps`, checks 1 and 3 through the installed `modelmapper.exe`, then `scripts/smoke_e2e.py` | PASS: install fine, checks 1 and 3 pass, smoke_e2e 6/6 stages |
| 8 | `scripts/export_release.sh` in the fresh clone (its own repo, placeholder identity) | PASS: 323 commits, one identity; the moved spot-check files, bulk FineWeb files and the export script are in no commit; `modelmapper/`, SCRIPTS.md and pyproject.toml are present |

## 1-3. Static and CLI checks

- `python scripts/check_imports.py` compiles every file under `modelmapper/` and `scripts/` and checks every import
  against the files on disk; it imports nothing. Its checks:
  - every `from modelmapper.x import y` names a module that exists, and `y` is defined there;
  - package modules use no old bare names (`from config import ...`);
  - the archive and the frozen ledger files use only bare names that `modelmapper.legacy` can resolve.

  Result: OK, here and in the fresh clone.
- The byte-identity check compares each archive / frozen-ledger / ledger-data file with the same file in the old
  tree. Result: 102 of 102 identical.
- `python scripts/smoke_cli.py` (and `--exe <venv>/Scripts/modelmapper.exe` in the fresh clone) runs `--help` for:
  - every top-level command, `ledger` and `dist` sub-command, experiment, and experiment's first stage;
  - every runnable core / loop / archive module, and `modelmapper run <old path> --help` for every runnable old path;
  - the listings: `settings`, `scripts`, `experiment list` and `ledger importer list`.

  Each runs in its own process with a fake `torch` and `transformers` first on the path that raise on import. So a
  pass also shows that `--help` only reads documentation and never starts a script.

  It then checks that 8 wrong inputs exit with a message, without running anything: an unknown command, experiment,
  stage, old path or archive name; a missing stage; and a library module passed to `run`.

  It checks 4 cases of argument handling in-process:
  - a script's own `--env` is passed through to it (`dist make-run` has one);
  - `--set-env NAME=VALUE` sets the variable;
  - `enrich --stages` sets `ENRICH_STAGES`;
  - old commands in ledger text are rewritten correctly.

  Finally, every one of the 227 old paths resolves to an existing file.

## 4. One real run per command group

| group | command (new) | what ran | result |
|---|---|---|---|
| core | `modelmapper stats Qwen/Qwen3-0.6B 20000`, `modelmapper atlas Qwen/Qwen3-0.6B 20000 20` | neuron ranges and the atlas on 25,129 FineWeb tokens | rc 0; 150 s and 147 s |
| experiment | `modelmapper experiment attrib_graph graphs --model Qwen/Qwen3-0.6B --out <dir> --limit 1` | attribution graph for the first pilot question (Chocolate): 300 nodes, 37,034 edges, gates g1/g2/g3 pass | rc 0; 40 s |
| loop | `modelmapper report <run>` | loop v2 report over a two-batch slice of run 1 (records copied to a scratch folder; the real run folder was not written) | rc 0 |
| loop + dist | `modelmapper dist make-run --kind map --specs <dir> ...`, then `modelmapper dist worker --store <dir> --once --no-terminate` | loop v2 `map_core` for one hand-written spec (Paris), through the runner | make-run rc 0. The map task loads the model and runs as far as `map_core.provenance()`, which calls `torch.cuda.get_device_name()`: **map tasks need a GPU**, in the old code too (finding 6). A CPU-only map run is not possible without changing an internal |
| dist | `modelmapper dist make-run --store <dir> --model Qwen/Qwen3-0.6B --shard-tokens 20000 --n-shards 2 --ablate 10:0-63`, then `modelmapper dist worker --store <dir> --no-terminate` | a local-store run: 2 observe shards, merge, 1 ablate task | rc 0; worker 1,106 s |
| ledger | `modelmapper ledger build --check`; `ledger import --no-export` (every importer, then build and verify); `ledger export`; `ledger query unit L31:8930`; `ledger pack validate modelmapper/ledger/contrib/research_claims` | against copies of the real packs in a private LEDGER_DIR; the shared ledger was only read | all rc 0; verify "ALL CHECKS PASSED"; export 54 experiments, 4,691,137 evidence rows, `scripts.jsonl` 227 rows |
| archive | `modelmapper archive indirect_path_summary`, `modelmapper archive cross_activation_summary` | unchanged archive scripts, including one importing another archive module by bare name (`families`) | rc 0 |
| all | `modelmapper --help`, `settings`, `scripts`, `experiment list`, ... | see check 3 | PASS |

## 5. Old vs new equivalence

Each row ran the old command in the old tree and the new command in this branch, on the same inputs, then compared
the outputs.

| comparison | old command | new command | result |
|---|---|---|---|
| core: neuron ranges | `python src/neuron_stats.py Qwen/Qwen3-0.6B 20000` | `modelmapper stats Qwen/Qwen3-0.6B 20000` | **identical**: every tensor in `stats_Qwen3-0.6B.pt` is equal (`torch.equal`) |
| core: atlas | `python src/build_atlas.py Qwen/Qwen3-0.6B 20000 20` | `modelmapper atlas Qwen/Qwen3-0.6B 20000 20` | **identical**: every tensor in `atlas_Qwen3-0.6B.pt`, every array in `atlas_Qwen3-0.6B_docs.npz` |
| experiment: attribution graph | `python src/loop_v2/attrib_graph.py graphs --model Qwen/Qwen3-0.6B --out <old> --limit 1` | `modelmapper experiment attrib_graph graphs ... --out <new> --limit 1` | **identical** apart from the wall-clock field: `graph.json` differs only in `seconds` (33.3 vs 33.4) |
| dist: runner | `python src/dist/make_run.py ...` + `python src/dist/worker.py ...` | `modelmapper dist make-run ...` + `modelmapper dist worker ...` | **identical results**: 99 of 113 store files byte-identical (corpus shards, promptsets, observe, merge and ablate results, manifest, config). The rest is run metadata: `git_commit` (empty for the old tree, which is a plain export, not a git checkout), `finished_at` times, and per-worker log folders named after the process id |
| loop: map_core via the runner | `python src/dist/make_run.py --kind map ...` + `worker.py --once` | `modelmapper dist make-run --kind map ...` + `dist worker --once` | **same**: the run stores (`config.json`, `manifest.json`, `map_index.json`, the uploaded spec) are byte-identical; both workers fail identically on CPU, with the same `AssertionError: Invalid device id` from `map_core.py` line 48 (`provenance`). The loop's map results themselves could not be compared on CPU |
| loop: report | `python src/loop_v2/report.py <run> <frame>` | `modelmapper report <run> <frame>` | **identical** `report.txt` |
| archive | `python src/indirect_path_summary.py`, `python src/cross_activation_summary.py` | `modelmapper archive indirect_path_summary`, `... cross_activation_summary` | **identical** output files (both rewrite a committed `results/*.txt`; the worktree was restored afterwards) |
| ledger: build | `python src/ledger/build.py --check` | `modelmapper ledger build --check` | **identical up to the build's own nondeterminism** (below). The same 18 tables, the same row count in each. All 12 base tables have identical content hashes (`evidence` 4,691,137 rows, `concepts`, `units`, `experiments`, `claims`, `artifacts` and their raw forms) |
| ledger: importers | `python src/ledger/import_all.py --no-export` | `modelmapper ledger import --no-export` | **same pack ids**: both skipped the same 5 importers as unchanged and minted the same 17 new pack ids. The one exception is `research_claims` (next section) |
| ledger: query / pack | `python src/ledger/query.py unit L31:8930`; `python src/ledger/pack.py validate ...` | `modelmapper ledger query unit L31:8930`; `modelmapper ledger pack validate ...` | **same output**: the query gives the same 205 lines, but tied rows can swap places; `validate` prints `ok` for both |

**The ledger's own nondeterminism (control run).** To tell refactor differences from noise, the OLD build was run
a second time on the same packs.

- Old run 1 and old run 2 differ in exactly the same derived tables as old and new: `unit_concept`, `unit_summary`,
  `concept_summary`, `layer_summary`, `head_summary` and `ledger_meta` (build time).
- With lists and maps sorted, `unit_concept` and `layer_summary` become equal in both comparisons.
- In `concept_summary`, `head_summary` and `unit_summary`, every top-N list holds the same drop values in every row,
  in both comparisons. Only which of several tied items fills a slot changes (e.g. 130 of 9,532 concepts, old vs old
  and old vs new alike).
- The `readout_tokens` column is `arg_max(value_text, count)` over tied counts: the old code alone gave different
  values in 149,767 of 454,507 rows between two runs (177,915 old vs new).
- `query unit` run three times with the old code gave three different line orders. Sorted, all three equal the new
  output.

So the old and new builds agree as closely as the old build agrees with itself.

## 6. Pack ids

- A pack id hashes the importer's source bytes, `ledger/core.py`, and each input's absolute path, size and mtime
  (`PackWriter`). So `core.py`, `pack.py` and every importer moved byte-for-byte. They read their paths from the
  environment, which `modelmapper.ledger` fills from `modelmapper.core.settings` with the same defaults.
- The old and new import runs minted the same ids, as shown above.
- The real ledger (`D:/already-clean-data/ledger`, read only) records a sha256 for each code file in every current
  pack. All 50 match the moved files' content.
- **Why the old code also re-minted 17 packs against the real ledger.** The research worktree that built them had
  LF line endings in some importer files; this checkout has CRLF (git `core.autocrlf`). The content is the same,
  but the bytes, and so the ids, differ. This predates the refactor: any fresh checkout re-mints. The re-minted
  packs have the same rows.
- **`research_claims` (the contrib pack)** is the only pack whose inputs live inside the code folder. Its id
  depends on the checkout's path and the files' mtimes, so it was already different in every checkout. It differs
  between old and new too (`src/ledger/contrib` vs `modelmapper/ledger/contrib`); its rows and claim ids are the
  same.
- The export maps each experiment row's old `script` / `reproduce` to the new command (`script_files`, `command`,
  `reproduce_command`) and writes `scripts.jsonl`. The packs themselves were not touched.
  - 39 of 54 rows get a command.
  - 2 name library modules (`dist/links.py`, `loop_v2/map_core.py`); `script_files` still records their new paths.
  - 2 name explainer scripts (`explainer/gen/map/...`), which didn't move.
  - 9 name `src/mechanisms/*.py`, which is **not on this branch** (it lives on `research/mechanisms`).

## 7-8. Fresh clone and export

- `git clone --branch claude/refactor-cli-b5646f` into a new folder on D:.
- A new venv from Python 3.11.9. A `.pth` file lets it see the existing venv's packages (torch 2.11.0+cu128,
  transformers 5.20.0); the shared venv itself was not modified.
- `pip install -e . --no-deps` installed `modelmapper-2026.10.18` and `modelmapper.exe`. From any folder, the package
  resolves to the clone, and `modelmapper settings` prints the clone as the repo root.
- Checks 1 and 3 pass there through the installed exe.
- `scripts/smoke_e2e.py` (every pipeline stage at toy scale on Qwen3-0.6B, CPU): **PASS 6/6 stages in 735 s**, run through the CLI
  (`modelmapper stats`, `atlas`, `wave`): environment, model files, neuron ranges (89 s), atlas (96 s), atlas readout
  and blind dossiers, recall + ablation + steering (539 s).
- The install leaves an untracked `modelmapper.egg-info/`, now gitignored.
- `scripts/export_release.sh` was run in the clone (a separate repo, so no branch was written in the shared one):
  - 323 commits, one identity;
  - zero commits touch `src/spotcheck_*`, `modelmapper/experiments/spotcheck_*`, `results/spliced_scan.json`,
    `results/luna_firing_judge.json` or `scripts/export_release.sh`;
  - `results/scan_*.json` and the dossier texts are absent; the `*_KEY.json` answer keys and `scan_*.log` logs are
    kept, as before;
  - `modelmapper/` (239 files), SCRIPTS.md and pyproject.toml are present.

## Found along the way (not changed; for James to decide)

1. **Ledger summaries are not reproducible.** Top-N slots and `readout_tokens` are picked among ties in whatever
   order DuckDB returns them. Rebuilding the same packs changes about 150k `readout_tokens` values and some top-N
   members. Ordering ties by a key (e.g. `unit_id`, `concept_id`) in `salience.sql` would fix it. That changes the
   derived views once, but no pack.
2. **Pack ids depend on line endings and checkout paths.** Hashing LF-normalised source would make ids stable across
   checkouts. It would also re-mint every pack once (a schema decision).
3. **`results/cross_activation_summary.txt` doesn't match its inputs.** The release took the 17 Oct re-run's
   `cross_activation.jsonl` (3,602 neurons) but not the regenerated summary, which still says 2,401. Old and new
   code both regenerate the 3,602 version.
4. **`results/interference.jsonl` is not in the release**, so `interference_summary` can't run (old or new alike).
5. ~~9 ledger experiments cite `src/mechanisms/*.py`, which was not on this branch.~~ Resolved in the second round:
   `research/attention` brought `src/mechanisms/`, now `modelmapper/experiments/mechanisms/`, with SCRIPTS.md rows.
6. **Loop v2 map tasks need a GPU.** `map_core.provenance()` calls `torch.cuda.get_device_name()` unconditionally, so a CPU or Mac run fails
   there. A guard (`if torch.cuda.is_available()`) would fix it; that's an internal, so it was left alone. On a CPU-only worker the
   task fails and the worker retries it every 5 s; with `--once` it exits instead.
7. **The settings defaults are the research machine's** (`D:/already-clean-data`, `C:/dev/already-clean/...`), as
   before; they are now all in one place, `modelmapper/core/settings.py`.

## How to re-run these checks

```bash
python scripts/check_imports.py
python scripts/smoke_cli.py                     # or --exe <installed modelmapper>
MODEL_MAPPER_DTYPE=float32 python scripts/smoke_e2e.py
```
The old-versus-new runs need an unrefactored tree (`git archive release-2026-10-18`). Their harness and logs are in
the release-prep notes folder, not in the repo.

## Second round: merging research/attention (0a4288b) and release-2026-10-18 (71cfe9c)

`release-2026-10-18` merged cleanly (explainer only). `research/attention` brought 11 commits. Here is where its
files went:

| came in as | now | how |
|---|---|---|
| `src/mechanisms/*.py` (9 modules) | `modelmapper/experiments/mechanisms/` | imports rewritten as before; `common.py` imports settings **after** its environment defaults (it sets `MODEL_MAPPER_MODELS`, which settings reads at import, as config did before); `rival_push`'s 8B path is `<MODEL_MAPPER_MODELS>/Qwen3-8B` (same default); `build_pack` writes to `modelmapper/ledger/contrib/`; old-path strings it writes into packs kept |
| `src/mechanisms/parallel_sentences.json`, `claims.jsonl` | same folder | byte-for-byte (read from next to the scripts) |
| `src/loop_v2/v3_snapshot_findings.py` | `modelmapper/experiments/` | `D:/already-clean-data` -> `settings.DATA_DIR` |
| `src/loop_v2/e7_pilot_after_enrich.sh`, `src/mechanisms/run_queue*.sh` | `scripts/` | commands use the CLI; the E7 waiter also recognises `modelmapper enrich run` |
| `src/ledger/contrib/mechanisms/`, `mechanisms_8b/` | `modelmapper/ledger/contrib/` | byte-for-byte (same git blobs as research/attention) |
| edits to `ledger/importers/loop_v3.py`, `vocab.yaml` | in place | merged files are research/attention's blobs, unchanged; `core.py` / `pack.py` unchanged too |
| edits to `loop/enrich.py` (E7), `experiments/jlens.py` (streamed merge) | in place | auto-merged; new lines use no bare imports or local paths |

Registry: `v3_snapshot_findings` and 8 `mechanisms.<name>` studies (`modelmapper experiment list --group mechanisms`).
SCRIPTS.md now covers 237 old paths.

A CLI bug, found and fixed while doing this:
- `runnable()` took 9 older experiments for libraries, because they do their work at the top level with no
  `__main__` block and no `sys.argv`: `auto_steer_jev_check`, `cr3_people_baseline`, `ctx_compare`,
  `default_heldout_regen`, `effort_compare`, `logodds_compare`, `logodds_summary`, `spotcheck_build`,
  `steer_retest_summary`.
- So `modelmapper run src/<one of them>.py` refused to run them. Their `experiment` commands were never affected.
- It now also counts top-level calls and loops. Set-up calls on `os` / `sys` / `torch` don't count, and
  `mechanisms/common` is listed as a library.
- `core/jev_test` gained its `modelmapper core jev_test` command. smoke_cli now compares the registry with the
  runnable experiment modules.

| check | result |
|---|---|
| Static import check | PASS: 254 files, 503 `modelmapper.*` imports; no bare `common` / `ledger` imports or `sys.path` edits left |
| CLI smoke (worktree, and the fresh clone at b742dac via `modelmapper.exe`) | PASS: 455 invocations, 237 old paths resolve, registry 73 = 73 runnable experiment modules, 4/4 argument checks |
| Ledger from the merged tree | PASS. `modelmapper ledger import --no-export` ran every importer, the 3 contrib packs, build and verify, into a private LEDGER_DIR holding a copy of tonight's real packs. Verify: **ALL CHECKS PASSED**, **5,601,506 evidence rows**, the same as the real ledger. Research's new `loop_v3` importer skipped as unchanged (same pack id as the real ledger) |
| Merged ledger vs the real ledger (`D:/already-clean-data/ledger`, read-only copy) | every table has the same row count. Leaving out the pack-id-derived columns (`pack_id`, `evidence_id`, `artifact_id`, `packs`) and build stamps: **every base table is identical** (`evidence`, `evidence_raw`, `ev_scored`, `concepts`, `concepts_raw`, `experiments`, `claims`, `artifacts`, `units`), and so are `unit_concept`, `layer_summary` and `ledger_meta`. `concept_summary`, `head_summary` and `unit_summary` differ only in tie choices, as in the first round. Pack ids differ for the importers re-minted by the CRLF / LF issue (finding 2) and the path-dependent contrib packs |
| `v3_snapshot_findings` (new CLI, inputs copied to a scratch `AC_DATA_DIR`) | **byte-identical** to the `snapshot_findings.txt` research wrote from the same snapshot. The old script hard-codes `D:/already-clean-data`, so it wasn't re-run (it would have overwritten research's file) |
| `mechanisms.ledger_checks`: old (`git archive research/attention`) vs new, on the merged ledger | rc 0 both. Not deterministic in the old code itself: 3 old runs gave 3 different files. The fields that vary old-vs-new are exactly the ones that vary old-vs-old (three top-15 lists cut at tied counts); nothing varies only in the new code |

Not run: the mechanisms model scripts (0.6B CPU runs of 10-60 min each that write to `<AC_DATA_DIR>/mechanisms`),
the E7 stage (needs the 5090) and the jlens merge (needs the shard files). Their changes are research's own lines,
merged as they were.

## Third round: research/attention 0975846 and release-2026-10-18 3d4efec

Both merged without conflicts.
- research/attention had moved one commit past the requested 6aab5a6, to 0975846 ("jlens merge: accept an earlier
  merge's output"). Its tip is in, so 6aab5a6 is too.
- The changes landed in the package paths: `experiments/jlens.py`, and `ledger/importers/jlens.py`, which is the
  same blob as research's. Nothing was recreated under `src/`.
- 3d4efec includes research/map-page 78c718c (`explainer/gen/map/push_words_jlens.py`). It gets no CLI entry: the
  map-page generators in `explainer/gen/map/` (`push_words.py` among them) stay outside the package and the CLI.

| check | result |
|---|---|
| Static import check | PASS: 254 files; the merged `experiments/jlens.py` has no bare imports or `sys.path` edits |
| CLI smoke | PASS: 455 invocations, 237 old paths, registry 73 = 73 |
| `v3_snapshot_findings` (same snapshot input, unchanged) | **byte-identical** to research's `snapshot_findings.txt` |

No ledger rebuild this round: the new jlens importer rows only appear once `D:/already-clean-data/jlens256` exists.
