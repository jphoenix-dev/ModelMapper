"""The `modelmapper` command: one entry point for every pipeline, loop stage, experiment, ledger and runner tool.

Each command runs an existing module exactly as its old `python src/... <args>` command did: the arguments after the
command are handed to the module unchanged (as its sys.argv), so outputs, defaults and logs are the same. SCRIPTS.md
maps every old path to its command. `--help` on any command prints that module's documentation and never runs it;
to reach a script's own argparse help, put `--` first: `modelmapper label -- --help`.

    modelmapper --help                      all commands
    modelmapper settings                    where every data path and model setting points, and why
    modelmapper scripts [text]              old path -> new module -> command (--markdown writes SCRIPTS.md's table)
    modelmapper run src/interference.py ... run any old script by its old path (or a new module name)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import textwrap
from pathlib import Path

from modelmapper import legacy
from modelmapper.experiments.registry import EXPERIMENTS

# command -> (module, one-line help, stages or None, {--option: ENV_VAR}, env vars it reads)
CORE = {
    "stats": ("modelmapper.core.neuron_stats", "every neuron's normal range on web text (data/stats_<model>.pt)", None, {}, ()),
    "atlas": ("modelmapper.core.build_atlas", "the activation atlas: top-K firings of every neuron (data/atlas_<model>.pt)", None, {}, ()),
    "scan": ("modelmapper.core.scan", "stream web text and keep chosen units' strongest firings, with context", None, {}, ()),
    "label": ("modelmapper.core.label_batch", "blind labelling of a neuron set: labeller explains, judge grades", None, {}, ()),
    "regrade": ("modelmapper.core.regrade", "re-grade explanations with the validated rubric grader", None, {}, ()),
    "concept": ("modelmapper.core.pipeline_fast", "concept pipeline: recall prompts -> attribution -> ablation -> steering",
                None, {}, ("RECALL_VARIANT", "CONCEPT_OUT_DIR", "ANSWER_DISCOVERY")),
    "wave": ("modelmapper.core.wave_free", "judge-free wave: recall, log-odds concentration, auto-selected steering",
             None, {}, ("RECALL_VARIANT", "CONCEPT_OUT_DIR", "MODEL")),
    "steer": ("modelmapper.core.steer_test", "clamp chosen neurons high and see what the model talks about", None, {}, ()),
    "dial": ("modelmapper.core.dial", "interactive steering dial (pick a concept, turn it up, chat)", None, {}, ()),
}
LOOP = {
    "map": ("modelmapper.loop.map_loop", "the mapping loop (labeller and GPU queues; was src/map_loop.py)",
            None, {}, ("RECALL_VARIANT", "CONCEPT_OUT_DIR", "KNOCK_BATCH", "MODEL")),
    "prepare": ("modelmapper.loop.prepare", "loop v2 prepare stage: specs and recall prompts for a frame", None, {}, ()),
    "enrich": ("modelmapper.loop.enrich", "loop v3 enrichment pass (LOOP_V3.md)", ["sample", "questions", "run", "summary"],
               {"--stages": "ENRICH_STAGES", "--name": "ENRICH_NAME"}, ("ENRICH_STAGES", "ENRICH_NAME", "ENRICH_CHEAP", "ENRICH_ALL")),
    "frame": ("modelmapper.loop.frame", "loop v2 frame: Vital Articles concept draw", ["types", "vital", "draw", "tag"], {}, ()),
    "report": ("modelmapper.loop.report", "loop v2 run report", None, {}, ()),
}
LEDGER = {
    "import": ("modelmapper.ledger.import_all", "run every importer (no-op when unchanged), then build, verify, export", None, {}, ()),
    "build": ("modelmapper.ledger.build", "build ledger.duckdb from the current packs ([--check])", None, {}, ()),
    "verify": ("modelmapper.ledger.verify", "check the built ledger against its sources", None, {}, ()),
    "export": ("modelmapper.ledger.export", "write the public export (Parquet + JSON)", None, {}, ()),
    "query": ("modelmapper.ledger.query", "look things up, e.g. `query unit L31:8930`", None, {}, ()),
    "pack": ("modelmapper.ledger.pack", "third-party packs: `pack validate <dir>`, `pack from-jsonl <dir> [--force]`",
             ["validate", "from-jsonl"], {}, ()),
}
DIST = {
    "launch": ("modelmapper.dist.launcher", "rent pods and run a store's tasks: run | status | terminate | billing",
               ["run", "status", "terminate", "billing"], {}, ()),
    "make-run": ("modelmapper.dist.make_run", "set up a run in a store (config, shards, manifest)", None, {}, ()),
    "worker": ("modelmapper.dist.worker", "claim and run tasks from a store (what each pod runs)", None, {}, ("DIST_STORE",)),
    "compare": ("modelmapper.dist.compare_runs", "compare two runs' results", None, {}, ()),
    "timeline": ("modelmapper.dist.timeline", "per-pod timeline of a run", None, {}, ()),
    "noise-floor": ("modelmapper.dist.noise_floor", "noise floor of one ablate task", None, {}, ()),
    "window-bench": ("modelmapper.dist.window_bench", "context-window benchmark", None, {}, ()),
}
GROUPS = {"ledger": LEDGER, "dist": DIST}
OTHER_LOOP = ["gate_repro", "p1_check", "run1_checks", "tag_check"]       # `modelmapper loop <name>`
STAGE_OPTIONAL = {"ranking_check"}                                        # its stage defaults to "check"


# ------------------------------------------------------------------------------------------------ module facts
def doc_of(mod):
    """A module's docstring, read without importing it."""
    try:
        return ast.get_docstring(ast.parse(legacy.module_file(mod).read_text(encoding="utf-8"))) or ""
    except (OSError, SyntaxError):
        return ""


def summary_of(mod):
    d = doc_of(mod).strip().split("\n\n")[0]
    return " ".join(d.split())


def runnable(mod):
    """True if the module does something when run: a __main__ block, argv handling, or work at the top level (many
    experiment scripts have no main() and just run: a top-level call, loop or `with`). False for libraries."""
    if mod in _LIBRARIES:
        return False
    s = legacy.module_file(mod).read_text(encoding="utf-8")
    if "__main__" in s or "sys.argv" in s:
        return True
    for n in ast.parse(s).body:
        if isinstance(n, (ast.For, ast.While, ast.With)):
            return True
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
            root = n.value.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if not (isinstance(root, ast.Name) and root.id in _SETUP_ROOTS):     # os.environ.setdefault(...) etc. is set-up
                return True
    return False


_SETUP_ROOTS = {"os", "sys", "warnings", "torch", "logging", "random", "np", "_settings", "_legacy"}
_LIBRARIES = {"modelmapper.experiments.mechanisms.common"}      # set-up at import (a loop over TMP / TEMP), no work


def uses_argparse(mod):
    return "argparse" in legacy.module_file(mod).read_text(encoding="utf-8")


def old_path(mod):
    return next((o for o, (m, _) in legacy.TABLE.items() if m == mod), None)


# ------------------------------------------------------------------------------------------------ commands per module
def command_for(mod, kind):
    """The CLI command that runs `mod`, or None for a library module."""
    for table, prefix in ((CORE, ""), (LOOP, "")):
        for c, ent in table.items():
            if ent[0] == mod:
                return f"modelmapper {c}"
    for g, table in GROUPS.items():
        for c, ent in table.items():
            if ent[0] == mod:
                return f"modelmapper {g} {c}"
    name = mod.rsplit(".", 1)[-1]
    exp = mod.removeprefix("modelmapper.experiments.")
    if kind == "experiments" and exp in EXPERIMENTS:
        return f"modelmapper experiment {exp}"
    if kind == "archive":
        return f"modelmapper archive {name}"
    if kind == "loop" and name in OTHER_LOOP:
        return f"modelmapper loop {name}"
    if kind == "core" and runnable(mod):
        return f"modelmapper core {name}"
    if kind == "ledger" and ".importers." in mod and name != "importers":
        return f"modelmapper ledger importer {name}"
    return None


_OLD_CMD = re.compile(r"python3?\s+(?:-u\s+)?(src/[A-Za-z0-9_/]+\.py)")


def new_command(old: str | None) -> str | None:
    """'src/loop_v2/era.py' -> 'modelmapper experiment era'; None for a library module or an unknown path."""
    if not old:
        return None
    try:
        mod, kind, _ = legacy.resolve(old)
    except KeyError:
        return None
    return command_for(mod, kind)


def new_files(text: str | None) -> list[str]:
    """Every old script path named in `text` -> its new file (repo-relative)."""
    out = []
    for o in re.findall(r"src/[A-Za-z0-9_/]+\.py", text or ""):
        try:
            out.append(legacy.module_file(legacy.resolve(o)[0]).relative_to(legacy.REPO_ROOT).as_posix())
        except KeyError:
            pass
    return out


def rewrite_commands(text: str | None) -> str | None:
    """Old commands in free text ('python src/loop_v2/jlens.py estimate ...') -> the CLI ('modelmapper experiment jlens
    estimate ...'); everything else is kept. Script paths named without `python` ('src/a.py + src/b.py') become the
    commands joined with ' ; '."""
    if not text:
        return text
    if "python" not in text:
        olds = re.findall(r"src/[A-Za-z0-9_/]+\.py", text)
        cmds = [c for c in (new_command(o) for o in olds) if c]
        return " ; ".join(cmds) if cmds else None
    return _OLD_CMD.sub(lambda m: new_command(m.group(1)) or m.group(0), text)


def stages_text(stages, optional=False):
    if not stages:
        return ""
    return "stages: " + " | ".join(stages) + (" (optional)" if optional else "")


def print_module_help(cmd, mod, stages=None, env_opts=None, env_vars=(), optional=False):
    old = old_path(mod)
    print(f"usage: {cmd}{' <stage>' if stages else ''} [args ...]")
    print()
    print(summary_of(mod) or "(no description)")
    print()
    print(f"  runs:  {mod}" + (f"   (was {old})" if old else ""))
    if stages:
        print(f"  {stages_text(stages, optional)}")
    for opt, var in (env_opts or {}).items():
        print(f"  {opt} VALUE   sets ${var}")
    print("  --set-env NAME=VALUE   set an environment variable for this run (repeatable; before the args)")
    if env_vars:
        print(f"  reads: {', '.join('$' + v for v in env_vars)}")
    print("  Arguments are passed to the script unchanged." +
          (" Its own option list: put `--` first, e.g. " + cmd + " -- --help" if uses_argparse(mod) else ""))
    d = doc_of(mod)
    if d:
        print("\n" + "-" * 30 + " module documentation " + "-" * 30)
        print(d)
    return 0


# ------------------------------------------------------------------------------------------------ running
def take_env(args, env_opts):
    """Strip leading `--set-env NAME=VALUE` and the command's own `--opt VALUE` env options; set them. Returns the rest.
    (Not `--env`: dist make-run has an `--env` option of its own, which must reach it.)"""
    args = list(args)
    while args:
        if args[0] == "--set-env" and len(args) > 1 and "=" in args[1]:
            k, v = args[1].split("=", 1); os.environ[k] = v; args = args[2:]
        elif args[0].startswith("--set-env=") and "=" in args[0][10:]:
            k, v = args[0][10:].split("=", 1); os.environ[k] = v; args = args[1:]
        elif env_opts and args[0] in env_opts and len(args) > 1:
            os.environ[env_opts[args[0]]] = args[1]; args = args[2:]
        else:
            break
    return args


def wants_help(args):
    return any(a in ("-h", "--help") for a in args)


def dispatch(cmd, mod, args, stages=None, env_opts=None, env_vars=(), optional=False, kind=None):
    """Run `mod` with `args` the way `python <old path> args` did, after the CLI's own checks."""
    if args and args[0] == "--":                     # verbatim from here on (e.g. `-- --help` for argparse scripts)
        args = args[1:]
        if wants_help(args) and not uses_argparse(mod):
            print(f"{mod} has no --help of its own (it would run); its documentation:\n")
            return print_module_help(cmd, mod, stages, env_opts, env_vars, optional)
    else:
        if wants_help(args):
            return print_module_help(cmd, mod, stages, env_opts, env_vars, optional)
        args = take_env(args, env_opts)
        if wants_help(args):
            return print_module_help(cmd, mod, stages, env_opts, env_vars, optional)
        if stages:
            if not args and not optional:
                print(f"{cmd}: give a stage: {' | '.join(stages)}  (see {cmd} --help)", file=sys.stderr)
                return 2
            if args and args[0] not in stages and not (optional and args[0].startswith("-")):
                print(f"{cmd}: unknown stage {args[0]!r}; stages: {' | '.join(stages)}", file=sys.stderr)
                return 2
    if kind == "archive":
        legacy.run_archive(mod, args)
    else:
        legacy.run_module(mod, args)
    return 0


# ------------------------------------------------------------------------------------------------ listings
def cmd_experiment(args):
    if not args or args[0] in ("-h", "--help"):
        print("usage: modelmapper experiment list [--group TEXT]\n"
              "       modelmapper experiment <name> [<stage>] [args ...]   (--help after a name shows its documentation)")
        return 0
    if args[0] == "list":
        ap = argparse.ArgumentParser(prog="modelmapper experiment list")
        ap.add_argument("--group", help="only groups whose name contains this text")
        a = ap.parse_args(args[1:])
        group = None
        for name, e in EXPERIMENTS.items():
            if a.group and a.group.lower() not in e["group"].lower():
                continue
            if e["group"] != group:
                group = e["group"]; print(f"\n{group}")
            st = (" [" + "|".join(e["stages"]) + "]") if e["stages"] else ""
            print(f"  {name + st:<58} {textwrap.shorten(summary_of(e['module']), 90)}")
        print("\nrun one: modelmapper experiment <name> [<stage>] [args ...]; --help for its documentation")
        return 0
    name = args[0]
    if name not in EXPERIMENTS:
        hits = [n for n in EXPERIMENTS if name in n]
        print(f"no experiment {name!r}" + (f"; did you mean {', '.join(hits)}?" if hits else "; see `modelmapper experiment list`"),
              file=sys.stderr)
        return 2
    e = EXPERIMENTS[name]
    return dispatch(f"modelmapper experiment {name}", e["module"], args[1:], e["stages"], None, e["env"],
                    optional=name in STAGE_OPTIONAL)


def cmd_table_group(group, table, args):
    if not args or args[0] in ("-h", "--help"):
        print(f"usage: modelmapper {group} <command> [args ...]\n")
        for c, ent in table.items():
            print(f"  {c:<14} {ent[1]}")
        if group == "ledger":
            print(f"  {'importer':<14} run one importer: importer <name> [--force]  (names: modelmapper ledger importer list)")
        print(f"\n--help after a command shows its documentation")
        return 0
    c = args[0]
    if group == "ledger" and c == "importer":
        return cmd_importer(args[1:])
    if c not in table:
        print(f"no command {group} {c!r}; see `modelmapper {group} --help`", file=sys.stderr)
        return 2
    mod, _, stages, env_opts, env_vars = table[c]
    return dispatch(f"modelmapper {group} {c}", mod, args[1:], stages, env_opts, env_vars)


def cmd_importer(args):
    names = sorted(p.stem for p in (legacy.PKG_DIR / "ledger" / "importers").glob("*.py") if p.stem != "__init__")
    if not args or args[0] in ("-h", "--help", "list"):
        print("usage: modelmapper ledger importer <name> [--force]\nimporters: " + " ".join(names))
        return 0
    if args[0] not in names:
        print(f"no importer {args[0]!r}; importers: {' '.join(names)}", file=sys.stderr)
        return 2
    import importlib
    importlib.import_module(f"modelmapper.ledger.importers.{args[0]}").main(force="--force" in args[1:])
    return 0


def cmd_loop(args):
    if not args or args[0] in ("-h", "--help"):
        print("usage: modelmapper loop <name> [args ...]   (the run checks; the main stages are top-level commands)\n")
        for n in OTHER_LOOP:
            print(f"  {n:<14} {textwrap.shorten(summary_of('modelmapper.loop.' + n), 100)}")
        for c, ent in LOOP.items():
            print(f"  {'(' + c + ')':<14} modelmapper {c} ...: {ent[1]}")
        return 0
    if args[0] in LOOP:
        return run_top(args[0], LOOP[args[0]], args[1:])
    if args[0] not in OTHER_LOOP:
        print(f"no loop command {args[0]!r}; see `modelmapper loop --help`", file=sys.stderr)
        return 2
    return dispatch(f"modelmapper loop {args[0]}", f"modelmapper.loop.{args[0]}", args[1:])


def cmd_core(args):
    mods = sorted(m for m, k in ((m, k) for _, (m, k) in legacy.TABLE.items()) if k == "core" and runnable(m))
    named = {ent[0]: c for c, ent in CORE.items()}
    if not args or args[0] in ("-h", "--help"):
        print("usage: modelmapper core <name> [args ...]   (any runnable core module)\n")
        for m in mods:
            n = m.rsplit(".", 1)[-1]
            alias = f" (= modelmapper {named[m]})" if m in named else ""
            print(f"  {n:<20} {textwrap.shorten(summary_of(m), 90)}{alias}")
        return 0
    m = f"modelmapper.core.{args[0]}"
    if m not in mods:
        print(f"no runnable core module {args[0]!r}; see `modelmapper core --help`", file=sys.stderr)
        return 2
    return dispatch(f"modelmapper core {args[0]}", m, args[1:])


def cmd_archive(args):
    names = sorted(m.rsplit(".", 1)[-1] for m, k in legacy.TABLE.values() if k == "archive")
    if not args or args[0] in ("-h", "--help", "list"):
        print("usage: modelmapper archive <name> [args ...]\n\nThe 16-17 Oct one-off analyses, unchanged (modelmapper/archive):\n")
        for n in names:
            print(f"  {n:<28} {textwrap.shorten(summary_of('modelmapper.archive.' + n), 90)}")
        return 0
    if args[0] not in names:
        print(f"no archive script {args[0]!r}; see `modelmapper archive list`", file=sys.stderr)
        return 2
    return dispatch(f"modelmapper archive {args[0]}", f"modelmapper.archive.{args[0]}", args[1:], kind="archive")


def cmd_run(args):
    if not args or args[0] in ("-h", "--help"):
        print("usage: modelmapper run <old path | module> [args ...]\n\n"
              "Runs any script by its old path (src/interference.py, src/loop_v2/era.py, src/ledger/build.py) or its new\n"
              "module (modelmapper.experiments.era), with the arguments the old command took. See SCRIPTS.md.")
        return 0
    try:
        mod, kind, old = legacy.resolve(args[0])
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    if not runnable(mod):
        print(f"{mod} is a library module (imported by others, nothing to run)", file=sys.stderr)
        return 2
    return dispatch(f"modelmapper run {args[0]}", mod, args[1:], kind=kind)


def stages_for(mod):
    """The stages (first argument choices) the CLI knows for a module, else None."""
    for table in (CORE, LOOP, *GROUPS.values()):
        for ent in table.values():
            if ent[0] == mod:
                return ent[2]
    exp = mod.removeprefix("modelmapper.experiments.")
    if exp != mod and exp in EXPERIMENTS:
        return EXPERIMENTS[exp]["stages"]
    return None


def scripts_rows():
    rows = []
    for old, (mod, kind) in sorted(legacy.TABLE.items()):
        cmd = command_for(mod, kind)
        rows.append((old, legacy.module_file(mod).relative_to(legacy.REPO_ROOT).as_posix(), kind, cmd))
    return rows


def cmd_scripts(args):
    ap = argparse.ArgumentParser(prog="modelmapper scripts", description="old script path -> new module -> command")
    ap.add_argument("filter", nargs="?", help="only rows containing this text")
    ap.add_argument("--markdown", action="store_true", help="print SCRIPTS.md's table")
    a = ap.parse_args(args)
    rows = [r for r in scripts_rows() if not a.filter or any(a.filter in (x or "") for x in r)]
    if a.markdown:
        print("| old path | new file | kind | command | stages |")
        print("|---|---|---|---|---|")
        for old, new, kind, cmd in rows:
            c = f"`{cmd}`" if cmd else "(library: imported by other modules)"
            st = stages_for(legacy.TABLE[old][0])
            print(f"| `{old}` | `{new}` | {kind} | {c} | {' &#124; '.join(st) if st else ''} |")
        return 0
    for old, new, kind, cmd in rows:
        print(f"{old:<45} {new:<52} {cmd or '(library)'}")
    return 0


def cmd_settings(args):
    if args and args[0] in ("-h", "--help"):
        print("usage: modelmapper settings [--knobs]\n\nPrints where every data path and model setting points and where the\n"
              "value came from (environment, .env or default). --knobs also lists the per-run environment knobs.\n"
              "API keys are never printed.")
        return 0
    from modelmapper.core import settings as S
    print(f"repo root: {S.REPO_ROOT}   (.env {'found' if S.DOTENV.exists() else 'not found'}; its values are not printed here)\n")
    for name, var, value, src in S.describe():
        print(f"  {var:<26} {value:<60} [{src}]")
    if "--knobs" in args:
        print("\nper-run knobs (read where they always were):")
        for var, (where, what) in S.KNOBS.items():
            print(f"  {var:<40} {where}: {what}")
    return 0


def run_top(c, ent, args):
    mod, _, stages, env_opts, env_vars = ent
    return dispatch(f"modelmapper {c}", mod, args, stages, env_opts, env_vars)


def main_help():
    p = argparse.ArgumentParser(prog="modelmapper", formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Model Mapper: one command for every pipeline, loop stage, experiment, "
                                            "ledger and runner tool (SCRIPTS.md maps the old src/ paths).",
                                epilog="Run `modelmapper <command> --help` for any command; it never starts a run.")
    sub = p.add_subparsers(title="commands", metavar="<command>")
    for c, ent in CORE.items():
        sub.add_parser(c, help=ent[1])
    for c, ent in LOOP.items():
        sub.add_parser(c, help=ent[1])
    sub.add_parser("loop", help="the loop's run checks: gate_repro, p1_check, run1_checks, tag_check")
    sub.add_parser("core", help="any other runnable core module (calibrate_clean, spliced_control, markers, ...)")
    sub.add_parser("experiment", help="experiment list | experiment <name> [<stage>] [args]")
    sub.add_parser("ledger", help="ledger import | build | verify | export | query | pack | importer")
    sub.add_parser("dist", help="dist launch | make-run | worker | compare | timeline | noise-floor | window-bench")
    sub.add_parser("archive", help="archive list | archive <name> [args]: the 16-17 Oct one-off analyses, unchanged")
    sub.add_parser("run", help="run any old script by its old path: run src/interference.py [args]")
    sub.add_parser("scripts", help="old path -> new module -> command (the table in SCRIPTS.md)")
    sub.add_parser("settings", help="where every data path and model setting points")
    p.add_argument("--version", action="version", version=f"modelmapper {__import__('modelmapper').__version__}")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "--version"):
        main_help().parse_args(argv or ["--help"])
        return 0
    c, rest = argv[0], argv[1:]
    if c in CORE:
        return run_top(c, CORE[c], rest)
    if c in LOOP:
        return run_top(c, LOOP[c], rest)
    if c in GROUPS:
        return cmd_table_group(c, GROUPS[c], rest)
    handlers = {"experiment": cmd_experiment, "loop": cmd_loop, "core": cmd_core, "archive": cmd_archive,
                "run": cmd_run, "scripts": cmd_scripts, "settings": cmd_settings}
    if c in handlers:
        return handlers[c](rest)
    print(f"modelmapper: unknown command {c!r} (see modelmapper --help)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
