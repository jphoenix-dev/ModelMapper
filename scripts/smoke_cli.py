"""CLI smoke test: `--help` for every command, group, experiment, core / loop / archive module and ledger importer.

  python scripts/smoke_cli.py [--exe modelmapper]     (default: python -m modelmapper from this checkout)

Each --help runs in its own process with a poisoned `torch` (and transformers) first on the path: importing it raises.
So a pass also proves that --help only reads documentation and never starts a script. Also checks that the old-path
resolver knows every file in SCRIPTS.md's table, and that unknown commands / stages fail cleanly without running.
Prints one line per failure and a summary; exit 1 on any failure. No model, no GPU, no API calls; ~2 minutes.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modelmapper import cli, legacy  # noqa: E402
from modelmapper.experiments.registry import EXPERIMENTS  # noqa: E402


def cases():
    c = [([], "commands"), (["--help"], "commands"), (["--version"], "modelmapper"),
         (["settings"], "repo root"), (["settings", "--knobs"], "per-run knobs"), (["settings", "--help"], "usage"),
         (["scripts"], "src/"), (["scripts", "--markdown"], "| old path"), (["scripts", "era"], "era"),
         (["experiment"], "usage"), (["experiment", "list"], "run one"), (["experiment", "list", "--group", "CR-3"], "cr3_pick"),
         (["loop", "--help"], "usage"), (["core", "--help"], "usage"), (["archive", "list"], "usage"), (["run", "--help"], "usage"),
         (["ledger", "--help"], "usage"), (["dist", "--help"], "usage"), (["ledger", "importer", "list"], "importers")]
    for k in list(cli.CORE) + list(cli.LOOP):
        c.append(([k, "--help"], "runs:"))
    for g, table in cli.GROUPS.items():
        for k in table:
            c.append(([g, k, "--help"], "runs:"))
    for n in EXPERIMENTS:
        c.append((["experiment", n, "--help"], "runs:"))
        if EXPERIMENTS[n]["stages"]:
            c.append((["experiment", n, EXPERIMENTS[n]["stages"][0], "--help"], "runs:"))
    for n in cli.OTHER_LOOP:
        c.append((["loop", n, "--help"], "runs:"))
    for m, k in legacy.TABLE.values():
        n = m.rsplit(".", 1)[-1]
        if k == "core" and cli.runnable(m):
            c.append((["core", n, "--help"], "runs:"))
        if k == "archive":
            c.append((["archive", n, "--help"], "runs:"))
    for old, (m, k) in legacy.TABLE.items():
        if cli.runnable(m):
            c.append((["run", old, "--help"], "runs:"))
    # failures that must not run anything (exit 2, message)
    for bad, msg in ((["nosuch"], "unknown command"), (["experiment", "nosuch"], "no experiment"),
                     (["experiment", "era", "nosuchstage"], "unknown stage"), (["experiment", "era"], "give a stage"),
                     (["ledger", "nosuch"], "no command"), (["run", "src/nosuch.py"], "not an old script"),
                     (["archive", "nosuch"], "no archive script"), (["run", "src/neurons.py"], "library module")):
        c.append((bad, "!" + msg))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", help="installed console script to test instead of python -m modelmapper")
    a = ap.parse_args()
    poison = Path(tempfile.mkdtemp(prefix="mm_poison_"))
    for mod in ("torch", "transformers"):
        (poison / f"{mod}.py").write_text(f"raise ImportError('{mod} imported during --help: a script was started')\n")
    env = dict(os.environ, PYTHONPATH=str(poison), PYTHONIOENCODING="utf-8")
    base = [a.exe] if a.exe else [sys.executable, "-X", "utf8", "-m", "modelmapper"]

    def one(case):
        args, want = case
        r = subprocess.run(base + args, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = r.stdout + r.stderr
        if want.startswith("!"):
            ok = r.returncode == 2 and want[1:] in out and "Traceback" not in out
        else:
            ok = r.returncode == 0 and want in out and "Traceback" not in out
        return ok, args, r.returncode, out[-600:]

    cs = cases()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, cs))
    bad = [r for r in res if not r[0]]
    for ok, args, rc, tail in bad:
        print(f"FAIL (rc {rc}): modelmapper {' '.join(args)}\n{tail}\n")
    # every old path resolves, and resolves to a file that exists
    unresolved = []
    for old, (mod, kind) in legacy.TABLE.items():
        try:
            m, k, o = legacy.resolve(old)
            if m != mod or not legacy.module_file(m).exists():
                unresolved.append(old)
        except KeyError:
            unresolved.append(old)
    for u in unresolved:
        print("UNRESOLVED old path:", u)
    # argument handling, in process: the CLI's own options never swallow a script's options
    unit = []
    if cli.take_env(["--env", "A=B", "x"], None) != ["--env", "A=B", "x"]:
        unit.append("take_env swallowed a script's own --env (dist make-run uses it)")
    os.environ.pop("MM_SMOKE_X", None)
    if cli.take_env(["--set-env", "MM_SMOKE_X=1", "build"], None) != ["build"] or os.environ.get("MM_SMOKE_X") != "1":
        unit.append("--set-env did not set the variable / strip itself")
    if cli.take_env(["--stages", "E0", "run"], {"--stages": "ENRICH_STAGES"}) != ["run"] or os.environ.get("ENRICH_STAGES") != "E0":
        unit.append("enrich --stages did not set ENRICH_STAGES")
    if cli.rewrite_commands("JL_ROWS=8 python src/loop_v2/jlens.py estimate --windows 32") != \
            "JL_ROWS=8 modelmapper experiment jlens estimate --windows 32":
        unit.append("rewrite_commands")
    for u in unit:
        print("FAIL (unit):", u)
    n_exp = len(EXPERIMENTS); n_files = sum(1 for m, k in legacy.TABLE.values() if k == "experiments" and cli.runnable(m))
    if n_exp != n_files:
        print(f"FAIL: registry has {n_exp} experiments, modelmapper/experiments has {n_files} modules")
    total_bad = len(bad) + len(unresolved) + (n_exp != n_files) + len(unit)
    print(f"{len(cs)} CLI invocations, {len(cs) - len(bad)} passed; {len(legacy.TABLE)} old paths resolve "
          f"({len(unresolved)} not); registry {n_exp} = {n_files} runnable experiment modules; {4 - len(unit)}/4 argument checks: "
          + ("PASS" if not total_bad else f"FAIL ({total_bad})"))
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
