"""Old script paths (before the 18 Oct 2026 package refactor) -> new modules, and a way to run them.

Until 18 Oct 2026 every script lived flat in src/ (or src/loop_v2, src/ledger, src/dist) and imported its neighbours
by bare name (`from config import env`, `import concept_pipeline`). RESULTS.md, the ledger's experiment rows and old
logs still name those paths. This module is the one table that maps each of them to its new home; SCRIPTS.md is
generated from it (`modelmapper scripts --markdown`).

Three kinds of file moved in three ways:
  core / loop / experiments / dist   imports rewritten to package form (`from modelmapper.core.config import env`),
                                     repo-root and data paths read from modelmapper.core.settings; nothing else changed.
  ledger                             moved byte-identical where the bytes feed pack ids (core.py, pack.py, importers/),
                                     so existing pack ids stay valid; their bare `ledger.*` / `config` imports resolve
                                     through install(only=...) below.
  archive                            moved byte-identical (the 16-17 Oct one-off analyses, kept for the record). They
                                     still use bare names and `Path(__file__).parents[1]` as the repo root, so they run
                                     through run_archive(): bare names resolve to the new modules, and the script sees
                                     __file__ = <repo>/src/<name>.py as before, so its paths land where they used to.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import runpy
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent

# --------------------------------------------------------------------------------------------- the table
# Flat src/<name>.py files, by destination. Everything else (src/loop_v2, src/ledger, src/dist) follows the rules in
# _build_table() below.
CORE = """
    config neurons oai jev cache atlas build_atlas neuron_stats scan merge_scans make_dossiers label label_batch
    intruder_test regrade grader_validation calibrate_clean spliced_control strata_scan strata_grade concept_pipeline
    pipeline_fast recall_variants steer_test strict_judge wave_free markers dial smoke_test jev_test
    wiki_sample wiki_sitelinks
""".split()
LOOP_FLAT = ["map_loop"]
EXPERIMENTS_FLAT = """
    answer_discovery attribute attribution_diag auto_steer_jev_check auto_steer_scale backup_routes cr3_jev
    cr3_people_baseline cr3_pick cr3_score ctx_compare default_heldout_regen effort_compare fluency_proxy gg_recognition
    gg_search grader2 h_depth k50_check label_match_luna label_rule lexical_check logodds_check logodds_compare
    logodds_summary make_units people_fluency people_prompts people_rules replicate_compare says_name shared_parts
    shared_parts_lexical shuffled_judge single_neuron_steer spotcheck_build spotcheck_score steer_retest
    steer_retest_summary steer_select summarise_concepts summarise_gg wave_summary
    concept_search_feasibility concept_atlas fame_summary
""".split()
# The loop proper (src/loop_v2 -> modelmapper/loop); every other src/loop_v2 module is a study -> experiments.
LOOP_V2_LOOP = "prepare map_core enrich frame report gate_repro p1_check run1_checks tag_check".split()
# Everything else flat in src/ is a 16-17 Oct one-off analysis -> archive (unchanged).

# added by the refactor; no old path
NEW_MODULES = {"modelmapper.core.settings", "modelmapper.experiments.registry", "modelmapper.experiments.mechanisms"}

# ledger files whose bytes feed pack ids (PackWriter hashes its importer's source and core.py): never edited
LEDGER_FROZEN = ("core.py", "pack.py", "importers/")


def _build_table(src_files):
    """{old path 'src/...': (new module 'modelmapper....', kind)} for the given old file list (repo-relative)."""
    t = {}
    for old in src_files:
        p = Path(old)
        parts = p.parts
        if p.suffix != ".py":
            continue
        if len(parts) == 2:                                   # src/<name>.py
            n = p.stem
            kind = ("core" if n in CORE else "loop" if n in LOOP_FLAT else "experiments" if n in EXPERIMENTS_FLAT
                    else "archive")
            t[old] = (f"modelmapper.{kind}.{n}", kind)
        elif parts[1] == "loop_v2":
            n = p.stem
            if n == "__init__":
                t[old] = ("modelmapper.loop", "loop")
            else:
                kind = "loop" if n in LOOP_V2_LOOP else "experiments"
                t[old] = (f"modelmapper.{kind}.{n}", kind)
        elif parts[1] == "mechanisms" and p.stem != "__init__":    # src/mechanisms/ (merged from research/attention)
            t[old] = (f"modelmapper.experiments.mechanisms.{p.stem}", "experiments")
        elif parts[1] in ("ledger", "dist"):
            mod = ".".join(("modelmapper",) + parts[1:-1] + (() if p.stem == "__init__" else (p.stem,)))
            t[old] = (mod, parts[1])
    return t


def module_file(mod: str) -> Path:
    """'modelmapper.core.config' -> <repo>/modelmapper/core/config.py (or .../__init__.py for a package)."""
    p = REPO_ROOT.joinpath(*mod.split("."))
    return p / "__init__.py" if p.is_dir() else p.with_suffix(".py")


def _discover():
    """The table, rebuilt from what is on disk: every module under modelmapper/{core,loop,experiments,archive,ledger,dist}
    gets its old path back (the rules above are a bijection, apart from loop_v2/feeders vs src/feeders)."""
    olds = []
    for kind in ("core", "loop", "experiments", "archive"):
        for f in sorted((PKG_DIR / kind).glob("*.py")):
            n = f.stem
            if f"modelmapper.{kind}.{n}" in NEW_MODULES:
                continue
            if n == "__init__":
                if kind == "loop":
                    olds.append("src/loop_v2/__init__.py")
                continue
            if kind in ("core", "archive") or (kind == "loop" and n in LOOP_FLAT) or (kind == "experiments" and n in EXPERIMENTS_FLAT):
                olds.append(f"src/{n}.py")
            else:
                olds.append(f"src/loop_v2/{n}.py")
    for f in sorted((PKG_DIR / "experiments" / "mechanisms").glob("*.py")):
        if f.stem != "__init__":
            olds.append(f"src/mechanisms/{f.name}")
    for kind in ("ledger", "dist"):
        for f in sorted((PKG_DIR / kind).rglob("*.py")):
            olds.append("src/" + f.relative_to(PKG_DIR).as_posix())
    t = _build_table(olds)
    return {k: v for k, v in t.items() if module_file(v[0]).exists()}


TABLE = _discover()


def resolve(name: str):
    """An old path ('src/era.py', 'src/loop_v2/era.py', 'loop_v2/era.py', 'era.py'), a bare old module name ('era') or a
    new dotted module -> (module, kind, old path or None). Raises KeyError with suggestions."""
    s = name.strip().replace("\\", "/")
    if s.startswith("modelmapper."):
        for old, (mod, kind) in TABLE.items():
            if mod == s:
                return mod, kind, old
        if module_file(s).exists():
            return s, s.split(".")[1], None
        raise KeyError(f"no module {s}")
    s = s.removeprefix("./")
    cands = [s, f"src/{s}", f"src/{s}.py", f"{s}.py", f"src/loop_v2/{s}.py"] if not s.startswith("src/") else [s, f"{s}.py"]
    for c in cands:
        if c in TABLE:
            mod, kind = TABLE[c]
            return mod, kind, c
    hits = [o for o in TABLE if Path(o).stem == Path(s).stem]
    raise KeyError(f"{name!r} is not an old script path" + (f"; did you mean {', '.join(hits)}?" if hits else ""))


# --------------------------------------------------------------------------------------------- bare-name imports
def _flat_names():
    """Bare module names the old code imported -> (new module, kind). src/ wins over src/loop_v2 (old sys.path order)."""
    flat, v2 = {}, {}
    for old, (mod, kind) in TABLE.items():
        parts = Path(old).parts
        if len(parts) == 2:
            flat[Path(old).stem] = (mod, kind)
        elif parts[1] == "loop_v2" and Path(old).stem != "__init__":
            v2[Path(old).stem] = (mod, kind)
    out = {**v2, **flat}
    out["loop_v2"] = ("modelmapper.loop", "loop")
    out["ledger"] = ("modelmapper.ledger", "ledger")
    out["dist"] = ("modelmapper.dist", "dist")
    return out


class _Alias(importlib.abc.Loader):
    """Loads an old bare name as the already-importable new module (same object, so module state is shared)."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self._spec = module.__spec__                     # module_from_spec overwrites __spec__; put it back below
        return module

    def exec_module(self, module):
        module.__spec__ = self._spec


class _ArchiveLoader(importlib.machinery.SourceFileLoader):
    """Loads an archive module under its old bare name with __file__ = <repo>/src/<name>.py (its ROOT stays the repo)."""

    def __init__(self, fullname, real_path):
        super().__init__(fullname, str(real_path))
        self.real_path = Path(real_path)

    def exec_module(self, module):
        module.__file__ = str(REPO_ROOT / "src" / self.real_path.name)
        code = compile(self.real_path.read_bytes(), str(self.real_path), "exec")
        exec(code, module.__dict__)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, only=None):
        self.names = _flat_names()
        self.only = set(only) if only else None

    def find_spec(self, fullname, path=None, target=None):
        top, _, rest = fullname.partition(".")
        if self.only is not None and top not in self.only:
            return None
        if top not in self.names:
            return None
        mod, kind = self.names[top]
        if rest:                                         # ledger.core, loop_v2.map_core, dist.storage ...
            if kind not in ("ledger", "dist", "loop") or not mod.count(".") == 1:
                return None
            target = f"{mod}.{rest}"
            return importlib.util.spec_from_loader(fullname, _Alias(target))
        if kind == "archive":
            real = module_file(mod)
            return importlib.util.spec_from_file_location(fullname, real, loader=_ArchiveLoader(fullname, real))
        return importlib.util.spec_from_loader(fullname, _Alias(mod))


_installed = {}


def install(only=None):
    """Make the old bare names importable (idempotent). only: limit to these top-level names (e.g. {'ledger', 'config'})."""
    key = frozenset(only) if only else None
    if key in _installed:
        return _installed[key]
    f = _Finder(only)
    sys.meta_path.insert(0, f)
    _installed[key] = f
    return f


# --------------------------------------------------------------------------------------------- running
def run_module(mod: str, argv: list[str], prog: str | None = None):
    """Run a (non-archive) module as __main__ with sys.argv = [its file, *argv], exactly like `python <file> argv`."""
    old_argv = sys.argv
    sys.argv = [prog or str(module_file(mod))] + list(argv)
    try:
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = old_argv


def run_archive(mod: str, argv: list[str]):
    """Run an archive script unchanged: bare imports resolve to the new modules, __file__ is its old src/ path."""
    install()
    real = module_file(mod)
    legacy_file = REPO_ROOT / "src" / real.name
    old_argv, old_main = sys.argv, sys.modules.get("__main__")
    sys.argv = [str(legacy_file)] + list(argv)
    import types
    m = types.ModuleType("__main__")
    m.__file__ = str(legacy_file)
    m.__builtins__ = __builtins__
    sys.modules["__main__"] = m
    try:
        exec(compile(real.read_bytes(), str(real), "exec"), m.__dict__)
    finally:
        sys.argv = old_argv
        if old_main is not None:
            sys.modules["__main__"] = old_main


def run(name: str, argv: list[str]):
    """Run any old path / bare name / new module with argv, the way its old `python src/...` command did."""
    mod, kind, _ = resolve(name)
    if kind == "archive":
        return run_archive(mod, argv)
    return run_module(mod, argv)
