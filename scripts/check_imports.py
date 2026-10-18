"""Static import check for the modelmapper package (no module is imported or run, so it is safe and instant).

  python scripts/check_imports.py

Checks, for every .py file under modelmapper/ and scripts/:
  1. it compiles;
  2. package modules (everything except archive/ and the byte-frozen ledger files) use no old bare names
     (`import config`, `from neurons import ...`, `from ledger.core import ...`);
  3. every `modelmapper.*` import names a module that exists, and every `from <module> import <name>` names something
     that module defines (a function, class, assignment, import) or a submodule;
  4. archive/ and frozen ledger files: every bare name they import is one that modelmapper.legacy resolves.
Exit 1 on any problem.
"""
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from modelmapper import legacy  # noqa: E402

PKG = REPO / "modelmapper"
FLAT = legacy._flat_names()
_defs = {}


def module_path(mod):
    p = REPO.joinpath(*mod.split("."))
    if p.is_dir() and (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def defined_names(path):
    if path not in _defs:
        names = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.For, ast.With)):
                for x in ast.walk(n):
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                        names.add(x.id)
        _defs[path] = names
    return _defs[path]


def frozen(rel):
    return rel.startswith("modelmapper/ledger/") and (rel.endswith(("ledger/core.py", "ledger/pack.py")) or "/importers/" in rel)


def main():
    problems = []
    files = sorted(PKG.rglob("*.py")) + sorted((REPO / "scripts").glob("*.py"))
    n_imports = 0
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), rel)
            compile(tree, rel, "exec")
        except SyntaxError as e:
            problems.append(f"{rel}: does not compile: {e}"); continue
        legacy_style = rel.startswith("modelmapper/archive/") or frozen(rel) or rel in (
            "scripts/patch_pipeline_once.py", "scripts/shorten_long_names.py")
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [(a.name, None) for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods = [(n.module, [a.name for a in n.names])]
            else:
                continue
            for mod, names in mods:
                top = mod.split(".")[0]
                if top == "modelmapper":
                    n_imports += 1
                    p = module_path(mod)
                    if p is None:
                        problems.append(f"{rel}:{n.lineno}: no module {mod}"); continue
                    for nm in names or []:
                        if nm == "*" or module_path(f"{mod}.{nm}") or nm in defined_names(p):
                            continue
                        problems.append(f"{rel}:{n.lineno}: {mod} has no {nm!r}")
                elif top in FLAT:
                    if not legacy_style:
                        problems.append(f"{rel}:{n.lineno}: old bare import {mod!r} in a package module")
                    else:
                        target, kind = FLAT[top]
                        rest = mod[len(top):]
                        if rest and not module_path(target + rest):
                            problems.append(f"{rel}:{n.lineno}: {mod} -> {target + rest} does not exist")
    print(f"{len(files)} files, {n_imports} modelmapper imports checked")
    for p in problems:
        print("PROBLEM", p)
    print("OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
