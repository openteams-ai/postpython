"""AST-based subset checker for POST Python.

Walks a Python AST and reports constructs that fall outside the
compilable subset.  A file with zero violations is a valid POST Python
source unit.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Violation dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    code: str          # short mnemonic, e.g. "PP001"
    message: str
    lineno: int
    col_offset: int
    filename: str = "<unknown>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}:{self.col_offset}: {self.code} {self.message}"


# ---------------------------------------------------------------------------
# Disqualified built-in function calls
# ---------------------------------------------------------------------------

#: Calls to these names are never allowed.
_BANNED_BUILTINS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
})

#: Calls to these names are disqualified when used dynamically.
#: (Static forms like getattr(obj, "known_name") are still banned —
#:  POST Python uses direct attribute access instead.)
_BANNED_REFLECTION: frozenset[str] = frozenset({
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
})


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class _Checker(ast.NodeVisitor):
    """Collect POST Python violations from an AST."""

    def __init__(self, filename: str = "<unknown>") -> None:
        self.filename = filename
        self.violations: list[Violation] = []
        # track function nesting for annotation checks
        self._func_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    # -- helpers -------------------------------------------------------------

    def _add(self, node: ast.AST, code: str, message: str) -> None:
        self.violations.append(Violation(
            code=code,
            message=message,
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            filename=self.filename,
        ))

    def _direct_call_name(self, node: ast.Call) -> str | None:
        """Return the direct function name for calls like f(...), or None."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        return None

    # -- dynamic execution ---------------------------------------------------

    def visit_Exec(self, node: ast.AST) -> None:           # Python 2 remnant in ast
        self._add(node, "PP001", "`exec` statement is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._direct_call_name(node)
        if name in _BANNED_BUILTINS:
            self._add(node, "PP002", f"call to `{name}()` is not allowed")
        elif name in _BANNED_REFLECTION:
            self._add(node, "PP003", f"call to `{name}()` is not allowed; use direct attribute access")
        elif name == "__import__":
            self._add(node, "PP004", "dynamic `__import__()` is not allowed; use static `import`")
        elif name == "type" and len(node.args) == 3:
            self._add(node, "PP005", "`type(name, bases, dict)` dynamic class creation is not allowed")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self._add(arg, "PP024", "starred splat in calls is not allowed; pass arguments explicitly")
        self.generic_visit(node)

    # -- dynamic namespace access --------------------------------------------

    def visit_Global(self, node: ast.Global) -> None:
        self._add(node, "PP006", "`global` statement is not allowed")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._add(node, "PP007", "`nonlocal` statement is not allowed")
        self.generic_visit(node)

    # -- dynamic imports -----------------------------------------------------

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self._add(node, "PP008", "relative imports are not allowed; use absolute imports")
        self.generic_visit(node)

    # -- dynamic class features ----------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for kw in node.keywords:
            if kw.arg == "metaclass":
                self._add(node, "PP009", f"metaclass on `{node.name}` is not allowed")
        # multiple inheritance is disqualified
        if len(node.bases) > 1:
            self._add(node, "PP010", f"`{node.name}` has multiple bases; multiple inheritance is not allowed")
        self.generic_visit(node)

    # -- async / coroutines --------------------------------------------------

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node, "PP011", f"`async def {node.name}` is not allowed in POST Python v1 subset")
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self._add(node, "PP012", "`await` expression is not allowed")
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._add(node, "PP013", "`async for` is not allowed")
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._add(node, "PP014", "`async with` is not allowed")
        self.generic_visit(node)

    # -- untyped function signatures -----------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_stack.append(node)
        self._check_func_annotations(node)
        self.generic_visit(node)
        self._func_stack.pop()

    def _check_func_annotations(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        args = node.args
        all_args = (
            args.posonlyargs
            + args.args
            + args.kwonlyargs
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        )
        for i, arg in enumerate(all_args):
            # self/cls are conventionally untyped; their type is the enclosing class
            if i == 0 and arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                self._add(
                    arg,
                    "PP020",
                    f"parameter `{arg.arg}` in `{node.name}` has no type annotation",
                )
        if node.returns is None:
            self._add(node, "PP021", f"`{node.name}` has no return type annotation")

    # -- variadic arguments --------------------------------------------------

    def visit_arguments(self, node: ast.arguments) -> None:
        if node.vararg is not None:
            self._add(node.vararg, "PP022", "`*args` (var-positional) is not allowed; use typed tuple parameters")
        if node.kwarg is not None:
            self._add(node.kwarg, "PP023", "`**kwargs` (var-keyword) is not allowed; use explicit keyword parameters")
        self.generic_visit(node)

    # -- del -----------------------------------------------------------------

    def visit_Delete(self, node: ast.Delete) -> None:
        self._add(node, "PP025", "`del` statement is not allowed")
        self.generic_visit(node)

    # -- walrus operator (allowed, but flag for awareness) -------------------
    # NamedExpr (:=) is fine — it has clear scoping and can be typed.

    # -- yield / generators --------------------------------------------------

    def visit_Yield(self, node: ast.Yield) -> None:
        self._add(node, "PP030", "`yield` is not allowed; generators are not in the compilable subset")
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._add(node, "PP031", "`yield from` is not allowed")
        self.generic_visit(node)

    # -- lambda --------------------------------------------------------------

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._add(node, "PP032", "`lambda` is not allowed; use a named function with annotations")
        self.generic_visit(node)

    # -- exception groups (3.11+) --------------------------------------------

    def visit_TryStar(self, node: ast.AST) -> None:
        self._add(node, "PP033", "`except*` (exception groups) are not allowed")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_source(source: str, filename: str = "<unknown>") -> list[Violation]:
    """Parse *source* and return all POST Python violations."""
    try:
        tree = ast.parse(source, filename=filename, type_comments=True)
    except SyntaxError as exc:
        return [Violation(
            code="PP000",
            message=f"syntax error: {exc.msg}",
            lineno=exc.lineno or 0,
            col_offset=exc.offset or 0,
            filename=filename,
        )]
    checker = _Checker(filename=filename)
    checker.visit(tree)
    return sorted(checker.violations, key=lambda v: (v.lineno, v.col_offset))


def check_file(path: str | Path) -> list[Violation]:
    """Read *path* and return all POST Python violations."""
    path = Path(path)
    return check_source(path.read_text(encoding="utf-8"), filename=str(path))


def is_valid(source: str, filename: str = "<unknown>") -> bool:
    """Return True if *source* has no POST Python violations."""
    return len(check_source(source, filename)) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="postpython-check",
        description="Check whether Python source files conform to the POST Python subset.",
    )
    parser.add_argument("files", nargs="+", metavar="FILE", help="source files to check")
    args = parser.parse_args(argv)

    total_violations = 0
    for path in args.files:
        violations = check_file(path)
        for v in violations:
            print(v)
        total_violations += len(violations)

    if total_violations:
        print(f"\n{total_violations} violation(s) found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
