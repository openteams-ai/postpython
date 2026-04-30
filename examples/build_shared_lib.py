"""POST Python → shared library: complete worked example.

Pipeline
--------
1. Check   postpython.checker      — reject non-compilable syntax (PP000-PP033)
2. Lower   postpython.compiler.frontend  — AST → POST Python IR
3. Emit    postpython.compiler.backend.c_backend  — IR → C99 source
4. Compile system C compiler       — C99 → native shared library
5. Load    ctypes                  — call compiled functions from Python
6. Batch   numpy                   — broadcast over arrays via the gufunc loop

Run from the project root:

    python examples/build_shared_lib.py
"""

from __future__ import annotations

import ctypes
import math
import sys
import textwrap
from pathlib import Path

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from postpython.checker import check_source
from postpython.compiler.frontend import compile_source
from postpython.compiler.backend.c_backend import emit_module
from postpython.build import build_file, BuildError

SOURCE = ROOT / "examples" / "gaussian.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def indent(text: str, n: int = 4) -> str:
    return textwrap.indent(text, " " * n)


# ---------------------------------------------------------------------------
# Step 1: POST Python compliance check
# ---------------------------------------------------------------------------

section("Step 1 — POST Python compliance check")

source = SOURCE.read_text()
violations = check_source(source, filename=str(SOURCE))

if violations:
    for v in violations:
        print(f"  FAIL  {v}")
    sys.exit(1)

print(f"  PASS  {SOURCE.name}  — no violations")


# ---------------------------------------------------------------------------
# Step 2: AST → IR
# ---------------------------------------------------------------------------

section("Step 2 — AST → POST Python IR")

module, errors = compile_source(source, filename=str(SOURCE))
if errors:
    for e in errors:
        print(f"  ERROR  {e}")
    sys.exit(1)

print(f"  Module '{module.name}'  ({len(module.functions)} functions)")
for fn in module.functions:
    sig = getattr(fn, "gufunc_sig", None)
    sig_str = f"  [{sig}]" if sig else ""
    param_str = ", ".join(f"{p.name}: {p.dtype.__name__}" for p in fn.params)
    print(f"    {fn.name}({param_str}) -> {fn.return_dtype.__name__ if fn.return_dtype else 'void'}{sig_str}")


# ---------------------------------------------------------------------------
# Step 3: IR → C99
# ---------------------------------------------------------------------------

section("Step 3 — IR → C99")

c_source = emit_module(module)

# Show a representative excerpt (skip the boilerplate preamble).
lines = c_source.splitlines()
body_start = next(i for i, l in enumerate(lines) if l.startswith("double") or l.startswith("static"))
print(indent("\n".join(lines[body_start : body_start + 35])))
print("    ...")
print(f"\n  Total C source: {len(c_source)} chars / {len(lines)} lines")


# ---------------------------------------------------------------------------
# Step 4: compile to shared library
# ---------------------------------------------------------------------------

section("Step 4 — C99 → shared library")

try:
    lib_path = build_file(SOURCE, keep_c=False)
except BuildError as exc:
    print(f"  BUILD FAILED:\n{exc}")
    sys.exit(1)

size_kb = lib_path.stat().st_size / 1024
print(f"  OK  {lib_path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Step 5: call via ctypes (scalar)
# ---------------------------------------------------------------------------

section("Step 5 — call compiled functions via ctypes")

lib = ctypes.CDLL(str(lib_path))

# square(double _x) -> double
lib.square.argtypes = [ctypes.c_double]
lib.square.restype  = ctypes.c_double

print("  square(x):")
for x in [0.0, 1.0, 3.0, -2.5, 1e6]:
    result = lib.square(x)
    expected = x * x
    ok = abs(result - expected) < 1e-10 * max(abs(expected), 1.0)
    mark = "OK" if ok else "FAIL"
    print(f"    [{mark}]  square({x}) = {result}  (expected {expected})")

# gaussian(double _x, double _mu, double _sigma) -> double
lib.gaussian.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.gaussian.restype  = ctypes.c_double

print("\n  gaussian(x, mu=0, sigma=1):")
mu, sigma = 0.0, 1.0
for x in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
    g = lib.gaussian(x, mu, sigma)
    expected = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    ok = abs(g - expected) < 1e-10
    mark = "OK" if ok else "FAIL"
    print(f"    [{mark}]  x={x:+.1f}  g={g:.8f}  expected={expected:.8f}")

# relu(double _x) -> double
lib.relu.argtypes = [ctypes.c_double]
lib.relu.restype  = ctypes.c_double

print("\n  relu(x):")
for x in [-3.0, -0.001, 0.0, 0.001, 5.0]:
    r = lib.relu(x)
    expected = max(0.0, x)
    ok = r == expected
    mark = "OK" if ok else "FAIL"
    print(f"    [{mark}]  relu({x}) = {r}  (expected {expected})")


# ---------------------------------------------------------------------------
# Step 6: call via NumPy (broadcast over arrays using interpreted gufunc)
# ---------------------------------------------------------------------------

section("Step 6 — NumPy broadcast via interpreted @gufunc")

try:
    import numpy as np  # type: ignore[import]
except ImportError:
    print("  (numpy not installed; skipping broadcast demo)")
    sys.exit(0)

# Import the POST Python module — the @gufunc decorator provides an
# interpreted broadcast path until the compiled gufunc loop is registered.
import importlib.util, types

spec = importlib.util.spec_from_file_location("gaussian", SOURCE)
assert spec and spec.loader
pp_mod: types.ModuleType = importlib.util.load_module_from_spec(spec, spec)
spec.loader.exec_module(pp_mod)  # type: ignore[union-attr]

xs = np.linspace(-3, 3, 7)

print(f"\n  xs = {xs}")

squares = pp_mod.square(xs)
print(f"\n  square(xs)   = {np.round(squares, 4)}")

gaussians = pp_mod.gaussian(xs, 0.0, 1.0)
print(f"  gaussian(xs) = {np.round(gaussians, 6)}")

relus = pp_mod.relu(xs)
print(f"  relu(xs)     = {np.round(relus, 4)}")

# Verify against numpy reference.
ref_g = np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi)
max_err = float(np.max(np.abs(gaussians - ref_g)))
print(f"\n  max |gaussian - scipy_ref| = {max_err:.2e}")

print("\n  All steps complete.")
