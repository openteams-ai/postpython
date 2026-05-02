"""POST Python build driver.

High-level pipeline: POST Python source → C99 → native shared library.

    from postpython.build import build_file, build_source

    lib_path = build_file("examples/gaussian.py")
    # → /tmp/gaussian-<hash>.so  (or .dylib on macOS)

The returned Path points to the compiled shared library.  Load it with
ctypes.CDLL or register ufunc loops with NumPy.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .checker import check_source, Violation
from .compiler.frontend import compile_source as _ir_compile
from .compiler.backend.c_backend import emit_module


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SO_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
_DEFAULT_CC = "cc"
_DEFAULT_CFLAGS = ["-O2", "-shared", "-fPIC", "-lm"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BuildError(RuntimeError):
    """Raised when the POST Python → shared-library pipeline fails."""


def build_source(
    source: str,
    *,
    filename: str = "<source>",
    output: Optional[Path] = None,
    cc: str = _DEFAULT_CC,
    cflags: list[str] | None = None,
    keep_c: bool = False,
    numpy_ufunc: bool = False,
) -> Path:
    """Compile *source* (POST Python text) to a shared library.

    Parameters
    ----------
    source      POST Python source text.
    filename    Name used in error messages.
    output      Path for the output .so / .dylib.  Defaults to a temp file.
    cc          C compiler command (default: ``cc``).
    cflags      Extra flags prepended to the compile command.
    keep_c      If True, do not delete the intermediate .c file.
    numpy_ufunc If True, pass ``-DNUMPY_UFUNC`` and include NumPy headers.

    Returns
    -------
    Path to the compiled shared library.
    """
    # ── 1. Checker ──────────────────────────────────────────────────────────
    violations = check_source(source, filename=filename)
    if violations:
        lines = "\n".join(f"  {v}" for v in violations)
        raise BuildError(f"POST Python violations in {filename!r}:\n{lines}")

    # ── 2. AST → IR ─────────────────────────────────────────────────────────
    module, errors = _ir_compile(source, filename=filename)
    if errors:
        lines = "\n".join(f"  {e}" for e in errors)
        raise BuildError(f"Compiler errors in {filename!r}:\n{lines}")

    # ── 3. IR → C99 ─────────────────────────────────────────────────────────
    c_source = emit_module(module)

    # Write C to a temp file (or a permanent location if keep_c is set).
    src_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    stem = Path(filename).stem if filename != "<source>" else "module"
    c_path = Path(tempfile.mktemp(prefix=f"{stem}-{src_hash}-", suffix=".c"))
    c_path.write_text(c_source, encoding="utf-8")

    # ── 4. C99 → shared library ─────────────────────────────────────────────
    if output is None:
        output = Path(tempfile.mktemp(prefix=f"{stem}-", suffix=_SO_SUFFIX))

    extra_flags: list[str] = list(cflags or [])
    if numpy_ufunc:
        import numpy as np  # type: ignore[import]
        numpy_inc = np.get_include()
        extra_flags += [f"-I{numpy_inc}", "-DNUMPY_UFUNC"]

    cmd = [cc, *extra_flags, *_DEFAULT_CFLAGS, "-o", str(output), str(c_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if not keep_c:
        c_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise BuildError(
            f"C compiler failed ({' '.join(cmd)}):\n{result.stderr}"
        )

    return output


def build_file(
    path: str | Path,
    **kwargs,
) -> Path:
    """Compile a POST Python source *file* to a shared library.

    All keyword arguments are forwarded to :func:`build_source`.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    kwargs.setdefault("filename", str(path))
    kwargs.setdefault("output", path.with_suffix(_SO_SUFFIX))
    return build_source(source, **kwargs)
