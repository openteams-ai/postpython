"""POST Python build driver.

High-level pipeline: POST Python source → C99 → native shared library.

    from postpython.build import build_file, build_source

    lib_path = build_file("examples/gaussian.py")
    # → /tmp/gaussian-<hash>.so  (or .dylib on macOS)

build_file compiles the entry file *and every POST module it imports*:
each translation unit is emitted as its own C source, compiled to an
object file, and the objects are linked into one shared library
(spec §9.1). build_source compiles a single translation unit from text.

The returned Path points to the compiled shared library.  Load it with
ctypes.CDLL or register ufunc loops with NumPy.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .checker import check_source
from .compiler.frontend import compile_source as _ir_compile, compile_program
from .compiler.backend.c_backend import emit_module
from .compiler.ir import Module


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SO_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
_DEFAULT_CC = "cc"
_COMPILE_FLAGS = ["-O2", "-fPIC"]
_LINK_FLAGS = ["-shared", "-lm"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BuildError(RuntimeError):
    """Raised when the POST Python → shared-library pipeline fails."""


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(
            f"C compiler failed ({' '.join(cmd)}):\n{result.stderr}"
        )


def _numpy_flags(numpy_ufunc: bool) -> list[str]:
    if not numpy_ufunc:
        return []
    import numpy as np  # type: ignore[import]
    return [f"-I{np.get_include()}", "-DNUMPY_UFUNC"]


def _link_modules(
    modules: list[Module],
    *,
    tag: str,
    output: Optional[Path],
    cc: str,
    cflags: list[str] | None,
    keep_c: bool,
    numpy_ufunc: bool,
) -> Path:
    """Emit each module to C, compile to objects, link a shared library."""
    if output is None:
        out_fd, out_str = tempfile.mkstemp(prefix=f"{tag}-", suffix=_SO_SUFFIX)
        os.close(out_fd)
        output = Path(out_str)

    extra_flags = [*(cflags or []), *_numpy_flags(numpy_ufunc)]
    work_dir = Path(tempfile.mkdtemp(prefix=f"pp-build-{tag}-"))
    objects: list[Path] = []
    c_paths: list[Path] = []
    try:
        for index, module in enumerate(modules):
            c_source = emit_module(module, dep_modules=module.dep_modules)
            c_path = work_dir / f"{index:02d}-{module.name}.c"
            c_path.write_text(c_source, encoding="utf-8")
            c_paths.append(c_path)

            obj_path = c_path.with_suffix(".o")
            _run([cc, *extra_flags, *_COMPILE_FLAGS, "-c", str(c_path), "-o", str(obj_path)])
            objects.append(obj_path)

        _run([cc, *extra_flags, *_LINK_FLAGS, "-o", str(output), *map(str, objects)])
    finally:
        if not keep_c:
            for p in (*c_paths, *objects):
                p.unlink(missing_ok=True)
            try:
                work_dir.rmdir()
            except OSError:
                pass

    return output


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
    """Compile *source* (a single POST Python translation unit) to a
    shared library.

    POST module imports are not resolved from bare source text; use
    :func:`build_file` when the code imports other POST translation units.

    Parameters
    ----------
    source      POST Python source text.
    filename    Name used in error messages.
    output      Path for the output .so / .dylib.  Defaults to a temp file.
    cc          C compiler command (default: ``cc``).
    cflags      Extra flags prepended to the compile command.
    keep_c      If True, do not delete the intermediate .c files.
    numpy_ufunc If True, pass ``-DNUMPY_UFUNC`` and include NumPy headers.

    Returns
    -------
    Path to the compiled shared library.
    """
    violations = check_source(source, filename=filename)
    if violations:
        lines = "\n".join(f"  {v}" for v in violations)
        raise BuildError(f"POST Python violations in {filename!r}:\n{lines}")

    module, errors = _ir_compile(source, filename=filename)
    if errors:
        lines = "\n".join(f"  {e}" for e in errors)
        raise BuildError(f"Compiler errors in {filename!r}:\n{lines}")

    src_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    stem = Path(filename).stem if filename != "<source>" else "module"
    return _link_modules(
        [module],
        tag=f"{stem}-{src_hash}",
        output=output,
        cc=cc,
        cflags=cflags,
        keep_c=keep_c,
        numpy_ufunc=numpy_ufunc,
    )


def build_file(
    path: str | Path,
    *,
    output: Optional[Path] = None,
    cc: str = _DEFAULT_CC,
    cflags: list[str] | None = None,
    keep_c: bool = False,
    numpy_ufunc: bool = False,
    search_paths: list[Path] | None = None,
) -> Path:
    """Compile a POST Python source *file* — and every POST module it
    imports — into one shared library.

    Each translation unit becomes its own object file; the objects are
    linked together, so cross-module calls resolve to the compiled POST
    functions rather than to any same-named libm symbol.

    POST imports resolve against the entry file's source root plus
    *search_paths*. Standard-library and site-packages modules are
    CPython-boundary imports and are never compiled implicitly; pass the
    package's source directory root in *search_paths* to opt a module in.
    """
    path = Path(path)
    modules, errors = compile_program(path, search_paths=search_paths)
    if errors:
        lines = "\n".join(f"  {e}" for e in errors)
        raise BuildError(f"Compiler errors building {str(path)!r}:\n{lines}")

    if output is None:
        output = path.with_suffix(_SO_SUFFIX)

    return _link_modules(
        modules,
        tag=path.stem,
        output=output,
        cc=cc,
        cflags=cflags,
        keep_c=keep_c,
        numpy_ufunc=numpy_ufunc,
    )
