"""POST Python build driver.

High-level pipeline: POST Python source → C99 → native shared library.

    from post_py.build import build_file, build_source

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
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .checker import check_source
from .compiler.frontend import compile_source as _ir_compile, compile_program
from .compiler.backend.abi import (
    collect_exports,
    emit_export_wrappers,
    emit_header as abi_emit_header,
    export_manifest,
)
from .compiler.backend.c_backend import emit_module
from .compiler.backend.ext_module import emit_ext_module, ExtModuleError
from .compiler.ir import Module


def _export_wrapper_source(modules: list[Module]) -> str:
    exports, abi_errors = collect_exports(modules)
    if abi_errors:
        lines = "\n".join(f"  {e}" for e in abi_errors)
        raise BuildError(f"Export ABI errors:\n{lines}")
    return emit_export_wrappers(exports)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SO_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
_DEFAULT_CC = "cc"
_COMPILE_FLAGS = ["-O2", "-fPIC"]
_LINK_FLAGS = ["-shared", "-lm"]
# Extension modules resolve CPython symbols at import time; on macOS that
# requires a bundle with dynamic lookup instead of a plain dylib.
_EXT_LINK_FLAGS = (
    ["-bundle", "-undefined", "dynamic_lookup", "-lm"]
    if platform.system() == "Darwin"
    else ["-shared", "-lm"]
)


def _ext_suffix() -> str:
    import importlib.machinery
    return importlib.machinery.EXTENSION_SUFFIXES[0]


def _python_include_flags() -> list[str]:
    import sysconfig
    import numpy as np  # type: ignore[import]
    return [f"-I{sysconfig.get_paths()['include']}", f"-I{np.get_include()}"]


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
    extra_c_sources: list[tuple[str, str]] | None = None,
) -> Path:
    """Emit each module to C, compile to objects, link a shared library.

    *extra_c_sources* are (stem, C source) pairs compiled and linked in —
    used for the stable-ABI export-wrapper translation unit.
    """
    if output is None:
        out_fd, out_str = tempfile.mkstemp(prefix=f"{tag}-", suffix=_SO_SUFFIX)
        os.close(out_fd)
        output = Path(out_str)

    extra_flags = [*(cflags or []), *_numpy_flags(numpy_ufunc)]
    work_dir = Path(tempfile.mkdtemp(prefix=f"pp-build-{tag}-"))
    objects: list[Path] = []
    c_paths: list[Path] = []
    try:
        sources = [
            (f"{index:02d}-{module.name}", emit_module(module, dep_modules=module.dep_modules))
            for index, module in enumerate(modules)
        ]
        sources.extend(extra_c_sources or [])
        for stem, c_source in sources:
            c_path = work_dir / f"{stem}.c"
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


def _build_extension(
    modules: list[Module],
    *,
    module_name: str,
    output: Path,
    cc: str,
    cflags: list[str] | None,
    keep_c: bool,
    extra_c_sources: list[tuple[str, str]] | None = None,
) -> Path:
    """Compile the program's translation units plus the PyInit shim and
    link them into an importable CPython extension module."""
    try:
        shim_source = emit_ext_module(modules, module_name)
    except ExtModuleError as exc:
        raise BuildError(str(exc)) from exc

    extra_flags = list(cflags or [])
    work_dir = Path(tempfile.mkdtemp(prefix=f"pp-ext-{module_name}-"))
    objects: list[Path] = []
    c_paths: list[Path] = []
    try:
        # Ordinary translation units: pure C, no Python headers needed.
        sources = [
            (f"{index:02d}-{module.name}", emit_module(module, dep_modules=module.dep_modules))
            for index, module in enumerate(modules)
        ]
        sources.extend(extra_c_sources or [])
        for stem, c_source in sources:
            c_path = work_dir / f"{stem}.c"
            c_path.write_text(c_source, encoding="utf-8")
            c_paths.append(c_path)
            obj_path = c_path.with_suffix(".o")
            _run([cc, *extra_flags, *_COMPILE_FLAGS, "-c", str(c_path), "-o", str(obj_path)])
            objects.append(obj_path)

        # The module shim needs CPython and NumPy headers.
        shim_path = work_dir / f"{module_name}_ext.c"
        shim_path.write_text(shim_source, encoding="utf-8")
        c_paths.append(shim_path)
        shim_obj = shim_path.with_suffix(".o")
        _run([
            cc, *extra_flags, *_python_include_flags(), *_COMPILE_FLAGS,
            "-c", str(shim_path), "-o", str(shim_obj),
        ])
        objects.append(shim_obj)

        _run([cc, *extra_flags, *_EXT_LINK_FLAGS, "-o", str(output), *map(str, objects)])
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
        extra_c_sources=[("pp_exports", _export_wrapper_source([module]))],
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
    ext_module: bool = False,
    module_name: Optional[str] = None,
    emit_header: bool = False,
    emit_manifest: bool = False,
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

    With ``ext_module=True`` the output is an importable CPython extension
    module (spec §9.3): every public ufunc of the entry translation unit —
    including ones it imports from other POST modules — is registered as a
    real ``numpy.ufunc``. *module_name* sets the importable name; it
    defaults to the entry file's stem, or the package directory's name
    when the entry is an ``__init__.py``.

    Every artifact also defines the stable C ABI symbols ``pp_<export>``
    (spec §9.1 Package ABI v1). With ``emit_header=True`` /
    ``emit_manifest=True`` the C header and the JSON export manifest are
    written next to the output as ``<output>.h`` / ``<output>.json``.
    """
    path = Path(path)
    modules, errors = compile_program(path, search_paths=search_paths)
    if errors:
        lines = "\n".join(f"  {e}" for e in errors)
        raise BuildError(f"Compiler errors building {str(path)!r}:\n{lines}")

    artifact = module_name or (
        path.resolve().parent.name if path.stem == "__init__" else path.stem
    )
    exports, abi_errors = collect_exports(modules)
    if abi_errors:
        lines = "\n".join(f"  {e}" for e in abi_errors)
        raise BuildError(f"Export ABI errors building {str(path)!r}:\n{lines}")
    wrapper_source = emit_export_wrappers(exports)

    if ext_module:
        if output is None:
            output = path.resolve().parent / f"{artifact}{_ext_suffix()}"
        result = _build_extension(
            modules,
            module_name=artifact,
            output=output,
            cc=cc,
            cflags=cflags,
            keep_c=keep_c,
            extra_c_sources=[("pp_exports", wrapper_source)],
        )
    else:
        if output is None:
            output = path.with_suffix(_SO_SUFFIX)
        result = _link_modules(
            modules,
            tag=path.stem,
            output=output,
            cc=cc,
            cflags=cflags,
            keep_c=keep_c,
            numpy_ufunc=numpy_ufunc,
            extra_c_sources=[("pp_exports", wrapper_source)],
        )

    if emit_header:
        result.with_suffix(".h").write_text(
            abi_emit_header(exports, artifact), encoding="utf-8",
        )
    if emit_manifest:
        result.with_suffix(".json").write_text(
            json.dumps(export_manifest(exports, artifact), indent=2) + "\n",
            encoding="utf-8",
        )
    return result
