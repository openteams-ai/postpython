"""Package __init__ as namespace manifest (spec §9.1) and __all__ narrowing.

An ``__init__.py`` may contain dynamic CPython-boundary code (native-module
loaders, ``globals()`` re-export machinery); the compiler consumes only its
declarative statements. Regular modules remain strictly checked.
"""

import ctypes
import importlib.util
import shutil

import pytest

from postpyc.build import build_file
from postpyc.checker import check_file, check_source
from postpyc.compiler.backend.abi import collect_exports
from postpyc.compiler.backend.ext_module import collect_registrations
from postpyc.compiler.frontend import compile_program

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")

KERNELS = """\
from postyp import Float64
from postpyc import vectorize

@vectorize
def double_it(x: Float64) -> Float64:
    \"\"\"Twice the input.\"\"\"
    return x * 2.0

@vectorize
def triple_it(x: Float64) -> Float64:
    return x * 3.0

twice = double_it
"""

# The ppspecial 0.1.1 pattern: dynamic loader machinery in __init__.
DYNAMIC_INIT = """\
\"\"\"Package with a dynamic native-preferring __init__.\"\"\"

from importlib import import_module as _import_module

from mypkg._kernels import double_it, twice

__all__ = ["double_it", "twice"]

__native_available__ = False


def _prefer_native() -> None:
    global __native_available__

    try:
        native = _import_module("mypkg_native")
    except ModuleNotFoundError:
        return

    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
    __native_available__ = True


_prefer_native()

del _prefer_native, _import_module
"""


def _write_pkg(tmp_path):
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "_kernels.py").write_text(KERNELS)
    (pkg / "__init__.py").write_text(DYNAMIC_INIT)
    return pkg


# ---------------------------------------------------------------------------
# Checker: manifests are exempt where modules are not
# ---------------------------------------------------------------------------

def test_dynamic_init_passes_checker(tmp_path):
    pkg = _write_pkg(tmp_path)
    assert check_file(pkg / "__init__.py") == []


def test_same_code_in_regular_module_still_fails():
    violations = check_source(DYNAMIC_INIT, filename="not_init.py")
    codes = {v.code for v in violations}
    assert "PP006" in codes and "PP025" in codes  # global, del


def test_manifest_declarative_parts_still_checked(tmp_path):
    pkg = tmp_path / "badpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import _kernels\n")
    violations = check_file(pkg / "__init__.py")
    assert any(v.code == "PP008" for v in violations)  # relative import


# ---------------------------------------------------------------------------
# Compilation: manifest namespace, __all__ narrowing
# ---------------------------------------------------------------------------

def test_package_with_dynamic_init_compiles(tmp_path):
    pkg = _write_pkg(tmp_path)
    modules, errors = compile_program(pkg / "__init__.py")
    assert errors == [], errors
    assert [m.name for m in modules] == ["_kernels", "__init__"]
    entry = modules[-1]
    assert entry.export_all == ["double_it", "twice"]
    assert entry.functions == []  # _prefer_native is boundary code


def test_all_narrows_exports_and_registrations(tmp_path):
    pkg = _write_pkg(tmp_path)
    modules, errors = compile_program(pkg / "__init__.py")
    assert errors == [], errors

    exports, abi_errors = collect_exports(modules)
    assert abi_errors == []
    names = {e.python_name for e in exports}
    # triple_it is imported nowhere and not in __all__; twice is an alias
    # imported through the manifest.
    assert names == {"double_it", "twice"}

    registered = {n for n, _ in collect_registrations(modules)}
    assert registered == {"double_it", "twice"}


def test_all_in_regular_module_narrows_too(tmp_path):
    (tmp_path / "solo.py").write_text(
        "from postyp import Float64\n"
        '__all__ = ["keep"]\n'
        "def keep(x: Float64) -> Float64:\n"
        "    return x\n"
        "def drop(x: Float64) -> Float64:\n"
        "    return x\n"
    )
    modules, errors = compile_program(tmp_path / "solo.py")
    assert errors == [], errors
    exports, _ = collect_exports(modules)
    assert {e.python_name for e in exports} == {"keep"}


def test_vectorized_kernel_in_manifest_is_diagnosed(tmp_path):
    pkg = tmp_path / "oops"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from postyp import Float64\n"
        "from postpyc import vectorize\n"
        "@vectorize\n"
        "def misplaced(x: Float64) -> Float64:\n"
        "    return x\n"
    )
    _, errors = compile_program(pkg / "__init__.py")
    assert any(
        e.code == "PP900" and "misplaced" in e.message and "manifest" in e.message
        for e in errors
    ), errors


# ---------------------------------------------------------------------------
# Explicit compile entry: __post__.py
# ---------------------------------------------------------------------------

def test_directory_build_prefers_post_entry(tmp_path):
    from postpyc.build import resolve_build_entry

    pkg = _write_pkg(tmp_path)
    # Without __post__.py: the manifest is the entry.
    assert resolve_build_entry(pkg).name == "__init__.py"
    # With it: __post__.py wins.
    (pkg / "__post__.py").write_text(
        "from mypkg._kernels import double_it\n"
        '__all__ = ["double_it"]\n'
    )
    assert resolve_build_entry(pkg).name == "__post__.py"


def test_post_entry_is_strictly_checked(tmp_path):
    pkg = _write_pkg(tmp_path)
    (pkg / "__post__.py").write_text(DYNAMIC_INIT.replace("mypkg._kernels", "mypkg._kernels"))
    from postpyc.build import BuildError, build_file
    with pytest.raises(BuildError) as exc:
        build_file(pkg, output=tmp_path / "x.so")
    assert "PP006" in str(exc.value)  # `global` in a strict entry


@needs_cc
def test_directory_build_via_post_entry_runs(tmp_path):
    pkg = _write_pkg(tmp_path)
    (pkg / "__post__.py").write_text(
        "from mypkg._kernels import double_it, triple_it\n"
        '__all__ = ["triple_it"]\n'
    )
    lib_path = build_file(pkg, output=tmp_path / "out.so", emit_manifest=True)
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.pp_triple_it
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(2.0) == 6.0
    import json
    manifest = json.loads((tmp_path / "out.json").read_text())
    # __post__'s own __all__ governs, independent of __init__'s.
    assert {e["name"] for e in manifest["exports"]} == {"triple_it"}


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@needs_cc
def test_dynamic_init_package_builds_and_runs(tmp_path):
    pkg = _write_pkg(tmp_path)
    lib_path = build_file(pkg / "__init__.py", output=tmp_path / "mypkg.so",
                          emit_manifest=True)
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.pp_double_it
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(2.5) == 5.0
    # alias exported; triple_it excluded by __all__
    assert lib.pp_twice(ctypes.c_double(2.0)) if False else True
    alias = lib.pp_twice
    alias.argtypes = [ctypes.c_double]
    alias.restype = ctypes.c_double
    assert alias(2.0) == 4.0
    import json
    manifest = json.loads((tmp_path / "mypkg.json").read_text())
    assert {e["name"] for e in manifest["exports"]} == {"double_it", "twice"}


@needs_cc
def test_dynamic_init_ext_module_registers_all_only(tmp_path):
    np = pytest.importorskip("numpy")
    pkg = _write_pkg(tmp_path)
    ext = build_file(pkg / "__init__.py", ext_module=True,
                     module_name="mypkg_native_test",
                     output=tmp_path / "mypkg_native_test.so")
    spec = importlib.util.spec_from_file_location("mypkg_native_test", str(ext))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert isinstance(mod.double_it, np.ufunc)
    assert isinstance(mod.twice, np.ufunc)
    assert not hasattr(mod, "triple_it")
    assert mod.double_it(np.float64(3.0)) == 6.0
