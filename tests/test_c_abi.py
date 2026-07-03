"""Stable C ABI v1 (spec §9.1.1; postpython#12).

Every artifact defines pp_<export> wrapper symbols for the entry unit's
namespace; headers and manifests describe them; module-level function
aliases (gammaln = lgamma) export natively and register as ufuncs.
"""

import ctypes
import json
import math
import shutil

import pytest

from postpython.build import build_file, BuildError
from postpython.compiler.backend.abi import (
    collect_exports,
    emit_export_wrappers,
    emit_header,
    export_manifest,
)
from postpython.compiler.frontend import compile_program, compile_source

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


LIB = """\
from postyp import Array, Float64
from postpython import vectorize, guvectorize

def _helper(x: Float64) -> Float64:
    return x * 0.5

@vectorize
def erf(x: Float64) -> Float64:
    \"\"\"Deliberately shadows a libm name.\"\"\"
    return x + 41.5

def lgamma_like(x: Float64) -> Float64:
    return x * 2.0

@guvectorize([], "(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * b[i]
    out[0] = acc

gammaln = lgamma_like
"""


def _exports_for(source: str):
    module, errors = compile_source(source)
    assert errors == [], errors
    exports, abi_errors = collect_exports([module])
    assert abi_errors == [], abi_errors
    return {e.python_name: e for e in exports}


# ---------------------------------------------------------------------------
# Export collection
# ---------------------------------------------------------------------------

def test_export_set_mirrors_namespace():
    exports = _exports_for(LIB)
    assert set(exports) == {"erf", "lgamma_like", "dot", "gammaln"}
    assert exports["erf"].kind == "ufunc"
    assert exports["erf"].c_symbol == "pp_erf"
    assert exports["erf"].kernel_symbol == "__pp_erf"  # libm-mangled kernel
    assert exports["lgamma_like"].kind == "function"
    assert exports["gammaln"].kind == "alias"
    assert exports["gammaln"].alias_of == "lgamma_like"
    assert "_helper" not in exports


def test_alias_collection_in_frontend():
    module, errors = compile_source(LIB)
    assert errors == [], errors
    assert module.function_aliases == {"gammaln": "lgamma_like"}


def test_alias_chain_resolves():
    exports = _exports_for(
        "from postyp import Float64\n"
        "def base(x: Float64) -> Float64:\n"
        "    return x\n"
        "middle = base\n"
        "outer = middle\n"
    )
    assert exports["outer"].alias_of == "base"
    assert exports["outer"].kernel_symbol == "base"


def test_alias_to_private_is_diagnosed():
    module, errors = compile_source(
        "from postyp import Float64\n"
        "def _hidden(x: Float64) -> Float64:\n"
        "    return x\n"
        "leaked = _hidden\n"
    )
    assert errors == [], errors
    _, abi_errors = collect_exports([module])
    assert any(e.code == "PP501" and "private" in e.message for e in abi_errors)


def test_export_collision_with_pp_named_function():
    module, errors = compile_source(
        "from postyp import Float64\n"
        "def erf(x: Float64) -> Float64:\n"
        "    return x\n"
        "def pp_erf(x: Float64) -> Float64:\n"
        "    return x\n"
    )
    assert errors == [], errors
    _, abi_errors = collect_exports([module])
    assert any(e.code == "PP501" for e in abi_errors)


def test_cross_module_imports_export_under_local_names(tmp_path):
    (tmp_path / "helper.py").write_text(
        "from postyp import Float64\n"
        "def double_it(x: Float64) -> Float64:\n"
        "    return x * 2.0\n"
    )
    (tmp_path / "main.py").write_text(
        "from postyp import Float64\n"
        "from helper import double_it as twice\n"
        "def quad(x: Float64) -> Float64:\n"
        "    return twice(twice(x))\n"
    )
    modules, errors = compile_program(tmp_path / "main.py")
    assert errors == [], errors
    exports, abi_errors = collect_exports(modules)
    assert abi_errors == [], abi_errors
    names = {e.python_name: e for e in exports}
    assert set(names) == {"twice", "quad"}
    assert names["twice"].module == "helper"


# ---------------------------------------------------------------------------
# Emission: wrappers, header, manifest
# ---------------------------------------------------------------------------

def test_wrapper_tu_delegates_to_kernel_symbols():
    module, _ = compile_source(LIB)
    exports, _ = collect_exports([module])
    c = emit_export_wrappers(exports)
    assert "double pp_erf(double _x)" in c
    assert "return __pp_erf(_x);" in c
    # gufunc wrapper forwards arrays and core dims, returns void.
    assert "void pp_dot(__pp_array* _a, __pp_array* _b, __pp_array* _out, int64_t _pp_dim_n)" in c
    assert "dot(_a, _b, _out, _pp_dim_n);" in c
    # alias wrapper targets the aliased kernel.
    assert "double pp_gammaln(double _x)" in c
    assert "return lgamma_like(_x);" in c


def test_header_is_self_contained_and_documented():
    module, _ = compile_source(LIB)
    exports, _ = collect_exports([module])
    h = emit_header(exports, "kern")
    assert "#ifndef PP_KERN_H" in h
    assert "typedef struct __pp_array" in h
    assert "double pp_erf(double x);" in h
    assert "(alias of lgamma_like)" in h
    assert "Deliberately shadows a libm name." in h
    assert "void erf_ufunc_loop(" in h
    assert 'extern "C"' in h


def test_manifest_schema():
    module, _ = compile_source(LIB)
    exports, _ = collect_exports([module])
    manifest = export_manifest(exports, "kern")
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "kern"
    by_name = {e["name"]: e for e in manifest["exports"]}
    erf = by_name["erf"]
    assert erf["c_symbol"] == "pp_erf"
    assert erf["kernel_symbol"] == "__pp_erf"
    assert erf["kind"] == "ufunc"
    assert erf["params"] == [
        {"name": "x", "dtype": "Float64", "is_array": False, "is_core_dim": False}
    ]
    assert erf["return_dtype"] == "Float64"
    assert erf["ufunc"] == {"loop_symbol": "erf_ufunc_loop", "signature": "()->()"}
    dot = by_name["dot"]
    assert dot["params"][-1] == {
        "name": "pp_dim_n", "dtype": "Int64", "is_array": False, "is_core_dim": True
    }
    assert dot["ufunc"]["signature"] == "(n),(n)->()"
    assert by_name["gammaln"]["kind"] == "alias"
    assert by_name["gammaln"]["alias_of"] == "lgamma_like"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@needs_cc
def test_pp_symbols_bypass_libm_ambiguity(tmp_path):
    (tmp_path / "kern.py").write_text(LIB)
    lib_path = build_file(
        tmp_path / "kern.py", output=tmp_path / "kern.so",
        emit_header=True, emit_manifest=True,
    )
    lib = ctypes.CDLL(str(lib_path))

    # `lib.erf` would fall through to libm (the POST kernel is __pp_erf);
    # `lib.pp_erf` is unambiguous — this is the Target 1 acceptance case.
    pp_erf = lib.pp_erf
    pp_erf.argtypes = [ctypes.c_double]
    pp_erf.restype = ctypes.c_double
    assert pp_erf(0.5) == 42.0
    assert abs(pp_erf(0.5) - math.erf(0.5)) > 40.0

    pp_gammaln = lib.pp_gammaln
    pp_gammaln.argtypes = [ctypes.c_double]
    pp_gammaln.restype = ctypes.c_double
    assert pp_gammaln(3.0) == 6.0

    # Header and manifest written next to the artifact.
    header = (tmp_path / "kern.h").read_text()
    assert "double pp_erf(double x);" in header
    manifest = json.loads((tmp_path / "kern.json").read_text())
    assert manifest["post_abi"] == 1

    # The manifest's loop symbol resolves in the artifact.
    loop_name = next(
        e["ufunc"]["loop_symbol"] for e in manifest["exports"] if e["name"] == "erf"
    )
    assert getattr(lib, loop_name) is not None


@needs_cc
def test_alias_and_target_agree_at_runtime(tmp_path):
    (tmp_path / "kern.py").write_text(LIB)
    lib = ctypes.CDLL(str(build_file(tmp_path / "kern.py", output=tmp_path / "k.so")))
    for name in ("pp_lgamma_like", "pp_gammaln"):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_double]
        fn.restype = ctypes.c_double
    assert lib.pp_gammaln(2.5) == lib.pp_lgamma_like(2.5) == 5.0


@needs_cc
def test_build_source_also_defines_pp_symbols():
    from postpython.build import build_source

    lib = ctypes.CDLL(str(build_source(
        "from postyp import Float64\n"
        "def halve(x: Float64) -> Float64:\n"
        "    return x / 2.0\n",
        filename="s.py",
    )))
    fn = lib.pp_halve
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(5.0) == 2.5
