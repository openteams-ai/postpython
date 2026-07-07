"""CPython extension-module output (spec §9.3).

build_file(ext_module=True) produces an importable extension whose public
ufuncs are registered with NumPy via PyUFunc_FromFuncAndData — real
numpy.ufunc objects with broadcasting, dtype dispatch, and docstrings.
"""

import importlib.util
import math
import shutil

import pytest

np = pytest.importorskip("numpy")

from postpyc.build import build_file, BuildError
from postpyc.compiler.backend.ext_module import (
    ExtModuleError,
    collect_registrations,
    emit_ext_module,
)
from postpyc.compiler.frontend import compile_program

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")


KERNELS = """\
from postyp import Array, Float64
from postpyc import vectorize, guvectorize
from postpyc.math import exp

@vectorize
def sigmoid(x: Float64) -> Float64:
    \"\"\"Logistic sigmoid, numerically stable.\"\"\"
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z: Float64 = exp(x)
    return z / (1.0 + z)

@guvectorize([], "(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    \"\"\"Inner product over the last axis.\"\"\"
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * b[i]
    out[0] = acc
"""


def _import_ext(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_ext(tmp_path, name, sources, entry="main.py"):
    for fname, text in sources.items():
        (tmp_path / fname).write_text(text)
    return build_file(
        tmp_path / entry,
        ext_module=True,
        module_name=name,
        output=tmp_path / f"{name}.so",
    )


# ---------------------------------------------------------------------------
# Shim emission
# ---------------------------------------------------------------------------

def test_shim_registers_vectorize_and_gufunc(tmp_path):
    (tmp_path / "kernels.py").write_text(KERNELS)
    modules, errors = compile_program(tmp_path / "kernels.py")
    assert errors == [], errors
    shim = emit_ext_module(modules, "kernels")

    assert "PyMODINIT_FUNC" in shim and "PyInit_kernels(void)" in shim
    # Element-wise kernel: FromFuncAndData with 1-in 1-out float64.
    assert "PyUFunc_FromFuncAndData(" in shim
    assert "static char _pp_sigmoid_types[2] = {NPY_FLOAT64, NPY_FLOAT64};" in shim
    # gufunc: FromFuncAndDataAndSignature carrying the layout signature.
    assert "PyUFunc_FromFuncAndDataAndSignature(" in shim
    assert '"(n),(n)->()"' in shim
    # Docstrings flow from source to registration.
    assert "Logistic sigmoid, numerically stable." in shim


def test_shim_mixed_dtype_types_array(tmp_path):
    (tmp_path / "k.py").write_text(
        "from postyp import Int64, Float64\n"
        "from postpyc import vectorize\n"
        "@vectorize\n"
        "def scale(n: Int64, x: Float64) -> Float64:\n"
        "    return n * x\n"
    )
    modules, _ = compile_program(tmp_path / "k.py")
    shim = emit_ext_module(modules, "k")
    assert "static char _pp_scale_types[3] = {NPY_INT64, NPY_FLOAT64, NPY_FLOAT64};" in shim


def test_registrations_mirror_entry_namespace(tmp_path):
    (tmp_path / "kernels.py").write_text(KERNELS)
    (tmp_path / "pkg.py").write_text(
        "from kernels import sigmoid as expit, dot\n"
    )
    modules, errors = compile_program(tmp_path / "pkg.py")
    assert errors == [], errors
    names = [name for name, _ in collect_registrations(modules)]
    # Imported ufuncs register under their local alias names.
    assert sorted(names) == ["dot", "expit"]


def test_function_aliases_register_as_ufuncs(tmp_path):
    (tmp_path / "k.py").write_text(
        "from postyp import Float64\n"
        "from postpyc import vectorize\n"
        "@vectorize\n"
        "def lgamma_like(x: Float64) -> Float64:\n"
        "    return x * 2.0\n"
        "gammaln = lgamma_like\n"
    )
    modules, _ = compile_program(tmp_path / "k.py")
    names = dict(collect_registrations(modules))
    assert set(names) == {"lgamma_like", "gammaln"}
    assert names["gammaln"] is names["lgamma_like"]


def test_private_ufuncs_are_not_registered(tmp_path):
    (tmp_path / "k.py").write_text(
        "from postyp import Float64\n"
        "from postpyc import vectorize\n"
        "@vectorize\n"
        "def _internal(x: Float64) -> Float64:\n"
        "    return x\n"
        "@vectorize\n"
        "def public(x: Float64) -> Float64:\n"
        "    return _internal(x)\n"
    )
    modules, _ = compile_program(tmp_path / "k.py")
    names = [name for name, _ in collect_registrations(modules)]
    assert names == ["public"]


def test_invalid_module_name_raises():
    with pytest.raises(ExtModuleError):
        emit_ext_module([], "not-an-identifier")


# ---------------------------------------------------------------------------
# Build, import, and execute
# ---------------------------------------------------------------------------

@needs_cc
def test_ext_module_imports_as_real_ufuncs(tmp_path):
    ext = _build_ext(tmp_path, "ppext_kernels", {"main.py": KERNELS})
    mod = _import_ext("ppext_kernels", ext)

    assert isinstance(mod.sigmoid, np.ufunc)
    assert mod.sigmoid.nin == 1 and mod.sigmoid.nout == 1
    assert "Logistic sigmoid" in mod.sigmoid.__doc__

    x = np.linspace(-4.0, 4.0, 9)
    assert np.allclose(mod.sigmoid(x), 1.0 / (1.0 + np.exp(-x)), rtol=1e-15)


@needs_cc
def test_ext_module_gufunc_broadcasts_and_honors_strides(tmp_path):
    ext = _build_ext(tmp_path, "ppext_gu", {"main.py": KERNELS})
    mod = _import_ext("ppext_gu", ext)

    assert isinstance(mod.dot, np.ufunc)
    assert mod.dot.signature == "(n),(n)->()"

    a = np.arange(12.0).reshape(3, 4)
    b = np.ones(4)
    assert np.allclose(mod.dot(a, b), a.sum(axis=1))

    # Non-contiguous core dimension: every other column. NumPy passes real
    # inner steps; a compact-layout assumption would read wrong elements.
    sliced = np.arange(24.0).reshape(3, 8)[:, ::2]
    assert not sliced.flags.c_contiguous
    assert np.allclose(mod.dot(sliced, b), (sliced * b).sum(axis=1))


@needs_cc
def test_ext_module_cross_module_ufuncs_and_aliases(tmp_path):
    ext = _build_ext(
        tmp_path,
        "ppext_pkg",
        {
            "kernels.py": KERNELS,
            "main.py": "from kernels import sigmoid as expit, dot\n",
        },
    )
    mod = _import_ext("ppext_pkg", ext)
    assert isinstance(mod.expit, np.ufunc)
    assert mod.expit(np.float64(0.0)) == 0.5
    assert mod.dot(np.ones(3), np.ones(3)) == 3.0
    assert not hasattr(mod, "sigmoid")  # registered under the alias only


@needs_cc
def test_ext_module_reserved_names_use_post_kernels_not_libm(tmp_path):
    # A ufunc named after a libm symbol must dispatch to the POST kernel:
    # registration binds the loop pointer directly, immune to symbol-level
    # fallthrough.
    ext = _build_ext(
        tmp_path,
        "ppext_erf",
        {
            "main.py": (
                "from postyp import Float64\n"
                "from postpyc import vectorize\n"
                "@vectorize\n"
                "def erf(x: Float64) -> Float64:\n"
                "    return x + 41.5\n"
            )
        },
    )
    mod = _import_ext("ppext_erf", ext)
    assert mod.erf(np.float64(0.5)) == 42.0
    assert abs(mod.erf(np.float64(0.5)) - math.erf(0.5)) > 40.0


@needs_cc
def test_ext_module_int_float_mixed_kernel(tmp_path):
    ext = _build_ext(
        tmp_path,
        "ppext_mixed",
        {
            "main.py": (
                "from postyp import Int64, Float64\n"
                "from postpyc import vectorize\n"
                "@vectorize\n"
                "def scale(n: Int64, x: Float64) -> Float64:\n"
                "    return n * x\n"
            )
        },
    )
    mod = _import_ext("ppext_mixed", ext)
    n = np.arange(4, dtype=np.int64)
    x = np.full(4, 2.5)
    assert np.allclose(mod.scale(n, x), n * x)
