"""Numba-shaped vectorize and guvectorize decorator tests."""

import pytest

from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_source
from postpython.compiler.ir import UFunc as UFuncIR
from postpython import guvectorize, vectorize
from postyp import Array, Float64


def compile_ok(source: str):
    module, errors = compile_source(source)
    assert errors == []
    return module


def test_vectorize_interpreted_broadcasts_scalar_kernel():
    np = pytest.importorskip("numpy")

    @vectorize
    def add(x: Float64, y: Float64) -> Float64:
        return x + y

    result = add(np.array([1.0, 2.0, 3.0]), 10.0)

    assert np.allclose(result, np.array([11.0, 12.0, 13.0]))
    assert add.__pp_sig__ == "(),()->()"


def test_vectorize_accepts_numba_style_signature_list_and_options():
    np = pytest.importorskip("numpy")

    @vectorize(["float64(float64, float64)"], target="cpu", nopython=True)
    def add(x: Float64, y: Float64) -> Float64:
        return x + y

    assert np.allclose(add(np.array([1.0]), np.array([2.0])), np.array([3.0]))
    assert add.__pp_type_signatures__ == ["float64(float64, float64)"]
    assert add.__pp_options__["target"] == "cpu"


def test_guvectorize_interpreted_uses_output_parameter():
    np = pytest.importorskip("numpy")

    @guvectorize([], "(n),()->(n)")
    def add_scalar(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
        for i in range(len(x)):
            out[i] = x[i] + y

    result = add_scalar(np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0)

    assert np.allclose(result, np.array([[11.0, 12.0], [13.0, 14.0]]))


def test_guvectorize_interpreted_supports_scalar_output_parameter():
    np = pytest.importorskip("numpy")

    @guvectorize([], "(n),()->()")
    def sum_plus(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
        acc: Float64 = 0.0
        for i in range(len(x)):
            acc += x[i] + y
        out[0] = acc

    result = sum_plus(np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0)

    assert np.allclose(result, np.array([23.0, 27.0]))


def test_compiler_lowers_vectorize_as_scalar_ufunc():
    source = """\
from postpython import vectorize
from postyp import Float64

@vectorize(["float64(float64, float64)"], target="cpu")
def add(x: Float64, y: Float64) -> Float64:
    return x + y
"""

    module = compile_ok(source)
    fn = module.functions[0]
    c_source = emit_module(module)

    assert isinstance(fn, UFuncIR)
    assert str(fn.ufunc_sig) == "(),()->()"
    assert "static void add_ufunc_loop(" in c_source


def test_compiler_lowers_guvectorize_numba_form():
    source = """\
from postpython import guvectorize
from postyp import Array, Float64

@guvectorize([], "(n),()->(n)", target="cpu")
def add_scalar(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
    for i in range(len(x)):
        out[i] = x[i] + y
"""

    module = compile_ok(source)
    fn = module.functions[0]
    c_source = emit_module(module)

    assert isinstance(fn, UFuncIR)
    assert str(fn.ufunc_sig) == "(n),()->(n)"
    assert "void add_scalar(__pp_array* _x, double _y, __pp_array* _out, int64_t _pp_dim_n)" in c_source


def test_guvectorize_requires_output_parameters():
    source = """\
from postpython import guvectorize
from postyp import Array, Float64

@guvectorize([], "(n)->()")
def norm(x: Array[Float64]) -> Float64:
    return 0.0
"""

    _, errors = compile_source(source)

    assert [error.code for error in errors] == ["PP100", "PP100"]
    assert "output parameter" in errors[0].message
    assert "return None" in errors[1].message
