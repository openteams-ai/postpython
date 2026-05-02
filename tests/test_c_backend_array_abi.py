"""C backend tests for the POST Array ABI."""

from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_source


def emit_c(source: str) -> str:
    module, errors = compile_source(source)
    assert errors == []
    return emit_module(module)


def test_guvectorize_inner_kernel_receives_array_view_with_strides():
    source = """\
from postyp import Array, Float64
from postpython import guvectorize

@guvectorize([], "(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    out[0] = result
"""

    c_source = emit_c(source)

    assert "void dot(__pp_array* _a, __pp_array* _b, __pp_array* _out, int64_t _pp_dim_n)" in c_source
    assert "int64_t _pp_shape_0[1] = {_pp_dim_n};" in c_source
    assert "int64_t _pp_strides_0[1] = {sizeof(double)};" in c_source
    assert "__pp_array _pp_arg_0 = {arg0 + _i * step0, 1, _pp_shape_0, _pp_strides_0, 0};" in c_source
    assert "dot(&_pp_arg_0, &_pp_arg_1, &_pp_arg_2, _pp_dim_n)" in c_source


def test_guvectorize_fortran_layout_view_uses_column_major_strides():
    source = """\
from postyp import Array, Float64, FOrder
from postpython import guvectorize

@guvectorize([], "(m,n)->()")
def first(x: Array[Float64, FOrder], out: Array[Float64]) -> None:
    out[0] = x[1][1]
"""

    c_source = emit_c(source)

    assert "int64_t _pp_strides_0[2] = {sizeof(double), _pp_dim_m * sizeof(double)};" in c_source
