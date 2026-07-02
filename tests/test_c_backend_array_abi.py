"""C backend tests for the POST Array ABI."""

from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_source


def emit_c(source: str) -> str:
    module, errors = compile_source(source)
    assert errors == []
    return emit_module(module)


def test_guvectorize_inner_kernel_receives_array_view_with_runtime_strides():
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
    # The view strides come from NumPy's inner (core-dimension) steps,
    # which follow the outer steps in the steps array: 3 args -> steps[3..].
    assert "int64_t cstep0_0 = (int64_t)steps[3];" in c_source
    assert "int64_t cstep1_0 = (int64_t)steps[4];" in c_source
    assert "int64_t _pp_strides_0[1] = {cstep0_0};" in c_source
    assert "int64_t _pp_strides_1[1] = {cstep1_0};" in c_source
    assert "__pp_array _pp_arg_0 = {arg0 + _i * step0, 1, _pp_shape_0, _pp_strides_0, 0};" in c_source
    assert "dot(&_pp_arg_0, &_pp_arg_1, &_pp_arg_2, _pp_dim_n)" in c_source
    # The kernel reads element strides from the view at runtime instead of
    # assuming a compact layout, so non-contiguous inputs index correctly.
    assert "__pp_array_stride(_a, 0)" in c_source


def test_guvectorize_2d_core_views_consume_steps_in_argument_order():
    source = """\
from postyp import Array, Float64, FOrder
from postpython import guvectorize

@guvectorize([], "(m,n)->()")
def first(x: Array[Float64, FOrder], out: Array[Float64]) -> None:
    out[0] = x[1][1]
"""

    c_source = emit_c(source)

    # Two core dims on argument 0; 2 args -> inner steps start at steps[2].
    assert "int64_t cstep0_0 = (int64_t)steps[2];" in c_source
    assert "int64_t cstep0_1 = (int64_t)steps[3];" in c_source
    assert "int64_t _pp_strides_0[2] = {cstep0_0, cstep0_1};" in c_source
    # The kernel indexes through the runtime strides of both axes.
    assert "__pp_array_stride(_x, 0)" in c_source
    assert "__pp_array_stride(_x, 1)" in c_source


def test_ufunc_loop_symbol_is_exported():
    source = """\
from postyp import Array, Float64
from postpython import guvectorize

@guvectorize([], "(n)->()")
def total(a: Array[Float64], out: Array[Float64]) -> None:
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i]
    out[0] = acc
"""

    c_source = emit_c(source)

    # The loop must be a visible symbol so it can be looked up via ctypes
    # and registered with PyUFunc_FromFuncAndData.
    assert "void total_ufunc_loop(" in c_source
    assert "static void total_ufunc_loop(" not in c_source
