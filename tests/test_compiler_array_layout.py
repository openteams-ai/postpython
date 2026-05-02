"""Compiler tests for POST Array layout metadata."""

from postpython.compiler.frontend import compile_source
from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.ir import ArrayLoad, ArrayStride, BinOp, BinOpInstr, Const
from postyp import COrder, FOrder, Strides


def compile_ok(source: str):
    module, errors = compile_source(source)
    assert errors == []
    return module


def error_codes(source: str) -> list[str]:
    _, errors = compile_source(source)
    return [error.code for error in errors]


def emit_c(source: str) -> str:
    return emit_module(compile_ok(source))


def array_load_byte_offsets(source: str) -> list[int]:
    module = compile_ok(source)
    values = {}
    pending_binops = []
    load_indexes = []
    for block in module.functions[0].blocks:
        for instr in block.instructions:
            if isinstance(instr, Const):
                values[instr.result.name] = instr.value
            elif isinstance(instr, BinOpInstr):
                pending_binops.append(instr)
            elif isinstance(instr, ArrayLoad):
                load_indexes.append(instr.index.name)

    changed = True
    while changed:
        changed = False
        for instr in pending_binops:
            if instr.result.name in values:
                continue
            left = values.get(instr.left.name)
            right = values.get(instr.right.name)
            if not isinstance(left, int) or not isinstance(right, int):
                continue
            if instr.op is BinOp.MUL:
                values[instr.result.name] = left * right
            elif instr.op is BinOp.ADD:
                values[instr.result.name] = left + right
            else:
                continue
            changed = True

    return [values[name] for name in load_indexes]


def test_array_param_defaults_to_c_order_layout():
    source = """\
from postyp import Array, Float64

def f(x: Array[Float64]) -> None:
    pass
"""

    module = compile_ok(source)

    assert module.functions[0].params[0].layout == COrder


def test_fortran_order_annotation_is_preserved_in_ir():
    source = """\
from postyp import Array, Float64, Shape, FOrder

def f(x: Array[Float64, Shape[3, 3], FOrder]) -> None:
    pass
"""

    module = compile_ok(source)

    assert module.functions[0].params[0].layout == FOrder


def test_explicit_strides_annotation_is_preserved_in_ir():
    source = """\
from postyp import Array, Float64, Shape, Strides

def f(x: Array[Float64, Shape[3, 3], Strides[24, 8]]) -> None:
    pass
"""

    module = compile_ok(source)

    assert module.functions[0].params[0].layout == Strides[24, 8]


def test_stride_rank_mismatch_is_rejected():
    source = """\
from postyp import Array, Float64, Shape, Strides

def f(x: Array[Float64, Shape[3, 3], Strides[1]]) -> None:
    pass
"""

    assert error_codes(source) == ["PP100"]


def test_c_order_multidimensional_indexing_uses_row_major_byte_stride():
    source = """\
from postyp import Array, Int64, Shape

def f(x: Array[Int64, Shape[2, 3]]) -> int:
    return x[1][1]
"""

    assert array_load_byte_offsets(source) == [32]


def test_fortran_order_multidimensional_indexing_uses_column_major_byte_stride():
    source = """\
from postyp import Array, Int64, Shape, FOrder

def f(x: Array[Int64, Shape[2, 3], FOrder]) -> int:
    return x[1][1]
"""

    assert array_load_byte_offsets(source) == [24]


def test_static_explicit_strides_are_used_for_indexing():
    source = """\
from postyp import Array, Int64, Shape, Strides

def f(x: Array[Int64, Shape[2, 3], Strides[80, 16]]) -> int:
    return x[1][1]
"""

    assert array_load_byte_offsets(source) == [96]


def test_dynamic_explicit_strides_lower_to_runtime_stride_metadata():
    source = """\
from postyp import Array, Int64, Shape, Strides

def f(x: Array[Int64, Shape[2, 3], Strides[None, 8]]) -> int:
    return x[1][1]
"""

    module = compile_ok(source)
    assert any(
        isinstance(instr, ArrayStride)
        for block in module.functions[0].blocks
        for instr in block.instructions
    )


def test_native_array_abi_includes_shape_and_strides():
    source = """\
from postyp import Array, Int64, Shape

def f(x: Array[Int64, Shape[2, 3]]) -> int:
    return x[1][1]
"""

    c_source = emit_c(source)

    assert "typedef struct __pp_array" in c_source
    assert "int64_t const *shape;" in c_source
    assert "int64_t const *strides;" in c_source
    assert "int64_t offset_bytes;" in c_source
    assert "int64_t f(__pp_array* _x)" in c_source
    assert "__pp_array_at(_x, int64_t" in c_source


def test_dynamic_len_uses_array_dim_from_abi():
    source = """\
from postyp import Array, Int64

def f(x: Array[Int64]) -> int:
    return len(x)
"""

    c_source = emit_c(source)

    assert "__pp_array_dim(_x, 0)" in c_source
