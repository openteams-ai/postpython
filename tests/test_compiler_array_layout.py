"""Compiler tests for POST Array layout metadata."""

from postpython.compiler.frontend import compile_source
from postpython.compiler.ir import BinOp, BinOpInstr, Const
from postyp import COrder, FOrder, Strides


def compile_ok(source: str):
    module, errors = compile_source(source)
    assert errors == []
    return module


def error_codes(source: str) -> list[str]:
    _, errors = compile_source(source)
    return [error.code for error in errors]


def stride_multiplier_constants(source: str) -> list[int]:
    module = compile_ok(source)
    constants = {}
    right_operands = []
    for block in module.functions[0].blocks:
        for instr in block.instructions:
            if isinstance(instr, Const):
                constants[instr.result.name] = instr.value
            elif isinstance(instr, BinOpInstr) and instr.op is BinOp.MUL:
                right_operands.append(instr.right.name)
    return [
        constants[name]
        for name in right_operands
        if isinstance(constants.get(name), int)
    ]


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

def f(x: Array[Float64, Shape[3, 3], Strides[3, 1]]) -> None:
    pass
"""

    module = compile_ok(source)

    assert module.functions[0].params[0].layout == Strides[3, 1]


def test_stride_rank_mismatch_is_rejected():
    source = """\
from postyp import Array, Float64, Shape, Strides

def f(x: Array[Float64, Shape[3, 3], Strides[1]]) -> None:
    pass
"""

    assert error_codes(source) == ["PP100"]


def test_c_order_multidimensional_indexing_uses_row_major_stride():
    source = """\
from postyp import Array, Int64, Shape

def f(x: Array[Int64, Shape[2, 3]]) -> int:
    return x[1][1]
"""

    assert stride_multiplier_constants(source) == [3, 1]


def test_fortran_order_multidimensional_indexing_uses_column_major_stride():
    source = """\
from postyp import Array, Int64, Shape, FOrder

def f(x: Array[Int64, Shape[2, 3], FOrder]) -> int:
    return x[1][1]
"""

    assert stride_multiplier_constants(source) == [1, 2]


def test_static_explicit_strides_are_used_for_indexing():
    source = """\
from postyp import Array, Int64, Shape, Strides

def f(x: Array[Int64, Shape[2, 3], Strides[10, 2]]) -> int:
    return x[1][1]
"""

    assert stride_multiplier_constants(source) == [10, 2]


def test_dynamic_explicit_strides_report_unsupported_runtime_metadata():
    source = """\
from postyp import Array, Int64, Shape, Strides

def f(x: Array[Int64, Shape[2, 3], Strides[None, 1]]) -> int:
    return x[1][1]
"""

    assert error_codes(source) == ["PP900"]
