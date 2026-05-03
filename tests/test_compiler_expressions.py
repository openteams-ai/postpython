"""Compiler expression-lowering tests for abs(), and/or, chained
compares, complex literals, and the related C backend emission.
"""

from postpython.compiler.backend.c_backend import emit_module
from postpython.compiler.frontend import compile_source
from postpython.compiler.ir import (
    BinOp,
    BinOpInstr,
    Const,
    UnaryOp,
    UnaryOpInstr,
)
from postyp import Bool, Complex128, Float32, Float64, Int64


def _module(source: str):
    module, errors = compile_source(source)
    assert errors == [], errors
    return module


def _instructions(source: str):
    module = _module(source)
    instrs = []
    for fn in module.functions:
        for block in fn.blocks:
            instrs.extend(block.instructions)
    return instrs


def _emit(source: str) -> str:
    return emit_module(_module(source))


# ---------------------------------------------------------------------------
# abs() lowering
# ---------------------------------------------------------------------------

def test_abs_of_int_lowers_to_unary_abs_with_same_dtype():
    instrs = _instructions(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    return abs(x)\n"
    )
    abs_ops = [i for i in instrs if isinstance(i, UnaryOpInstr) and i.op == UnaryOp.ABS]
    assert len(abs_ops) == 1
    assert abs_ops[0].result.dtype is Int64


def test_abs_of_float_lowers_to_unary_abs_with_same_dtype():
    instrs = _instructions(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return abs(x)\n"
    )
    abs_ops = [i for i in instrs if isinstance(i, UnaryOpInstr) and i.op == UnaryOp.ABS]
    assert len(abs_ops) == 1
    assert abs_ops[0].result.dtype is Float64


def test_abs_of_complex128_yields_float64_result():
    instrs = _instructions(
        "from postyp import Complex128\n"
        "def f(x: Complex128) -> Complex128:\n"
        "    y: Complex128 = x\n"
        "    z = abs(y)\n"
        "    return y\n"
    )
    abs_ops = [i for i in instrs if isinstance(i, UnaryOpInstr) and i.op == UnaryOp.ABS]
    assert len(abs_ops) == 1
    assert abs_ops[0].operand.dtype is Complex128
    assert abs_ops[0].result.dtype is Float64


def test_c_backend_abs_int_uses_llabs():
    c = _emit(
        "from postyp import Int64\n"
        "def f(x: Int64) -> Int64:\n"
        "    return abs(x)\n"
    )
    assert "llabs(" in c
    assert "/* abs */" not in c  # no placeholder leftover


def test_c_backend_abs_float_uses_fabs():
    c = _emit(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return abs(x)\n"
    )
    assert "fabs(" in c


def test_c_backend_abs_float32_uses_fabsf():
    c = _emit(
        "from postyp import Float32\n"
        "def f(x: Float32) -> Float32:\n"
        "    return abs(x)\n"
    )
    assert "fabsf(" in c


def test_c_backend_abs_complex_uses_cabs():
    c = _emit(
        "from postyp import Complex128, Float64\n"
        "def f(x: Complex128) -> Float64:\n"
        "    return abs(x)\n"
    )
    assert "cabs(" in c


# ---------------------------------------------------------------------------
# BoolOp (and / or)
# ---------------------------------------------------------------------------

def test_bool_and_lowers_to_binop_and():
    instrs = _instructions(
        "from postyp import Bool\n"
        "def f(a: Bool, b: Bool) -> Bool:\n"
        "    return a and b\n"
    )
    ops = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.AND]
    assert len(ops) == 1
    assert ops[0].result.dtype is Bool


def test_bool_or_lowers_to_binop_or():
    instrs = _instructions(
        "from postyp import Bool\n"
        "def f(a: Bool, b: Bool) -> Bool:\n"
        "    return a or b\n"
    )
    ops = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.OR]
    assert len(ops) == 1


def test_bool_and_chain_emits_pairwise_ands():
    instrs = _instructions(
        "from postyp import Bool\n"
        "def f(a: Bool, b: Bool, c: Bool) -> Bool:\n"
        "    return a and b and c\n"
    )
    ops = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.AND]
    assert len(ops) == 2


def test_c_backend_emits_short_circuit_operators_for_and_or():
    c = _emit(
        "from postyp import Bool\n"
        "def f(a: Bool, b: Bool) -> Bool:\n"
        "    return a and b\n"
    )
    assert "&&" in c


# ---------------------------------------------------------------------------
# Chained comparisons
# ---------------------------------------------------------------------------

def test_chained_compare_lowers_to_two_comparisons_and_one_and():
    instrs = _instructions(
        "from postyp import Int64, Bool\n"
        "def f(x: Int64) -> Bool:\n"
        "    return 0 < x < 10\n"
    )
    cmps = [
        i for i in instrs
        if isinstance(i, BinOpInstr) and i.op in (BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE, BinOp.EQ, BinOp.NE)
    ]
    ands = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.AND]
    assert len(cmps) == 2
    assert len(ands) == 1


def test_three_term_chain_lowers_to_three_comparisons():
    instrs = _instructions(
        "from postyp import Int64, Bool\n"
        "def f(a: Int64, b: Int64, c: Int64, d: Int64) -> Bool:\n"
        "    return a < b < c < d\n"
    )
    cmps = [
        i for i in instrs
        if isinstance(i, BinOpInstr) and i.op == BinOp.LT
    ]
    ands = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.AND]
    assert len(cmps) == 3
    assert len(ands) == 2


def test_simple_compare_unchanged():
    instrs = _instructions(
        "from postyp import Int64, Bool\n"
        "def f(x: Int64) -> Bool:\n"
        "    return x < 10\n"
    )
    cmps = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.LT]
    ands = [i for i in instrs if isinstance(i, BinOpInstr) and i.op == BinOp.AND]
    assert len(cmps) == 1
    assert len(ands) == 0


# ---------------------------------------------------------------------------
# Complex literals
# ---------------------------------------------------------------------------

def test_complex_literal_lowers_to_const_with_complex_value():
    # `2.0j` is a single complex literal in the Python AST; `1.0 + 2.0j`
    # would be a BinOp of (float, complex), which is a separate concern.
    instrs = _instructions(
        "from postyp import Complex128\n"
        "def f() -> Complex128:\n"
        "    z: Complex128 = 2.0j\n"
        "    return z\n"
    )
    complex_consts = [
        i for i in instrs
        if isinstance(i, Const) and isinstance(i.value, complex)
    ]
    assert len(complex_consts) == 1
    assert complex_consts[0].value == complex(0.0, 2.0)
    assert complex_consts[0].result.dtype is Complex128


def test_c_backend_emits_complex_literal_with_imaginary_unit():
    c = _emit(
        "from postyp import Complex128\n"
        "def f() -> Complex128:\n"
        "    return 4.0j\n"
    )
    assert "_Complex_I" in c
    assert "#include <complex.h>" in c


# ---------------------------------------------------------------------------
# Floor division (//) — Python semantics, not C truncation
# ---------------------------------------------------------------------------

def test_c_backend_signed_floor_div_uses_python_semantic_helper():
    c = _emit(
        "from postyp import Int64\n"
        "def f(a: Int64, b: Int64) -> Int64:\n"
        "    return a // b\n"
    )
    assert "__pp_floordiv_si(" in c
    # And the helper is defined in the preamble:
    assert "#define __pp_floordiv_si" in c


def test_c_backend_unsigned_floor_div_uses_plain_division():
    c = _emit(
        "from postyp import UInt64\n"
        "def f(a: UInt64, b: UInt64) -> UInt64:\n"
        "    return a // b\n"
    )
    # The signed helper macro is defined in the preamble for any TU; what we
    # care about is that the unsigned function body does not call it.
    body = c.split("uint64_t f(", 1)[1]
    assert "__pp_floordiv_si(" not in body


def test_c_backend_float_floor_div_uses_floor():
    c = _emit(
        "from postyp import Float64\n"
        "def f(a: Float64, b: Float64) -> Float64:\n"
        "    return a // b\n"
    )
    assert "floor(" in c


# ---------------------------------------------------------------------------
# Power (**) — integer-specific path
# ---------------------------------------------------------------------------

def test_c_backend_integer_pow_uses_integer_helper_not_libm_pow():
    c = _emit(
        "from postyp import Int64\n"
        "def f(a: Int64, b: Int64) -> Int64:\n"
        "    return a ** b\n"
    )
    assert "__pp_ipow_i64(" in c
    # And the helper is defined in the preamble:
    assert "static int64_t __pp_ipow_i64(" in c


def test_c_backend_uint_pow_uses_unsigned_helper():
    c = _emit(
        "from postyp import UInt64\n"
        "def f(a: UInt64, b: UInt64) -> UInt64:\n"
        "    return a ** b\n"
    )
    assert "__pp_ipow_u64(" in c


def test_c_backend_float_pow_still_uses_libm_pow():
    c = _emit(
        "from postyp import Float64\n"
        "def f(a: Float64, b: Float64) -> Float64:\n"
        "    return a ** b\n"
    )
    # Integer pow helpers are defined in the preamble unconditionally;
    # verify the float function body calls libm's pow, not the int helper.
    body = c.split("double f(", 1)[1]
    assert "pow(" in body
    assert "__pp_ipow" not in body
