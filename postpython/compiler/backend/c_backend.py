"""C99 backend for the POST Python reference compiler.

Lowers a POST Python IR Module to a C99 source string.  The output is
intended to be compiled by the system C compiler (cc / clang / gcc).

For gufunc outputs, the emitted C conforms to the NumPy generalized
ufunc C API so the resulting shared library can be registered with
numpy.lib.add_newdoc_ufunc or loaded via ctypes.
"""

from __future__ import annotations

import io
from typing import Optional

from ..ir import (
    Module, Function, GUFunc, GUFuncSignature,
    BasicBlock, Value, Param,
    Const, BinOpInstr, UnaryOpInstr, ArrayLoad, ArrayStore,
    Call, Cast, Alloc, Return, Branch, CondBranch,
    BinOp, UnaryOp,
    Instruction, Terminator,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from postyp import (
    DType,
    Bool, Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, Complex64, Complex128,
    Str, Bytes,
)


# ---------------------------------------------------------------------------
# Dtype → C type string
# ---------------------------------------------------------------------------

_C_TYPE: dict[type[DType], str] = {
    Bool:       "bool",
    Int8:       "int8_t",
    Int16:      "int16_t",
    Int32:      "int32_t",
    Int64:      "int64_t",
    UInt8:      "uint8_t",
    UInt16:     "uint16_t",
    UInt32:     "uint32_t",
    UInt64:     "uint64_t",
    Float16:    "uint16_t",   # no native float16 in C99; stored as uint16
    Float32:    "float",
    Float64:    "double",
    Complex64:  "float _Complex",
    Complex128: "double _Complex",
    Str:        "const char*",
    Bytes:      "const uint8_t*",
}


def c_type(dtype: type[DType]) -> str:
    return _C_TYPE.get(dtype, "void*")


def c_type_ptr(dtype: type[DType]) -> str:
    return c_type(dtype) + "*"


# ---------------------------------------------------------------------------
# Code emitter
# ---------------------------------------------------------------------------

class CEmitter:
    def __init__(self) -> None:
        self._buf = io.StringIO()
        self._indent = 0

    def write(self, text: str) -> None:
        self._buf.write(text)

    def line(self, text: str = "") -> None:
        if text:
            self._buf.write("    " * self._indent + text + "\n")
        else:
            self._buf.write("\n")

    def indent(self) -> None:
        self._indent += 1

    def dedent(self) -> None:
        self._indent -= 1

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Instruction emission
# ---------------------------------------------------------------------------

def _emit_value(v: Value) -> str:
    return f"_{v.name}"


def _emit_const_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _emit_binop(op: BinOp) -> str:
    return {
        BinOp.ADD:  "+",   BinOp.SUB:  "-",
        BinOp.MUL:  "*",   BinOp.DIV:  "/",
        BinOp.FDIV: "/",   BinOp.MOD:  "%",
        BinOp.POW:  "/* ** */",
        BinOp.EQ:   "==",  BinOp.NE:   "!=",
        BinOp.LT:   "<",   BinOp.LE:   "<=",
        BinOp.GT:   ">",   BinOp.GE:   ">=",
        BinOp.AND:  "&&",  BinOp.OR:   "||",
    }.get(op, "/*?*/")


def emit_instruction(instr: Instruction, em: CEmitter) -> None:
    if isinstance(instr, Const):
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {_emit_const_value(instr.value)};")

    elif isinstance(instr, BinOpInstr):
        if instr.op == BinOp.POW:
            em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = pow({_emit_value(instr.left)}, {_emit_value(instr.right)});")
        else:
            em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {_emit_value(instr.left)} {_emit_binop(instr.op)} {_emit_value(instr.right)};")

    elif isinstance(instr, UnaryOpInstr):
        prefix = {UnaryOp.NEG: "-", UnaryOp.NOT: "!", UnaryOp.ABS: "/* abs */"}[instr.op]
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {prefix}{_emit_value(instr.operand)};")

    elif isinstance(instr, ArrayLoad):
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {_emit_value(instr.array)}[{_emit_value(instr.index)}];")

    elif isinstance(instr, ArrayStore):
        em.line(f"{_emit_value(instr.array)}[{_emit_value(instr.index)}] = {_emit_value(instr.value)};")

    elif isinstance(instr, Call):
        args_str = ", ".join(_emit_value(a) for a in instr.args)
        if instr.func == "__pp_len__":
            # len(array) — for fixed arrays this is a static constant;
            # for dynamic arrays it's stored alongside the pointer.
            em.line(f"int64_t {_emit_value(instr.result)} = __pp_array_len({args_str});")
        elif instr.result:
            em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {instr.func}({args_str});")
        else:
            em.line(f"{instr.func}({args_str});")

    elif isinstance(instr, Cast):
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = ({c_type(instr.result.dtype)}){_emit_value(instr.operand)};")

    elif isinstance(instr, Alloc):
        em.line(f"{c_type_ptr(instr.result.dtype)} {_emit_value(instr.result)} = malloc({_emit_value(instr.length)} * sizeof({c_type(instr.result.dtype)}));")


def emit_terminator(term: Terminator, em: CEmitter) -> None:
    if isinstance(term, Return):
        if term.value:
            em.line(f"return {_emit_value(term.value)};")
        else:
            em.line("return;")
    elif isinstance(term, Branch):
        em.line(f"goto {term.target};")
    elif isinstance(term, CondBranch):
        em.line(f"if ({_emit_value(term.cond)}) goto {term.true_target}; else goto {term.false_target};")


# ---------------------------------------------------------------------------
# Function emission
# ---------------------------------------------------------------------------

def emit_function_signature(fn: Function, em: CEmitter, *, declaration: bool = False) -> None:
    ret = c_type(fn.return_dtype) if fn.return_dtype else "void"
    params = ", ".join(
        f"{c_type(p.dtype)}* _{p.name}" if _is_array_param(p) else f"{c_type(p.dtype)} _{p.name}"
        for p in fn.params
    )
    semi = ";" if declaration else ""
    em.line(f"{ret} {fn.name}({params}){semi}")


def _is_array_param(p: Param) -> bool:
    # In a real compiler this would check if p.dtype is an array type.
    # Skeleton: check by name convention.
    return False


def emit_function(fn: Function, em: CEmitter) -> None:
    emit_function_signature(fn, em)
    em.line("{")
    em.indent()
    for bb in fn.blocks:
        em.line(f"{bb.label}:;")
        em.indent()
        for instr in bb.instructions:
            emit_instruction(instr, em)
        if bb.terminator:
            emit_terminator(bb.terminator, em)
        em.dedent()
    em.dedent()
    em.line("}")


# ---------------------------------------------------------------------------
# GUFunc emission (NumPy ufunc protocol)
# ---------------------------------------------------------------------------

def emit_gufunc(fn: GUFunc, em: CEmitter) -> None:
    """Emit the inner scalar function and the NumPy ufunc wrapper."""
    sig = fn.gufunc_sig
    if sig is None:
        emit_function(fn, em)
        return

    # 1. Emit the inner (scalar / core-array) function.
    emit_function(fn, em)
    em.line()

    # 2. Emit the NumPy gufunc wrapper.
    n_in  = len(sig.inputs)
    n_out = len(sig.outputs)
    out_dtype = fn.return_dtype  # dtype written to output pointer(s); None = void

    em.line(f"/* NumPy generalized ufunc wrapper for {fn.name} */")
    em.line(f"/* Signature: {sig} */")
    em.line(f"static void {fn.name}_gufunc_loop(")
    em.indent()
    em.line("char **args,")
    em.line("npy_intp const *dimensions,")
    em.line("npy_intp const *steps,")
    em.line("void *NPY_UNUSED(data)")
    em.dedent()
    em.line(") {")
    em.indent()

    em.line("npy_intp n = dimensions[0];  /* outer broadcast loop length */")

    # Core dimension sizes.
    for i, dim_name in enumerate(sig.core_dims):
        if dim_name:
            em.line(f"npy_intp {dim_name} = dimensions[{i + 1}];")

    em.line()

    # Input arg pointers — one per function parameter.
    for i, p in enumerate(fn.params):
        em.line(f"{c_type_ptr(p.dtype)} arg{i} = ({c_type_ptr(p.dtype)})args[{i}];")
        em.line(f"npy_intp step{i} = steps[{i}] / sizeof({c_type(p.dtype)});")

    # Output arg pointer(s) — come after inputs in the args/steps arrays.
    if out_dtype is not None:
        for j in range(n_out):
            k = n_in + j
            em.line(f"{c_type_ptr(out_dtype)} arg{k} = ({c_type_ptr(out_dtype)})args[{k}];")
            em.line(f"npy_intp step{k} = steps[{k}] / sizeof({c_type(out_dtype)});")

    em.line()
    em.line("for (npy_intp _i = 0; _i < n; _i++) {")
    em.indent()

    # Call the inner scalar function with dereferenced input scalars.
    call_args = ", ".join(f"arg{i}[_i * step{i}]" for i in range(n_in))
    if out_dtype is not None and n_out > 0:
        # Capture the return value and write it to the output pointer.
        em.line(f"arg{n_in}[_i * step{n_in}] = {fn.name}({call_args});")
    else:
        em.line(f"{fn.name}({call_args});")

    em.dedent()
    em.line("}")  # for loop
    em.dedent()
    em.line("}")  # wrapper function


# ---------------------------------------------------------------------------
# Module emission
# ---------------------------------------------------------------------------

_PREAMBLE = """\
/* AUTO-GENERATED by POST Python reference compiler. DO NOT EDIT. */
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* NumPy ufunc protocol.
   When built with -DNUMPY_GUFUNC the real NumPy headers supply npy_intp and
   NPY_UNUSED.  Otherwise we provide ABI-compatible fallback definitions so
   the gufunc loop functions compile without NumPy installed; they can still
   be called directly or registered with numpy.ctypeslib later. */
#ifdef NUMPY_GUFUNC
#  include <numpy/ndarraytypes.h>
#  include <numpy/ufuncobject.h>
#else
#  ifndef NPY_INTP_DEFINED
     typedef ssize_t npy_intp;
#    define NPY_INTP_DEFINED
#  endif
#  ifndef NPY_UNUSED
#    define NPY_UNUSED(x) x
#  endif
#endif

/* POST Python runtime helpers */
#define __pp_array_len(a) ((a)->length)

"""


def emit_module(module: Module) -> str:
    """Emit a complete C99 translation unit for *module*."""
    em = CEmitter()
    em.write(_PREAMBLE)

    # Forward declarations.
    for fn in module.functions:
        emit_function_signature(fn, em, declaration=True)
    em.line()

    # Definitions.
    for fn in module.functions:
        if isinstance(fn, GUFunc):
            emit_gufunc(fn, em)
        else:
            emit_function(fn, em)
        em.line()

    return em.getvalue()
