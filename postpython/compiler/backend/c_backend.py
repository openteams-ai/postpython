"""C99 backend for the POST Python reference compiler.

Lowers a POST Python IR Module to a C99 source string.  The output is
intended to be compiled by the system C compiler (cc / clang / gcc).

For vectorized outputs, the emitted C conforms to the NumPy
ufunc C API so the resulting shared library can be registered with
numpy.lib.add_newdoc_ufunc or loaded via ctypes.
"""

from __future__ import annotations

import io
from typing import Optional

from ..ir import (
    Module, Function, UFunc, UFuncSignature,
    BasicBlock, Value, Param,
    Const, BinOpInstr, UnaryOpInstr,
    ArrayLoad, ArrayStore, ArrayDim, ArrayStride,
    Call, Cast, AssignValue, Select, Alloc, Return, Branch, CondBranch,
    BinOp, UnaryOp,
    Instruction, Terminator,
)

# sys.path setup happens once in postpython/__init__.py.
import postpython  # noqa: F401  -- ensure path setup runs
from postyp import (
    DType,
    Bool, Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, Complex64, Complex128,
    Str, Bytes,
    COrder, FOrder, Strides,
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


def c_value_type(v: Value | Param) -> str:
    if getattr(v, "is_array", False):
        return "__pp_array*"
    if getattr(v, "is_output", False):
        return c_type_ptr(v.dtype)
    return c_type(v.dtype)


_RESERVED_C_SYMBOLS: frozenset[str] = frozenset({
    "acos", "asin", "atan", "atan2",
    "ceil", "cos", "cosh",
    "erf", "erfc", "exp", "exp2", "expm1",
    "fabs", "floor", "fma", "fmod", "frexp",
    "gamma", "hypot", "j0", "j1",
    "ldexp", "lgamma", "log", "log10", "log1p", "log2",
    "modf", "pow",
    "sin", "sinh", "sqrt",
    "tan", "tanh", "tgamma", "trunc",
    "y0", "y1",
})


def c_symbol(name: str) -> str:
    return f"__pp_{name}" if name in _RESERVED_C_SYMBOLS else name


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
    if isinstance(v, complex):
        return f"({_emit_const_value(v.real)} + {_emit_const_value(v.imag)} * _Complex_I)"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _emit_abs(instr: "UnaryOpInstr") -> str:
    """Return the C statement for ``result = abs(operand)``.

    The libm spelling depends on the operand dtype:
      - signed int: abs / labs / llabs by width
      - unsigned / bool: identity
      - float: fabsf / fabs
      - complex: cabsf / cabs (result type is the underlying float)
    """
    dtype = instr.operand.dtype
    operand = _emit_value(instr.operand)
    result = _emit_value(instr.result)
    res_t = c_type(instr.result.dtype)
    kind = dtype.kind
    if kind == "f":
        fn = "fabsf" if dtype is Float32 else "fabs"
        return f"{res_t} {result} = {fn}({operand});"
    if kind == "c":
        fn = "cabsf" if dtype is Complex64 else "cabs"
        return f"{res_t} {result} = {fn}({operand});"
    if kind in ("u", "b"):
        return f"{res_t} {result} = {operand};"
    # Signed integer.
    if dtype is Int64:
        fn = "llabs"
    elif dtype is Int32:
        fn = "labs"
    else:
        fn = "abs"
    return f"{res_t} {result} = {fn}({operand});"


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


def emit_instruction(instr: Instruction, em: CEmitter, symbol_map: dict[str, str] | None = None) -> None:
    symbol_map = symbol_map or {}
    if isinstance(instr, Const):
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {_emit_const_value(instr.value)};")

    elif isinstance(instr, BinOpInstr):
        res_t = c_type(instr.result.dtype)
        result = _emit_value(instr.result)
        left = _emit_value(instr.left)
        right = _emit_value(instr.right)
        kind = instr.result.dtype.kind
        if instr.op == BinOp.POW:
            if kind == "i":
                em.line(f"{res_t} {result} = ({res_t})__pp_ipow_i64({left}, {right});")
            elif kind == "u":
                em.line(f"{res_t} {result} = ({res_t})__pp_ipow_u64({left}, {right});")
            elif kind == "c":
                fn = "cpowf" if instr.result.dtype is Complex64 else "cpow"
                em.line(f"{res_t} {result} = {fn}({left}, {right});")
            else:
                fn = "powf" if instr.result.dtype is Float32 else "pow"
                em.line(f"{res_t} {result} = {fn}({left}, {right});")
        elif instr.op == BinOp.MOD and kind == "f":
            fn = "fmodf" if instr.result.dtype is Float32 else "fmod"
            em.line(f"{res_t} {result} = {fn}({left}, {right});")
        elif instr.op == BinOp.FDIV:
            # Python floor-division semantics, dispatched per dtype.
            if kind == "i":
                em.line(f"{res_t} {result} = __pp_floordiv_si({left}, {right});")
            elif kind in ("u", "b"):
                em.line(f"{res_t} {result} = ({left}) / ({right});")
            else:
                fn = "floorf" if instr.result.dtype is Float32 else "floor"
                em.line(f"{res_t} {result} = {fn}(({res_t})({left}) / ({res_t})({right}));")
        else:
            em.line(f"{res_t} {result} = {left} {_emit_binop(instr.op)} {right};")

    elif isinstance(instr, UnaryOpInstr):
        if instr.op == UnaryOp.ABS:
            em.line(_emit_abs(instr))
        else:
            prefix = {UnaryOp.NEG: "-", UnaryOp.NOT: "!"}[instr.op]
            em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {prefix}{_emit_value(instr.operand)};")

    elif isinstance(instr, ArrayLoad):
        em.line(
            f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = "
            f"__pp_array_at({_emit_value(instr.array)}, {c_type(instr.result.dtype)}, {_emit_value(instr.index)});"
        )

    elif isinstance(instr, ArrayStore):
        em.line(
            f"__pp_array_at({_emit_value(instr.array)}, {c_type(instr.value.dtype)}, {_emit_value(instr.index)}) = "
            f"{_emit_value(instr.value)};"
        )

    elif isinstance(instr, ArrayDim):
        em.line(
            f"int64_t {_emit_value(instr.result)} = "
            f"__pp_array_dim({_emit_value(instr.array)}, {instr.axis});"
        )

    elif isinstance(instr, ArrayStride):
        em.line(
            f"int64_t {_emit_value(instr.result)} = "
            f"__pp_array_stride({_emit_value(instr.array)}, {instr.axis});"
        )

    elif isinstance(instr, Call):
        args_str = ", ".join(_emit_value(a) for a in instr.args)
        if instr.func == "__pp_len__":
            # len(array) — for fixed arrays this is a static constant;
            # for dynamic arrays it's stored alongside the pointer.
            em.line(f"int64_t {_emit_value(instr.result)} = __pp_array_len({args_str});")
        elif instr.result:
            em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = {symbol_map.get(instr.func, instr.func)}({args_str});")
        else:
            em.line(f"{symbol_map.get(instr.func, instr.func)}({args_str});")

    elif isinstance(instr, Cast):
        em.line(f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = ({c_type(instr.result.dtype)}){_emit_value(instr.operand)};")

    elif isinstance(instr, AssignValue):
        if instr.declare:
            em.line(f"{c_value_type(instr.target)} {_emit_value(instr.target)} = {_emit_value(instr.value)};")
        elif instr.target.is_output and not instr.target.is_array:
            em.line(f"*{_emit_value(instr.target)} = {_emit_value(instr.value)};")
        else:
            em.line(f"{_emit_value(instr.target)} = {_emit_value(instr.value)};")

    elif isinstance(instr, Select):
        em.line(
            f"{c_type(instr.result.dtype)} {_emit_value(instr.result)} = "
            f"{_emit_value(instr.cond)} ? {_emit_value(instr.if_true)} : {_emit_value(instr.if_false)};"
        )

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
        f"{c_value_type(p)} _{p.name}"
        for p in [*fn.params, *fn.core_dim_params]
    )
    semi = ";" if declaration else ""
    em.line(f"{ret} {c_symbol(fn.name)}({params}){semi}")


def _is_array_param(p: Param) -> bool:
    return p.is_array


def _c_array_literal(name: str, values: list[str]) -> str:
    if not values:
        return f"int64_t {name}[1] = {{0}};"
    return f"int64_t {name}[{len(values)}] = {{{', '.join(values)}}};"


def _byte_stride_expr(element_factors: list[str], dtype: type[DType]) -> str:
    return " * ".join([*element_factors, f"sizeof({c_type(dtype)})"])


def _ufunc_stride_exprs(p: Param, dims: list[str]) -> list[str]:
    """Return byte-stride expressions for a ufunc core view."""
    ndim = len(dims)
    layout = p.layout

    if isinstance(layout, Strides) and len(layout.strides) >= ndim:
        result: list[str] = []
        for stride in layout.strides[:ndim]:
            if stride is None:
                # The simplified reference ufunc ABI cannot recover dynamic
                # core strides from NumPy's steps array yet, so keep the view
                # compact for now.
                result.append(f"sizeof({c_type(p.dtype)})")
            else:
                result.append(str(stride))
        return result

    if layout == FOrder:
        result = []
        for axis in range(ndim):
            factors = [f"_pp_dim_{dim}" for dim in dims[:axis]]
            result.append(_byte_stride_expr(factors, p.dtype))
        return result

    result = []
    for axis in range(ndim):
        factors = [f"_pp_dim_{dim}" for dim in dims[axis + 1:]]
        result.append(_byte_stride_expr(factors, p.dtype))
    return result


def emit_function(fn: Function, em: CEmitter, symbol_map: dict[str, str] | None = None) -> None:
    emit_function_signature(fn, em)
    em.line("{")
    em.indent()
    for bb in fn.blocks:
        em.line(f"{bb.label}:;")
        em.indent()
        for instr in bb.instructions:
            emit_instruction(instr, em, symbol_map)
        if bb.terminator:
            emit_terminator(bb.terminator, em)
        em.dedent()
    em.dedent()
    em.line("}")


# ---------------------------------------------------------------------------
# UFunc emission (NumPy ufunc protocol)
# ---------------------------------------------------------------------------

def emit_ufunc(fn: UFunc, em: CEmitter, symbol_map: dict[str, str] | None = None) -> None:
    """Emit the inner kernel and the NumPy ufunc wrapper."""
    sig = fn.ufunc_sig
    if sig is None:
        emit_function(fn, em, symbol_map)
        return

    # 1. Emit the inner (scalar / core-array) function.
    emit_function(fn, em, symbol_map)
    em.line()

    # 2. Emit the NumPy ufunc wrapper.
    n_in  = len(sig.inputs)
    n_out = len(sig.outputs)
    output_params = fn.params[n_in:n_in + n_out]
    out_dtype = fn.return_dtype  # dtype written to returned scalar output(s); None = output buffers

    em.line(f"/* NumPy ufunc wrapper for {fn.name} */")
    em.line(f"/* Signature: {sig} */")
    em.line(f"static void {fn.name}_ufunc_loop(")
    em.indent()
    em.line("char **args,")
    em.line("npy_intp const *dimensions,")
    em.line("npy_intp const *steps,")
    em.line("void *NPY_UNUSED(data)")
    em.dedent()
    em.line(") {")
    em.indent()

    em.line("npy_intp _pp_outer_n = dimensions[0];  /* outer broadcast loop length */")

    # Core dimension sizes.
    for i, dim_name in enumerate(sig.core_dims):
        if dim_name:
            em.line(f"int64_t _pp_dim_{dim_name} = (int64_t)dimensions[{i + 1}];")

    em.line()

    # Input arg pointers.
    for i, p in enumerate(fn.params[:n_in]):
        em.line(f"char *arg{i} = args[{i}];")
        em.line(f"npy_intp step{i} = steps[{i}];")

    # Output arg pointer(s) — come after inputs in the args/steps arrays.
    for j in range(n_out):
        k = n_in + j
        dtype = out_dtype if out_dtype is not None else output_params[j].dtype
        em.line(f"char *arg{k} = args[{k}];")
        em.line(f"npy_intp step{k} = steps[{k}];")

    em.line()
    em.line("for (npy_intp _i = 0; _i < _pp_outer_n; _i++) {")
    em.indent()

    call_parts: list[str] = []
    for i, p in enumerate(fn.params[:n_in]):
        if p.is_array or sig.inputs[i]:
            dims = sig.inputs[i]
            shape_name = f"_pp_shape_{i}"
            strides_name = f"_pp_strides_{i}"
            view_name = f"_pp_arg_{i}"
            em.line(_c_array_literal(shape_name, [f"_pp_dim_{dim}" for dim in dims]))
            em.line(_c_array_literal(strides_name, _ufunc_stride_exprs(p, dims)))
            em.line(
                f"__pp_array {view_name} = "
                f"{{arg{i} + _i * step{i}, {len(dims)}, {shape_name}, {strides_name}, 0}};"
            )
            call_parts.append(f"&{view_name}")
        else:
            call_parts.append(f"*(({c_type(p.dtype)} *)(arg{i} + _i * step{i}))")

    if out_dtype is None:
        for j, p in enumerate(output_params):
            k = n_in + j
            if p.is_array or sig.outputs[j]:
                dims = sig.outputs[j]
                shape_name = f"_pp_shape_{k}"
                strides_name = f"_pp_strides_{k}"
                view_name = f"_pp_arg_{k}"
                em.line(_c_array_literal(shape_name, [f"_pp_dim_{dim}" for dim in dims]))
                em.line(_c_array_literal(strides_name, _ufunc_stride_exprs(p, dims)))
                em.line(
                    f"__pp_array {view_name} = "
                    f"{{arg{k} + _i * step{k}, {len(dims)}, {shape_name}, {strides_name}, 0}};"
                )
                call_parts.append(f"&{view_name}")
            else:
                call_parts.append(f"({c_type_ptr(p.dtype)})(arg{k} + _i * step{k})")

    call_parts.extend(_emit_value(Value(p.name, p.dtype)) for p in fn.core_dim_params)
    call_args = ", ".join(call_parts)

    if out_dtype is not None and n_out > 0:
        # Capture the return value and write it to the output pointer.
        em.line(
            f"*(({c_type(out_dtype)} *)(arg{n_in} + _i * step{n_in})) = "
            f"{c_symbol(fn.name)}({call_args});"
        )
    else:
        em.line(f"{c_symbol(fn.name)}({call_args});")

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
#include <complex.h>

/* Python-semantic floor division for signed integers: rounds toward -inf,
   unlike C's `/` which truncates toward zero. */
#define __pp_floordiv_si(a, b) \\
    (((a) / (b)) - ((((a) % (b)) != 0) && (((a) < 0) != ((b) < 0)) ? 1 : 0))

/* Integer power (exponentiation by squaring). Negative exponents return 0
   to match Python's truncating int ** int < 0 not being representable. */
static int64_t __pp_ipow_i64(int64_t base, int64_t exp) {
    if (exp < 0) return 0;
    int64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        exp >>= 1;
        if (exp) base *= base;
    }
    return result;
}

static uint64_t __pp_ipow_u64(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        exp >>= 1;
        if (exp) base *= base;
    }
    return result;
}

/* NumPy ufunc protocol.
   When built with -DNUMPY_UFUNC the real NumPy headers supply npy_intp and
   NPY_UNUSED.  Otherwise we provide ABI-compatible fallback definitions so
   the ufunc loop functions compile without NumPy installed; they can still
   be called directly or registered with numpy.ctypeslib later. */
#ifdef NUMPY_UFUNC
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
typedef struct __pp_array {
    void *data;
    int64_t ndim;
    int64_t const *shape;
    int64_t const *strides;  /* byte strides, NumPy-compatible */
    int64_t offset_bytes;
} __pp_array;

#define __pp_array_dim(a, axis) ((a)->shape[(axis)])
#define __pp_array_stride(a, axis) ((a)->strides[(axis)])
#define __pp_array_len(a) __pp_array_dim((a), 0)
#define __pp_array_at(a, type, byte_index) (*(type *)((char *)((a)->data) + (a)->offset_bytes + (byte_index)))

"""


def emit_module(module: Module) -> str:
    """Emit a complete C99 translation unit for *module*."""
    em = CEmitter()
    em.write(_PREAMBLE)
    symbol_map = {fn.name: c_symbol(fn.name) for fn in module.functions}

    # Forward declarations.
    for fn in module.functions:
        emit_function_signature(fn, em, declaration=True)
    em.line()

    # Definitions.
    for fn in module.functions:
        if isinstance(fn, UFunc):
            emit_ufunc(fn, em, symbol_map)
        else:
            emit_function(fn, em, symbol_map)
        em.line()

    return em.getvalue()
