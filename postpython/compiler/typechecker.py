"""POST Python type checker and inference engine.

Operates on Python AST nodes after the checker (postpython.checker) has
confirmed the source is in the compilable subset.  Produces a type
environment mapping AST node ids to postyp DType subclasses.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from postyp import (
    DType, Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128,
    Str, Bytes,
    Array, Shape, AnyShape,
)


# ---------------------------------------------------------------------------
# Type error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TypeError_PP:
    code: str
    message: str
    lineno: int
    col_offset: int

    def __str__(self) -> str:
        return f"{self.lineno}:{self.col_offset}: {self.code} {self.message}"


@dataclass(frozen=True)
class ResolvedAnnotation:
    """Compiler-facing annotation metadata."""
    dtype: Optional[type[DType]]
    shape: Shape = AnyShape
    is_array: bool = False


# ---------------------------------------------------------------------------
# Name → scalar dtype mapping (for annotation resolution)
# ---------------------------------------------------------------------------

_ANNOTATION_MAP: dict[str, type[DType]] = {
    # postyp names
    "Bool":       Bool,
    "Int8":       Int8,  "Int16": Int16, "Int32": Int32, "Int64": Int64,
    "UInt8":      UInt8, "UInt16": UInt16, "UInt32": UInt32, "UInt64": UInt64,
    "Float16":    Float16, "Float32": Float32, "Float64": Float64,
    "Complex64":  Complex64, "Complex128": Complex128,
    "Str":        Str, "Bytes": Bytes,
    # Aliases
    "Int":        Int64, "Float": Float64, "Complex": Complex128,
    # Python built-ins → canonical POST Python types
    "bool":       Bool,
    "int":        Int64,
    "float":      Float64,
    "complex":    Complex128,
    "str":        Str,
    "bytes":      Bytes,
}


def _resolve_dtype_expr(node: ast.expr) -> Optional[type[DType]]:
    if isinstance(node, ast.Name):
        return _ANNOTATION_MAP.get(node.id)
    if isinstance(node, ast.Attribute):
        # e.g. postyp.Float64 — just use the attribute name
        return _ANNOTATION_MAP.get(node.attr)
    return None


def _resolve_shape_expr(node: ast.expr) -> Shape:
    if isinstance(node, ast.Name) and node.id == "AnyShape":
        return AnyShape
    if isinstance(node, ast.Subscript):
        base = node.value
        if (
            (isinstance(base, ast.Name) and base.id == "Shape")
            or (isinstance(base, ast.Attribute) and base.attr == "Shape")
        ):
            raw_dims = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            dims: list[int | None] = []
            for dim in raw_dims:
                if isinstance(dim, ast.Constant):
                    if dim.value is Ellipsis:
                        return AnyShape
                    if dim.value is None:
                        dims.append(None)
                    elif isinstance(dim.value, int):
                        dims.append(dim.value)
                elif isinstance(dim, ast.Name) and dim.id == "None":
                    dims.append(None)
            return Shape(*dims) if dims else AnyShape
    return AnyShape


def resolve_annotation_info(node: ast.expr) -> ResolvedAnnotation:
    """Best-effort resolution of an annotation node.

    Scalar annotations resolve to a dtype. Array annotations resolve to their
    element dtype plus shape metadata so the compiler can distinguish pointer
    values from scalar values.
    """
    if isinstance(node, ast.Subscript):
        base = node.value
        is_array = (
            (isinstance(base, ast.Name) and base.id == "Array")
            or (isinstance(base, ast.Attribute) and base.attr == "Array")
        )
        if is_array:
            parts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            dtype = _resolve_dtype_expr(parts[0]) if parts else None
            shape = _resolve_shape_expr(parts[1]) if len(parts) > 1 else AnyShape
            return ResolvedAnnotation(dtype=dtype, shape=shape, is_array=True)

    dtype = _resolve_dtype_expr(node)
    if dtype is not None:
        return ResolvedAnnotation(dtype=dtype)
    if isinstance(node, ast.Constant) and node.value is None:
        return ResolvedAnnotation(dtype=None)   # 'None' return type → void
    return ResolvedAnnotation(dtype=None)


def resolve_annotation(node: ast.expr) -> Optional[type[DType]]:
    """Best-effort resolution of a type annotation node to a postyp DType."""
    return resolve_annotation_info(node).dtype


# ---------------------------------------------------------------------------
# Numeric promotion rules (mirrors array-api)
# ---------------------------------------------------------------------------

# Precedence: higher index wins in mixed-kind arithmetic.
_INT_RANK = [Int8, Int16, Int32, Int64]
_UINT_RANK = [UInt8, UInt16, UInt32, UInt64]
_FLOAT_RANK = [Float16, Float32, Float64]
_COMPLEX_RANK = [Complex64, Complex128]


def promote(a: type[DType], b: type[DType]) -> type[DType]:
    """Return the result dtype when mixing a and b in arithmetic."""
    if a is b:
        return a

    def _rank_in(t: type[DType], lst: list) -> int:
        try:
            return lst.index(t)
        except ValueError:
            return -1

    # Complex wins over everything
    for t in (a, b):
        if t in _COMPLEX_RANK:
            ca, cb = _rank_in(a, _COMPLEX_RANK), _rank_in(b, _COMPLEX_RANK)
            return _COMPLEX_RANK[max(ca, cb, 0)]

    # Float wins over int/uint
    for t in (a, b):
        if t in _FLOAT_RANK:
            fa, fb = _rank_in(a, _FLOAT_RANK), _rank_in(b, _FLOAT_RANK)
            return _FLOAT_RANK[max(fa, fb, 0)]

    # Mixed signed/unsigned: promote to signed with one extra bit
    ra, rb = _rank_in(a, _INT_RANK), _rank_in(b, _INT_RANK)
    ua, ub = _rank_in(a, _UINT_RANK), _rank_in(b, _UINT_RANK)
    if ra >= 0 and rb >= 0:
        return _INT_RANK[max(ra, rb)]
    if ua >= 0 and ub >= 0:
        return _UINT_RANK[max(ua, ub)]
    # mixed signed + unsigned — widen signed
    signed_rank = max(ra, 0)
    uint_rank = max(ua, ub, 0)
    result_rank = min(max(signed_rank, uint_rank + 1), len(_INT_RANK) - 1)
    return _INT_RANK[result_rank]


# ---------------------------------------------------------------------------
# Type environment
# ---------------------------------------------------------------------------

class TypeEnv:
    """Scope-aware mapping from variable name → DType.

    Scopes are stacked; inner scopes shadow outer ones.
    """

    def __init__(self) -> None:
        self._scopes: list[dict[str, type[DType]]] = [{}]
        self.errors: list[TypeError_PP] = []

    # -- scope management ---------------------------------------------------

    def push(self) -> None:
        self._scopes.append({})

    def pop(self) -> None:
        self._scopes.pop()

    # -- lookup / binding ---------------------------------------------------

    def get(self, name: str) -> Optional[type[DType]]:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def bind(self, name: str, dtype: type[DType]) -> None:
        self._scopes[-1][name] = dtype

    def error(self, node: ast.AST, code: str, msg: str) -> None:
        self.errors.append(TypeError_PP(
            code=code,
            message=msg,
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
        ))


# ---------------------------------------------------------------------------
# Inference visitor
# ---------------------------------------------------------------------------

class TypeInferencer(ast.NodeVisitor):
    """Walk a function body and infer types for all sub-expressions.

    After calling `infer(func_node)`, `type_of` maps ast node id → DType.
    """

    def __init__(self, env: TypeEnv, return_dtype: Optional[type[DType]]) -> None:
        self.env = env
        self.return_dtype = return_dtype
        self.type_of: dict[int, type[DType]] = {}

    def _set(self, node: ast.AST, dtype: type[DType]) -> type[DType]:
        self.type_of[id(node)] = dtype
        return dtype

    # -- expression inference -----------------------------------------------

    def infer_expr(self, node: ast.expr) -> Optional[type[DType]]:
        """Return the inferred dtype for an expression node."""
        if isinstance(node, ast.Constant):
            return self._infer_constant(node)
        if isinstance(node, ast.Name):
            dtype = self.env.get(node.id)
            if dtype is not None:
                self._set(node, dtype)
            return dtype
        if isinstance(node, ast.BinOp):
            return self._infer_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._infer_unary(node)
        if isinstance(node, ast.Call):
            return self._infer_call(node)
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, ast.Compare):
            self.infer_expr(node.left)
            for c in node.comparators:
                self.infer_expr(c)
            return self._set(node, Bool)
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                self.infer_expr(v)
            return self._set(node, Bool)
        if isinstance(node, ast.IfExp):
            self.infer_expr(node.test)
            t = self.infer_expr(node.body)
            f = self.infer_expr(node.orelse)
            if t and f:
                result = promote(t, f)
                return self._set(node, result)
        return None

    def _infer_constant(self, node: ast.Constant) -> type[DType]:
        v = node.value
        if isinstance(v, bool):
            return self._set(node, Bool)
        if isinstance(v, int):
            return self._set(node, Int64)
        if isinstance(v, float):
            return self._set(node, Float64)
        if isinstance(v, complex):
            return self._set(node, Complex128)
        if isinstance(v, str):
            return self._set(node, Str)
        if isinstance(v, bytes):
            return self._set(node, Bytes)
        return self._set(node, Int64)  # fallback

    def _infer_binop(self, node: ast.BinOp) -> Optional[type[DType]]:
        lt = self.infer_expr(node.left)
        rt = self.infer_expr(node.right)
        if lt is None or rt is None:
            return None
        result = promote(lt, rt)
        return self._set(node, result)

    def _infer_unary(self, node: ast.UnaryOp) -> Optional[type[DType]]:
        t = self.infer_expr(node.operand)
        if t is None:
            return None
        if isinstance(node.op, ast.Not):
            return self._set(node, Bool)
        return self._set(node, t)

    def _infer_call(self, node: ast.Call) -> Optional[type[DType]]:
        for arg in node.args:
            self.infer_expr(arg)
        # Built-in math functions: infer from first arg
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in ("len", "range"):
                return self._set(node, Int64)
            if name in ("abs", "round"):
                if node.args:
                    t = self.infer_expr(node.args[0])
                    if t:
                        return self._set(node, t)
        return None

    def _infer_subscript(self, node: ast.Subscript) -> Optional[type[DType]]:
        vt = self.infer_expr(node.value)
        self.infer_expr(node.slice)
        # Array element access → element dtype
        if vt is not None and issubclass(vt, DType):
            return self._set(node, vt)
        return None

    # -- statement traversal ------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        dtype = self.infer_expr(node.value)
        if dtype is not None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env.bind(target.id, dtype)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        annotated = resolve_annotation(node.annotation)
        if node.value:
            inferred = self.infer_expr(node.value)
            dtype = annotated or inferred
        else:
            dtype = annotated
        if dtype is not None and isinstance(node.target, ast.Name):
            self.env.bind(node.target.id, dtype)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.infer_expr(node.value)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value:
            inferred = self.infer_expr(node.value)
            if self.return_dtype and inferred and inferred is not self.return_dtype:
                # Implicit numeric cast on return — record but don't error yet.
                pass

    def visit_For(self, node: ast.For) -> None:
        # range() loops: bind the loop variable as Int64
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and isinstance(node.target, ast.Name)
        ):
            self.env.bind(node.target.id, Int64)
        self.env.push()
        for stmt in node.body:
            self.visit(stmt)
        self.env.pop()

    def visit_If(self, node: ast.If) -> None:
        self.infer_expr(node.test)
        self.env.push()
        for stmt in node.body:
            self.visit(stmt)
        self.env.pop()
        self.env.push()
        for stmt in node.orelse:
            self.visit(stmt)
        self.env.pop()

    def visit_While(self, node: ast.While) -> None:
        self.infer_expr(node.test)
        self.env.push()
        for stmt in node.body:
            self.visit(stmt)
        self.env.pop()


def infer_function(
    node: ast.FunctionDef,
    param_types: dict[str, type[DType]],
    return_dtype: Optional[type[DType]],
) -> tuple[dict[int, type[DType]], list[TypeError_PP]]:
    """Infer types for all sub-expressions in a function body.

    Returns (type_map, errors).
    """
    env = TypeEnv()
    env.push()
    for name, dtype in param_types.items():
        env.bind(name, dtype)

    inferencer = TypeInferencer(env, return_dtype)
    for stmt in node.body:
        inferencer.visit(stmt)

    return inferencer.type_of, env.errors
