"""POST Python frontend: Python AST → POST Python IR.

Pipeline:
  source text
    → ast.parse                      (Python stdlib)
    → postpython.checker violations  (reject non-compilable syntax)
    → FunctionLifter per function    (build IR Function / GUFunc)
    → Module                         (collect into a translation unit)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from ..checker import check_source, Violation
from .ir import (
    Module, Function, GUFunc, GUFuncSignature,
    BasicBlock, Param, Value,
    Const, BinOpInstr, UnaryOpInstr, ArrayLoad, ArrayStore, Call, Cast, Alloc,
    BinOp, UnaryOp, Return, Branch, CondBranch,
)
from .typechecker import resolve_annotation, infer_function, promote

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from postyp import DType, Int64, Float64, Bool, AnyShape, Shape, Array


# ---------------------------------------------------------------------------
# Gufunc signature parser (shared between frontend and decorator)
# ---------------------------------------------------------------------------

def parse_gufunc_sig(sig: str) -> GUFuncSignature:
    """Parse a gufunc signature string into a GUFuncSignature.

    Example: "(m,k),(k,n)->(m,n)" → GUFuncSignature(inputs=[['m','k'],['k','n']], outputs=[['m','n']])
    """
    if "->" not in sig:
        raise ValueError(f"Invalid gufunc signature (missing '->'): {sig!r}")
    lhs, rhs = sig.split("->", 1)

    def _parse_groups(s: str) -> list[list[str]]:
        groups = []
        s = s.strip()
        if not s:
            return groups
        # Split on '),' boundary
        parts = []
        depth = 0
        current: list[str] = []
        for ch in s:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                current.append(ch)
                if depth == 0:
                    parts.append("".join(current).strip())
                    current = []
                continue
            if depth > 0 or ch not in (",", " "):
                current.append(ch)
        for p in parts:
            inner = p.strip("() \t")
            dims = [d.strip() for d in inner.split(",") if d.strip()]
            groups.append(dims)
        return groups

    inputs = _parse_groups(lhs)
    outputs = _parse_groups(rhs)
    return GUFuncSignature(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# IR builder helpers
# ---------------------------------------------------------------------------

class Builder:
    """Stateful IR builder for a single function."""

    def __init__(self, fn: Function) -> None:
        self._fn = fn
        self._block: Optional[BasicBlock] = None
        self._counter = 0

    @property
    def current_block(self) -> BasicBlock:
        assert self._block is not None, "no active block"
        return self._block

    def set_block(self, block: BasicBlock) -> None:
        self._block = block

    def fresh(self, prefix: str = "t") -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    def make_value(self, prefix: str, dtype: type[DType]) -> Value:
        return Value(self.fresh(prefix), dtype)

    def emit(self, instr: object) -> None:
        self.current_block.append(instr)  # type: ignore[arg-type]

    def terminate(self, term: object) -> None:
        self.current_block.terminate(term)  # type: ignore[arg-type]

    def new_block(self, label: str | None = None) -> BasicBlock:
        if label is None:
            label = self.fresh("bb")
        bb = self._fn.new_block(label)
        return bb


# ---------------------------------------------------------------------------
# AST → IR lifter for a single function
# ---------------------------------------------------------------------------

_AST_BINOP: dict[type, BinOp] = {
    ast.Add:  BinOp.ADD,  ast.Sub:  BinOp.SUB,
    ast.Mult: BinOp.MUL,  ast.Div:  BinOp.DIV,
    ast.FloorDiv: BinOp.FDIV, ast.Mod: BinOp.MOD,
    ast.Pow:  BinOp.POW,
    ast.Eq:   BinOp.EQ,   ast.NotEq: BinOp.NE,
    ast.Lt:   BinOp.LT,   ast.LtE:  BinOp.LE,
    ast.Gt:   BinOp.GT,   ast.GtE:  BinOp.GE,
}

_AST_UNOP: dict[type, UnaryOp] = {
    ast.USub: UnaryOp.NEG,
    ast.Not:  UnaryOp.NOT,
}


class FunctionLifter:
    """Lift one ast.FunctionDef into a Function IR node."""

    def __init__(
        self,
        node: ast.FunctionDef,
        module: Module,
        gufunc_sig: Optional[str] = None,
    ) -> None:
        self._node = node
        self._module = module
        self._gufunc_sig = gufunc_sig
        self._locals: dict[str, Value] = {}  # name → current SSA Value
        self._type_map: dict[int, type[DType]] = {}
        self._errors: list[str] = []

    def lift(self) -> Function:
        node = self._node

        # Resolve parameter types from annotations.
        params: list[Param] = []
        param_types: dict[str, type[DType]] = {}
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            dtype = resolve_annotation(arg.annotation) if arg.annotation else Int64
            if dtype is None:
                dtype = Int64  # fallback; checker already enforced annotation exists
            params.append(Param(arg.arg, dtype))
            param_types[arg.arg] = dtype

        return_dtype = resolve_annotation(node.returns) if node.returns else None

        # Build the IR function (or gufunc).
        if self._gufunc_sig is not None:
            sig = parse_gufunc_sig(self._gufunc_sig)
            fn: Function = GUFunc(
                name=node.name,
                params=params,
                return_dtype=return_dtype,
                gufunc_sig=sig,
            )
        else:
            fn = Function(name=node.name, params=params, return_dtype=return_dtype)

        # Run type inference over the body.
        self._type_map, tc_errors = infer_function(node, param_types, return_dtype)

        # Build IR.
        builder = Builder(fn)
        entry = fn.new_block("entry")
        builder.set_block(entry)

        # Seed locals with parameter values.
        for p in params:
            v = Value(p.name, p.dtype)
            self._locals[p.name] = v

        # Lower the body.
        self._builder = builder
        for stmt in node.body:
            self._lower_stmt(stmt)

        # Ensure the last block has a terminator.
        if builder.current_block.terminator is None:
            builder.terminate(Return(None))

        return fn

    # ---- statement lowering -----------------------------------------------

    def _lower_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Return):
            self._lower_return(stmt)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            self._lower_assign(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._lower_aug_assign(stmt)
        elif isinstance(stmt, ast.For):
            self._lower_for(stmt)
        elif isinstance(stmt, ast.While):
            self._lower_while(stmt)
        elif isinstance(stmt, ast.If):
            self._lower_if(stmt)
        elif isinstance(stmt, ast.Expr):
            self._lower_expr(stmt.value)
        # Other stmts (pass, assert) are no-ops in the IR.

    def _lower_return(self, stmt: ast.Return) -> None:
        if stmt.value:
            val = self._lower_expr(stmt.value)
            self._builder.terminate(Return(val))
        else:
            self._builder.terminate(Return(None))

    def _lower_assign(self, stmt: ast.Assign | ast.AnnAssign) -> None:
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return
            val = self._lower_expr(stmt.value)
            if val and isinstance(stmt.target, ast.Name):
                self._locals[stmt.target.id] = val
        else:
            val = self._lower_expr(stmt.value)
            if val:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self._locals[target.id] = val

    def _lower_aug_assign(self, stmt: ast.AugAssign) -> None:
        if not isinstance(stmt.target, ast.Name):
            return
        name = stmt.target.id
        left = self._locals.get(name)
        right = self._lower_expr(stmt.value)
        if left is None or right is None:
            return
        op_type = type(stmt.op)
        if op_type not in _AST_BINOP:
            return
        result_dtype = promote(left.dtype, right.dtype)
        result = self._builder.make_value("aug", result_dtype)
        self._builder.emit(BinOpInstr(result, _AST_BINOP[op_type], left, right))
        self._locals[name] = result

    def _lower_for(self, stmt: ast.For) -> None:
        # Only handle range() loops in the skeleton.
        if not (
            isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
            and isinstance(stmt.target, ast.Name)
        ):
            return

        b = self._builder
        loop_var = stmt.target.id

        # Resolve range args.
        range_args = stmt.iter.args
        if len(range_args) == 1:
            start_v = Value("_c0", Int64)
            b.emit(Const(start_v, 0))
            stop_v  = self._lower_expr(range_args[0]) or Value("_stop", Int64)
            step_v  = Value("_c1", Int64)
            b.emit(Const(step_v, 1))
        elif len(range_args) == 2:
            start_v = self._lower_expr(range_args[0]) or Value("_start", Int64)
            stop_v  = self._lower_expr(range_args[1]) or Value("_stop", Int64)
            step_v  = Value("_c1", Int64)
            b.emit(Const(step_v, 1))
        else:
            start_v = self._lower_expr(range_args[0]) or Value("_start", Int64)
            stop_v  = self._lower_expr(range_args[1]) or Value("_stop", Int64)
            step_v  = self._lower_expr(range_args[2]) or Value("_step", Int64)

        idx = Value(loop_var, Int64)
        b.emit(Const(idx, 0))  # placeholder; real SSA would use phi
        self._locals[loop_var] = idx

        cond_bb  = b.new_block("for_cond")
        body_bb  = b.new_block("for_body")
        after_bb = b.new_block("for_after")

        b.terminate(Branch(cond_bb.label))

        # Condition block.
        b.set_block(cond_bb)
        cond_val = b.make_value("cond", Bool)
        b.emit(BinOpInstr(cond_val, BinOp.LT, idx, stop_v))
        b.terminate(CondBranch(cond_val, body_bb.label, after_bb.label))

        # Body block.
        b.set_block(body_bb)
        for s in stmt.body:
            self._lower_stmt(s)
        # increment
        if b.current_block.terminator is None:
            next_idx = b.make_value("idx_next", Int64)
            b.emit(BinOpInstr(next_idx, BinOp.ADD, idx, step_v))
            self._locals[loop_var] = next_idx
            b.terminate(Branch(cond_bb.label))

        b.set_block(after_bb)

    def _lower_while(self, stmt: ast.While) -> None:
        b = self._builder
        cond_bb  = b.new_block("while_cond")
        body_bb  = b.new_block("while_body")
        after_bb = b.new_block("while_after")

        b.terminate(Branch(cond_bb.label))
        b.set_block(cond_bb)
        cond_val = self._lower_expr(stmt.test)
        if cond_val:
            b.terminate(CondBranch(cond_val, body_bb.label, after_bb.label))

        b.set_block(body_bb)
        for s in stmt.body:
            self._lower_stmt(s)
        if b.current_block.terminator is None:
            b.terminate(Branch(cond_bb.label))

        b.set_block(after_bb)

    def _lower_if(self, stmt: ast.If) -> None:
        b = self._builder
        cond_val = self._lower_expr(stmt.test)
        then_bb  = b.new_block("if_then")
        else_bb  = b.new_block("if_else") if stmt.orelse else None
        after_bb = b.new_block("if_after")

        if cond_val:
            b.terminate(CondBranch(
                cond_val,
                then_bb.label,
                (else_bb.label if else_bb else after_bb.label),
            ))

        b.set_block(then_bb)
        for s in stmt.body:
            self._lower_stmt(s)
        if b.current_block.terminator is None:
            b.terminate(Branch(after_bb.label))

        if else_bb:
            b.set_block(else_bb)
            for s in stmt.orelse:
                self._lower_stmt(s)
            if b.current_block.terminator is None:
                b.terminate(Branch(after_bb.label))

        b.set_block(after_bb)

    # ---- expression lowering ----------------------------------------------

    def _lower_expr(self, node: ast.expr) -> Optional[Value]:
        b = self._builder

        if isinstance(node, ast.Constant):
            return self._lower_const(node)

        if isinstance(node, ast.Name):
            return self._locals.get(node.id)

        if isinstance(node, ast.BinOp):
            left  = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            if left is None or right is None:
                return None
            result_dtype = promote(left.dtype, right.dtype)
            op_type = type(node.op)
            if op_type not in _AST_BINOP:
                return None
            result = b.make_value("v", result_dtype)
            b.emit(BinOpInstr(result, _AST_BINOP[op_type], left, right))
            return result

        if isinstance(node, ast.UnaryOp):
            operand = self._lower_expr(node.operand)
            if operand is None:
                return None
            op_type = type(node.op)
            result_dtype = Bool if op_type is ast.Not else operand.dtype
            result = b.make_value("v", result_dtype)
            b.emit(UnaryOpInstr(result, _AST_UNOP.get(op_type, UnaryOp.NEG), operand))
            return result

        if isinstance(node, ast.Subscript):
            arr = self._lower_expr(node.value)
            idx = self._lower_expr(node.slice)
            if arr is None or idx is None:
                return None
            result = b.make_value("elem", arr.dtype)
            b.emit(ArrayLoad(result, arr, idx))
            return result

        if isinstance(node, ast.Call):
            return self._lower_call(node)

        if isinstance(node, ast.Compare):
            left = self._lower_expr(node.left)
            result = b.make_value("cmp", Bool)
            if left and node.comparators:
                right = self._lower_expr(node.comparators[0])
                if right:
                    op_type = type(node.ops[0])
                    b.emit(BinOpInstr(result, _AST_BINOP.get(op_type, BinOp.EQ), left, right))
            return result

        return None

    def _lower_const(self, node: ast.Constant) -> Value:
        b = self._builder
        v = node.value
        if isinstance(v, bool):
            val = b.make_value("b", Bool)
            b.emit(Const(val, int(v)))
        elif isinstance(v, int):
            val = b.make_value("i", Int64)
            b.emit(Const(val, v))
        elif isinstance(v, float):
            val = b.make_value("f", Float64)
            b.emit(Const(val, v))
        else:
            val = b.make_value("c", Int64)
            b.emit(Const(val, 0))
        return val

    def _lower_call(self, node: ast.Call) -> Optional[Value]:
        b = self._builder
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "len" and node.args:
                arr = self._lower_expr(node.args[0])
                result = b.make_value("len", Int64)
                if arr:
                    b.emit(Call(result, "__pp_len__", [arr]))
                return result
            if name == "range":
                return None  # handled by for-loop lowering
        args = [v for a in node.args if (v := self._lower_expr(a)) is not None]
        func_name = (
            node.func.id if isinstance(node.func, ast.Name)
            else str(ast.unparse(node.func))
        )
        # Infer return dtype: promote argument types, falling back to Float64
        # for known math functions and Int64 for pure integer operations.
        if args:
            ret_dtype = args[0].dtype
            for a in args[1:]:
                ret_dtype = promote(ret_dtype, a.dtype)
        else:
            ret_dtype = Float64
        result = b.make_value("ret", ret_dtype)
        b.emit(Call(result, func_name, args))
        return result


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

_GUFUNC_DECORATOR = "gufunc"


def _extract_gufunc_sig(decorator: ast.expr) -> Optional[str]:
    """Extract the signature string from a @gufunc("...") decorator node."""
    if isinstance(decorator, ast.Call):
        if (
            isinstance(decorator.func, ast.Name)
            and decorator.func.id == _GUFUNC_DECORATOR
        ):
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                return str(decorator.args[0].value)
        if (
            isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == _GUFUNC_DECORATOR
        ):
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                return str(decorator.args[0].value)
    return None


def compile_source(source: str, filename: str = "<unknown>") -> tuple[Module, list]:
    """Parse and lower *source* to a POST Python Module.

    Returns (module, errors) where errors is a list of Violation or
    type-error strings.  An empty error list means the translation
    succeeded cleanly.
    """
    # 1. Checker pass.
    violations = check_source(source, filename=filename)
    if violations:
        return Module(name=filename), violations

    # 2. Parse to AST.
    tree = ast.parse(source, filename=filename, type_comments=True)

    # 3. Build module IR.
    stem = Path(filename).stem if filename != "<unknown>" else "module"
    module = Module(name=stem)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip nested functions (they'll be lifted by their parent).
            pass

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            # Check for @gufunc decorator.
            gufunc_sig: Optional[str] = None
            for dec in node.decorator_list:
                sig = _extract_gufunc_sig(dec)
                if sig is not None:
                    gufunc_sig = sig
                    break

            lifter = FunctionLifter(node, module, gufunc_sig)
            fn = lifter.lift()
            module.add_function(fn)

    return module, []


def compile_file(path: str | Path) -> tuple[Module, list]:
    path = Path(path)
    return compile_source(path.read_text(encoding="utf-8"), filename=str(path))
