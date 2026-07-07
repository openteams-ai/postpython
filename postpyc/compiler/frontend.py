"""POST Python frontend: Python AST → POST Python IR.

Pipeline:
  source text
    → ast.parse                      (Python stdlib)
    → postpyc.checker violations  (reject non-compilable syntax)
    → FunctionLifter per function    (build IR Function / UFunc)
    → Module                         (collect into a translation unit)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..checker import check_source, Violation
from .ir import (
    Module, ImportedName, Function, UFunc, UFuncSignature,
    BasicBlock, Param, Value,
    Const, BinOpInstr, UnaryOpInstr,
    ArrayLoad, ArrayStore, ArrayDim, ArrayStride,
    Call, Cast, Alloc,
    AssignValue, Select,
    BinOp, UnaryOp, Return, Branch, CondBranch,
)
from .typechecker import (
    TypeError_PP,
    resolve_annotation,
    resolve_annotation_info,
    infer_function,
    promote,
)

# sys.path setup happens once in postpyc/__init__.py.
import postpyc  # noqa: F401  -- ensure path setup runs
from postyp import (
    DType,
    Int64,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Bool,
    AnyShape,
    Shape,
    Array,
    ArrayLayout,
    COrder,
    FOrder,
    Strides,
)


# ---------------------------------------------------------------------------
# Ufunc layout signature parser (shared between frontend and decorator)
# ---------------------------------------------------------------------------

def parse_ufunc_sig(sig: str) -> UFuncSignature:
    """Parse a ufunc layout signature string into a UFuncSignature.

    Example: "(m,k),(k,n)->(m,n)" → UFuncSignature(inputs=[['m','k'],['k','n']], outputs=[['m','n']])
    """
    if sig.count("->") != 1:
        raise ValueError(f"Invalid ufunc layout signature (missing '->'): {sig!r}")
    lhs, rhs = sig.split("->", 1)

    name_re = re.compile(r"[a-z][a-z0-9_]*\Z")

    def _parse_groups(s: str, side: str) -> list[list[str]]:
        groups: list[list[str]] = []
        pos = 0
        length = len(s)

        def skip_ws() -> None:
            nonlocal pos
            while pos < length and s[pos].isspace():
                pos += 1

        skip_ws()
        if pos == length:
            raise ValueError(f"Invalid ufunc layout signature (empty {side} side): {sig!r}")

        while pos < length:
            if s[pos] != "(":
                raise ValueError(
                    f"Invalid ufunc layout signature (expected '(' in {side} side): {sig!r}"
                )
            pos += 1
            start = pos
            while pos < length and s[pos] != ")":
                if s[pos] == "(":
                    raise ValueError(
                        f"Invalid ufunc layout signature (nested '(' in {side} side): {sig!r}"
                    )
                pos += 1
            if pos == length:
                raise ValueError(
                    f"Invalid ufunc layout signature (unclosed group in {side} side): {sig!r}"
                )

            inner = s[start:pos].strip()
            pos += 1
            if inner:
                dims = [part.strip() for part in inner.split(",")]
                if any(not dim for dim in dims):
                    raise ValueError(
                        f"Invalid ufunc layout signature (empty dimension name in {side} side): {sig!r}"
                    )
                for dim in dims:
                    if not name_re.fullmatch(dim):
                        raise ValueError(
                            f"Invalid ufunc layout signature dimension name {dim!r}: {sig!r}"
                        )
            else:
                dims = []
            groups.append(dims)

            skip_ws()
            if pos == length:
                break
            if s[pos] != ",":
                raise ValueError(
                    f"Invalid ufunc layout signature (expected ',' in {side} side): {sig!r}"
                )
            pos += 1
            skip_ws()
            if pos == length:
                raise ValueError(
                    f"Invalid ufunc layout signature (trailing ',' in {side} side): {sig!r}"
                )

        return groups

    inputs = _parse_groups(lhs, "input")
    outputs = _parse_groups(rhs, "output")
    input_dims = {dim for group in inputs for dim in group}
    for dim in (dim for group in outputs for dim in group):
        if dim not in input_dims:
            raise ValueError(
                f"Invalid ufunc layout signature output dimension {dim!r} "
                f"does not appear in an input: {sig!r}"
            )
    return UFuncSignature(inputs=inputs, outputs=outputs)


# ---------------------------------------------------------------------------
# IR builder helpers
# ---------------------------------------------------------------------------

class Builder:
    """Stateful IR builder for a single function.

    *reserved* holds every user-visible name in the function (parameters
    and assignment targets) so generated temporaries never collide with
    them — a user local named ``c1`` must not clash with the ``c``-prefix
    constant temp ``c1``.
    """

    def __init__(self, fn: Function, reserved: set[str] | None = None) -> None:
        self._fn = fn
        self._block: Optional[BasicBlock] = None
        self._counter = 0
        self._reserved = reserved or set()

    @property
    def current_block(self) -> BasicBlock:
        assert self._block is not None, "no active block"
        return self._block

    def set_block(self, block: BasicBlock) -> None:
        self._block = block

    def fresh(self, prefix: str = "t") -> str:
        while True:
            self._counter += 1
            name = f"{prefix}{self._counter}"
            if name not in self._reserved:
                return name

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

# Builtins lowered directly to libm calls. `round` maps to C round(),
# which rounds half away from zero rather than Python's half-to-even;
# code sensitive to that difference should avoid round() for now.
_BUILTIN_LIBM_CALLS: dict[str, str] = {
    "round": "round",
}

# Modules whose imports provide compile-time vocabulary (types, decorators)
# rather than POST translation units or intrinsics.
_COMPILE_TIME_MODULES: frozenset[str] = frozenset({
    "postyp", "dataclasses", "typing", "__future__",
})

_INTRINSIC_MODULE = "postpyc.math"


class FunctionLifter:
    """Lift one ast.FunctionDef into a Function IR node.

    Lifting happens in two phases so that calls can resolve against every
    function in the module (forward references) and against imported POST
    modules: ``declare()`` builds the typed Function shell from the
    signature; ``lower()`` fills in the body once all declarations exist.
    """

    def __init__(
        self,
        node: ast.FunctionDef,
        module: Module,
        ufunc_sig: Optional[str] = None,
        ufunc_kind: Optional[str] = None,
        program: Optional[dict[str, Module]] = None,
    ) -> None:
        self._node = node
        self._module = module
        self._ufunc_sig = ufunc_sig
        self._ufunc_kind = ufunc_kind
        self._program = program or {}
        self._locals: dict[str, Value] = {}  # name → current SSA Value
        self._array_dims: dict[str, list[Value]] = {}
        self._implicit_array_return: str | None = None
        self._type_map: dict[int, type[DType]] = {}
        self._errors: list[TypeError_PP] = []
        self._loop_stack: list[tuple[str, str]] = []
        self._param_types: dict[str, type[DType]] = {}
        self._assigned_names: set[str] = set()

    @property
    def errors(self) -> list[TypeError_PP]:
        return self._errors

    def _annotation_error(
        self,
        node: ast.AST,
        message: str,
    ) -> None:
        self._compiler_error(node, "PP100", message)

    def _compiler_error(
        self,
        node: ast.AST,
        code: str,
        message: str,
    ) -> None:
        self._errors.append(TypeError_PP(
            code=code,
            message=message,
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
        ))

    def _unsupported_feature_error(self, node: ast.AST, feature: str) -> None:
        self._compiler_error(
            node,
            "PP900",
            f"{feature} is valid POST Python but is not lowered by this compiler yet",
        )

    def lift(self) -> Function:
        """Declare and lower in one step (single-function convenience)."""
        fn = self.declare()
        self.lower()
        return fn

    def declare(self) -> Function:
        """Phase 1: build the typed Function shell from the signature."""
        node = self._node

        parsed_ufunc_sig = parse_ufunc_sig(self._ufunc_sig) if self._ufunc_sig is not None else None
        core_dim_values: dict[str, Value] = {}
        core_dim_params: list[Param] = []
        if parsed_ufunc_sig is not None:
            for dim_name in parsed_ufunc_sig.core_dims:
                value = Value(f"pp_dim_{dim_name}", Int64)
                core_dim_values[dim_name] = value
                core_dim_params.append(Param(value.name, Int64))

        # Resolve parameter types from annotations.
        params: list[Param] = []
        param_types: dict[str, type[DType]] = {}
        user_arg_index = 0
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            annotation = resolve_annotation_info(arg.annotation) if arg.annotation else None
            if annotation is not None and not annotation.is_valid:
                self._annotation_error(
                    arg,
                    f"unsupported annotation for parameter `{arg.arg}` in `{node.name}`",
                )
            if annotation is not None and not annotation.is_supported:
                self._compiler_error(
                    arg,
                    "PP900",
                    annotation.unsupported_reason
                    or f"annotation for parameter `{arg.arg}` in `{node.name}` is not lowered by this compiler yet",
                )
            if annotation is not None and annotation.is_none:
                self._annotation_error(
                    arg,
                    f"`None` is not a valid parameter annotation for `{arg.arg}` in `{node.name}`",
                )
            dtype = annotation.dtype if annotation and annotation.dtype is not None else Int64
            is_array = bool(annotation and annotation.is_array)
            shape = annotation.shape if annotation else AnyShape
            layout = annotation.layout if annotation else COrder
            is_output = False
            if parsed_ufunc_sig is not None:
                output_index = user_arg_index - len(parsed_ufunc_sig.inputs)
                is_output = 0 <= output_index < len(parsed_ufunc_sig.outputs)
            params.append(Param(arg.arg, dtype, shape, layout, is_array, is_output))
            param_types[arg.arg] = dtype

            if is_array and parsed_ufunc_sig is not None:
                dim_names: list[str] = []
                if user_arg_index < len(parsed_ufunc_sig.inputs):
                    dim_names = parsed_ufunc_sig.inputs[user_arg_index]
                else:
                    output_index = user_arg_index - len(parsed_ufunc_sig.inputs)
                    if output_index < len(parsed_ufunc_sig.outputs):
                        dim_names = parsed_ufunc_sig.outputs[output_index]
                self._array_dims[arg.arg] = [core_dim_values[d] for d in dim_names]
            user_arg_index += 1

        return_info = resolve_annotation_info(node.returns) if node.returns else None
        if return_info is not None and not return_info.is_valid:
            self._annotation_error(
                node,
                f"unsupported return annotation in `{node.name}`",
            )
        if return_info is not None and not return_info.is_supported:
            self._compiler_error(
                node,
                "PP900",
                return_info.unsupported_reason
                or f"return annotation in `{node.name}` is not lowered by this compiler yet",
            )
        returns_array = bool(return_info and return_info.is_array)
        return_dtype = (
            None
            if return_info is None or return_info.is_none
            else return_info.dtype
        )

        if parsed_ufunc_sig is not None and self._ufunc_kind == "vectorize":
            if any(param.is_array for param in params):
                self._compiler_error(
                    node,
                    "PP100",
                    "@vectorize kernels must take scalar parameters; use @guvectorize for core array arguments",
                )
            if returns_array or return_dtype is None:
                self._compiler_error(
                    node,
                    "PP100",
                    "@vectorize kernels must return a scalar dtype",
                )

        if parsed_ufunc_sig is not None and self._ufunc_kind == "guvectorize":
            output_param_count = max(0, len(params) - len(parsed_ufunc_sig.inputs))
            if output_param_count != len(parsed_ufunc_sig.outputs):
                self._compiler_error(
                    node,
                    "PP100",
                    "@guvectorize kernels must declare one trailing output parameter for each output in the layout signature",
                )
            for param in params[len(parsed_ufunc_sig.inputs):]:
                if not param.is_array:
                    self._compiler_error(
                        node,
                        "PP100",
                        f"@guvectorize output parameter `{param.name}` must be annotated as Array[...]",
                    )
            if return_dtype is not None or returns_array:
                self._compiler_error(
                    node,
                    "PP100",
                    "@guvectorize kernels must return None and write results through output parameters",
                )

        if parsed_ufunc_sig is not None and returns_array:
            output_index = len(params) - len(parsed_ufunc_sig.inputs)
            output_dims = (
                parsed_ufunc_sig.outputs[output_index]
                if 0 <= output_index < len(parsed_ufunc_sig.outputs)
                else []
            )
            self._implicit_array_return = "__pp_return"
            output_param = Param(
                self._implicit_array_return,
                return_dtype or Float64,
                return_info.shape if return_info else AnyShape,
                return_info.layout if return_info else COrder,
                True,
                True,
            )
            params.append(output_param)
            self._array_dims[output_param.name] = [core_dim_values[d] for d in output_dims]
            return_dtype = None

        # Build the IR function (or ufunc).
        doc = ast.get_docstring(node)
        if parsed_ufunc_sig is not None:
            fn: Function = UFunc(
                name=node.name,
                params=params,
                return_dtype=return_dtype,
                core_dim_params=core_dim_params,
                ufunc_sig=parsed_ufunc_sig,
                doc=doc,
            )
        else:
            fn = Function(
                name=node.name, params=params, return_dtype=return_dtype, doc=doc,
            )

        self._current_fn = fn
        self._param_types = param_types
        return fn

    def lower(self) -> Function:
        """Phase 2: lower the body into the declared Function."""
        node = self._node
        fn = self._current_fn

        # Run type inference over the body.
        constant_types = {
            name: dtype for name, (dtype, _) in self._module.constants.items()
        }
        self._type_map, tc_errors = infer_function(
            node, self._param_types, fn.return_dtype, constants=constant_types,
        )
        self._errors.extend(tc_errors)

        # Build IR. Reserve every user-visible name so generated temps
        # can't collide with locals (e.g. a user variable named `c1`).
        # Names stored anywhere in the body are function-local everywhere
        # in it (Python scoping) — track them so module constants never
        # shadow through.
        user_names = {p.name for p in fn.params}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                user_names.add(sub.id)
                self._assigned_names.add(sub.id)
            elif isinstance(sub, ast.arg):
                user_names.add(sub.arg)
        builder = Builder(fn, reserved=user_names)
        entry = fn.new_block("entry")
        builder.set_block(entry)

        # Seed locals with parameter values.
        for p in fn.params:
            v = Value(p.name, p.dtype, p.shape, p.layout, p.is_array, p.is_output)
            self._locals[p.name] = v
        for p in fn.core_dim_params:
            self._locals[p.name] = Value(p.name, p.dtype)

        # Lower the body.
        self._builder = builder
        self._lower_stmt_list(node.body)

        # Ensure the last block has a terminator.
        if builder.current_block.terminator is None:
            builder.terminate(Return(None))

        return fn

    # ---- statement lowering -----------------------------------------------

    def _lower_stmt_list(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if self._builder.current_block.terminator is not None:
                break
            self._lower_stmt(stmt)

    def _lower_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Return):
            self._lower_return(stmt)
        elif isinstance(stmt, ast.Break):
            self._lower_break(stmt)
        elif isinstance(stmt, ast.Continue):
            self._lower_continue(stmt)
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
        elif isinstance(stmt, (ast.Pass, ast.Assert)):
            return
        elif isinstance(stmt, ast.With):
            self._unsupported_feature_error(stmt, "`with` statements")
        elif isinstance(stmt, ast.Try):
            self._unsupported_feature_error(stmt, "`try` statements")
        elif hasattr(ast, "Match") and isinstance(stmt, ast.Match):
            self._unsupported_feature_error(stmt, "`match` statements")
        else:
            self._unsupported_feature_error(
                stmt,
                f"`{type(stmt).__name__}` statements",
            )

    def _lower_return(self, stmt: ast.Return) -> None:
        if stmt.value:
            val = self._lower_expr(stmt.value)
            if val is not None and (val.is_array or val.is_output) and self._current_fn.return_dtype is None:
                self._builder.terminate(Return(None))
            else:
                self._builder.terminate(Return(val))
        else:
            self._builder.terminate(Return(None))

    def _lower_break(self, stmt: ast.Break) -> None:
        if not self._loop_stack:
            self._compiler_error(stmt, "PP900", "`break` used outside a loop")
            return
        break_label, _ = self._loop_stack[-1]
        self._builder.terminate(Branch(break_label))

    def _lower_continue(self, stmt: ast.Continue) -> None:
        if not self._loop_stack:
            self._compiler_error(stmt, "PP900", "`continue` used outside a loop")
            return
        _, continue_label = self._loop_stack[-1]
        self._builder.terminate(Branch(continue_label))

    def _lower_assign(self, stmt: ast.Assign | ast.AnnAssign) -> None:
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return
            annotation = resolve_annotation_info(stmt.annotation)
            if (
                annotation.is_array
                and isinstance(stmt.target, ast.Name)
                and self._try_alias_array_allocation(stmt.target.id, stmt.value)
            ):
                return
            val = self._lower_expr(stmt.value)
            if val is None:
                return
            if isinstance(stmt.target, ast.Name):
                self._assign_name(
                    stmt.target.id,
                    val,
                    dtype=annotation.dtype,
                    shape=annotation.shape,
                    layout=annotation.layout,
                    is_array=annotation.is_array,
                )
            elif isinstance(stmt.target, ast.Subscript):
                self._lower_store(stmt.target, val)
            else:
                self._unsupported_feature_error(
                    stmt.target, "assignment targets of this form"
                )
        else:
            val = self._lower_expr(stmt.value)
            if val:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self._assign_name(target.id, val)
                    elif isinstance(target, ast.Subscript):
                        self._lower_store(target, val)
                    else:
                        self._unsupported_feature_error(
                            target,
                            "assignment targets of this form (tuple unpacking, attributes)",
                        )

    def _assign_name(
        self,
        name: str,
        val: Value,
        *,
        dtype: type[DType] | None = None,
        shape: Shape = AnyShape,
        layout: ArrayLayout = COrder,
        is_array: bool = False,
    ) -> Value:
        existing = self._locals.get(name)
        if existing is not None:
            self._builder.emit(AssignValue(existing, val))
            return existing

        target = Value(name, dtype or val.dtype, shape, layout, is_array)
        self._builder.emit(AssignValue(target, val, declare=True))
        self._locals[name] = target
        return target

    def _lower_aug_assign(self, stmt: ast.AugAssign) -> None:
        op_type = type(stmt.op)
        if op_type not in _AST_BINOP:
            self._unsupported_feature_error(
                stmt, f"augmented assignment with the `{type(stmt.op).__name__}` operator"
            )
            return
        bin_op = _AST_BINOP[op_type]

        if isinstance(stmt.target, ast.Subscript):
            left = self._lower_expr(stmt.target)
            right = self._lower_expr(stmt.value)
            if left is None or right is None:
                return
            result_dtype = self._binop_result_dtype(bin_op, left, right)
            result = self._builder.make_value("aug", result_dtype)
            self._builder.emit(BinOpInstr(result, bin_op, left, right))
            self._lower_store(stmt.target, result)
            return

        if not isinstance(stmt.target, ast.Name):
            self._unsupported_feature_error(stmt, "augmented assignment to this target form")
            return
        name = stmt.target.id
        left = self._locals.get(name)
        if left is None:
            self._compiler_error(
                stmt, "PP900",
                f"augmented assignment to unknown name `{name}`",
            )
            return
        right = self._lower_expr(stmt.value)
        if right is None:
            return
        result_dtype = self._binop_result_dtype(bin_op, left, right)
        result = self._builder.make_value("aug", result_dtype)
        self._builder.emit(BinOpInstr(result, bin_op, left, right))
        self._builder.emit(AssignValue(left, result))
        self._locals[name] = left

    def _lower_for(self, stmt: ast.For) -> None:
        # Only handle range() loops in the skeleton.
        if not (
            isinstance(stmt.iter, ast.Call)
            and isinstance(stmt.iter.func, ast.Name)
            and stmt.iter.func.id == "range"
            and isinstance(stmt.target, ast.Name)
        ):
            self._unsupported_feature_error(
                stmt,
                "`for` loops over non-range iterables",
            )
            return

        b = self._builder
        loop_var = stmt.target.id

        # Resolve range args.
        range_args = stmt.iter.args
        if len(range_args) == 1:
            start_v = b.make_value("c", Int64)
            b.emit(Const(start_v, 0))
            stop_v  = self._lower_expr(range_args[0]) or Value("_stop", Int64)
            step_v  = b.make_value("c", Int64)
            b.emit(Const(step_v, 1))
        elif len(range_args) == 2:
            start_v = self._lower_expr(range_args[0]) or Value("_start", Int64)
            stop_v  = self._lower_expr(range_args[1]) or Value("_stop", Int64)
            step_v  = b.make_value("c", Int64)
            b.emit(Const(step_v, 1))
        else:
            start_v = self._lower_expr(range_args[0]) or Value("_start", Int64)
            stop_v  = self._lower_expr(range_args[1]) or Value("_stop", Int64)
            step_v  = self._lower_expr(range_args[2]) or Value("_step", Int64)

        # Reuse the existing binding when the loop variable was already
        # declared (a prior loop or assignment); redeclaring it would emit
        # a duplicate C declaration in the shared function scope.
        existing = self._locals.get(loop_var)
        if existing is not None:
            idx = existing
            # Python evaluates range() before rebinding the loop variable,
            # so snapshot bounds that alias the variable itself
            # (e.g. `for n in range(n)`).
            if stop_v is idx:
                stop_v = self._copy_to_temp(stop_v)
            if step_v is idx:
                step_v = self._copy_to_temp(step_v)
            b.emit(AssignValue(idx, start_v))
        else:
            idx = Value(loop_var, Int64)
            b.emit(AssignValue(idx, start_v, declare=True))
        self._locals[loop_var] = idx

        cond_bb  = b.new_block(b.fresh("for_cond"))
        body_bb  = b.new_block(b.fresh("for_body"))
        step_bb  = b.new_block(b.fresh("for_step"))
        after_bb = b.new_block(b.fresh("for_after"))

        b.terminate(Branch(cond_bb.label))

        # Condition block.
        b.set_block(cond_bb)
        cond_val = b.make_value("cond", Bool)
        b.emit(BinOpInstr(cond_val, BinOp.LT, idx, stop_v))
        b.terminate(CondBranch(cond_val, body_bb.label, after_bb.label))

        # Body block.
        b.set_block(body_bb)
        self._loop_stack.append((after_bb.label, step_bb.label))
        self._lower_stmt_list(stmt.body)
        self._loop_stack.pop()
        if b.current_block.terminator is None:
            b.terminate(Branch(step_bb.label))

        # Increment block.
        b.set_block(step_bb)
        if b.current_block.terminator is None:
            next_idx = b.make_value("idx_next", Int64)
            b.emit(BinOpInstr(next_idx, BinOp.ADD, idx, step_v))
            b.emit(AssignValue(idx, next_idx))
            self._locals[loop_var] = idx
            b.terminate(Branch(cond_bb.label))

        b.set_block(after_bb)

    def _lower_while(self, stmt: ast.While) -> None:
        b = self._builder
        cond_bb  = b.new_block(b.fresh("while_cond"))
        body_bb  = b.new_block(b.fresh("while_body"))
        after_bb = b.new_block(b.fresh("while_after"))

        b.terminate(Branch(cond_bb.label))
        b.set_block(cond_bb)
        cond_val = self._lower_expr(stmt.test)
        if cond_val:
            b.terminate(CondBranch(cond_val, body_bb.label, after_bb.label))

        b.set_block(body_bb)
        self._loop_stack.append((after_bb.label, cond_bb.label))
        self._lower_stmt_list(stmt.body)
        self._loop_stack.pop()
        if b.current_block.terminator is None:
            b.terminate(Branch(cond_bb.label))

        b.set_block(after_bb)

    def _lower_if(self, stmt: ast.If) -> None:
        b = self._builder
        cond_val = self._lower_expr(stmt.test)
        then_bb  = b.new_block(b.fresh("if_then"))
        else_bb  = b.new_block(b.fresh("if_else")) if stmt.orelse else None
        after_bb = b.new_block(b.fresh("if_after"))

        if cond_val:
            b.terminate(CondBranch(
                cond_val,
                then_bb.label,
                (else_bb.label if else_bb else after_bb.label),
            ))

        b.set_block(then_bb)
        self._lower_stmt_list(stmt.body)
        if b.current_block.terminator is None:
            b.terminate(Branch(after_bb.label))

        if else_bb:
            b.set_block(else_bb)
            self._lower_stmt_list(stmt.orelse)
            if b.current_block.terminator is None:
                b.terminate(Branch(after_bb.label))

        b.set_block(after_bb)

    # ---- expression lowering ----------------------------------------------

    def _binop_result_dtype(self, op: BinOp, left: Value, right: Value) -> type[DType]:
        """Result dtype for a binary operation on two lowered values.

        Python true division always yields a float, so integer / integer
        promotes to Float64 rather than staying integral.
        """
        result = promote(left.dtype, right.dtype)
        if op == BinOp.DIV and result.kind in ("i", "u", "b"):
            return Float64
        return result

    def _lower_expr(self, node: ast.expr) -> Optional[Value]:
        b = self._builder

        if isinstance(node, ast.Constant):
            return self._lower_const(node)

        if isinstance(node, ast.Name):
            value = self._locals.get(node.id)
            if value is not None:
                return value
            # Python scoping: a name assigned anywhere in the function is
            # local everywhere in it — reading it before assignment is an
            # error, never a fall-through to module scope.
            if node.id in self._assigned_names:
                self._compiler_error(
                    node,
                    "PP100",
                    f"local variable `{node.id}` referenced before assignment",
                )
                return None
            const = self._module.constants.get(node.id)
            if const is not None:
                dtype, pyval = const
                result = b.make_value("k", dtype)
                b.emit(Const(result, pyval))
                return result
            if node.id in self._module.post_imports:
                self._compiler_error(
                    node,
                    "PP900",
                    f"`{node.id}` names a POST function; function references "
                    "as values are not lowered by this compiler yet",
                )
                return None
            boundary = self._module.boundary_imports.get(node.id)
            if boundary is not None:
                self._compiler_error(
                    node,
                    "PP900",
                    f"`{node.id}` is imported from `{boundary.module_name}`, "
                    "which was not resolved as a POST translation unit",
                )
                return None
            self._compiler_error(
                node,
                "PP900",
                f"name `{node.id}` is not a parameter, local, module "
                "constant, or imported constant",
            )
            return None

        if isinstance(node, ast.NamedExpr):
            # Walrus operator: assign and yield the value (spec §5.1).
            value = self._lower_expr(node.value)
            if value is None or not isinstance(node.target, ast.Name):
                return None
            return self._assign_name(node.target.id, value)

        if isinstance(node, ast.BinOp):
            left  = self._lower_expr(node.left)
            right = self._lower_expr(node.right)
            if left is None or right is None:
                return None
            op_type = type(node.op)
            if op_type not in _AST_BINOP:
                self._unsupported_feature_error(
                    node, f"the `{type(node.op).__name__}` operator"
                )
                return None
            bin_op = _AST_BINOP[op_type]
            result_dtype = self._binop_result_dtype(bin_op, left, right)
            result = b.make_value("v", result_dtype)
            b.emit(BinOpInstr(result, bin_op, left, right))
            return result

        if isinstance(node, ast.UnaryOp):
            operand = self._lower_expr(node.operand)
            if operand is None:
                return None
            op_type = type(node.op)
            if op_type is ast.UAdd:
                return operand  # unary plus is the identity
            un_op = _AST_UNOP.get(op_type)
            if un_op is None:
                self._unsupported_feature_error(
                    node, f"the `{type(node.op).__name__}` unary operator"
                )
                return None
            result_dtype = Bool if op_type is ast.Not else operand.dtype
            result = b.make_value("v", result_dtype)
            b.emit(UnaryOpInstr(result, un_op, operand))
            return result

        if isinstance(node, ast.Subscript):
            return self._lower_subscript_load(node)

        if isinstance(node, ast.Call):
            return self._lower_call(node)

        if isinstance(node, ast.Compare):
            return self._lower_compare(node)

        if isinstance(node, ast.BoolOp):
            return self._lower_boolop(node)

        if isinstance(node, ast.IfExp):
            cond = self._lower_expr(node.test)
            if_true = self._lower_expr(node.body)
            if_false = self._lower_expr(node.orelse)
            if cond is None or if_true is None or if_false is None:
                return None
            result_dtype = promote(if_true.dtype, if_false.dtype)
            result = b.make_value("select", result_dtype)
            b.emit(Select(result, cond, if_true, if_false))
            return result

        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
            self._unsupported_feature_error(node, "comprehensions")
            return None

        # Diagnose instead of silently dropping: any expression form
        # without a lowering above must not vanish from the output.
        self._unsupported_feature_error(
            node, f"`{type(node).__name__}` expressions"
        )
        return None

    def _lower_const(self, node: ast.Constant) -> Value:
        b = self._builder
        v = node.value
        if isinstance(v, bool):
            val = b.make_value("b", Bool)
            b.emit(Const(val, bool(v)))
        elif isinstance(v, int):
            val = b.make_value("i", Int64)
            b.emit(Const(val, v))
        elif isinstance(v, float):
            val = b.make_value("f", Float64)
            b.emit(Const(val, v))
        elif isinstance(v, complex):
            val = b.make_value("z", Complex128)
            b.emit(Const(val, v))
        else:
            val = b.make_value("c", Int64)
            b.emit(Const(val, 0))
        return val

    def _lower_compare(self, node: ast.Compare) -> Optional[Value]:
        """Lower comparisons, including chains like ``a < b < c``.

        A chain ``a op0 b op1 c`` is lowered to ``(a op0 b) and (b op1 c)``,
        evaluating each comparand exactly once. POST Python expressions are
        side-effect free at the boundaries we lower, so non-short-circuit
        evaluation is observably equivalent.
        """
        b = self._builder
        left = self._lower_expr(node.left)
        if left is None or not node.comparators:
            return None

        comparisons: list[Value] = []
        prev = left
        for op_node, comparator in zip(node.ops, node.comparators):
            right = self._lower_expr(comparator)
            if right is None:
                return None
            op_type = type(op_node)
            cmp_op = _AST_BINOP.get(op_type)
            if cmp_op is None:
                self._unsupported_feature_error(
                    node, f"`{type(op_node).__name__}` comparisons"
                )
                return None
            cmp_val = b.make_value("cmp", Bool)
            b.emit(BinOpInstr(cmp_val, cmp_op, prev, right))
            comparisons.append(cmp_val)
            prev = right

        result = comparisons[0]
        for next_cmp in comparisons[1:]:
            combined = b.make_value("and", Bool)
            b.emit(BinOpInstr(combined, BinOp.AND, result, next_cmp))
            result = combined
        return result

    def _lower_boolop(self, node: ast.BoolOp) -> Optional[Value]:
        """Lower ``and`` / ``or`` to chained binary ops on Bool values."""
        b = self._builder
        op = BinOp.AND if isinstance(node.op, ast.And) else BinOp.OR
        operands: list[Value] = []
        for value in node.values:
            v = self._lower_expr(value)
            if v is None:
                return None
            operands.append(v)
        if not operands:
            return None
        result = operands[0]
        for nxt in operands[1:]:
            new = b.make_value("bool", Bool)
            b.emit(BinOpInstr(new, op, result, nxt))
            result = new
        return result

    def _lower_call(self, node: ast.Call) -> Optional[Value]:
        b = self._builder
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "len" and node.args:
                known_len = self._resolve_len_arg(node.args[0])
                if known_len is not None:
                    return known_len
                arr = self._lower_expr(node.args[0])
                result = b.make_value("len", Int64)
                if arr:
                    b.emit(Call(result, "__pp_len__", [arr]))
                return result
            if name == "abs" and len(node.args) == 1:
                operand = self._lower_expr(node.args[0])
                if operand is None:
                    return None
                # abs of complex collapses to the underlying float type.
                if operand.dtype is Complex128:
                    result_dtype: type[DType] = Float64
                elif operand.dtype is Complex64:
                    result_dtype = Float32
                else:
                    result_dtype = operand.dtype
                result = b.make_value("abs", result_dtype)
                b.emit(UnaryOpInstr(result, UnaryOp.ABS, operand))
                return result
            if name == "range":
                # For-loop lowering consumes range() directly; reaching here
                # means range was used as a value, which has no lowering.
                self._compiler_error(
                    node,
                    "PP900",
                    "`range()` is only supported as a `for` loop iterable",
                )
                return None
            cast_dtype = resolve_annotation(ast.Name(id=name, ctx=ast.Load()))
            if cast_dtype is not None and node.args:
                operand = self._lower_expr(node.args[0])
                if operand is None:
                    return None
                result = b.make_value("cast", cast_dtype)
                b.emit(Cast(result, operand))
                return result
        elif isinstance(node.func, ast.Attribute):
            self._compiler_error(
                node,
                "PP900",
                f"attribute calls (`{ast.unparse(node.func)}(...)`) are not "
                "lowered by this compiler yet; import the function by name",
            )
            return None
        args = [v for a in node.args if (v := self._lower_expr(a)) is not None]
        func_name = (
            node.func.id if isinstance(node.func, ast.Name)
            else str(ast.unparse(node.func))
        )

        symbol, callee = self._resolve_callee(node, func_name)
        if symbol is None:
            return None  # diagnostic already recorded

        if callee is not None and callee.core_dim_params:
            args.extend(self._resolve_call_core_dims(callee, node.args))

        if callee is not None:
            # POST function (this module or an imported one): the callee's
            # signature is the source of truth for the result dtype.
            if callee.return_dtype is None:
                b.emit(Call(None, symbol, args))
                return None
            result = b.make_value("ret", callee.return_dtype)
            b.emit(Call(result, symbol, args))
            return result

        # Intrinsic (postpyc.math / libm): infer the return dtype by
        # promoting argument types, falling back to Float64.
        if args:
            ret_dtype = args[0].dtype
            for a in args[1:]:
                ret_dtype = promote(ret_dtype, a.dtype)
        else:
            ret_dtype = Float64
        result = b.make_value("ret", ret_dtype)
        b.emit(Call(result, symbol, args))
        return result

    def _resolve_callee(self, node: ast.Call, name: str) -> tuple[Optional[str], Optional[Function]]:
        """Resolve a called name to (symbol, callee IR).

        Resolution order: functions of this module (including forward
        references, since all declarations exist before bodies are
        lowered), names imported from POST modules (following aliases),
        postpyc.math intrinsics, then a small builtin whitelist.
        Anything else is diagnosed — per spec §9.1, a call must not be
        silently compiled against an arbitrary imported (or libm) symbol.
        """
        fn = self._module.get_function(name)
        if fn is not None:
            return name, fn

        imp = self._module.post_imports.get(name)
        if imp is not None:
            dep = self._program.get(imp.module_name)
            callee = dep.get_function(imp.source_name) if dep is not None else None
            if callee is None:
                self._compiler_error(
                    node,
                    "PP502",
                    f"cannot find function `{imp.source_name}` in POST module "
                    f"`{imp.module_name}`",
                )
                return None, None
            if imp.source_name.startswith("_"):
                self._compiler_error(
                    node,
                    "PP503",
                    f"`{imp.source_name}` is private to POST module "
                    f"`{imp.module_name}` and cannot be called across modules",
                )
                return None, None
            return imp.source_name, callee

        intrinsic = self._module.intrinsic_imports.get(name)
        if intrinsic is not None:
            return intrinsic, None

        boundary = self._module.boundary_imports.get(name)
        if boundary is not None:
            self._compiler_error(
                node,
                "PP900",
                f"call to `{name}` imported from `{boundary.module_name}`, "
                "which was not resolved as a POST translation unit; "
                "CPython-boundary calls are not lowered by this compiler yet",
            )
            return None, None

        if name in _BUILTIN_LIBM_CALLS:
            return _BUILTIN_LIBM_CALLS[name], None

        self._compiler_error(
            node,
            "PP502",
            f"call to unknown function `{name}`; it is not defined in this "
            "module and not imported from a POST module",
        )
        return None, None

    def _resolve_call_core_dims(self, callee: Function, arg_nodes: list[ast.expr]) -> list[Value]:
        resolved: list[Value] = []
        seen: set[str] = set()
        for arg_node in arg_nodes:
            for dim in self._resolve_array_dims_arg(arg_node):
                if dim.name in seen:
                    continue
                resolved.append(dim)
                seen.add(dim.name)
                if len(resolved) == len(callee.core_dim_params):
                    return resolved
        return resolved

    def _resolve_array_dims_arg(self, node: ast.expr) -> list[Value]:
        if isinstance(node, ast.Name):
            return list(self._array_dims.get(node.id, []))
        if isinstance(node, ast.Subscript):
            root = self._subscript_root_and_indices(node)
            if root is None:
                return []
            name, index_nodes = root
            dims = self._array_dims.get(name, [])
            return list(dims[len(index_nodes):])
        return []

    def _try_alias_array_allocation(self, name: str, value: ast.expr) -> bool:
        if isinstance(value, ast.Name):
            existing = self._locals.get(value.id)
            if existing is not None and existing.is_array:
                self._locals[name] = existing
                if value.id in self._array_dims:
                    self._array_dims[name] = self._array_dims[value.id]
                return True

        if not self._is_array_allocation_expr(value):
            return False

        if self._implicit_array_return is None:
            return False

        output = self._locals.get(self._implicit_array_return)
        if output is None:
            return False

        self._locals[name] = output
        if self._implicit_array_return in self._array_dims:
            self._array_dims[name] = self._array_dims[self._implicit_array_return]
        return True

    def _is_array_allocation_expr(self, value: ast.expr) -> bool:
        if isinstance(value, ast.List):
            return True
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
            return isinstance(value.left, ast.List) or isinstance(value.right, ast.List)
        return False

    def _subscript_root_and_indices(self, node: ast.Subscript) -> tuple[str, list[ast.expr]] | None:
        indices: list[ast.expr] = []
        current: ast.expr = node
        while isinstance(current, ast.Subscript):
            indices.append(current.slice)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        indices.reverse()
        return current.id, indices

    def _copy_to_temp(self, v: Value) -> Value:
        temp = self._builder.make_value("tmp", v.dtype)
        self._builder.emit(AssignValue(temp, v, declare=True))
        return temp

    def _const_int_value(self, value: int, prefix: str = "dim") -> Value:
        result = self._builder.make_value(prefix, Int64)
        self._builder.emit(Const(result, value))
        return result

    def _dim_value_for_axis(self, name: str, arr: Value, axis: int) -> Optional[Value]:
        dims = self._array_dims.get(name)
        if dims and axis < len(dims):
            return dims[axis]
        if arr.shape is not AnyShape and axis < len(arr.shape.dims):
            dim = arr.shape.dims[axis]
            if dim is not None:
                return self._const_int_value(dim)
        if arr.is_array:
            result = self._builder.make_value("dim", Int64)
            self._builder.emit(ArrayDim(result, arr, axis))
            return result
        return None

    def _mul_values(self, values: list[Value]) -> Value:
        if not values:
            return self._const_int_value(1)
        result = values[0]
        for value in values[1:]:
            next_result = self._builder.make_value("stride", Int64)
            self._builder.emit(BinOpInstr(next_result, BinOp.MUL, result, value))
            result = next_result
        return result

    def _scale_stride_to_bytes(self, stride: Value, arr: Value, node: ast.AST) -> Optional[Value]:
        itemsize = getattr(arr.dtype, "itemsize", 0)
        if itemsize <= 0:
            self._compiler_error(
                node,
                "PP900",
                f"array dtype `{arr.dtype.__name__}` does not have a fixed byte size",
            )
            return None
        if itemsize == 1:
            return stride
        itemsize_value = self._const_int_value(itemsize, "itemsize")
        result = self._builder.make_value("stride", Int64)
        self._builder.emit(BinOpInstr(result, BinOp.MUL, stride, itemsize_value))
        return result

    def _array_stride_values(
        self,
        name: str,
        arr: Value,
        ndim: int,
        node: ast.AST,
    ) -> Optional[list[Value]]:
        layout = arr.layout

        if isinstance(layout, Strides):
            if len(layout.strides) < ndim:
                self._compiler_error(
                    node,
                    "PP300",
                    f"array `{name}` has {len(layout.strides)} stride value(s), "
                    f"but {ndim} index value(s) were used",
                )
                return None
            result: list[Value] = []
            for axis, stride in enumerate(layout.strides[:ndim]):
                if stride is None:
                    stride_value = self._builder.make_value("stride", Int64)
                    self._builder.emit(ArrayStride(stride_value, arr, axis))
                    result.append(stride_value)
                else:
                    result.append(self._const_int_value(stride, "stride"))
            return result

        if self._ufunc_sig is not None and arr.is_array:
            # Ufunc kernels receive views built by the wrapper from NumPy's
            # steps array, so strides must be read at runtime: NumPy does
            # not guarantee compact core slices (e.g. transposed inputs).
            runtime_strides: list[Value] = []
            for axis in range(ndim):
                stride_value = self._builder.make_value("stride", Int64)
                self._builder.emit(ArrayStride(stride_value, arr, axis))
                runtime_strides.append(stride_value)
            return runtime_strides

        dim_values = [self._dim_value_for_axis(name, arr, axis) for axis in range(ndim)]
        dims = [dim for dim in dim_values if dim is not None]

        if layout == COrder:
            strides = [self._mul_values(dims[axis + 1:]) for axis in range(ndim)]
            byte_strides: list[Value] = []
            for stride in strides:
                byte_stride = self._scale_stride_to_bytes(stride, arr, node)
                if byte_stride is None:
                    return None
                byte_strides.append(byte_stride)
            return byte_strides
        if layout == FOrder:
            strides = [self._mul_values(dims[:axis]) for axis in range(ndim)]
            byte_strides = []
            for stride in strides:
                byte_stride = self._scale_stride_to_bytes(stride, arr, node)
                if byte_stride is None:
                    return None
                byte_strides.append(byte_stride)
            return byte_strides

        self._compiler_error(
            node,
            "PP900",
            f"array layout `{layout!r}` is not lowered by this compiler yet",
        )
        return None

    def _flatten_array_index(self, name: str, index_nodes: list[ast.expr]) -> tuple[Value, Value] | None:
        arr = self._locals.get(name)
        if arr is None:
            self._compiler_error(
                index_nodes[0] if index_nodes else self._node,
                "PP900",
                f"subscripted name `{name}` is not a lowered local array",
            )
            return None
        lowered = [self._lower_expr(index) for index in index_nodes]
        if any(index is None for index in lowered):
            return None
        indices = [index for index in lowered if index is not None]
        if not indices:
            return None

        strides = self._array_stride_values(name, arr, len(indices), index_nodes[0])
        if strides is None:
            return None

        flat: Value | None = None
        for index, stride in zip(indices, strides):
            if isinstance(stride, Value):
                term = self._builder.make_value("idx", Int64)
                self._builder.emit(BinOpInstr(term, BinOp.MUL, index, stride))
            else:
                term = index
            if flat is None:
                flat = term
            else:
                next_flat = self._builder.make_value("idx", Int64)
                self._builder.emit(BinOpInstr(next_flat, BinOp.ADD, flat, term))
                flat = next_flat
        if flat is None:
            return None
        return arr, flat

    def _lower_subscript_load(self, node: ast.Subscript) -> Optional[Value]:
        root = self._subscript_root_and_indices(node)
        if root is None:
            self._unsupported_feature_error(
                node, "subscripts of non-name expressions"
            )
            return None
        name, index_nodes = root
        flattened = self._flatten_array_index(name, index_nodes)
        if flattened is None:
            return None
        arr, idx = flattened
        result = self._builder.make_value("elem", arr.dtype)
        self._builder.emit(ArrayLoad(result, arr, idx))
        return result

    def _lower_store(self, target: ast.Subscript, val: Value) -> None:
        root = self._subscript_root_and_indices(target)
        if root is None:
            self._unsupported_feature_error(
                target, "subscript stores to non-name expressions"
            )
            return
        name, index_nodes = root
        flattened = self._flatten_array_index(name, index_nodes)
        if flattened is None:
            return
        arr, idx = flattened
        self._builder.emit(ArrayStore(arr, idx, val))

    def _resolve_len_arg(self, node: ast.expr) -> Optional[Value]:
        if isinstance(node, ast.Name):
            arr = self._locals.get(node.id)
            if arr is not None:
                dim = self._dim_value_for_axis(node.id, arr, 0)
                if dim is not None:
                    return dim
            dims = self._array_dims.get(node.id)
            if dims:
                return dims[0]
        if isinstance(node, ast.Subscript):
            root = self._subscript_root_and_indices(node)
            if root is None:
                return None
            name, index_nodes = root
            arr = self._locals.get(name)
            if arr is not None:
                dim = self._dim_value_for_axis(name, arr, len(index_nodes))
                if dim is not None:
                    return dim
            dims = self._array_dims.get(name)
            if dims and len(index_nodes) < len(dims):
                return dims[len(index_nodes)]
        return None


# ---------------------------------------------------------------------------
# Module builder
# ---------------------------------------------------------------------------

_UFUNC_DECORATORS = frozenset({"vectorize", "guvectorize"})


@dataclass(frozen=True)
class _UFuncDecorator:
    kind: str
    signature: str | None = None


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _constant_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_layout_signature(call: ast.Call) -> str | None:
    for arg in call.args:
        value = _constant_string(arg)
        if value is not None and "->" in value:
            return value
    for keyword in call.keywords:
        if keyword.arg in {"signature", "layout", "sig"}:
            value = _constant_string(keyword.value)
            if value is not None and "->" in value:
                return value
    return None


def _scalar_ufunc_signature(node: ast.FunctionDef) -> str:
    n_inputs = sum(1 for arg in node.args.args if arg.arg not in {"self", "cls"})
    return ",".join("()" for _ in range(n_inputs)) + "->()"


def _extract_ufunc_decorator(node: ast.FunctionDef) -> _UFuncDecorator | None:
    for decorator in node.decorator_list:
        name = _decorator_name(decorator)
        if name not in _UFUNC_DECORATORS:
            continue
        if name == "vectorize":
            return _UFuncDecorator("vectorize", _scalar_ufunc_signature(node))
        if isinstance(decorator, ast.Call):
            return _UFuncDecorator(name, _extract_layout_signature(decorator))
        return _UFuncDecorator(name, None)
    return None


def _iter_import_froms(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node


# ---------------------------------------------------------------------------
# Module-level constants (spec §3: top-level constant definitions)
# ---------------------------------------------------------------------------

_CONST_DTYPES: dict[type, type[DType]] = {
    bool: Bool,
    int: Int64,
    float: Float64,
    complex: Complex128,
}

_FOLD_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _fold_constant_expr(node: ast.expr, constants: dict[str, tuple[type[DType], object]]):
    """Evaluate a module-level constant expression at compile time.

    Supports numeric literals, references to previously defined constants,
    unary +/-, and literal arithmetic (matching interpreted semantics).
    Returns the Python value, or None when the expression is not a
    compile-time constant.
    """
    if isinstance(node, ast.Constant) and type(node.value) in _CONST_DTYPES:
        return node.value
    if isinstance(node, ast.Name):
        entry = constants.get(node.id)
        return entry[1] if entry is not None else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _fold_constant_expr(node.operand, constants)
        if operand is None:
            return None
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        fold = _FOLD_BINOPS.get(type(node.op))
        if fold is None:
            return None
        left = _fold_constant_expr(node.left, constants)
        right = _fold_constant_expr(node.right, constants)
        if left is None or right is None:
            return None
        try:
            return fold(left, right)
        except (ArithmeticError, TypeError):
            return None
    return None


def _collect_module_constants(tree: ast.Module, module: Module) -> None:
    """Record foldable module-level constant definitions on *module*.

    Runs after import classification so constant expressions can reference
    compile-time imports (e.g. ``TWO_PI = 2.0 * PI``). Non-constant
    top-level assignments (function aliases, containers) are skipped; the
    alias export policy is separate work (postpython#12).
    """
    for node in tree.body:
        target = None
        annotated_dtype: Optional[type[DType]] = None
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                target = node.target.id
                info = resolve_annotation_info(node.annotation)
                if info.is_array or not info.is_valid or info.dtype is None:
                    continue
                annotated_dtype = info.dtype
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
        if target is None:
            continue
        value = _fold_constant_expr(node.value, module.constants)
        if value is None:
            continue
        dtype = annotated_dtype or _CONST_DTYPES[type(value)]
        module.constants[target] = (dtype, value)


def _classify_imports(
    tree: ast.Module,
    module: Module,
    program: dict[str, Module],
    errors: list,
) -> None:
    """Sort ``from X import name`` bindings into POST / intrinsic / boundary.

    A module counts as a POST module import only when it has already been
    compiled into *program* (compile_program does this, dependencies
    first). Everything unresolved is a CPython-boundary import: legal to
    import, diagnosed only if called.
    """
    for node in _iter_import_froms(tree):
        mod = node.module
        if mod == _INTRINSIC_MODULE:
            # Compile-time imports (spec §9.1): functions lower to libm
            # calls; numeric constants (PI, E, ...) fold to their values.
            import postpyc.math as _pp_math
            for alias in node.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                attr = getattr(_pp_math, alias.name, None)
                if type(attr) in _CONST_DTYPES:
                    module.constants[local] = (_CONST_DTYPES[type(attr)], attr)
                else:
                    module.intrinsic_imports[local] = alias.name
            continue
        if mod in _COMPILE_TIME_MODULES or mod.split(".")[0] == "postpyc":
            continue
        for alias in node.names:
            if alias.name == "*":
                errors.append(TypeError_PP(
                    code="PP900",
                    message=f"`from {mod} import *` is not supported for POST module imports",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ))
                continue
            local = alias.asname or alias.name
            imported = ImportedName(local, mod, alias.name)
            if mod in program:
                dep = program[mod]
                dep_const = dep.constants.get(alias.name)
                if dep_const is not None and dep.get_function(alias.name) is None:
                    # Imported POST constant: fold the value in directly.
                    module.constants[local] = dep_const
                    continue
                module.post_imports[local] = imported
                if mod not in module.dependencies:
                    module.dependencies.append(mod)
            else:
                module.boundary_imports[local] = imported


def compile_source(
    source: str,
    filename: str = "<unknown>",
    *,
    program: Optional[dict[str, Module]] = None,
) -> tuple[Module, list]:
    """Parse and lower *source* to a POST Python Module.

    Returns (module, errors) where errors is a list of Violation or
    type-error strings.  An empty error list means the translation
    succeeded cleanly.

    *program* maps dotted module names to already-compiled POST Modules;
    calls to names imported from those modules resolve against their IR
    (see compile_program). Without it, POST-style imports are treated as
    CPython-boundary imports.
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
    errors = []
    program = program or {}
    _classify_imports(tree, module, program, errors)
    _collect_module_constants(tree, module)

    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Expr):
            if (
                index == 0
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            errors.append(TypeError_PP(
                code="PP900",
                message="top-level executable expressions are not lowered by this compiler yet",
                lineno=node.lineno,
                col_offset=node.col_offset,
            ))
        elif isinstance(node, ast.ClassDef):
            errors.append(TypeError_PP(
                code="PP900",
                message="class/dataclass definitions are valid POST Python but are not lowered by this compiler yet",
                lineno=node.lineno,
                col_offset=node.col_offset,
            ))
        elif isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.With,
                ast.Try,
            ),
        ) or (hasattr(ast, "Match") and isinstance(node, ast.Match)):
            errors.append(TypeError_PP(
                code="PP900",
                message="top-level executable statements are not lowered by this compiler yet",
                lineno=node.lineno,
                col_offset=node.col_offset,
            ))

    # Phase 1: declare every function so calls (including forward
    # references) can resolve against complete signatures.
    lifters: list[FunctionLifter] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            ufunc_decorator = _extract_ufunc_decorator(node)
            ufunc_sig = ufunc_decorator.signature if ufunc_decorator else None
            ufunc_kind = ufunc_decorator.kind if ufunc_decorator else None
            if ufunc_decorator is not None and ufunc_sig is None:
                errors.append(TypeError_PP(
                    code="PP100",
                    message=f"@{ufunc_kind} requires a NumPy ufunc layout signature",
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ))
                continue
            if ufunc_sig is not None:
                try:
                    parse_ufunc_sig(ufunc_sig)
                except ValueError as exc:
                    errors.append(TypeError_PP(
                        code="PP100",
                        message=str(exc),
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    ))
                    continue

            lifter = FunctionLifter(node, module, ufunc_sig, ufunc_kind, program=program)
            module.add_function(lifter.declare())
            lifters.append(lifter)

    # Module-level function aliases (``gammaln = lgamma``) — collected
    # after declarations so alias targets are recognizable. Constant
    # definitions were already claimed by the constant collector.
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
            and node.targets[0].id not in module.constants
        ):
            target_name = node.value.id
            if (
                module.get_function(target_name) is not None
                or target_name in module.post_imports
                or target_name in module.function_aliases
            ):
                module.function_aliases[node.targets[0].id] = target_name

    # Phase 2: lower bodies.
    for lifter in lifters:
        lifter.lower()
        errors.extend(lifter.errors)

    return module, errors


# ---------------------------------------------------------------------------
# Program compilation (multiple translation units)
# ---------------------------------------------------------------------------

def _resolve_post_module(
    dotted: str,
    importer: Path,
    search_paths: list[Path] | None = None,
) -> Optional[Path]:
    """Locate the source file for an absolute POST module import.

    Resolution is deliberately narrow: the importer's source root (the
    first ancestor directory that is not a package) plus any explicitly
    supplied *search_paths*. Installed site-packages and the standard
    library are NOT searched — an import that does not resolve here is a
    CPython-boundary import (spec §9.1), not a translation unit to
    compile. Arbitrary installed Python must never be pulled into a POST
    build implicitly.
    """
    root = importer.resolve().parent
    while (root / "__init__.py").exists() and root.parent != root:
        root = root.parent

    parts = dotted.split(".")
    for base in [root, *(Path(p).resolve() for p in search_paths or [])]:
        candidate = base.joinpath(*parts).with_suffix(".py")
        if candidate.is_file():
            return candidate
        package_init = base.joinpath(*parts, "__init__.py")
        if package_init.is_file():
            return package_init
    return None


def compile_program(
    path: str | Path,
    *,
    search_paths: list[Path] | None = None,
) -> tuple[list[Module], list]:
    """Compile *path* and every POST module it imports, dependencies first.

    Returns (modules, errors) where *modules* is dependency-ordered (the
    entry module last). Each imported POST translation unit is checked and
    compiled once; import cycles are diagnosed as PP500.

    POST imports resolve against the entry file's source root plus
    *search_paths*; anything else (standard library, site-packages) is a
    CPython-boundary import and is never compiled implicitly.
    """
    errors: list = []
    program: dict[str, Module] = {}
    order: list[Module] = []
    visiting: set[Path] = set()
    compiled: dict[Path, str] = {}  # resolved file → registered dotted name

    def compile_unit(file_path: Path, dotted: Optional[str]) -> None:
        file_path = file_path.resolve()
        if file_path in compiled:
            # Same file reachable under a second dotted spelling.
            if dotted is not None and dotted not in program:
                program[dotted] = program[compiled[file_path]]
            return
        if file_path in visiting:
            errors.append(TypeError_PP(
                code="PP500",
                message=f"circular POST module import involving `{file_path}`",
                lineno=0,
                col_offset=0,
            ))
            return
        visiting.add(file_path)

        source = file_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(file_path), type_comments=True)
        except SyntaxError:
            tree = None  # checker will report PP000 below

        # Depth-first: resolve and compile POST dependencies first.
        if tree is not None:
            for node in _iter_import_froms(tree):
                mod = node.module
                if (
                    mod == _INTRINSIC_MODULE
                    or mod in _COMPILE_TIME_MODULES
                    or mod.split(".")[0] == "postpyc"
                ):
                    continue
                dep_path = _resolve_post_module(mod, file_path, search_paths)
                if dep_path is not None and dep_path != file_path:
                    compile_unit(dep_path, mod)

        module, unit_errors = compile_source(
            source, filename=str(file_path), program=program,
        )
        module.dep_modules = [
            program[dep] for dep in module.dependencies if dep in program
        ]
        name = dotted if dotted is not None else file_path.stem
        program[name] = module
        order.append(module)
        errors.extend(unit_errors)
        visiting.discard(file_path)
        compiled[file_path] = name

    compile_unit(Path(path), None)

    # Public function names must be unique across the linked program: every
    # unit's publics become extern C symbols in one artifact. Diagnose at
    # compile time (PP501) rather than surfacing a linker error; a stable
    # module-qualified symbol ABI is future spec work (§9.1).
    seen_public: dict[str, str] = {}
    for module in order:
        for fn in module.functions:
            if fn.name.startswith("_"):
                continue
            if fn.name in seen_public and seen_public[fn.name] != module.name:
                errors.append(TypeError_PP(
                    code="PP501",
                    message=(
                        f"public function `{fn.name}` is defined by both POST "
                        f"modules `{seen_public[fn.name]}` and `{module.name}`; "
                        "public names must be unique across a linked program"
                    ),
                    lineno=0,
                    col_offset=0,
                ))
            else:
                seen_public.setdefault(fn.name, module.name)
    return order, errors


def compile_file(path: str | Path) -> tuple[Module, list]:
    path = Path(path)
    return compile_source(path.read_text(encoding="utf-8"), filename=str(path))
