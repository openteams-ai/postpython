"""POST Python Intermediate Representation.

A typed, SSA-inspired three-address IR.  The design is intentionally
simple: basic blocks, typed values, explicit control flow.  Later passes
(optimisation, backend lowering) operate on this structure.

Terminology
-----------
* Module   — a translation unit (one .py file).
* Function — a typed function definition with a list of BasicBlocks.
* UFunc   — a Function annotated with a ufunc layout signature.
* BasicBlock — a straight-line sequence of Instructions ending in a Terminator.
* Value    — a named, typed SSA value.
* Instruction — a single operation that produces at most one Value.
* Terminator — the last instruction in a BasicBlock (Return, Branch, CondBranch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union

# sys.path setup happens once in postpython/__init__.py.
import postpython  # noqa: F401  -- ensure path setup runs
from postyp import DType, AnyShape, Shape, ArrayLayout, COrder


# ---------------------------------------------------------------------------
# Typed values
# ---------------------------------------------------------------------------

@dataclass
class Value:
    """An SSA value: a name and a type."""
    name: str
    dtype: type[DType]
    # For array values, shape carries dimension info.
    shape: Shape = field(default_factory=lambda: AnyShape)
    layout: ArrayLayout = COrder
    is_array: bool = False
    is_output: bool = False

    def __repr__(self) -> str:
        shape_str = f", {self.shape}" if self.shape is not AnyShape else ""
        layout_str = f", {self.layout}" if self.layout != COrder else ""
        array_str = "[]" if self.is_array else ""
        output_str = "&" if self.is_output else ""
        return f"%{self.name}: {output_str}{self.dtype.__name__}{array_str}{shape_str}{layout_str}"


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

class BinOp(Enum):
    ADD  = "+"
    SUB  = "-"
    MUL  = "*"
    DIV  = "/"
    FDIV = "//"
    MOD  = "%"
    POW  = "**"
    EQ   = "=="
    NE   = "!="
    LT   = "<"
    LE   = "<="
    GT   = ">"
    GE   = ">="
    AND  = "and"
    OR   = "or"


class UnaryOp(Enum):
    NEG  = "-"
    NOT  = "not"
    ABS  = "abs"


@dataclass
class Const:
    """result = literal_value"""
    result: Value
    value: Union[int, float, bool, complex, str, bytes]


@dataclass
class BinOpInstr:
    """result = left op right"""
    result: Value
    op: BinOp
    left: Value
    right: Value


@dataclass
class UnaryOpInstr:
    """result = op operand"""
    result: Value
    op: UnaryOp
    operand: Value


@dataclass
class ArrayLoad:
    """result = array[index]  — index is a byte offset from the view origin"""
    result: Value
    array: Value
    index: Value


@dataclass
class ArrayStore:
    """array[index] = value  — index is a byte offset from the view origin"""
    array: Value
    index: Value
    value: Value


@dataclass
class ArrayDim:
    """result = array.shape[axis]"""
    result: Value
    array: Value
    axis: int


@dataclass
class ArrayStride:
    """result = array.strides[axis] in bytes"""
    result: Value
    array: Value
    axis: int


@dataclass
class Call:
    """result = func(args...)  — typed function call"""
    result: Optional[Value]   # None for void calls
    func: str                 # function name (resolved at link time)
    args: list[Value]


@dataclass
class Cast:
    """result = (dtype) operand  — numeric cast"""
    result: Value
    operand: Value


@dataclass
class AssignValue:
    """target = value, optionally declaring target first."""
    target: Value
    value: Value
    declare: bool = False


@dataclass
class Select:
    """result = cond ? if_true : if_false"""
    result: Value
    cond: Value
    if_true: Value
    if_false: Value


@dataclass
class Alloc:
    """result = alloc(dtype, length)  — allocate an array on the POST Python heap"""
    result: Value
    length: Value             # number of elements


@dataclass
class GetField:
    """result = aggregate.field_name"""
    result: Value
    aggregate: Value
    field_name: str


@dataclass
class SetField:
    """aggregate.field_name = value"""
    aggregate: Value
    field_name: str
    value: Value


# Union of all non-terminator instruction types.
Instruction = Union[
    Const, BinOpInstr, UnaryOpInstr,
    ArrayLoad, ArrayStore, ArrayDim, ArrayStride,
    Call, Cast, AssignValue, Select, Alloc,
    GetField, SetField,
]


# ---------------------------------------------------------------------------
# Terminators
# ---------------------------------------------------------------------------

@dataclass
class Return:
    """return value  (or return None for void)"""
    value: Optional[Value]


@dataclass
class Branch:
    """Unconditional jump to target."""
    target: str   # label of the target BasicBlock


@dataclass
class CondBranch:
    """if cond goto true_target else false_target"""
    cond: Value
    true_target: str
    false_target: str


Terminator = Union[Return, Branch, CondBranch]


# ---------------------------------------------------------------------------
# Basic block
# ---------------------------------------------------------------------------

@dataclass
class BasicBlock:
    label: str
    instructions: list[Instruction] = field(default_factory=list)
    terminator: Optional[Terminator] = None   # set after construction

    def append(self, instr: Instruction) -> None:
        self.instructions.append(instr)

    def terminate(self, term: Terminator) -> None:
        assert self.terminator is None, f"Block {self.label!r} already has a terminator"
        self.terminator = term


# ---------------------------------------------------------------------------
# Function and UFunc
# ---------------------------------------------------------------------------

@dataclass
class Param:
    """A typed function parameter."""
    name: str
    dtype: type[DType]
    shape: Shape = field(default_factory=lambda: AnyShape)
    layout: ArrayLayout = COrder
    is_array: bool = False
    is_output: bool = False


@dataclass
class Function:
    """A typed POST Python function."""
    name: str
    params: list[Param]
    return_dtype: Optional[type[DType]]   # None → void
    return_shape: Shape = field(default_factory=lambda: AnyShape)
    core_dim_params: list[Param] = field(default_factory=list)
    blocks: list[BasicBlock] = field(default_factory=list)

    @property
    def entry(self) -> BasicBlock:
        return self.blocks[0]

    def new_block(self, label: str) -> BasicBlock:
        bb = BasicBlock(label)
        self.blocks.append(bb)
        return bb


@dataclass
class UFuncSignature:
    """Parsed ufunc layout signature, e.g. '(m,k),(k,n)->(m,n)'."""
    inputs: list[list[str]]   # list of core-dim name lists, one per input
    outputs: list[list[str]]  # list of core-dim name lists, one per output

    @property
    def core_dims(self) -> list[str]:
        """All unique named core dimensions in declaration order."""
        seen: dict[str, None] = {}
        for dims in self.inputs + self.outputs:
            for d in dims:
                seen[d] = None
        return list(seen)

    def __str__(self) -> str:
        def fmt(groups: list[list[str]]) -> str:
            return ",".join("(" + ",".join(g) + ")" for g in groups)
        return fmt(self.inputs) + "->" + fmt(self.outputs)


@dataclass
class UFunc(Function):
    """A vectorized function with a broadcast layout signature."""
    ufunc_sig: Optional[UFuncSignature] = None


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

@dataclass
class Module:
    """A POST Python translation unit."""
    name: str                               # typically the source filename stem
    functions: list[Function] = field(default_factory=list)
    # Imports visible to this module (name → resolved module path).
    imports: dict[str, str] = field(default_factory=dict)

    def add_function(self, fn: Function) -> None:
        self.functions.append(fn)

    def get_function(self, name: str) -> Optional[Function]:
        for fn in self.functions:
            if fn.name == name:
                return fn
        return None

    @property
    def ufuncs(self) -> list[UFunc]:
        return [f for f in self.functions if isinstance(f, UFunc)]
