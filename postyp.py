"""postyp — POST Python type library.

Defines the scalar, array, and dataframe types that form the type
vocabulary of POST Python source files.  Import from here in any POST
Python module:

    from postyp import Float64, Array, DataFrame, Shape

Design notes
------------
* Scalar dtypes mirror the array-api standard (data-apis.org) so that
  POST Python's numeric tower is compatible with NumPy, CuPy, JAX, etc.
* Array[DType] / Array[DType, Shape(...)] is the compile-time array type.
  At runtime it is a thin wrapper; the compiler replaces it with native
  memory layouts.
* DataFrame / LazyFrame / Series wrap narwhals types so that dataframe
  code is backend-agnostic (pandas, polars, modin, …).
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar, Generic, Optional, Tuple, TypeVar, Union

# ---------------------------------------------------------------------------
# Sentinel for "no value" at the type level
# ---------------------------------------------------------------------------

_MISSING = object()


# ---------------------------------------------------------------------------
# Dtype base and scalar dtype hierarchy
# (mirrors array-api standard: https://data-apis.org/array-api/latest/API_specification/data_types.html)
# ---------------------------------------------------------------------------

class DType:
    """Abstract base for all POST Python dtypes.

    Subclasses represent concrete scalar types.  They are never
    instantiated — they are used as type parameters only.
    """
    # Compiler-facing metadata
    itemsize: ClassVar[int]       # bytes per element
    signed: ClassVar[bool]        # meaningful for integer types
    kind: ClassVar[str]           # 'i' int, 'u' uint, 'f' float, 'c' complex, 'b' bool, 's' str

    def __class_getitem__(cls, item: Any) -> Any:
        # Allow DType[...] syntax for future extensions
        return cls

    def __init_subclass__(cls, itemsize: int = 0, kind: str = "", signed: bool = True, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        cls.itemsize = itemsize
        cls.kind = kind
        cls.signed = signed


# -- Boolean -----------------------------------------------------------------

class Bool(DType, itemsize=1, kind='b', signed=False):
    """Boolean (True / False), 1 byte."""


# -- Signed integers ---------------------------------------------------------

class Int8(DType, itemsize=1, kind='i', signed=True):
    """Signed 8-bit integer  [-128, 127]."""

class Int16(DType, itemsize=2, kind='i', signed=True):
    """Signed 16-bit integer [-32 768, 32 767]."""

class Int32(DType, itemsize=4, kind='i', signed=True):
    """Signed 32-bit integer [-2³¹, 2³¹-1]."""

class Int64(DType, itemsize=8, kind='i', signed=True):
    """Signed 64-bit integer [-2⁶³, 2⁶³-1]."""


# -- Unsigned integers -------------------------------------------------------

class UInt8(DType, itemsize=1, kind='u', signed=False):
    """Unsigned 8-bit integer [0, 255]."""

class UInt16(DType, itemsize=2, kind='u', signed=False):
    """Unsigned 16-bit integer [0, 65 535]."""

class UInt32(DType, itemsize=4, kind='u', signed=False):
    """Unsigned 32-bit integer [0, 2³²-1]."""

class UInt64(DType, itemsize=8, kind='u', signed=False):
    """Unsigned 64-bit integer [0, 2⁶⁴-1]."""


# -- Floating point ----------------------------------------------------------

class Float16(DType, itemsize=2, kind='f', signed=True):
    """IEEE 754 half-precision float (binary16)."""

class Float32(DType, itemsize=4, kind='f', signed=True):
    """IEEE 754 single-precision float (binary32)."""

class Float64(DType, itemsize=8, kind='f', signed=True):
    """IEEE 754 double-precision float (binary64)."""


# -- Complex -----------------------------------------------------------------

class Complex64(DType, itemsize=8, kind='c', signed=True):
    """Complex number: two Float32 components (real, imag)."""

class Complex128(DType, itemsize=16, kind='c', signed=True):
    """Complex number: two Float64 components (real, imag)."""


# -- Text / bytes ------------------------------------------------------------

class Str(DType, itemsize=0, kind='s', signed=False):
    """Variable-length UTF-8 string.

    Note: in compiled POST Python, string values are immutable and
    passed by reference.  itemsize=0 signals variable-width.
    """

class Bytes(DType, itemsize=0, kind='s', signed=False):
    """Variable-length byte sequence."""


# ---------------------------------------------------------------------------
# Convenience aliases — preferred defaults
# ---------------------------------------------------------------------------

#: Default integer type (64-bit signed, matches Python int semantics).
Int = Int64

#: Default floating-point type.
Float = Float64

#: Default complex type.
Complex = Complex128

#: Default boolean alias (mirrors Python's bool).
# Bool is already defined above.


# ---------------------------------------------------------------------------
# All public dtype names (for introspection / checker use)
# ---------------------------------------------------------------------------

SCALAR_DTYPES: tuple[type[DType], ...] = (
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128,
    Str, Bytes,
)


# ---------------------------------------------------------------------------
# Shape type
# ---------------------------------------------------------------------------

class Shape:
    """Compile-time shape descriptor for Array.

    Examples::

        Array[Float64, Shape[10]]           # 1-D, length 10
        Array[Float64, Shape[3, 3]]         # 2-D, 3×3
        Array[Float64, Shape[None, 128]]    # 2-D, dynamic first dim
        Array[Float64, Shape[...]]          # any rank / any shape

    None in a dimension means "dynamic size" (not known at compile time).
    Ellipsis (...) means fully dynamic rank.
    """

    dims: tuple[int | None, ...]

    def __init__(self, *dims: int | None) -> None:
        self.dims = dims

    def __class_getitem__(cls, item: Any) -> "Shape":
        if item is Ellipsis:
            return cls()  # fully dynamic
        if isinstance(item, tuple):
            return cls(*item)
        return cls(item)

    @property
    def ndim(self) -> int | None:
        """Number of dimensions, or None if fully dynamic."""
        return len(self.dims) if self.dims else None

    def __repr__(self) -> str:
        if not self.dims:
            return "Shape[...]"
        return f"Shape[{', '.join('?' if d is None else str(d) for d in self.dims)}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Shape) and self.dims == other.dims

    def __hash__(self) -> int:
        return hash(self.dims)


#: Fully-dynamic shape sentinel — use when rank and sizes are unknown.
AnyShape = Shape()


# ---------------------------------------------------------------------------
# Array type
# ---------------------------------------------------------------------------

DT = TypeVar("DT", bound=DType)


class Array(Generic[DT]):
    """POST Python array type (array-api compatible).

    Use as a type annotation; never instantiate directly.

    Examples::

        def scale(a: Array[Float64], factor: Float64) -> Array[Float64]: ...

        # With shape constraints:
        def dot(a: Array[Float64, Shape[3]], b: Array[Float64, Shape[3]]) -> Float64: ...

    The compiler maps Array[DType] to native memory layouts (e.g.
    contiguous row-major buffers) and lowers operations to BLAS / SIMD
    intrinsics as appropriate.

    Runtime behaviour (when running under the standard interpreter) is
    provided by the array-api-compat layer over whatever backend is active.
    """

    dtype: ClassVar[type[DType]]
    shape: ClassVar[Shape]

    def __class_getitem__(cls, params: Any) -> type["Array[Any]"]:
        if not isinstance(params, tuple):
            params = (params,)

        if len(params) == 1:
            dtype_param, shape_param = params[0], AnyShape
        elif len(params) == 2:
            dtype_param, shape_param = params
            if not isinstance(shape_param, Shape):
                raise TypeError(
                    f"Array second parameter must be a Shape, got {type(shape_param).__name__!r}"
                )
        else:
            raise TypeError(f"Array takes 1 or 2 type parameters, got {len(params)}")

        if not (isinstance(dtype_param, type) and issubclass(dtype_param, DType)):
            raise TypeError(
                f"Array first parameter must be a DType subclass, got {dtype_param!r}"
            )

        ns = {"dtype": dtype_param, "shape": shape_param, "__orig_class__": cls}
        return type(
            f"Array[{dtype_param.__name__}{',' + repr(shape_param) if shape_param is not AnyShape else ''}]",
            (cls,),
            ns,
        )


# -- Convenience array aliases -----------------------------------------------

BoolArray    = Array[Bool]
Int8Array    = Array[Int8]
Int16Array   = Array[Int16]
Int32Array   = Array[Int32]
Int64Array   = Array[Int64]
UInt8Array   = Array[UInt8]
UInt16Array  = Array[UInt16]
UInt32Array  = Array[UInt32]
UInt64Array  = Array[UInt64]
Float16Array = Array[Float16]
Float32Array = Array[Float32]
Float64Array = Array[Float64]
Complex64Array  = Array[Complex64]
Complex128Array = Array[Complex128]

#: Most common aliases
IntArray   = Int64Array
FloatArray = Float64Array


# ---------------------------------------------------------------------------
# DataFrame / Series types (narwhals-based)
# ---------------------------------------------------------------------------

try:
    import narwhals as nw
    _HAS_NARWHALS = True
except ModuleNotFoundError:
    _HAS_NARWHALS = False

#: Schema is a mapping of column name → DType subclass.
Schema = dict[str, type[DType]]

_NARWHALS_TO_POSTYP: "dict[Any, type[DType]]"
_POSTYP_TO_NARWHALS: "dict[type[DType], Any]"

if _HAS_NARWHALS:
    import narwhals as nw

    _NARWHALS_TO_POSTYP = {
        nw.Boolean:    Bool,
        nw.Int8:       Int8,
        nw.Int16:      Int16,
        nw.Int32:      Int32,
        nw.Int64:      Int64,
        nw.UInt8:      UInt8,
        nw.UInt16:     UInt16,
        nw.UInt32:     UInt32,
        nw.UInt64:     UInt64,
        nw.Float32:    Float32,
        nw.Float64:    Float64,
        nw.String:     Str,
    }
    _POSTYP_TO_NARWHALS = {v: k for k, v in _NARWHALS_TO_POSTYP.items()}


def narwhals_dtype_to_postyp(nw_dtype: Any) -> type[DType]:
    """Convert a narwhals dtype to the equivalent postyp DType."""
    if not _HAS_NARWHALS:
        raise ImportError("narwhals is not installed")
    result = _NARWHALS_TO_POSTYP.get(type(nw_dtype))
    if result is None:
        raise TypeError(f"No postyp equivalent for narwhals dtype {nw_dtype!r}")
    return result


def postyp_dtype_to_narwhals(dtype: type[DType]) -> Any:
    """Convert a postyp DType to the equivalent narwhals dtype."""
    if not _HAS_NARWHALS:
        raise ImportError("narwhals is not installed")
    result = _POSTYP_TO_NARWHALS.get(dtype)
    if result is None:
        raise TypeError(f"No narwhals equivalent for postyp dtype {dtype.__name__!r}")
    return result


class DataFrame:
    """POST Python DataFrame type annotation.

    Backend-agnostic; backed by narwhals at runtime.

    Examples::

        def process(df: DataFrame) -> DataFrame: ...

        # With schema (column→dtype mapping):
        MyFrame = DataFrame.with_schema({"x": Float64, "y": Float64, "label": Int32})
        def cluster(df: MyFrame) -> MyFrame: ...
    """

    schema: ClassVar[Optional[Schema]] = None

    @classmethod
    def with_schema(cls, schema: Schema) -> type["DataFrame"]:
        """Return a DataFrame subtype bound to a specific column schema."""
        name = "DataFrame[" + ", ".join(f"{k}:{v.__name__}" for k, v in schema.items()) + "]"
        return type(name, (cls,), {"schema": schema})

    @classmethod
    def from_narwhals(cls, nw_df: Any) -> "DataFrame":
        """Wrap a narwhals DataFrame for use in POST Python code."""
        if not _HAS_NARWHALS:
            raise ImportError("narwhals is not installed")
        obj = cls.__new__(cls)
        object.__setattr__(obj, "_nw_df", nw_df)
        return obj

    def to_narwhals(self) -> Any:
        if not _HAS_NARWHALS:
            raise ImportError("narwhals is not installed")
        return object.__getattribute__(self, "_nw_df")


class LazyFrame:
    """POST Python LazyFrame type annotation (deferred computation).

    Backed by narwhals LazyFrame at runtime.
    """

    schema: ClassVar[Optional[Schema]] = None

    @classmethod
    def with_schema(cls, schema: Schema) -> type["LazyFrame"]:
        name = "LazyFrame[" + ", ".join(f"{k}:{v.__name__}" for k, v in schema.items()) + "]"
        return type(name, (cls,), {"schema": schema})


class Series:
    """POST Python Series type annotation — a single typed column.

    Examples::

        def normalize(s: Series[Float64]) -> Series[Float64]: ...
    """

    dtype: ClassVar[Optional[type[DType]]] = None

    def __class_getitem__(cls, dtype: type[DType]) -> type["Series"]:
        if not (isinstance(dtype, type) and issubclass(dtype, DType)):
            raise TypeError(f"Series parameter must be a DType, got {dtype!r}")
        return type(f"Series[{dtype.__name__}]", (cls,), {"dtype": dtype})


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    # DType base
    "DType",
    # Scalar dtypes
    "Bool",
    "Int8", "Int16", "Int32", "Int64",
    "UInt8", "UInt16", "UInt32", "UInt64",
    "Float16", "Float32", "Float64",
    "Complex64", "Complex128",
    "Str", "Bytes",
    # Aliases
    "Int", "Float", "Complex",
    # All scalar dtypes
    "SCALAR_DTYPES",
    # Shape
    "Shape", "AnyShape",
    # Array
    "Array",
    "BoolArray", "IntArray", "FloatArray",
    "Int8Array", "Int16Array", "Int32Array", "Int64Array",
    "UInt8Array", "UInt16Array", "UInt32Array", "UInt64Array",
    "Float16Array", "Float32Array", "Float64Array",
    "Complex64Array", "Complex128Array",
    # DataFrame types
    "Schema",
    "DataFrame", "LazyFrame", "Series",
    # Narwhals bridge
    "narwhals_dtype_to_postyp",
    "postyp_dtype_to_narwhals",
]
