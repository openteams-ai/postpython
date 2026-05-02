"""Tests for postyp — POST Python type library."""

import pytest
from postyp import (
    Bool,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128,
    Str, Bytes,
    Int, Float, Complex,
    DType, SCALAR_DTYPES,
    Shape, AnyShape,
    COrder, FOrder, Strides,
    Array, FloatArray, Float64Array, IntArray,
    DataFrame, LazyFrame, Series,
)


# ---------------------------------------------------------------------------
# DType metadata
# ---------------------------------------------------------------------------

class TestDTypeMetadata:
    def test_bool_metadata(self):
        assert Bool.itemsize == 1
        assert Bool.kind == 'b'
        assert Bool.signed is False

    def test_int8_metadata(self):
        assert Int8.itemsize == 1
        assert Int8.kind == 'i'
        assert Int8.signed is True

    def test_uint64_metadata(self):
        assert UInt64.itemsize == 8
        assert UInt64.signed is False

    def test_float32_metadata(self):
        assert Float32.itemsize == 4
        assert Float32.kind == 'f'

    def test_float64_metadata(self):
        assert Float64.itemsize == 8

    def test_complex64_two_float32(self):
        # complex64 = 2 × float32 = 8 bytes
        assert Complex64.itemsize == 8

    def test_complex128_two_float64(self):
        assert Complex128.itemsize == 16

    def test_str_variable_width(self):
        assert Str.itemsize == 0

    def test_all_dtypes_are_dtype_subclasses(self):
        for dt in SCALAR_DTYPES:
            assert issubclass(dt, DType)

    def test_scalar_dtypes_count(self):
        # 1 bool + 4 signed + 4 unsigned + 3 float + 2 complex + 2 text = 16
        assert len(SCALAR_DTYPES) == 16


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------

class TestAliases:
    def test_int_is_int64(self):
        assert Int is Int64

    def test_float_is_float64(self):
        assert Float is Float64

    def test_complex_is_complex128(self):
        assert Complex is Complex128


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

class TestShape:
    def test_1d_shape(self):
        s = Shape[10]
        assert s.dims == (10,)
        assert s.ndim == 1

    def test_2d_shape(self):
        s = Shape[3, 4]
        assert s.dims == (3, 4)
        assert s.ndim == 2

    def test_dynamic_dim(self):
        s = Shape[None, 128]
        assert s.dims == (None, 128)

    def test_fully_dynamic(self):
        s = Shape[...]
        assert s.ndim is None

    def test_any_shape_sentinel(self):
        assert AnyShape.ndim is None

    def test_equality(self):
        assert Shape[3, 3] == Shape[3, 3]
        assert Shape[3, 3] != Shape[3, 4]

    def test_repr_1d(self):
        assert "10" in repr(Shape[10])

    def test_repr_dynamic(self):
        assert "..." in repr(Shape[...])


# ---------------------------------------------------------------------------
# Array parametrization
# ---------------------------------------------------------------------------

class TestArray:
    def test_array_dtype_param(self):
        A = Array[Float64]
        assert A.dtype is Float64
        assert A.layout == COrder

    def test_array_with_shape(self):
        A = Array[Float64, Shape[3, 3]]
        assert A.dtype is Float64
        assert A.shape == Shape[3, 3]
        assert A.layout == COrder

    def test_array_no_shape_has_any_shape(self):
        A = Array[Float64]
        assert A.shape == AnyShape

    def test_array_with_fortran_order(self):
        A = Array[Float64, Shape[3, 3], FOrder]
        assert A.dtype is Float64
        assert A.shape == Shape[3, 3]
        assert A.layout == FOrder

    def test_array_with_layout_without_shape(self):
        A = Array[Float64, FOrder]
        assert A.shape == AnyShape
        assert A.layout == FOrder

    def test_array_with_explicit_strides(self):
        strides = Strides[None, 8]
        A = Array[Float64, Shape[None, None], strides]
        assert strides.strides == (None, 8)
        assert A.layout == strides

    def test_convenience_alias(self):
        assert FloatArray.dtype is Float64
        assert Float64Array.dtype is Float64
        assert IntArray.dtype is Int64

    def test_bad_dtype_raises(self):
        with pytest.raises(TypeError):
            Array[int]  # plain Python int, not a DType

    def test_bad_shape_raises(self):
        with pytest.raises(TypeError):
            Array[Float64, (3, 3)]  # tuple, not a Shape

    def test_bad_layout_raises(self):
        with pytest.raises(TypeError):
            Array[Float64, Shape[3, 3], "C"]

    def test_stride_rank_must_match_shape_rank(self):
        with pytest.raises(TypeError):
            Array[Float64, Shape[3, 3], Strides[1]]

    def test_bad_stride_value_raises(self):
        with pytest.raises(TypeError):
            Strides["x"]


# ---------------------------------------------------------------------------
# DataFrame / LazyFrame / Series
# ---------------------------------------------------------------------------

class TestDataFrame:
    def test_with_schema(self):
        MyFrame = DataFrame.with_schema({"x": Float64, "y": Float64})
        assert MyFrame.schema == {"x": Float64, "y": Float64}
        assert issubclass(MyFrame, DataFrame)

    def test_schema_name_contains_columns(self):
        MyFrame = DataFrame.with_schema({"a": Int32})
        assert "a" in MyFrame.__name__
        assert "Int32" in MyFrame.__name__

    def test_lazyframe_with_schema(self):
        LF = LazyFrame.with_schema({"v": Float32})
        assert issubclass(LF, LazyFrame)
        assert LF.schema == {"v": Float32}


class TestSeries:
    def test_series_dtype(self):
        S = Series[Float64]
        assert S.dtype is Float64
        assert issubclass(S, Series)

    def test_series_bad_dtype_raises(self):
        with pytest.raises(TypeError):
            Series[float]

    def test_series_name(self):
        S = Series[Int32]
        assert "Int32" in S.__name__


# ---------------------------------------------------------------------------
# Narwhals bridge (skipped if narwhals not installed)
# ---------------------------------------------------------------------------

narwhals_available = pytest.importorskip  # marker below

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("narwhals"),
    reason="narwhals not installed",
)
class TestNarwhalsBridge:
    def test_round_trip_float64(self):
        import narwhals as nw
        from postyp import narwhals_dtype_to_postyp, postyp_dtype_to_narwhals
        assert narwhals_dtype_to_postyp(nw.Float64()) is Float64
        assert postyp_dtype_to_narwhals(Float64) == nw.Float64

    def test_unknown_narwhals_dtype_raises(self):
        import narwhals as nw
        from postyp import narwhals_dtype_to_postyp
        with pytest.raises(TypeError):
            narwhals_dtype_to_postyp(nw.Date())
