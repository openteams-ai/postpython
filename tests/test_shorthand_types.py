"""Short-hand dtype spellings (spec §4.1): i32, u16, f64, c128, …

Each short-hand is the same class object as its canonical name, so the
spellings are interchangeable in annotations (compiler path) and at
runtime (interpreted mode) with no special-casing anywhere downstream.
"""

import ast
import ctypes
import shutil

import pytest

import postyp
from postyp import (
    SHORTHAND_DTYPES,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64,
    Complex64, Complex128,
)
from postpyc.build import build_file
from postpyc.compiler.frontend import compile_source
from postpyc.compiler.typechecker import resolve_annotation, resolve_annotation_info

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")

EXPECTED = {
    "i8": Int8, "i16": Int16, "i32": Int32, "i64": Int64,
    "u8": UInt8, "u16": UInt16, "u32": UInt32, "u64": UInt64,
    "f16": Float16, "f32": Float32, "f64": Float64,
    "c64": Complex64, "c128": Complex128,
}


# ---------------------------------------------------------------------------
# postyp: aliases exist, are the canonical classes, and are exported
# ---------------------------------------------------------------------------

def test_shorthands_are_the_canonical_classes():
    for name, canonical in EXPECTED.items():
        assert getattr(postyp, name) is canonical, name


def test_shorthand_table_matches_and_is_exported():
    assert SHORTHAND_DTYPES == EXPECTED
    for name in EXPECTED:
        assert name in postyp.__all__
    assert "SHORTHAND_DTYPES" in postyp.__all__


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,canonical", sorted(EXPECTED.items()))
def test_every_shorthand_resolves_as_annotation(name, canonical):
    node = ast.parse(name, mode="eval").body
    assert resolve_annotation(node) is canonical


def test_attribute_form_resolves():
    node = ast.parse("postyp.f32", mode="eval").body
    assert resolve_annotation(node) is Float32


def test_array_of_shorthand_resolves():
    node = ast.parse("Array[f32, Shape[3]]", mode="eval").body
    info = resolve_annotation_info(node)
    assert info.is_array and info.dtype is Float32
    assert info.shape.dims == (3,)


# ---------------------------------------------------------------------------
# Compilation: shorthand annotations lower identically to canonical ones
# ---------------------------------------------------------------------------

def test_scalar_shorthand_kernel_compiles():
    module, errors = compile_source(
        "from postyp import f64, i32\n"
        "def nth_power(x: f64, n: i32) -> f64:\n"
        "    acc: f64 = 1.0\n"
        "    for _ in range(n):\n"
        "        acc = acc * x\n"
        "    return acc\n"
    )
    assert errors == [], errors
    fn = module.functions[0]
    assert fn.params[0].dtype is Float64
    assert fn.params[1].dtype is Int32
    assert fn.return_dtype is Float64


def test_array_shorthand_kernel_compiles():
    module, errors = compile_source(
        "from postyp import Array, f32\n"
        "def first(a: Array[f32]) -> f32:\n"
        "    return a[0]\n"
    )
    assert errors == [], errors
    fn = module.functions[0]
    assert fn.params[0].is_array and fn.params[0].dtype is Float32


# ---------------------------------------------------------------------------
# Interpreted mode: the ufunc wrappers see the same class objects
# ---------------------------------------------------------------------------

def test_vectorize_interpreted_mode_with_shorthands():
    np = pytest.importorskip("numpy")
    from postpyc import vectorize
    from postyp import f32

    @vectorize
    def halve(x: f32) -> f32:
        return x * 0.5

    out = halve(np.array([2.0, 5.0], dtype=np.float32))
    assert out.dtype == np.float32
    assert out.tolist() == [1.0, 2.5]


# ---------------------------------------------------------------------------
# Native build: end to end through the C backend
# ---------------------------------------------------------------------------

@needs_cc
def test_shorthand_kernel_builds_and_runs(tmp_path):
    src = tmp_path / "short.py"
    src.write_text(
        "from postyp import f64\n"
        "def double_it(x: f64) -> f64:\n"
        "    return x * 2.0\n"
    )
    lib = ctypes.CDLL(str(build_file(src, output=tmp_path / "short.so")))
    fn = lib.pp_double_it
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(3.5) == 7.0
