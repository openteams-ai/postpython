"""Numba-shaped vectorize and guvectorize decorator tests."""

import pytest

from postpyc.compiler.backend.c_backend import emit_module
from postpyc.compiler.frontend import compile_source
from postpyc.compiler.ir import UFunc as UFuncIR
from postpyc import guvectorize, vectorize
from postyp import Array, Float64, Int64


def compile_ok(source: str):
    module, errors = compile_source(source)
    assert errors == []
    return module


def test_vectorize_interpreted_broadcasts_scalar_kernel():
    np = pytest.importorskip("numpy")

    @vectorize
    def add(x: Float64, y: Float64) -> Float64:
        return x + y

    result = add(np.array([1.0, 2.0, 3.0]), 10.0)

    assert np.allclose(result, np.array([11.0, 12.0, 13.0]))
    assert add.__pp_sig__ == "(),()->()"


def test_vectorize_accepts_numba_style_signature_list_and_options():
    np = pytest.importorskip("numpy")

    @vectorize(["float64(float64, float64)"], target="cpu", nopython=True)
    def add(x: Float64, y: Float64) -> Float64:
        return x + y

    assert np.allclose(add(np.array([1.0]), np.array([2.0])), np.array([3.0]))
    assert add.__pp_type_signatures__ == ["float64(float64, float64)"]
    assert add.__pp_options__["target"] == "cpu"


def test_guvectorize_interpreted_uses_output_parameter():
    np = pytest.importorskip("numpy")

    @guvectorize([], "(n),()->(n)")
    def add_scalar(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
        for i in range(len(x)):
            out[i] = x[i] + y

    result = add_scalar(np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0)

    assert np.allclose(result, np.array([[11.0, 12.0], [13.0, 14.0]]))


def test_guvectorize_interpreted_supports_scalar_output_parameter():
    np = pytest.importorskip("numpy")

    @guvectorize([], "(n),()->()")
    def sum_plus(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
        acc: Float64 = 0.0
        for i in range(len(x)):
            acc += x[i] + y
        out[0] = acc

    result = sum_plus(np.array([[1.0, 2.0], [3.0, 4.0]]), 10.0)

    assert np.allclose(result, np.array([23.0, 27.0]))


def test_compiler_lowers_vectorize_as_scalar_ufunc():
    source = """\
from postpyc import vectorize
from postyp import Float64

@vectorize(["float64(float64, float64)"], target="cpu")
def add(x: Float64, y: Float64) -> Float64:
    return x + y
"""

    module = compile_ok(source)
    fn = module.functions[0]
    c_source = emit_module(module)

    assert isinstance(fn, UFuncIR)
    assert str(fn.ufunc_sig) == "(),()->()"
    assert "void add_ufunc_loop(" in c_source


def test_compiler_lowers_guvectorize_numba_form():
    source = """\
from postpyc import guvectorize
from postyp import Array, Float64

@guvectorize([], "(n),()->(n)", target="cpu")
def add_scalar(x: Array[Float64], y: Float64, out: Array[Float64]) -> None:
    for i in range(len(x)):
        out[i] = x[i] + y
"""

    module = compile_ok(source)
    fn = module.functions[0]
    c_source = emit_module(module)

    assert isinstance(fn, UFuncIR)
    assert str(fn.ufunc_sig) == "(n),()->(n)"
    assert "void add_scalar(__pp_array* _x, double _y, __pp_array* _out, int64_t _pp_dim_n)" in c_source


def test_guvectorize_requires_output_parameters():
    source = """\
from postpyc import guvectorize
from postyp import Array, Float64

@guvectorize([], "(n)->()")
def norm(x: Array[Float64]) -> Float64:
    return 0.0
"""

    _, errors = compile_source(source)

    assert [error.code for error in errors] == ["PP100", "PP100"]
    assert "output parameter" in errors[0].message
    assert "return None" in errors[1].message


# ---------------------------------------------------------------------------
# Computed output core dimensions — interpreted mode (issue #38, slice 38b)
# ---------------------------------------------------------------------------

def _pdist_kernel():
    """Condensed pairwise-distance gufunc: (n,d)->(m=n*(n-1)//2)."""
    from postpyc.math import sqrt

    @guvectorize([], "(n,d)->(m=n*(n-1)//2)")
    def pdist(points: Array[Float64], out: Array[Float64]) -> None:
        n: Int64 = len(points)
        k: Int64 = 0
        for i in range(n):
            for j in range(i + 1, n):
                acc: Float64 = 0.0
                diff: Float64 = 0.0
                for c in range(len(points[i])):
                    diff = points[i][c] - points[j][c]
                    acc += diff * diff
                out[k] = sqrt(acc)
                k += 1

    return pdist


def _brute_pdist(points):
    """Reference condensed pdist, upper triangle in row-major order."""
    import math

    out = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(points[i], points[j]))))
    return out


def test_pdist_interpreted_auto_allocates_computed_dim():
    np = pytest.importorskip("numpy")
    pdist = _pdist_kernel()

    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]])
    result = pdist(points)

    # n=4 -> m = 4*3//2 = 6
    assert result.shape == (6,)
    assert np.allclose(result, _brute_pdist(points))


def test_pdist_interpreted_degenerate_sizes_give_empty_output():
    np = pytest.importorskip("numpy")
    pdist = _pdist_kernel()

    # n=1 -> m = 1*0//2 = 0; n=0 (no points, but 2-D shape) -> m = 0.
    assert pdist(np.array([[1.0, 2.0]])).shape == (0,)
    assert pdist(np.empty((0, 2))).shape == (0,)


def test_pdist_interpreted_batches_over_leading_dims():
    np = pytest.importorskip("numpy")
    pdist = _pdist_kernel()

    batch = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    result = pdist(batch)

    # batch (2,) + core (4,3) -> output batch (2,) + core (m=6,)
    assert result.shape == (2, 6)
    assert np.allclose(result[0], _brute_pdist(batch[0]))
    assert np.allclose(result[1], _brute_pdist(batch[1]))


def test_pdist_interpreted_validates_explicit_out():
    np = pytest.importorskip("numpy")
    pdist = _pdist_kernel()

    points = np.array([[0.0, 0.0], [3.0, 4.0]])  # n=2 -> m=1
    out = np.empty(1)
    assert pdist(points, out) is out
    assert np.allclose(out, [5.0])

    with pytest.raises(ValueError, match=r"shape"):
        pdist(points, np.empty(2))  # wrong computed size


def test_convolve_interpreted_matches_numpy():
    np = pytest.importorskip("numpy")

    @guvectorize([], "(n),(m)->(k=n+m-1)")
    def convolve(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
        n: Int64 = len(a)
        m: Int64 = len(b)
        for i in range(n + m - 1):
            acc: Float64 = 0.0
            for j in range(m):
                if 0 <= i - j < n:
                    acc += a[i - j] * b[j]
            out[i] = acc

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 0.5])
    result = convolve(a, b)

    assert result.shape == (5,)  # n+m-1 = 3+3-1
    assert np.allclose(result, np.convolve(a, b))


def test_computed_dim_negative_size_raises_clear_error():
    np = pytest.importorskip("numpy")

    # n - 5 goes negative for small n, so the computed output size is invalid.
    @guvectorize([], "(n)->(m=n-5)")
    def shrink(x: Array[Float64], out: Array[Float64]) -> None:
        for i in range(len(out)):
            out[i] = x[i]

    with pytest.raises(ValueError, match=r"negative size.*n=3"):
        shrink(np.zeros(3))
