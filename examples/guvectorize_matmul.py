"""Example: matrix multiply as a POST Python @guvectorize kernel.

Signature: (m,k),(k,n)->(m,n)

The most classic 2-D generalized ufunc shape.  The shared dimension `k` must match between
the two inputs.  Outer dimensions (batch) are broadcast automatically.

This example also demonstrates:
  * Two named core dimensions in each input.
  * A 2-D array output.
  * How the ufunc broadcast loop wraps the inner triple-loop kernel.
"""

from postyp import Array, Float64, Int64
from postpython import guvectorize


@guvectorize([], "(m,k),(k,n)->(m,n)")
def matmul(
    a: Array[Float64],
    b: Array[Float64],
    out: Array[Float64],
) -> None:
    """Compute out = a @ b for 2-D arrays.

    The output array is passed as an argument (write-through) rather than
    returned, which maps directly to the NumPy ufunc protocol where output
    buffers are provided by the caller.
    """
    m: Int64 = len(a)
    k: Int64 = len(b)
    n: Int64 = len(out[0]) if m > 0 else 0  # type: ignore[index]

    for i in range(m):
        for j in range(n):
            acc: Float64 = 0.0
            for p in range(k):
                acc += a[i][p] * b[p][j]  # type: ignore[index]
            out[i][j] = acc  # type: ignore[index]


if __name__ == "__main__":
    try:
        import numpy as np

        A = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
        B = np.array([[5.0, 6.0],
                      [7.0, 8.0]])
        out = np.zeros((2, 2))

        # Call in interpreted mode (inner triple-loop runs in Python).
        matmul(A, B, out)
        print("matmul result:")
        print(out)
        # expected:
        # [[19. 22.]
        #  [43. 50.]]

        # NumPy reference check.
        ref = A @ B
        assert np.allclose(out, ref), f"mismatch:\n{out}\nvs\n{ref}"
        print("matches numpy.matmul ✓")

    except ImportError:
        # Pure-Python fallback demonstration (scalar 2×2 only).
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        out_py = [[0.0, 0.0], [0.0, 0.0]]
        matmul(A, B, out_py)
        print("matmul result:", out_py)
