"""Example: L2 norm as a POST Python @guvectorize kernel.

Signature: (n)->()

Demonstrates a reduction ufunc with a single array input and scalar output.
The norm broadcasts naturally over batches: norm([[3,4],[5,12]]) → [5., 13.]
"""

from postyp import Array, Float64
from postpython import guvectorize


@guvectorize([], "(n)->()")
def norm(a: Array[Float64], out: Array[Float64]) -> None:
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * a[i]
    out[0] = acc ** 0.5


@guvectorize([], "(n)->(n)")
def normalize(a: Array[Float64], out: Array[Float64]) -> None:
    """Return a unit vector in the direction of *a*."""
    norm_out: Array[Float64] = [0.0]  # type: ignore[assignment]
    norm(a, norm_out)
    n: Float64 = norm_out[0]
    for i in range(len(a)):
        out[i] = a[i] / n


if __name__ == "__main__":
    v = [3.0, 4.0]
    print(f"norm({v}) = {norm(v)}")   # expected: 5.0

    v2 = [5.0, 12.0]
    print(f"norm({v2}) = {norm(v2)}")  # expected: 13.0

    try:
        import numpy as np
        batch = np.array([[3.0, 4.0], [5.0, 12.0]])
        print(f"batch norm:\n{norm.as_numpy_ufunc()(batch)}")  # expected: [ 5. 13.]
    except ImportError:
        print("(NumPy not installed; skipping batch test)")
