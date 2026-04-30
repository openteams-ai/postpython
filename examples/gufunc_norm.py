"""Example: L2 norm as a POST Python generalized ufunc.

Signature: (n)->()

Demonstrates a reduction ufunc with a single array input and scalar output.
The norm broadcasts naturally over batches: norm([[3,4],[5,12]]) → [5., 13.]
"""

from postyp import Array, Float64
from postpython.gufunc import gufunc


@gufunc("(n)->()")
def norm(a: Array[Float64]) -> Float64:
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * a[i]
    return acc ** 0.5


@gufunc("(n)->(n)")
def normalize(a: Array[Float64]) -> Array[Float64]:
    """Return a unit vector in the direction of *a*."""
    n: Float64 = norm(a)
    result: Array[Float64] = [0.0] * len(a)  # type: ignore[assignment]
    for i in range(len(a)):
        result[i] = a[i] / n
    return result


if __name__ == "__main__":
    v = [3.0, 4.0]
    print(f"norm({v}) = {norm(v)}")   # expected: 5.0

    v2 = [5.0, 12.0]
    print(f"norm({v2}) = {norm(v2)}")  # expected: 13.0

    try:
        import numpy as np
        batch = np.array([[3.0, 4.0], [5.0, 12.0]])
        print(f"batch norm:\n{norm.as_numpy_gufunc()(batch)}")  # expected: [ 5. 13.]
    except ImportError:
        print("(NumPy not installed; skipping batch test)")
