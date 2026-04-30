"""Example: dot product as a POST Python generalized ufunc.

Signature: (n),(n)->()

This is the simplest non-trivial gufunc: a reduction over a single shared
core dimension.  It demonstrates:

  * The @gufunc decorator with a reduction signature.
  * Typed loop-variable inference (i: Int64 is inferred from range()).
  * Accumulator pattern (result starts at 0.0, updated in place).
  * Broadcasting: dot([[1,2],[3,4]], [[1,0],[0,1]]) → [1.0, 3.0]
"""

from postyp import Array, Float64
from postpython.gufunc import gufunc


@gufunc("(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64]) -> Float64:
    result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


if __name__ == "__main__":
    # Scalar (interpreted) test — works without NumPy.
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    print(f"dot({a}, {b}) = {dot(a, b)}")        # expected: 32.0

    a2 = [1.0, 0.0]
    b2 = [0.0, 1.0]
    print(f"dot({a2}, {b2}) = {dot(a2, b2)}")    # expected: 0.0

    # NumPy batch test (if available).
    try:
        import numpy as np
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])
        print(f"batch dot:\n{dot.as_numpy_gufunc()(A, B)}")  # expected: [2. 3.]
    except ImportError:
        print("(NumPy not installed; skipping batch test)")
