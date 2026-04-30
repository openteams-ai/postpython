"""Ternary and type-II ufuncs — signatures (),(),()->() and (),()->(),()

These cover the less common but important ufunc shapes:

  * Three-input scalar ufuncs (e.g. fma, clip, lerp, where_select).
  * Two-output scalar ufuncs (e.g. modf — integral and fractional parts,
    frexp — mantissa and exponent).

NumPy supports multiple outputs via the `out` tuple argument; the POST
Python gufunc compiler handles this by treating trailing output args as
write-through pointer parameters in the emitted C.
"""

from postyp import Float64, Int64, Bool
from postpython.gufunc import gufunc


# ---------------------------------------------------------------------------
# Three-input, one-output: (),(),()->()
# ---------------------------------------------------------------------------

@gufunc("(),(),()->()")
def fma(a: Float64, b: Float64, c: Float64) -> Float64:
    """Fused multiply-add: a*b + c

    In compiled output this lowers to the hardware FMA instruction
    (via C99 fma()) — no intermediate rounding.  Critical for BLAS
    and ML weight-update kernels.
    """
    return a * b + c


@gufunc("(),(),()->()")
def clip(x: Float64, lo: Float64, hi: Float64) -> Float64:
    """Clamp x to [lo, hi]  (mirrors np.clip element-wise)."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@gufunc("(),(),()->()")
def lerp(a: Float64, b: Float64, t: Float64) -> Float64:
    """Linear interpolation: a + t*(b - a), t ∈ [0, 1]."""
    return a + t * (b - a)


@gufunc("(),(),()->()")
def where_select(condition: Bool, x: Float64, y: Float64) -> Float64:
    """Element-wise conditional select: condition ? x : y

    Mirrors np.where(condition, x, y) as a ufunc.
    """
    if condition:
        return x
    return y


@gufunc("(),(),()->()")
def muladd_clamp(x: Float64, scale: Float64, bias: Float64) -> Float64:
    """(x * scale + bias) clamped to [0, 1].

    Common in quantization and normalization layers.
    """
    result: Float64 = x * scale + bias
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


@gufunc("(),(),()->()")
def smooth_step(edge0: Float64, edge1: Float64, x: Float64) -> Float64:
    """Hermite interpolation between edge0 and edge1 (GLSL smoothstep).

    Returns 0 for x ≤ edge0, 1 for x ≥ edge1, smooth cubic in between.
    """
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t: Float64 = (x - edge0) / (edge1 - edge0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Two-output ufuncs — signature ()->(),()
# ---------------------------------------------------------------------------

@gufunc("()->(),() ")
def modf(x: Float64, int_part: Float64, frac_part: Float64) -> None:
    """Split x into its integer and fractional parts (mirrors np.modf).

    Both parts carry the sign of x.
    int_part and frac_part are write-through output arguments.
    """
    if x >= 0.0:
        ipart: Float64 = x - (x % 1.0)
    else:
        ipart = x - (x % 1.0)
        if x % 1.0 != 0.0:
            ipart = ipart - 1.0
    int_part = ipart
    frac_part = x - ipart


@gufunc("()->(),() ")
def frexp(x: Float64, mantissa: Float64, exponent: Int64) -> None:
    """Decompose x = mantissa * 2**exponent, mantissa ∈ [0.5, 1.0).

    Mirrors np.frexp.  In compiled output this lowers to frexp() from
    libm.
    """
    if x == 0.0:
        mantissa = 0.0
        exponent = 0
        return
    # Iterative extraction (reference; compiler replaces with frexp()).
    m: Float64 = x
    e: Int64 = 0
    while m >= 1.0:
        m = m * 0.5
        e = e + 1
    while m < 0.5:
        m = m * 2.0
        e = e - 1
    mantissa = m
    exponent = e


# ---------------------------------------------------------------------------
# Self-test (interpreted mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EPS: Float64 = 1e-9

    def close(a: Float64, b: Float64) -> Bool:
        diff: Float64 = a - b
        if diff < 0.0:
            diff = -diff
        return diff < EPS

    cases = [
        ("fma(2,3,4)",           fma(2.0, 3.0, 4.0),            10.0),
        ("clip(5, 0, 3)",        clip(5.0, 0.0, 3.0),             3.0),
        ("clip(-1, 0, 3)",       clip(-1.0, 0.0, 3.0),            0.0),
        ("clip(1.5, 0, 3)",      clip(1.5, 0.0, 3.0),             1.5),
        ("lerp(0,10,0.3)",       lerp(0.0, 10.0, 0.3),            3.0),
        ("lerp(0,10,1.0)",       lerp(0.0, 10.0, 1.0),           10.0),
        ("where_select(T,1,2)",  where_select(True,  1.0, 2.0),   1.0),
        ("where_select(F,1,2)",  where_select(False, 1.0, 2.0),   2.0),
        ("smooth_step(0,1,0.5)", smooth_step(0.0, 1.0, 0.5),      0.5),
        ("smooth_step(0,1,0.0)", smooth_step(0.0, 1.0, 0.0),      0.0),
        ("smooth_step(0,1,1.0)", smooth_step(0.0, 1.0, 1.0),      1.0),
        ("muladd_clamp(2,1,0)",  muladd_clamp(2.0, 1.0, 0.0),     1.0),
    ]

    all_ok: Bool = True
    for label, got, expected in cases:
        ok: Bool = close(got, expected)
        status: str = "ok  " if ok else "FAIL"
        print(f"  {status}  {label} = {got}  (expected {expected})")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All ternary ufunc tests passed.")
    else:
        print("Some tests FAILED.")
