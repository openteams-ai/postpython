# Getting Started

## Install

With [pixi](https://pixi.sh/) (recommended — also provides the C
compiler native builds need):

```bash
git clone https://github.com/openteams-ai/postpython.git
cd postpython
pixi install -e dev
pixi run -e dev test        # 360 tests, including end-to-end native builds
```

With pip (bring your own `cc`):

```bash
python -m pip install postpyc       # from PyPI
```

or from a checkout:

```bash
python -m pip install ./postyp-dist -e ".[dev]"
pytest
```

(Distribution, import, and CLI all share the name: `pip install postpyc`,
`import postpyc`, `postpyc build`.)

Both paths install the `postpyc` package, the `postyp` type
vocabulary, and the `post-py` CLI.

## Write a kernel

POST Python is ordinary Python with complete type annotations at function
boundaries (the checker enforces this):

```python
# kernels.py
from postyp import Array, Float64
from postpyc import vectorize, guvectorize
from postpyc.math import exp

HALF: Float64 = 0.5          # module constants fold at compile time

@vectorize
def sigmoid(x: Float64) -> Float64:
    """Logistic sigmoid, numerically stable."""
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    z: Float64 = exp(x)
    return z / (1.0 + z)

@guvectorize([], "(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    """Inner product over the last axis."""
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i] * b[i]
    out[0] = acc
```

Run it interpreted right now — no compiler involved:

```python
>>> from kernels import sigmoid, dot
>>> sigmoid(0.0)
0.5
>>> import numpy as np
>>> dot(np.arange(12.0).reshape(3, 4), np.ones(4))
array([ 6., 22., 38.])
```

## Check conformance

```bash
post-py check kernels.py
```

Violations of the compilable subset are reported with `PP`-prefixed
diagnostic codes defined by the [specification](spec.md).

## Compile it — three ways

**A shared library with a stable C ABI:**

```bash
post-py build kernels.py --emit-header --emit-manifest
```

produces `kernels.dylib`/`.so` (plus `kernels.h` and `kernels.json`)
exporting `pp_sigmoid` and `pp_dot` — callable from C, Rust, Julia, R,
or ctypes:

```python
>>> import ctypes
>>> lib = ctypes.CDLL("./kernels.dylib")
>>> lib.pp_sigmoid.restype = ctypes.c_double
>>> lib.pp_sigmoid(ctypes.c_double(0.0))
0.5
```

**A NumPy ufunc extension module:**

```bash
post-py build kernels.py --ext-module
```

```python
>>> import kernels_native  # the compiled artifact
>>> type(kernels_native.sigmoid)
<class 'numpy.ufunc'>
>>> kernels_native.sigmoid(np.linspace(-4, 4, 9))   # native speed
```

**The package-manager layout** (what a conda/nix recipe calls):

```bash
post-py build mypkg/__init__.py --prefix $PREFIX
# → $PREFIX/lib/libmypkg.so, $PREFIX/include/mypkg.h,
#   $PREFIX/share/post-py/mypkg.json
```

## Multi-module programs

POST imports between your own modules compile as separate translation
units and link together (spec §9.1) — `from mypkg._erf import erfc` in
one module calls the *compiled* `erfc` from the other, never a
same-named libm symbol:

```bash
post-py build mypkg/__init__.py    # compiles the whole package
```

## Learn the language

The [specification](spec.md) is the normative reference: the type
system (§4), the permitted subset (§5), the memory model (§7),
vectorized functions (§8), and the compilation model (§9). For a large
worked example, read
[ppspecial](https://github.com/openteams-ai/ppspecial) — 26 special
functions written entirely in POST Python.
