# PostPython

PostPython is an early reference project for **POST Python**: Performance
Optimized Statically Typed Python.

The goal is to define a clear, portable subset of Python that can be compiled
ahead of time to native code, in the spirit of tools like Numba, Cython, Codon,
Pythran, taichi-lang, and related compiled Python variants. A POST Python source
file is still valid Python, but the language subset, type vocabulary, array ABI,
and vectorized kernel model are specified so multiple compiler implementations
can target the same standard.

The current specification is a draft. See [docs/spec.md](docs/spec.md).

## Project Status

This repository contains:

- A draft language specification for POST Python 0.2.
- A structural checker for the compilable Python subset.
- A typed frontend that lowers Python AST into a small IR.
- A C99 backend that emits native shared-library code.
- A `postyp` type vocabulary for scalar dtypes, arrays, shapes, layouts,
  dataframes, and series.
- Numba-shaped `@vectorize` and `@guvectorize` decorators for NumPy-compatible
  ufunc-style kernels.
- A small `ppspecial` library written in typed POST Python.
- Tests for checker behavior, compiler lowering, array layout/ABI behavior,
  vectorized decorators, and numerical special functions.

PostPython is not production-ready. It is a reference implementation and design
vehicle for the standard.

## Language Sketch

POST Python code uses ordinary Python syntax with explicit type annotations:

```python
from postpython import vectorize
from postyp import Float64
from postpython.math import exp


@vectorize
def gaussian(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    z: Float64 = (x - mu) / sigma
    return exp(-0.5 * z * z) / (sigma * 2.5066282746310002)
```

Generalized vectorized kernels use Numba-style `@guvectorize` with output
parameters:

```python
from postpython import guvectorize
from postyp import Array, Float64


@guvectorize([], "(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64], out: Array[Float64]) -> None:
    result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    out[0] = result
```

## Repository Layout

```text
docs/spec.md              Draft language specification
postyp.py                 Type vocabulary and annotations
postpython/checker.py     Structural subset checker
postpython/compiler/      AST frontend, IR, and C backend
postpython/ufunc.py       @vectorize and @guvectorize runtime wrappers
postpython/build.py       POST Python to C99 to shared-library build helper
postpython/math.py        Typed scalar math wrappers
ppspecial/                Example special-function library
examples/                 Example POST Python source files
tests/                    Reference test suite
```

## Installation

PostPython ships as a regular Python package and can be installed with either
`pip` or [pixi](https://pixi.sh/). Both paths install three importable units:
the `postpython` package, the `postyp` type module, and the `ppspecial`
example library.

A working C compiler (`cc`, `clang`, or `gcc`) is required to compile POST
Python sources to native code. The pixi environment installs one for you;
under pip you need a system compiler.

### With pip

Install the latest release from a local checkout (a future release will be
published to PyPI):

```bash
python -m pip install .
```

For development — including `pytest`, `numpy`, and `narwhals` — install the
`dev` extra in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

### With pixi

`pyproject.toml` contains a `[tool.pixi]` workspace. Pixi resolves
conda-forge dependencies (Python, NumPy, narwhals, a C compiler) and installs
PostPython itself as an editable PyPI package, so any source changes are
picked up immediately.

```bash
pixi install            # default environment
pixi install -e dev     # development environment with pytest etc.
```

Run a defined task:

```bash
pixi run -e dev test                # pytest tests/
pixi run -e dev check FILE.py       # postpython-check on a source file
pixi run -e dev build-example       # python examples/build_shared_lib.py
```

Or drop into a shell with the environment activated:

```bash
pixi shell -e dev
```

## Quick Start

After installing (or with `pixi shell -e dev` active), build one of the
examples to a native shared library:

```bash
python examples/build_shared_lib.py
```

Or call the build helper directly:

```python
from postpython.build import build_file

lib_path = build_file("examples/gaussian.py")
print(lib_path)
```

## Design Highlights

- **Python syntax, static subset:** POST Python files remain `.py` files, but
  unsupported dynamic constructs are rejected by the checker or compiler.
- **Typed native values:** scalar types such as `Float64`, `Int64`, and `Bool`
  are fixed-width native dtypes.
- **Array ABI:** arrays carry shape, byte-stride, layout, and offset metadata so
  C-order, Fortran-order, and strided views can be represented portably.
- **Numba-shaped vectorization:** `@vectorize` defines scalar elementwise
  kernels; `@guvectorize` defines kernels over core dimensions with explicit
  output arrays.
- **Modular standard:** conformance is organized into profiles such as POST
  Core, POST Array, POST DataFrame, POST Ufunc ABI, CPython Extension, and
  Accelerator Extension.
- **Interpreter compatibility:** decorators provide interpreted-mode behavior
  so examples can be run under CPython while the compiler path matures.

## Current Limitations

The implementation is intentionally small and incomplete. Some features
described in the specification are not lowered yet and should produce explicit
unsupported-feature diagnostics rather than being silently accepted. The
reference compiler currently emits C99 and shared libraries; broader executable,
extension-module, dataframe, and accelerator support are still design and
implementation work.

## Contributing Direction

This project is useful as both a language design artifact and a testbed for
compiler behavior. Good contributions include:

- Tightening the specification.
- Adding conformance tests.
- Improving diagnostics for unsupported-but-valid POST Python features.
- Expanding array layout and ABI coverage.
- Building out examples that stress native-code lowering.
- Comparing behavior against existing compiled Python tools.

The most important rule for the reference implementation is simple: reject
unsupported semantics clearly rather than accepting code and changing behavior.
