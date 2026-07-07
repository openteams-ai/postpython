# POST Python

**Performance Optimized Statically Typed Python** — a defined, compilable
subset of Python with a normative specification and a reference
ahead-of-time compiler.

A POST Python source file is valid Python. It runs unmodified under the
standard CPython interpreter — and a conforming compiler translates the
same file to native code with no Python runtime in the binary.

```python
from postyp import Float64
from postpyc import vectorize
from postpyc.math import exp

@vectorize
def gaussian(x: Float64, mu: Float64, sigma: Float64) -> Float64:
    """Normal probability density."""
    z: Float64 = (x - mu) / sigma
    return exp(-0.5 * z * z) / (sigma * 2.5066282746310002)
```

That one definition is, today:

- **an interpreted Python function** — callable immediately, NumPy
  broadcasting included;
- **a native C kernel** — `postpyc build` emits C99, compiles each
  module as its own translation unit, and links a shared library with a
  [stable C ABI](toolchain.md): `pp_gaussian` callable from C, Rust,
  Julia, R, or ctypes;
- **a real `numpy.ufunc`** — `postpyc build --ext-module` produces an
  importable CPython extension with full broadcasting, `out=`, dtype
  handling, and the original docstring.

One code base. One artifact per audience. No vendored binaries.

## Why a standard, not just a compiler

Python has many compilation projects — Cython, mypyc, Numba, Codon,
Pythran, taichi — each defining its own informal subset. POST Python
inverts that: the [specification](spec.md) is normative, organized into
conformance profiles (POST Core, POST Array, POST Ufunc ABI, CPython
Extension, …), and the compiler in this repository is a *reference
implementation*, not the definition. Existing tools are invited to claim
conformance for the profiles they support.

The reference implementation follows one cardinal rule: **reject
unsupported semantics clearly rather than accepting code and changing
behavior.** Valid-but-unimplemented POST Python produces an explicit
diagnostic, never a silent rewrite.

## Proving ground: rebuilding SciPy

The primary way the language and compiler grow is the
[PostSciPy effort](postscipy.md) — recreating SciPy one subpackage at a
time as pure POST Python libraries
([ppspecial](https://github.com/openteams-ai/ppspecial) for
`scipy.special`, with thirteen more `pp*` packages scaffolded). Real
numerical code discovers what the language is missing; those gaps become
compiler and specification work.

ppspecial today: 26 special functions (error functions, gamma family,
Bessel, statistical) — every module compiles natively, cross-module calls
link per the spec's translation-unit model, and the whole package builds
into a single library and an importable NumPy extension.

## Status

POST Python is early and moving fast. Working today in the reference
implementation:

| Area | State |
|---|---|
| Structural checker (subset enforcement, `PP0xx` diagnostics) | ✅ |
| Scalar kernels, control flow, module constants | ✅ |
| `@vectorize` / `@guvectorize` with NumPy-conformant gufunc ABI | ✅ |
| Cross-module compilation and linking (one object per translation unit) | ✅ |
| CPython extension-module output (real `numpy.ufunc` registration) | ✅ |
| Stable C ABI: `pp_*` exports, generated headers, export manifests | ✅ |
| Structs, executables, callable parameters, local array allocation | 🔜 spec'd, not yet lowered |

The specification is a **v0.2 draft**. Interfaces will change.

## Where to go next

- [Getting started](getting-started.md) — install, write a kernel,
  compile it three ways.
- [The specification](spec.md) — the normative document.
- [Toolchain & C ABI](toolchain.md) — the CLI, headers, and manifests.
- [Distribution policy](distribution.md) — source-only PyPI; binaries
  through package managers that treat native code honestly.
