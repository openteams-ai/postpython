# POST Python Language Specification

**Version:** 0.1 (draft)
**Status:** Work in progress

---

## 1. Overview

POST Python (Performance Optimized Statically Typed Python) is a defined subset of the Python language with mandatory type annotations that enables ahead-of-time (AOT) compilation to native executables and shared libraries.  A POST Python source file is valid Python; it can be run under the standard CPython interpreter without modification.  A conforming compiler additionally translates it to native code without requiring the Python runtime.

### 1.1 Goals

1. **Standardize** a compilable Python subset so that tools such as Cython, mypyc, Numba, and taichi-lang can implement a common target.
2. **Enable AOT compilation** to standalone executables and shared libraries (including CPython extension modules) without the Python runtime.
3. **Be the go-to language for extension modules**, replacing typical uses of C, Rust, Go, or Zig for that purpose.
4. **Provide built-in array and dataframe types** grounded in the array-api (data-apis.org) and narwhals standards, making numerical and data-intensive code first-class.
5. **Support generalized ufuncs** as the primary abstraction for data-parallel computation.
6. **Serve as a base layer for DSLs** — GPU kernels, SIMD kernels, distributed compute — by keeping the subset minimal and the type system extensible.

### 1.2 Non-Goals

- Replacing CPython for general scripting.
- Supporting the full Python object model in compiled output.
- Providing a runtime garbage collector in compiled binaries.

---

## 2. Conformance

A **POST Python source file** is a `.py` file that passes the POST Python checker (Section 5) with zero violations and where every function boundary carries complete type annotations (Section 6).

A **conforming compiler** is a tool that:
1. Accepts POST Python source files.
2. Produces native object code, shared libraries, or executables with semantics identical to running the source under CPython (within the bounds of numeric precision as specified in Section 4).
3. Does not require the CPython runtime to be present in the final binary (except for extension module builds, which link `libpython` by definition).
4. Implements the generalized ufunc ABI described in Section 8.

A **conforming interpreter** is a tool that accepts POST Python source files and executes them with CPython-compatible semantics, providing the same type-checking guarantees as a conforming compiler at runtime.

Existing tools (Cython, mypyc, Numba, taichi-lang, etc.) are encouraged to implement this standard and claim conformance for the subset they support.

---

## 3. Source Files

- Encoding: UTF-8.
- File extension: `.py` (same as Python; no new extension is introduced).
- A POST Python file must be parseable by the CPython `ast` module with `type_comments=True`.
- The first-party checker (`postpython-check`) is the normative conformance gate.

---

## 4. Type System

### 4.1 Scalar Dtypes

POST Python defines the following scalar types in `postyp`, which mirror the array-api dtype specification:

| Name         | Kind    | Width | Notes                              |
|--------------|---------|-------|------------------------------------|
| `Bool`       | boolean | 8-bit | Maps to C `bool` / `_Bool`        |
| `Int8`       | signed  | 8     |                                    |
| `Int16`      | signed  | 16    |                                    |
| `Int32`      | signed  | 32    |                                    |
| `Int64`      | signed  | 64    | Default `Int` alias                |
| `UInt8`      | unsigned| 8     |                                    |
| `UInt16`     | unsigned| 16    |                                    |
| `UInt32`     | unsigned| 32    |                                    |
| `UInt64`     | unsigned| 64    |                                    |
| `Float16`    | float   | 16    | IEEE 754 binary16                  |
| `Float32`    | float   | 32    | IEEE 754 binary32                  |
| `Float64`    | float   | 64    | IEEE 754 binary64; default `Float` |
| `Complex64`  | complex | 64    | Two `Float32` components           |
| `Complex128` | complex | 128   | Two `Float64`; default `Complex`   |
| `Str`        | text    | var   | Immutable UTF-8 string             |
| `Bytes`      | bytes   | var   | Immutable byte sequence            |

Python built-in types map to POST Python types as follows:

| Python  | POST Python |
|---------|-------------|
| `bool`  | `Bool`      |
| `int`   | `Int64`     |
| `float` | `Float64`   |
| `complex`| `Complex128`|
| `str`   | `Str`       |
| `bytes` | `Bytes`     |

A conforming compiler must respect IEEE 754 semantics for all floating-point operations.

### 4.2 Array Type

The `Array[DType]` and `Array[DType, Shape[...]]` types represent N-dimensional homogeneous arrays, following the array-api standard.

```python
from postyp import Array, Float64, Shape

# 1-D array of unknown length
def scale(a: Array[Float64], factor: Float64) -> Array[Float64]: ...

# Statically shaped: 3×3 matrix
def det3(m: Array[Float64, Shape[3, 3]]) -> Float64: ...

# Mixed: dynamic first dim, fixed second
def batch_norm(x: Array[Float64, Shape[None, 128]]) -> Array[Float64, Shape[None, 128]]: ...
```

`Shape` dimensions:
- Positive integer — statically known size; compiler may use this for bounds-elimination and SIMD.
- `None` — dynamic size; checked at runtime in debug builds.
- `Shape[...]` (or `AnyShape`) — fully dynamic rank; no static shape information.

Array memory layout defaults to row-major (C order).  Compilers may support additional layout annotations in future versions.

### 4.3 DataFrame and Series Types

`DataFrame`, `LazyFrame`, and `Series` wrap the narwhals abstraction layer, making POST Python dataframe code backend-agnostic (pandas, polars, modin, etc.).

```python
from postyp import DataFrame, Series, Float64, Schema

Trades = DataFrame.with_schema({"price": Float64, "volume": Float64})

def vwap(trades: Trades) -> Float64: ...

def prices(trades: Trades) -> Series[Float64]: ...
```

### 4.4 Aggregate Types

`dataclass`-decorated classes with fully annotated fields are valid POST Python aggregate types.  They compile to structs.  Inheritance from exactly one base is permitted; multiple inheritance is not (Section 5.2, PP010).

```python
from dataclasses import dataclass
from postyp import Float64

@dataclass
class Point:
    x: Float64
    y: Float64
```

### 4.5 Type Inference

Within a function body, types may be inferred from:
- Literal values (`0`, `0.0`, `True`, `b""`, `""`)
- Arithmetic expressions where all operands have known types
- Function return types
- Array element access

Inferred types do not need annotation.  Type annotations at function boundaries (parameters and return values) are always required (Section 6.1).

---

## 5. Language Subset

### 5.1 Permitted Constructs

The following constructs are in the POST Python subset:

- Module-level `import` and `from … import` (absolute only)
- `def` with fully annotated signatures
- `class` (single inheritance, no metaclass, `@dataclass` recommended)
- `if` / `elif` / `else`
- `for` over ranges and arrays
- `while`
- `with` (for resource management)
- `try` / `except` / `finally` (scalar exceptions only; no exception groups)
- `return`, `break`, `continue`, `pass`
- `assert` (compiled to no-op in release builds)
- Annotated assignment (`x: Float64 = 0.0`)
- Augmented assignment (`+=`, `-=`, etc.)
- Comprehensions (list, dict, set) over statically-typed iterables
- Walrus operator (`:=`)
- `match` / `case` (structural pattern matching over scalar and aggregate types)
- `@dataclass`, `@staticmethod`, `@classmethod`
- `@gufunc` (Section 8)

### 5.2 Disqualified Constructs

The following constructs are outside the compilable subset.  The POST Python checker emits the listed violation code for each:

| Code  | Construct                                      |
|-------|------------------------------------------------|
| PP000 | Syntax error (cannot parse)                   |
| PP001 | `exec` statement                              |
| PP002 | Call to `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `dir`, `breakpoint` |
| PP003 | Call to `getattr`, `setattr`, `delattr`, `hasattr` |
| PP004 | Dynamic `__import__()`                        |
| PP005 | `type(name, bases, dict)` — dynamic class creation |
| PP006 | `global` statement                            |
| PP007 | `nonlocal` statement                          |
| PP008 | Relative import (`from . import …`)           |
| PP009 | Metaclass on a class definition               |
| PP010 | Multiple inheritance                          |
| PP011 | `async def`                                   |
| PP012 | `await` expression                            |
| PP013 | `async for`                                   |
| PP014 | `async with`                                  |
| PP020 | Unannotated function parameter (excluding `self`/`cls`) |
| PP021 | Missing return type annotation                |
| PP022 | `*args` (variadic positional)                 |
| PP023 | `**kwargs` (variadic keyword)                 |
| PP024 | Starred splat in call (`f(*lst)`)             |
| PP025 | `del` statement                               |
| PP030 | `yield`                                       |
| PP031 | `yield from`                                  |
| PP032 | `lambda`                                      |
| PP033 | `except*` (exception groups, PEP 654)         |

---

## 6. Type Annotations

### 6.1 Required Annotations

Every function or method visible at module scope must carry:
- A type annotation on every parameter except `self` and `cls`.
- A return type annotation.

Violation of either is reported as PP020 / PP021 respectively.

### 6.2 Annotation Grammar

```
annotation ::= scalar_type
             | Array "[" dtype "]"
             | Array "[" dtype "," shape "]"
             | DataFrame
             | DataFrame.with_schema(schema)
             | LazyFrame
             | LazyFrame.with_schema(schema)
             | Series "[" dtype "]"
             | dataclass_type
             | "None"
             | "Optional" "[" annotation "]"
             | "Union" "[" annotation ("," annotation)+ "]"
             | "Tuple" "[" annotation ("," annotation)* "]"
             | "List" "[" annotation "]"

scalar_type ::= "Bool" | "Int8" | "Int16" | "Int32" | "Int64"
              | "UInt8" | "UInt16" | "UInt32" | "UInt64"
              | "Float16" | "Float32" | "Float64"
              | "Complex64" | "Complex128"
              | "Str" | "Bytes"
              | "Int" | "Float" | "Complex"   # aliases
              | "bool" | "int" | "float" | "complex" | "str" | "bytes"  # Python built-ins

dtype     ::= scalar_type
shape     ::= "Shape" "[" dim ("," dim)* "]"
            | "Shape" "[" "..." "]"
            | "AnyShape"
dim       ::= integer | "None"
schema    ::= "{" str ":" dtype ("," str ":" dtype)* "}"
```

---

## 7. Memory Model

### 7.1 Two Heaps

POST Python recognises two distinct memory regions:

| Heap           | Owner              | Lifetime              | Management          |
|----------------|--------------------|-----------------------|---------------------|
| **POST Python heap** | POST Python runtime | RAII / scope-bound  | Deterministic free  |
| **CPython heap**     | CPython interpreter | Reference-counted   | CPython GC          |

RAII applies **only** to objects allocated on the POST Python heap.  CPython-owned objects are accessed through typed handles that respect CPython's reference counting protocol; they are never freed by POST Python code directly.

### 7.2 RAII (Resource Acquisition Is Initialization)

POST Python heap objects use deterministic, scope-based lifetime.  Every POST Python-owned value's storage is released when the owning scope exits.  There is no garbage collector for POST Python heap objects.

- Array and aggregate allocations are freed when the binding goes out of scope.
- `with` blocks implement RAII for external resources.
- Exceptions unwind the stack with deterministic cleanup (analogous to C++ stack unwinding).

### 7.3 Ownership

Each POST Python heap value has exactly one owner at any point in time.  Ownership transfers on assignment across scope boundaries (function calls, return).  Shared read-only access is permitted.

Compilers may implement ownership transfer as move semantics (zero-copy) for arrays and aggregates.  The reference compiler uses reference-counted handles at scope boundaries; future optimization passes may elide the count updates.

### 7.4 Single-Writer Concurrency

POST Python's threading model:

- **Multiple readers, one writer**: a value may be read concurrently by any number of threads, but at most one thread holds write access at any time.
- Write access is acquired implicitly when a mutable binding is created in a thread's scope.
- Read-sharing across threads is done via explicit `share(value)` — a future built-in that promotes a value to shared-read status.
- This model is designed for efficient multi-core scaling without a global interpreter lock in compiled output.

The reference compiler enforces single-writer as a runtime assertion in debug builds and a no-op in release builds (trusting the programmer), while the long-term goal is compile-time enforcement.

### 7.5 CPython Heap Boundary

POST Python code frequently needs to read CPython-owned objects — for example, a narwhals DataFrame backed by pandas, or any Python object passed in from an extension module caller.  The rules are:

**Reading CPython objects (POST Python → CPython direction)**

POST Python code may read CPython-owned objects through typed *borrow handles*.  A borrow handle:
- Does not transfer ownership.
- Increments the CPython refcount on acquisition and decrements it on release (RAII-managed).
- Is read-only by default; write access requires an explicit `mut_borrow()` and is only safe when the caller guarantees exclusive access (checked in debug builds).

**Writing CPython objects**

POST Python may write into CPython-owned mutable containers (lists, arrays, DataFrames) via borrow handles.  Structural mutation (resizing, type change) is not permitted through a borrow handle.

**Passing POST Python objects to CPython (POST Python → CPython direction)**

When a POST Python value must be passed to a CPython-callable (e.g. a Python function argument), the compiler wraps it in a CPython object:
- The POST Python value's lifetime is extended until the CPython object's refcount drops to zero.
- Compilers may use a zero-copy path for types that map directly to buffer-protocol objects (e.g. `Array[Float64]` → `numpy.ndarray`).

**Returning CPython objects from CPython calls**

When CPython returns an object to POST Python code, the compiler inserts a typed unwrap thunk.  If the runtime type does not match the annotation, a `TypeError` is raised (this is the only place runtime type checking occurs in POST Python).

**Extension module build mode**

When compiling a CPython extension module (`--ext-module`), POST Python code runs *inside* CPython's interpreter loop.  In this mode:
- The POST Python heap still uses RAII.
- CPython heap objects are accessed via the standard C API (`PyObject*`), managed with `Py_INCREF`/`Py_DECREF` wrappers generated by the compiler.
- The GIL is released automatically around POST Python-heap-only operations (pure numeric kernels, gufunc inner loops) and reacquired before any CPython API call.

---

## 8. Generalized Ufuncs

### 8.1 Overview

A **generalized ufunc** (gufunc) is a function that operates on sub-arrays of specified rank, broadcast by the runtime over outer (batch) dimensions.  POST Python adopts the NumPy gufunc signature convention and compiles gufuncs to the NumPy C ufunc protocol, making them usable from any Python code.

```python
from postyp import Array, Float64
from postpython.gufunc import gufunc

@gufunc("(n),(n)->()")
def dot(a: Array[Float64], b: Array[Float64]) -> Float64:
    result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
```

### 8.2 Signature Syntax

```
gufunc_sig  ::= input_sigs "->" output_sigs
input_sigs  ::= "(" dim_names ")" ("," "(" dim_names ")")*
output_sigs ::= "(" dim_names ")" ("," "(" dim_names ")")*
dim_names   ::= ""                      # scalar output/input
              | name ("," name)*        # named dimensions
name        ::= [a-z][a-z0-9_]*
```

Named dimensions that appear in both input and output signatures are **core dimensions** — their size must be consistent across all arguments.

Examples:

| Signature             | Operation                        |
|-----------------------|----------------------------------|
| `"()->()"`            | scalar element-wise              |
| `"(n)->(n)"`          | element-wise on 1-D array        |
| `"(n)->()"`           | reduction over last axis         |
| `"(n),(n)->()"`       | inner product                    |
| `"(m,k),(k,n)->(m,n)"`| matrix multiply                  |
| `"(n,n)->()"`         | square-matrix scalar (e.g. det)  |

### 8.3 Type Constraints

- Input arrays' element dtype must match the `Array[DType]` annotation.
- Core dimension sizes must agree across all array arguments at call time.
- If a gufunc's signature includes scalar outputs (`"->()"`), the return annotation must be the corresponding scalar dtype, not `Array`.

### 8.4 Lowering to NumPy Ufunc Protocol

A conforming compiler lowers `@gufunc` functions to the NumPy generalized ufunc C API:

```c
void fn_gufunc(
    char **args,
    npy_intp const *dimensions,
    npy_intp const *steps,
    void *data
);
```

Where:
- `dimensions[0]` is the number of outer (broadcast) iterations.
- `dimensions[1..n]` are the core dimensions in signature order.
- `steps[i]` is the stride in bytes for argument `i`.
- `args[i]` points to the first element of argument `i`.

The compiler emits the outer broadcast loop and maps the inner body to the user function.

### 8.5 Interpreted Mode

When a POST Python source file is run under the standard interpreter (not compiled), `@gufunc` wraps the function in a Python-level broadcast loop.  If NumPy is available, it uses `numpy.lib.stride_tricks` to implement the broadcast semantics.  The function remains callable and testable without compilation.

---

## 9. Compilation Model

### 9.1 Translation Units

A POST Python translation unit is a single `.py` source file.  A **package** is a directory of translation units with an `__init__.py`.  The compiler processes one translation unit at a time and produces one object file (`.o`) per translation unit.

### 9.2 Outputs

| Mode              | Flag           | Output                              |
|-------------------|----------------|-------------------------------------|
| Object file       | `-c`           | `.o`                                |
| Shared library    | `--shared`     | `.so` / `.dylib` / `.dll`           |
| CPython extension | `--ext-module` | `<name>.cpython-<tag>.so`           |
| Executable        | (default)      | native binary, entry point = `main` |

### 9.3 Entry Point

When compiling to an executable, the compiler looks for a top-level `def main() -> Int:` function as the entry point.  Its return value becomes the process exit code.

### 9.4 Backends

The reference compiler emits C99 as its intermediate output, then invokes the system C compiler (`cc`/`clang`/`gcc`).  Alternate backends (LLVM IR, WebAssembly) may be added without changing the language specification.

### 9.5 Debug vs. Release Builds

| Feature                    | Debug (`-g`) | Release (`-O2`) |
|----------------------------|-------------|-----------------|
| Bounds checks              | yes          | configurable    |
| Single-writer assertions   | yes          | no              |
| `assert` statements        | yes          | elided          |
| Numeric overflow checks    | yes          | no              |

---

## 10. Standard Library

POST Python's standard library consists of:

1. **`postyp`** — the type vocabulary (scalar dtypes, `Array`, `DataFrame`, `Series`, `Shape`).
2. **`postpython.gufunc`** — the `@gufunc` decorator and signature utilities.
3. **`postpython.math`** — scalar math functions (`sqrt`, `sin`, `cos`, `exp`, `log`, …), lowered to `libm`.
4. **`postpython.mem`** — explicit memory utilities (`alloc`, `free`, `share`) for advanced use.

The standard library does not include I/O, networking, or threading primitives; those are accessed through the CPython boundary.

---

## 11. Implementation Guidance

This section is non-normative.

- Implementors targeting CPython extension output should follow the [CPython limited API](https://docs.python.org/3/c-api/stable.html) to maximize ABI stability.
- The `postyp` module is the canonical source of type metadata; compilers should import and introspect it rather than duplicating dtype definitions.
- The reference compiler (this repository) serves as the normative executable specification.  When the prose spec and the reference compiler disagree, file a bug against the spec.
- Implementors of existing tools (Cython, mypyc, Numba) implementing this standard should document which violation codes they enforce and which parts of the gufunc ABI they support.

---

## Appendix A: Violation Code Registry

See Section 5.2 for the full table.  Codes PP000–PP099 are reserved for structural/syntax violations.  PP100–PP199 are reserved for type errors.  PP200–PP299 are reserved for memory model violations (future).

## Appendix B: Revision History

| Version | Date       | Notes                       |
|---------|------------|-----------------------------|
| 0.1     | 2026-04-30 | Initial draft                |
