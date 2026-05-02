# POST Python Language Specification

**Version:** 0.2 (draft)
**Status:** Work in progress

---

## 1. Overview

POST Python (Performance Optimized Statically Typed Python) is a defined subset of the Python language with mandatory type annotations that enables ahead-of-time (AOT) compilation to native executables and shared libraries.  A POST Python source file is valid Python; it can be run under the standard CPython interpreter without modification.  A conforming compiler additionally translates it to native code without requiring the Python runtime.

### 1.1 Design Contract

POST Python is designed to be:

- **Syntactically Python** — POST Python source files are valid `.py` files and should remain usable in interpreted compatibility mode.
- **Statically typed at boundaries** — function and method boundaries are fully annotated; local types may be inferred.
- **Native-code first** — compiled output uses native memory layouts and does not depend on the CPython object model unless explicitly crossing the CPython boundary.
- **Array and dataframe first** — arrays, series, and dataframes are part of the standard language model, not optional foreign-library objects.
- **Smaller than Python** — dynamic language features that prevent portable AOT compilation are excluded.
- **Extensible by design** — accelerator, SIMD, dataframe, and domain-specific compilers can extend the base language through typed, compiler-visible APIs.

Where CPython behavior conflicts with fixed-width native types, deterministic memory management, or explicitly specified POST Python semantics, this specification governs compiled behavior.  Interpreted compatibility mode should remain as close to CPython behavior as practical while preserving the same type and subset guarantees.

### 1.2 Goals

1. **Standardize** a compilable Python subset so that tools such as Cython, mypyc, Numba, Codon, Pythran, taichi-lang, and related projects can implement a common target.
2. **Enable AOT compilation** to standalone executables and shared libraries (including CPython extension modules) without the Python runtime.
3. **Be the go-to language for extension modules**, replacing typical uses of C, Rust, Go, or Zig when Python syntax and ecosystem compatibility are desired.
4. **Provide built-in array and dataframe types** grounded in the array-api, Arrow-like columnar memory, and narwhals-style backend-neutral semantics, making numerical and data-intensive code first-class.
5. **Support generalized ufuncs** as the primary abstraction for data-parallel computation.
6. **Serve as a base layer for DSLs** — GPU kernels, SIMD kernels, distributed compute — by keeping the subset minimal and the type system extensible.

### 1.3 Non-Goals

- Replacing CPython for general scripting.
- Supporting the full Python object model in compiled output.
- Providing a runtime garbage collector in compiled binaries.

---

## 2. Conformance

A **POST Python source file** is a `.py` file that:

1. Is parseable by the CPython `ast` module with `type_comments=True`.
2. Passes the POST Python structural checker (Section 5) with zero structural violations.
3. Carries complete type annotations at every function boundary (Section 6).
4. Uses only annotations, operations, and standard-library features defined by this specification or by an explicitly enabled POST Python extension.

Conformance is intentionally modular.  A compiler may claim conformance to one or more named profiles:

| Profile | Required support |
|---------|------------------|
| **POST Core** | Scalar types, functions, modules, structs/dataclasses, control flow, type checking, and native object/shared-library/executable output. |
| **POST Array** | POST Core plus `Array[DType, Shape]`, array indexing, array allocation, shape checking, array iteration, and array-compatible scalar math. |
| **POST DataFrame** | POST Core plus `Series`, `DataFrame`, `LazyFrame`, typed schemas, and the standard dataframe algebra defined by this specification. |
| **POST Ufunc ABI** | POST Array plus Numba-shaped `@vectorize`/`@guvectorize` decorators, layout signature checking, interpreted-mode behavior, and the native ufunc ABI. |
| **POST CPython Extension** | POST Core plus CPython extension-module output and CPython heap-boundary semantics. |
| **POST Accelerator Extension** | A named extension profile for GPU, SIMD, distributed, or other domain-specific lowering.  Such profiles must state their additional types, decorators, memory spaces, and fallback behavior. |

A compiler that does not implement an optional profile must reject code requiring that profile with a clear diagnostic rather than silently changing semantics.

A **conforming compiler** is a tool that:
1. Accepts POST Python source files for the conformance profile(s) it claims.
2. Produces native object code, shared libraries, executables, or CPython extension modules with semantics defined by this specification.
3. Does not require the CPython runtime to be present in the final binary except in CPython extension builds or when explicitly crossing the CPython heap boundary.
4. Rejects unsupported valid POST Python features with an implementation-support diagnostic, rather than accepting and dropping or rewriting behavior.

A **conforming interpreter** is a tool that accepts POST Python source files and executes them with CPython-compatible behavior where possible, while enforcing the same POST Python type, subset, and profile guarantees as a conforming compiler.

Existing tools (Cython, mypyc, Numba, Codon, Pythran, taichi-lang, etc.) are encouraged to implement this standard and claim conformance for the profiles they support.

The prose specification is normative.  The reference checker, reference compiler, and conformance test suite are executable aids for implementors.  When the reference implementation and this specification disagree, the specification governs unless the specification is amended.

---

## 3. Source Files

- Encoding: UTF-8.
- File extension: `.py` (same as Python; no new extension is introduced).
- A POST Python file must be parseable by the CPython `ast` module with `type_comments=True`.
- The first-party checker (`postpython-check`) is the normative structural checker for the reference implementation.  Full conformance also requires type, semantic, memory, and profile-specific validation.
- Top-level executable statements are implementation-defined in v0.1 except for imports, type aliases, constant definitions, class definitions, and function definitions.  Portable POST Python packages should put executable logic behind typed functions.

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

### 4.1.1 Numeric Semantics

POST Python numeric types are fixed-width native scalar types.  They do not inherit Python's arbitrary-precision `int` semantics in compiled mode.

Unless otherwise specified by a future profile:

- Signed and unsigned integer arithmetic is performed at the declared width.
- Debug builds must diagnose integer overflow where practical; release builds may use the target platform's native overflow behavior, but compilers must document whether they wrap, trap, or assume no overflow for optimization.
- Floating-point operations follow IEEE 754 for the declared width.  Compilers must not enable transformations that violate required IEEE behavior unless the user explicitly enables a non-conforming fast-math mode.
- `NaN`, infinities, and signed zero follow IEEE comparison and arithmetic behavior.
- Integer division, floor division, and remainder must be specified by the POST Python arithmetic rules for the operand dtypes.  Until those rules are finalized, portable code should avoid relying on edge cases involving negative integer `//` and `%`.
- Numeric casts must be explicit when they may lose precision, change signedness, or narrow width.  Debug builds should diagnose out-of-range narrowing casts where practical.
- Scalar promotion rules are part of the type system and must be consistent across scalar expressions, array expressions, and vectorized kernels.

### 4.2 Array Type

The `Array[DType]`, `Array[DType, Shape[...]]`, and layout-qualified array types represent N-dimensional homogeneous arrays, following the array-api standard and NumPy-compatible layout concepts.

```python
from postyp import Array, Float64, Shape, COrder, FOrder, Strides

# 1-D array of unknown length
def scale(a: Array[Float64], factor: Float64) -> Array[Float64]: ...

# Statically shaped: 3×3 matrix
def det3(m: Array[Float64, Shape[3, 3]]) -> Float64: ...

# Mixed: dynamic first dim, fixed second
def batch_norm(x: Array[Float64, Shape[None, 128]]) -> Array[Float64, Shape[None, 128]]: ...

# Fortran-contiguous matrix
def col_major_sum(x: Array[Float64, Shape[3, 3], FOrder]) -> Float64: ...

# Explicitly strided 2-D view, strides are measured in bytes
def view_sum(x: Array[Float64, Shape[None, None], Strides[None, 8]]) -> Float64: ...
```

`Shape` dimensions:
- Positive integer — statically known size; compiler may use this for bounds-elimination and SIMD.
- `None` — dynamic size; checked at runtime in debug builds.
- `Shape[...]` (or `AnyShape`) — fully dynamic rank; no static shape information.

Array memory layout defaults to row-major C order when no layout qualifier is supplied.

The POST Array profile should distinguish logical arrays from physical layout:

- `Array[DType, Shape]` denotes an owned or borrowed homogeneous array value with a known element dtype and optional shape constraints.
- The default physical layout is C-contiguous row-major storage (`COrder`).
- `FOrder` denotes Fortran-contiguous column-major storage.
- `Strides[...]` denotes explicit per-axis strides.  Strides are specified in bytes, following NumPy's convention.  `None` in a stride position means the stride is dynamic and supplied at runtime.
- C-order and Fortran-order arrays are special cases of strided arrays with statically derivable strides.
- Negative strides are permitted for borrowed views when the implementation can preserve bounds and lifetime safety; owned arrays should use non-negative compact strides unless explicitly constructed otherwise.
- Slices and views borrow storage by default and carry shape, stride, offset, and mutability metadata.  Operations that require compact storage must either prove contiguity or make an explicit copy.
- Implementations must expose enough runtime metadata for dynamic-rank or dynamic-stride arrays to support shape checks, bounds checks, and correct element-address calculation.
- Bounds checks are required in debug builds and configurable in release builds.
- Mutation requires write access to the array value under the ownership model in Section 7.
- Broadcasting outside vectorized functions is reserved for a future array-expression standard.

### 4.3 DataFrame and Series Types

`DataFrame`, `LazyFrame`, and `Series` define a typed logical dataframe model inspired by narwhals and backed in compiled mode by a native columnar runtime with Arrow-like behavior.  Interpreted compatibility mode may delegate to narwhals-compatible backends such as pandas, polars, or modin, but compiled POST DataFrame code must not require pandas or another interpreted dataframe engine at runtime.

```python
from postyp import DataFrame, Series, Float64, Schema

Trades = DataFrame.with_schema({"price": Float64, "volume": Float64})

def vwap(trades: Trades) -> Float64: ...

def prices(trades: Trades) -> Series[Float64]: ...
```

The default POST DataFrame runtime is columnar:

- Each `Series[DType]` is a homogeneous logical column with a validity bitmap for nullable values when nullability is enabled.
- A `DataFrame` is an ordered mapping of UTF-8 column names to `Series` values of equal length.
- Physical layout should be Arrow-compatible where practical: contiguous buffers, optional offsets for variable-width values, optional validity bitmaps, and zero-copy interchange when layout constraints are met.
- Implementations may use alternate physical layouts internally if observable semantics and the standard interchange ABI are preserved.

The POST DataFrame profile should define a portable dataframe algebra.  The initial standard library should prioritize:

- projection and column selection
- row filtering
- computed columns
- scalar and element-wise expressions
- aggregation
- `group_by` aggregation
- equi-join operations
- sorting
- null propagation and null-aware comparisons
- schema-preserving and schema-transforming operations

`LazyFrame` represents an optimizable logical query plan.  A conforming POST DataFrame compiler may lower `LazyFrame` operations to native loops, vectorized kernels, SQL-like plans, Arrow compute kernels, accelerator kernels, or other equivalent execution strategies.

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
- `@vectorize` and `@guvectorize` (Section 8)

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

POST Python code may need to read CPython-owned objects — for example, an interpreted-mode dataframe backend, an object passed in from a CPython extension caller, or a Python object explicitly accessed through a borrow handle.  The rules are:

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
- The GIL is released automatically around POST Python-heap-only operations (pure numeric kernels, vectorized inner loops) and reacquired before any CPython API call.

---

## 8. Vectorized Functions and Generalized Ufuncs

### 8.1 Overview

POST Python adopts Numba's public decorator model for NumPy-compatible ufuncs:

- `@vectorize` defines an element-wise scalar kernel.  The user function receives scalar values and returns one scalar value.  The compiler supplies the broadcast loop.
- `@guvectorize` defines a generalized ufunc kernel.  The user function receives scalar values and/or core array views, and writes results through trailing output array parameters.

The decorators may be imported from `postpython` or `postpython.ufunc`.

Scalar element-wise example:

```python
from postpython import vectorize
from postyp import Float64

@vectorize(["float64(float64, float64)"], target="cpu")
def add(x: Float64, y: Float64) -> Float64:
    return x + y
```

Generalized ufunc example:

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

### 8.2 Decorator Forms

`@vectorize` accepts the following forms:

```python
@vectorize
@vectorize()
@vectorize([type_signature, ...], target="cpu", nopython=True)
```

The concrete type signature list follows Numba's convention.  POST Python annotations remain normative; the signature list is used for compatibility, dispatch metadata, and future multi-specialization support.  A `@vectorize` function must have scalar parameters and a scalar return annotation.  Its implicit layout signature is `(),...->()`.

`@guvectorize` accepts the following forms:

```python
@guvectorize(layout_signature)
@guvectorize([type_signature, ...], layout_signature, target="cpu", nopython=True)
```

The `layout_signature` is the NumPy ufunc layout signature string defined below.  A `@guvectorize` function must return `None`; output values are trailing `Array[...]` parameters, including scalar outputs, which are represented as one-element output arrays and written with `out[0]`.

The standard CPU target is `"cpu"`.  `"parallel"`, `"cuda"`, and other targets are accelerator-extension concerns unless a compiler explicitly claims those profiles.

### 8.3 Layout Signature Syntax

```
layout_sig  ::= input_sigs "->" output_sigs
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

### 8.4 Type Constraints

- Input arrays' element dtype must match the `Array[DType]` annotation.
- Core dimension sizes must agree across all array arguments at call time.
- `@vectorize` parameters and returns must be scalar dtype annotations.
- `@guvectorize` outputs must be trailing `Array[...]` parameters.  Scalar layout outputs such as `"->()"` are still output array parameters; the kernel writes the value at index `0`.
- `@guvectorize` kernels must return `None`.

### 8.5 Lowering to NumPy Ufunc Protocol

A conforming compiler lowers `@vectorize` and `@guvectorize` functions to the NumPy ufunc C API.  `@vectorize` is equivalent to a vectorized function whose inputs and single output all have scalar layout `()`:

```c
void fn_ufunc(
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

The compiler emits the outer broadcast loop and maps each array argument to the POST Array ABI view described in Section 9.2 before calling the user kernel.  This preserves shape, stride, and offset metadata inside the native kernel instead of passing only a raw data pointer.

### 8.6 Interpreted Mode

When a POST Python source file is run under the standard interpreter (not compiled), `@vectorize` and `@guvectorize` wrap the function in a Python-level broadcast loop.  If NumPy is available, it uses NumPy arrays and broadcasting semantics.  The function remains callable and testable without compilation.

---

## 9. Compilation Model

### 9.1 Translation Units

A POST Python translation unit is a single `.py` source file.  A **package** is a directory of translation units with an `__init__.py`.  The compiler processes one translation unit at a time and produces one object file (`.o`) per translation unit.

Module imports have three roles:

- **Compile-time imports** provide POST Python types, decorators, constants, and functions visible to the compiler.
- **POST module imports** refer to other POST Python translation units that are type-checked and linked into the output artifact.
- **CPython boundary imports** refer to Python modules used only in interpreted mode or through explicit CPython heap-boundary handles.

Portable POST Python code should make CPython boundary crossings explicit.  A compiler must not silently compile a call to an arbitrary imported Python function as native POST code unless that function is available as a checked POST translation unit or as a declared foreign function.

Public symbols are top-level functions, dataclasses, type aliases, and constants not prefixed with `_`.  A future package ABI will define stable symbol names, version metadata, and cross-module incremental compilation behavior.

### 9.2 Native Array ABI

The POST Array ABI represents each array value as a view with explicit shape and stride metadata.  The reference C ABI uses the following logical layout:

```c
typedef struct __pp_array {
    void *data;
    int64_t ndim;
    int64_t const *shape;
    int64_t const *strides;  /* byte strides */
    int64_t offset_bytes;
} __pp_array;
```

Fields:

- `data` points to the first byte of the underlying allocation or exported buffer.
- `ndim` is the runtime rank.
- `shape[i]` is the logical extent of axis `i`.
- `strides[i]` is the byte stride for axis `i`, matching NumPy's `ndarray.strides` convention.
- `offset_bytes` is the byte offset from `data` to the first logical element in the view.

C-contiguous and Fortran-contiguous arrays are represented as regular `__pp_array` values with derived byte strides.  Slices and other views may share the same `data` pointer while changing `shape`, `strides`, and `offset_bytes`.

Typed element access is computed as:

```c
*(T *)((char *)array.data + array.offset_bytes + byte_index)
```

where `byte_index` is produced from the logical index tuple and the view's byte strides.  Implementations may use equivalent target-specific ABI layouts, but they must preserve the same logical fields at ABI boundaries.

### 9.3 Outputs

| Mode              | Flag           | Output                              |
|-------------------|----------------|-------------------------------------|
| Object file       | `-c`           | `.o`                                |
| Shared library    | `--shared`     | `.so` / `.dylib` / `.dll`           |
| CPython extension | `--ext-module` | `<name>.cpython-<tag>.so`           |
| Executable        | (default)      | native binary, entry point = `main` |

### 9.4 Entry Point

When compiling to an executable, the compiler looks for a top-level `def main() -> Int:` function as the entry point.  Its return value becomes the process exit code.

### 9.5 Backends

The reference compiler emits C99 as its intermediate output, then invokes the system C compiler (`cc`/`clang`/`gcc`).  Alternate backends (LLVM IR, WebAssembly) may be added without changing the language specification.

### 9.6 Debug vs. Release Builds

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
2. **`postpython` / `postpython.ufunc`** — the `@vectorize` and `@guvectorize` decorators and signature utilities.
3. **`postpython.math`** — scalar math functions (`sqrt`, `sin`, `cos`, `exp`, `log`, …), lowered to `libm`.
4. **`postpython.mem`** — explicit memory utilities (`alloc`, `free`, `share`) for advanced use.

The standard library does not include I/O, networking, or threading primitives; those are accessed through the CPython boundary.

---

## 11. Extension Model

POST Python is intended to support multiple compiler implementations and domain-specific extensions such as Triton-like GPU kernels, Helion-like tensor kernels, SIMD kernels, distributed compute, and dataframe query engines.

An extension profile may add:

- decorators
- types
- address spaces or memory layouts
- intrinsic functions
- compiler passes
- backend-specific ABIs

Every extension profile must specify:

- the conformance profile(s) it depends on
- how extension syntax remains valid Python
- whether interpreted compatibility mode is required and what fallback it uses
- which operations are portable across implementations and which are implementation-defined
- how extension values cross into POST Core, POST Array, POST DataFrame, and CPython boundary code

Extensions must not change the meaning of POST Core programs unless explicitly enabled by the user.

---

## 12. Implementation Guidance

This section is non-normative.

- Implementors targeting CPython extension output should follow the [CPython limited API](https://docs.python.org/3/c-api/stable.html) to maximize ABI stability.
- The `postyp` module is the canonical source of type metadata; compilers should import and introspect it rather than duplicating dtype definitions.
- The reference compiler (this repository) serves as an executable implementation aid and conformance-test target.  The prose specification is normative.
- Implementors of existing tools (Cython, mypyc, Numba, Codon, Pythran, taichi-lang, etc.) implementing this standard should document which conformance profiles, violation codes, and ABIs they support.

---

## Appendix A: Violation Code Registry

See Section 5.2 for the current structural table.  Diagnostic ranges are reserved as follows:

| Range | Category |
|-------|----------|
| PP000–PP099 | Structural and syntax violations |
| PP100–PP199 | Type and annotation errors |
| PP200–PP299 | Ownership and memory model violations |
| PP300–PP399 | Array, shape, and broadcasting errors |
| PP400–PP499 | DataFrame, Series, schema, and query-plan errors |
| PP500–PP599 | ABI, module, linking, and build errors |
| PP900–PP999 | Implementation-defined or unsupported valid POST Python features |

## Appendix B: Revision History

| Version | Date       | Notes                       |
|---------|------------|-----------------------------|
| 0.2     | 2026-05-02 | Modular conformance profiles, native dataframe runtime guidance, and expanded array layout semantics |
| 0.1     | 2026-04-30 | Initial draft                |
