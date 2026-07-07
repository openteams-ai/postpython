# Toolchain & C ABI

## The `post-py` CLI

```
post-py check FILE...          structural subset checking
post-py build FILE [options]   compile to native artifacts
```

Build options:

| Option | Effect |
|---|---|
| `--output PATH` | artifact path (default: next to the source) |
| `--ext-module` | build an importable CPython extension registering NumPy ufuncs |
| `--module-name NAME` | artifact / importable name (defaults to the file stem, or the package directory for `__init__.py`) |
| `--emit-header` | write the C ABI header next to the output |
| `--emit-manifest` | write the JSON export manifest next to the output |
| `--prefix PREFIX` | install the `libpp<name>` package-manager layout |
| `--search-path DIR` | additional POST module source root (repeatable) |
| `--cc`, `--cflags`, `--keep-c` | toolchain control |

The same pipeline is available as a Python API:
`post_py.build.build_file` / `build_source`.

## What a build does

1. **Check** — the structural checker rejects non-subset constructs
   (`PP0xx`).
2. **Compile the program** — the entry file and every POST module it
   imports, dependencies first, one translation unit each
   ([spec §9.1](spec.md#9-compilation-model)). Imports from the standard
   library or site-packages are CPython-boundary imports and are never
   compiled implicitly.
3. **Emit C99, compile, link** — one object file per translation unit,
   linked into a single shared library (or extension module). Private
   (`_`-prefixed) functions get internal linkage; public names must be
   unique across the program (`PP501`).

## The stable C ABI (spec §9.1.1)

Every artifact defines `pp_<name>` wrapper symbols for its export set —
the entry unit's public functions, its POST imports under their local
names, and module-level aliases like `gammaln = lgamma`. Kernel symbol
names underneath are implementation detail (libm-colliding names such as
`j0` are mangled to `__pp_j0`); **`pp_j0` is the contract.**

**The header** (`--emit-header`) is self-contained C99: the
`__pp_array` view struct (spec §9.2), one declaration per export with
provenance and docstring comments, and the NumPy ufunc loop symbols
(spec §8.5).

**The manifest** (`--emit-manifest`) is versioned JSON:

```json
{
  "post_abi": 1,
  "artifact": "ppspecial",
  "exports": [
    {
      "name": "gammaln",
      "c_symbol": "pp_gammaln",
      "kernel_symbol": "__pp_lgamma",
      "module": "_gamma",
      "kind": "alias",
      "alias_of": "lgamma",
      "params": [{"name": "x", "dtype": "Float64", "is_array": false,
                  "is_core_dim": false}],
      "return_dtype": "Float64",
      "ufunc": {"loop_symbol": "lgamma_ufunc_loop", "signature": "()->()"},
      "doc": "..."
    }
  ]
}
```

Packaging recipes and foreign-language bindings should consume the
manifest rather than guessing symbol names.

## Array ABI

Arrays cross the C boundary as views with explicit metadata
(spec §9.2):

```c
typedef struct __pp_array {
    void *data;
    int64_t ndim;
    int64_t const *shape;
    int64_t const *strides;   /* byte strides, NumPy-compatible */
    int64_t offset_bytes;
} __pp_array;
```

Vectorized functions additionally export `<name>_ufunc_loop` with the
NumPy generalized-ufunc calling convention — outer broadcast steps
followed by per-argument core-dimension strides — so loops registered
via `PyUFunc_FromFuncAndData` handle non-contiguous inputs correctly.
