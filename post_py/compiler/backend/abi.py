"""Stable C ABI for compiled POST programs (spec §9.1 Package ABI v1).

Three artifacts make a compiled program consumable from C, C++, Rust, Go,
Zig, Julia, ctypes, cffi, and friends without knowing compiler internals:

- **Export wrappers**: every export gets a thin C function named
  ``pp_<name>`` with the export's exact signature. Kernel symbols
  (possibly mangled to avoid libc/libm collisions, e.g. ``__pp_j0``)
  remain implementation detail; ``pp_j0`` is the contract.
- **Header** (``emit_header``): self-contained C99 declarations for every
  export plus the ``__pp_array`` view struct (spec §9.2) and the NumPy
  ufunc loop symbols (spec §8.5).
- **Manifest** (``export_manifest``): machine-readable JSON
  (``"post_abi": 1``) mapping Python names to C symbols, dtypes, kinds,
  aliases, and ufunc layout signatures.

The export set mirrors the entry translation unit's Python namespace:
its own public functions, functions imported from POST dependencies
(under local alias names), and module-level function aliases such as
``gammaln = lgamma`` — which become real exported symbols and manifest
entries of kind ``"alias"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..ir import Function, Module, UFunc
from ..typechecker import TypeError_PP
from .c_backend import (
    CEmitter,
    c_symbol,
    c_type,
    c_value_type,
    emit_function_signature,
)


@dataclass(frozen=True)
class Export:
    python_name: str          # name in the artifact's namespace
    c_symbol: str             # stable ABI symbol: "pp_" + python_name
    kernel_symbol: str        # the defining function's emitted C symbol
    module: str               # defining module's name
    kind: str                 # "function" | "ufunc" | "alias"
    target: Function          # resolved IR of the defining function
    alias_of: Optional[str] = None   # target python name when kind == "alias"


def _abi_error(message: str) -> TypeError_PP:
    return TypeError_PP(code="PP501", message=message, lineno=0, col_offset=0)


def _is_ufunc(fn: Function) -> bool:
    return isinstance(fn, UFunc) and fn.ufunc_sig is not None


def resolve_function(
    module: Module, name: str, *, _seen: frozenset = frozenset(),
) -> tuple[Optional[Function], Optional[Module]]:
    """Resolve *name* in *module*'s namespace to its defining Function.

    Follows module-level aliases and POST imports transitively (an
    ``__init__`` importing ``gammaln`` that is itself ``gammaln = lgamma``
    in the defining module resolves to ``lgamma``). Returns (None, None)
    for unresolvable names and cycles.
    """
    key = (id(module), name)
    if key in _seen:
        return None, None
    _seen = _seen | {key}

    fn = module.get_function(name)
    if fn is not None:
        return fn, module
    aliased = module.function_aliases.get(name)
    if aliased is not None:
        return resolve_function(module, aliased, _seen=_seen)
    imported = module.post_imports.get(name)
    if imported is not None:
        by_dotted = dict(zip(module.dependencies, module.dep_modules))
        dep = by_dotted.get(imported.module_name)
        if dep is not None:
            return resolve_function(dep, imported.source_name, _seen=_seen)
    return None, None


def collect_exports(modules: list[Module]) -> tuple[list[Export], list[TypeError_PP]]:
    """Resolve the entry translation unit's namespace into an export set.

    The entry unit is the last module in program order. Private names
    (underscore-prefixed) are never exported; a public name whose
    resolution chain ends at a private function is diagnosed rather than
    dropped. An export whose Python name differs from the defining
    function's name has kind ``"alias"``.
    """
    entry = modules[-1]
    errors: list[TypeError_PP] = []
    exports: dict[str, Export] = {}

    def add(python_name: str) -> None:
        if python_name.startswith("_") or python_name in exports:
            return
        fn, module = resolve_function(entry, python_name)
        if fn is None or module is None:
            return
        if fn.name.startswith("_"):
            errors.append(_abi_error(
                f"public name `{python_name}` resolves to private function "
                f"`{fn.name}`, which cannot be exported (spec §9.1)"
            ))
            return
        is_alias = python_name != fn.name
        exports[python_name] = Export(
            python_name=python_name,
            c_symbol=f"pp_{python_name}",
            kernel_symbol=c_symbol(fn.name),
            module=module.name,
            kind="alias" if is_alias else ("ufunc" if _is_ufunc(fn) else "function"),
            target=fn,
            alias_of=fn.name if is_alias else None,
        )

    # Namespace order: imports, own public functions, module aliases.
    for local in entry.post_imports:
        add(local)
    for fn in entry.functions:
        add(fn.name)
    for alias in entry.function_aliases:
        add(alias)

    result = list(exports.values())

    # The pp_* namespace must not collide with kernel symbols.
    kernel_symbols = {
        c_symbol(fn.name) for m in modules for fn in m.functions
    }
    for export in result:
        if export.c_symbol in kernel_symbols:
            errors.append(_abi_error(
                f"export symbol `{export.c_symbol}` collides with a "
                "compiled function of the same name"
            ))
    return result, errors


# ---------------------------------------------------------------------------
# Wrapper translation unit
# ---------------------------------------------------------------------------

_WRAPPER_PREAMBLE = """\
/* AUTO-GENERATED by Post-Py reference compiler. DO NOT EDIT.
   Stable C ABI export wrappers (spec §9.1 Package ABI v1). */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

typedef struct __pp_array {
    void *data;
    int64_t ndim;
    int64_t const *shape;
    int64_t const *strides;  /* byte strides, NumPy-compatible */
    int64_t offset_bytes;
} __pp_array;

"""


def _params_of(fn: Function) -> list:
    return [*fn.params, *fn.core_dim_params]


def emit_export_wrappers(exports: list[Export]) -> str:
    """Emit the C translation unit defining the ``pp_*`` export symbols."""
    em = CEmitter()
    em.write(_WRAPPER_PREAMBLE)

    em.line("/* Kernel symbols defined by the program's translation units. */")
    declared: set[str] = set()
    for export in exports:
        if export.kernel_symbol not in declared:
            emit_function_signature(export.target, em, declaration=True)
            declared.add(export.kernel_symbol)
    em.line()

    for export in exports:
        fn = export.target
        ret = c_type(fn.return_dtype) if fn.return_dtype else "void"
        params = ", ".join(
            f"{c_value_type(p)} _{p.name}" for p in _params_of(fn)
        )
        args = ", ".join(f"_{p.name}" for p in _params_of(fn))
        em.line(f"/* {export.module}.{export.python_name}"
                + (f" (alias of {export.alias_of})" if export.alias_of else "")
                + " */")
        em.line(f"{ret} {export.c_symbol}({params})")
        em.line("{")
        em.indent()
        call = f"{export.kernel_symbol}({args});"
        em.line(f"return {call}" if fn.return_dtype else call)
        em.dedent()
        em.line("}")
        em.line()

    return em.getvalue()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def emit_header(exports: list[Export], artifact_name: str) -> str:
    """Emit a self-contained C99 header for the artifact's stable ABI."""
    guard = f"PP_{artifact_name.upper()}_H"
    em = CEmitter()
    em.line(f"/* AUTO-GENERATED by Post-Py reference compiler. DO NOT EDIT.")
    em.line(f"   Stable C ABI for `{artifact_name}` (spec §9.1 Package ABI v1).")
    em.line("   Link against the artifact's shared library; the pp_* symbols")
    em.line("   below are the supported interface. */")
    em.line(f"#ifndef {guard}")
    em.line(f"#define {guard}")
    em.line()
    em.line("#include <stdint.h>")
    em.line("#include <stdbool.h>")
    em.line("#include <stddef.h>")
    em.line("#include <complex.h>")
    em.line()
    em.line("#ifdef __cplusplus")
    em.line('extern "C" {')
    em.line("#endif")
    em.line()
    em.line("/* POST array view (spec §9.2). Strides are in bytes. */")
    em.line("typedef struct __pp_array {")
    em.indent()
    em.line("void *data;")
    em.line("int64_t ndim;")
    em.line("int64_t const *shape;")
    em.line("int64_t const *strides;")
    em.line("int64_t offset_bytes;")
    em.dedent()
    em.line("} __pp_array;")
    em.line()

    for export in exports:
        fn = export.target
        doc = (fn.doc or "").strip().splitlines()
        note = f" — {doc[0]}" if doc else ""
        alias = f" (alias of {export.alias_of})" if export.alias_of else ""
        em.line(f"/* {export.module}.{export.python_name}{alias}{note} */")
        ret = c_type(fn.return_dtype) if fn.return_dtype else "void"
        params = ", ".join(
            f"{c_value_type(p)} {p.name}" for p in _params_of(fn)
        ) or "void"
        em.line(f"{ret} {export.c_symbol}({params});")
        em.line()

    loops = [e for e in exports if _is_ufunc(e.target) and e.kind != "alias"]
    if loops:
        em.line("/* NumPy ufunc inner loops (spec §8.5). The pointer type of")
        em.line("   dimensions/steps is ABI-compatible with npy_intp. */")
        for export in loops:
            em.line(
                f"void {export.target.name}_ufunc_loop("
                "char **args, intptr_t const *dimensions, "
                "intptr_t const *steps, void *data);"
            )
        em.line()

    em.line("#ifdef __cplusplus")
    em.line("}")
    em.line("#endif")
    em.line()
    em.line(f"#endif /* {guard} */")
    return em.getvalue()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def export_manifest(exports: list[Export], artifact_name: str) -> dict:
    """Machine-readable description of the artifact's stable ABI."""
    entries = []
    for export in exports:
        fn = export.target
        params = [
            {
                "name": p.name,
                "dtype": p.dtype.__name__,
                "is_array": p.is_array,
                "is_core_dim": False,
            }
            for p in fn.params
        ] + [
            {
                "name": p.name,
                "dtype": p.dtype.__name__,
                "is_array": False,
                "is_core_dim": True,
            }
            for p in fn.core_dim_params
        ]
        ufunc_info = None
        if _is_ufunc(fn):
            ufunc_info = {
                "loop_symbol": f"{fn.name}_ufunc_loop",
                "signature": str(fn.ufunc_sig),
            }
        entries.append({
            "name": export.python_name,
            "c_symbol": export.c_symbol,
            "kernel_symbol": export.kernel_symbol,
            "module": export.module,
            "kind": export.kind,
            "alias_of": export.alias_of,
            "params": params,
            "return_dtype": fn.return_dtype.__name__ if fn.return_dtype else None,
            "ufunc": ufunc_info,
            "doc": fn.doc,
        })
    return {
        "post_abi": 1,
        "artifact": artifact_name,
        "exports": entries,
    }
