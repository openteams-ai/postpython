"""CPython extension-module shim emitter (spec §9.3, §7.5).

Generates the C source for an importable extension module that registers
each compiled ufunc loop with NumPy via PyUFunc_FromFuncAndData /
PyUFunc_FromFuncAndDataAndSignature. The loops themselves live in the
program's ordinary translation units (emitted by c_backend); this shim
only declares them extern and builds the module namespace.

Registered names mirror the entry translation unit's Python namespace:
its own public ufuncs plus ufuncs it imports from other POST modules
(under their local alias names).

The GIL contract of §7.5 comes from NumPy itself: the ufunc machinery
releases the GIL around registered inner loops for non-object dtypes.

Note: the NumPy ufunc C API is not part of the CPython limited API, so
the shim targets the full API (see spec §12).
"""

from __future__ import annotations

from .. import dimexpr
from ..ir import Module, UFunc

# sys.path setup happens once in postpyc/__init__.py.
import postpyc  # noqa: F401  -- ensure path setup runs
from postyp import (
    DType,
    Bool, Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, Complex64, Complex128,
)


_NPY_TYPE: dict[type[DType], str] = {
    Bool:       "NPY_BOOL",
    Int8:       "NPY_INT8",
    Int16:      "NPY_INT16",
    Int32:      "NPY_INT32",
    Int64:      "NPY_INT64",
    UInt8:      "NPY_UINT8",
    UInt16:     "NPY_UINT16",
    UInt32:     "NPY_UINT32",
    UInt64:     "NPY_UINT64",
    Float16:    "NPY_HALF",
    Float32:    "NPY_FLOAT32",
    Float64:    "NPY_FLOAT64",
    Complex64:  "NPY_COMPLEX64",
    Complex128: "NPY_COMPLEX128",
}


class ExtModuleError(ValueError):
    """A ufunc cannot be registered with the NumPy type system."""


def _c_string(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _npy_type(dtype: type[DType], where: str) -> str:
    try:
        return _NPY_TYPE[dtype]
    except KeyError:
        raise ExtModuleError(
            f"dtype `{dtype.__name__}` in {where} has no NumPy ufunc type"
        ) from None


def _ufunc_type_chars(fn: UFunc) -> tuple[int, int, list[str]]:
    """Return (nin, nout, NPY type constants) for a ufunc registration."""
    sig = fn.ufunc_sig
    assert sig is not None
    n_in = len(sig.inputs)
    n_out = len(sig.outputs)
    types: list[str] = []
    for p in fn.params[:n_in]:
        types.append(_npy_type(p.dtype, f"ufunc `{fn.name}` input `{p.name}`"))
    if fn.return_dtype is not None:
        # @vectorize: scalar return is the single output.
        types.append(_npy_type(fn.return_dtype, f"ufunc `{fn.name}` return"))
    else:
        for p in fn.params[n_in:n_in + n_out]:
            types.append(_npy_type(p.dtype, f"ufunc `{fn.name}` output `{p.name}`"))
    return n_in, n_out, types


def collect_registrations(modules: list[Module]) -> list[tuple[str, UFunc]]:
    """(python_name, ufunc) pairs exposed by the entry translation unit.

    The entry unit is the last module in program order. Its namespace is
    mirrored: ufuncs imported from POST dependencies register under their
    local names, the unit's own public ufuncs, and module-level function
    aliases (``gammaln = lgamma``) as separately-named ufuncs — including
    aliases defined in imported modules and re-exported by the entry.
    """
    from .abi import resolve_function

    entry = modules[-1]
    registrations: dict[str, UFunc] = {}

    def add(python_name: str) -> None:
        if python_name.startswith("_") or python_name in registrations:
            return
        fn, _ = resolve_function(entry, python_name)
        if isinstance(fn, UFunc) and fn.ufunc_sig is not None and not fn.name.startswith("_"):
            registrations[python_name] = fn

    for local in entry.post_imports:
        add(local)
    for fn in entry.functions:
        add(fn.name)
    for alias in entry.function_aliases:
        add(alias)
    if entry.export_all is not None:
        allowed = set(entry.export_all)
        return [(n, fn) for n, fn in registrations.items() if n in allowed]
    return list(registrations.items())


def _process_core_dims_symbol(py_name: str) -> str:
    return f"_pp_{py_name}_process_core_dims"


def _emit_process_core_dims(py_name: str, fn: UFunc) -> list[str]:
    """Emit a NumPy process_core_dims_func for *fn*'s computed output dims.

    The callback runs before the gufunc allocates outputs: NumPy fills
    ``core_dim_sizes`` with the input-derived dimensions, leaving any
    output-only dimension the caller did not size as -1. We evaluate each
    computed expression (the same dimexpr the interpreted path uses) and
    fill the slot, or validate a caller-provided output whose size is fixed.
    """
    sig = fn.ufunc_sig
    assert sig is not None and sig.computed_dims
    order = sig.core_dims  # NumPy assigns core_dim_ix in this same order.

    needed = sorted(
        {name for expr in sig.computed_dims.values() for name in dimexpr.free_names(expr)}
    )
    lines: list[str] = []
    w = lines.append
    w(f"static int")
    w(f"{_process_core_dims_symbol(py_name)}(PyUFuncObject *ufunc, npy_intp *core_dim_sizes)")
    w("{")
    w("    (void)ufunc;")
    w("    int _pp_ovf = 0;")
    for name in needed:
        w(f"    int64_t _pp_cd_{name} = core_dim_sizes[{order.index(name)}];")
    for cd_name, expr in sig.computed_dims.items():
        ix = order.index(cd_name)
        rendered = dimexpr.render(expr)
        cexpr = dimexpr.to_c_checked(expr, lambda nm: f"_pp_cd_{nm}", "_pp_ovf")
        var = f"_pp_computed_{cd_name}"
        w(f"    int64_t {var} = {cexpr};")
        w(f"    if (_pp_ovf) {{")
        w(f"        PyErr_Format(PyExc_OverflowError,")
        w(f'            "{py_name}: computed output core dimension \'{cd_name}\' '
          f'({rendered}) overflowed int64");')
        w(f"        return -1;")
        w(f"    }}")
        w(f"    if ({var} < 0) {{")
        w(f"        PyErr_Format(PyExc_ValueError,")
        w(f'            "{py_name}: computed output core dimension \'{cd_name}\' '
          f'({rendered}) is negative (%zd)",')
        w(f"            (Py_ssize_t){var});")
        w(f"        return -1;")
        w(f"    }}")
        w(f"    if (core_dim_sizes[{ix}] == -1) {{")
        w(f"        core_dim_sizes[{ix}] = {var};")
        w(f"    }} else if (core_dim_sizes[{ix}] != {var}) {{")
        w(f"        PyErr_Format(PyExc_ValueError,")
        w(f'            "{py_name}: output core dimension \'{cd_name}\' has size %zd, '
          f'but the layout signature requires {rendered} = %zd",')
        w(f"            (Py_ssize_t)core_dim_sizes[{ix}], (Py_ssize_t){var});")
        w(f"        return -1;")
        w(f"    }}")
    w("    return 0;")
    w("}")
    return lines


def emit_ext_module(modules: list[Module], module_name: str) -> str:
    """Emit the extension-module shim C source for a compiled program."""
    if not module_name.isidentifier():
        raise ExtModuleError(
            f"extension module name {module_name!r} is not a valid identifier"
        )
    registrations = collect_registrations(modules)

    # Gufuncs with computed output core dims need process_core_dims_func,
    # a PyUFuncObject field added in the NumPy 2.1 C API — so the shim must
    # target at least that version to see it. Ordinary builds keep the
    # broadly-compatible 1.7 target.
    computed = [
        (py_name, fn) for py_name, fn in registrations
        if fn.ufunc_sig is not None and fn.ufunc_sig.computed_dims
    ]

    lines: list[str] = []
    w = lines.append
    w("/* AUTO-GENERATED by POST Python reference compiler. DO NOT EDIT. */")
    w("#define PY_SSIZE_T_CLEAN")
    w("#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
    if computed:
        w("#define NPY_TARGET_VERSION NPY_2_1_API_VERSION")
    w("#include <Python.h>")
    w("#include <numpy/arrayobject.h>")
    w("#include <numpy/ufuncobject.h>")
    w("")
    w("/* Ufunc loops defined in the program's translation units. */")
    for _, fn in registrations:
        w(
            f"extern void {fn.name}_ufunc_loop("
            "char **args, npy_intp const *dimensions, "
            "npy_intp const *steps, void *data);"
        )
    w("")

    # Computed output core dims: a process_core_dims_func per gufunc that
    # declares one, so NumPy sizes the output before the loop runs.
    if computed:
        w(dimexpr.FLOORDIV_SI_C)
        w(dimexpr.CHECKED_ARITH_C)
        for py_name, fn in computed:
            lines.extend(_emit_process_core_dims(py_name, fn))
            w("")

    for py_name, fn in registrations:
        _, _, types = _ufunc_type_chars(fn)
        w(f"static PyUFuncGenericFunction _pp_{py_name}_funcs[1] = {{&{fn.name}_ufunc_loop}};")
        w(f"static char _pp_{py_name}_types[{len(types)}] = {{{', '.join(types)}}};")
        w(f"static void *_pp_{py_name}_data[1] = {{NULL}};")
        w("")

    w("static struct PyModuleDef _pp_moduledef = {")
    w("    PyModuleDef_HEAD_INIT,")
    w(f"    {_c_string(module_name)},")
    w(f"    {_c_string('POST Python compiled extension module.')},")
    w("    -1,")
    w("    NULL, NULL, NULL, NULL, NULL,")
    w("};")
    w("")
    w("PyMODINIT_FUNC")
    w(f"PyInit_{module_name}(void)")
    w("{")
    w("    PyObject *m = PyModule_Create(&_pp_moduledef);")
    w("    if (m == NULL) return NULL;")
    w("    import_array();")
    w("    import_umath();")
    w("")
    w("    PyObject *uf;")
    for py_name, fn in registrations:
        n_in, n_out, _ = _ufunc_type_chars(fn)
        doc = _c_string(fn.doc or f"POST Python ufunc `{py_name}`.")
        sig = fn.ufunc_sig
        assert sig is not None
        is_gufunc = any(sig.inputs) or any(sig.outputs)
        if is_gufunc:
            w("    uf = PyUFunc_FromFuncAndDataAndSignature(")
            w(f"        _pp_{py_name}_funcs, _pp_{py_name}_data, _pp_{py_name}_types,")
            w(f"        1, {n_in}, {n_out}, PyUFunc_None,")
            w(f"        {_c_string(py_name)}, {doc}, 0, {_c_string(str(sig))});")
        else:
            w("    uf = PyUFunc_FromFuncAndData(")
            w(f"        _pp_{py_name}_funcs, _pp_{py_name}_data, _pp_{py_name}_types,")
            w(f"        1, {n_in}, {n_out}, PyUFunc_None,")
            w(f"        {_c_string(py_name)}, {doc}, 0);")
        if sig.computed_dims:
            w("    if (uf != NULL) {")
            w(f"        ((PyUFuncObject *)uf)->process_core_dims_func = "
              f"&{_process_core_dims_symbol(py_name)};")
            w("    }")
        w(f"    if (uf == NULL || PyModule_AddObject(m, {_c_string(py_name)}, uf) < 0) {{")
        w("        Py_XDECREF(uf);")
        w("        Py_DECREF(m);")
        w("        return NULL;")
        w("    }")
        w("")
    w("    return m;")
    w("}")
    return "\n".join(lines) + "\n"
