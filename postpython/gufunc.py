"""POST Python @gufunc decorator.

Marks a function as a generalized ufunc with a NumPy-compatible broadcast
signature.  At runtime (interpreted mode) the decorator wraps the function
in a pure-Python broadcast loop so it remains callable and testable without
the AOT compiler.  When the POST Python compiler processes the source file it
recognises the decorator and lowers the function to the NumPy gufunc C ABI.

Usage::

    from postyp import Array, Float64
    from postpython.gufunc import gufunc

    @gufunc("(n),(n)->()")
    def dot(a: Array[Float64], b: Array[Float64]) -> Float64:
        result: Float64 = 0.0
        for i in range(len(a)):
            result += a[i] * b[i]
        return result
"""

from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Callable, get_type_hints

from .compiler.ir import GUFuncSignature


# ---------------------------------------------------------------------------
# Signature parsing (re-exported from frontend for convenience)
# ---------------------------------------------------------------------------

def parse_gufunc_sig(sig: str) -> GUFuncSignature:
    """Parse a gufunc signature string into a GUFuncSignature."""
    from .compiler.frontend import parse_gufunc_sig as _parse
    return _parse(sig)


def _positional_params(fn: Callable) -> list[inspect.Parameter]:
    """Return positional parameters accepted by *fn*."""
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    return [
        param
        for param in inspect.signature(fn).parameters.values()
        if param.kind in positional_kinds
    ]


def _resolved_type_hints(fn: Callable) -> dict[str, Any]:
    """Return resolved annotations, falling back to raw annotations."""
    try:
        return get_type_hints(fn)
    except Exception:
        return getattr(fn, "__annotations__", {})


def _np_dtype_for_annotation(annotation: Any, np: Any) -> Any:
    """Map a postyp scalar or array annotation to a NumPy dtype."""
    try:
        from postyp import Array, Bool, DType
    except ImportError:
        return None

    dtype = None
    if isinstance(annotation, type):
        if issubclass(annotation, DType):
            dtype = annotation
        elif issubclass(annotation, Array):
            dtype = getattr(annotation, "dtype", None)

    if dtype is None:
        return None
    if dtype is Bool:
        return np.dtype(np.bool_)
    if dtype.itemsize == 0 or dtype.kind not in {"i", "u", "f", "c"}:
        return None
    return np.dtype(f"{dtype.kind}{dtype.itemsize}")


def _output_dtypes(
    fn: Callable,
    sig: GUFuncSignature,
    n_inputs: int,
    output_param_count: int,
    np: Any,
) -> list[Any]:
    """Infer NumPy dtypes for gufunc outputs from function annotations."""
    hints = _resolved_type_hints(fn)
    n_out = len(sig.outputs)

    if output_param_count:
        params = _positional_params(fn)
        anns = [
            hints.get(param.name)
            for param in params[n_inputs : n_inputs + n_out]
        ]
    elif n_out == 1:
        anns = [hints.get("return")]
    else:
        anns = []

    dtypes = [_np_dtype_for_annotation(ann, np) for ann in anns]
    return dtypes + [None] * (n_out - len(dtypes))


# ---------------------------------------------------------------------------
# Runtime broadcast loop (interpreted mode)
# ---------------------------------------------------------------------------

def _broadcast_call(fn: Callable, sig: GUFuncSignature, args: tuple) -> Any:
    """Execute *fn* with a pure-Python broadcast loop.

    This mirrors what the compiled gufunc wrapper does in C, for use when
    running under the standard interpreter.

    Requires NumPy for ndarray broadcasting if array inputs are present;
    falls back to direct calls when NumPy is unavailable.
    """
    n_inputs = len(sig.inputs)

    # Array inputs: attempt NumPy-backed broadcast.
    try:
        import numpy as np
        input_arrays = args[:n_inputs]
        output_args  = args[n_inputs:]  # pre-allocated output arrays, if any
        n_out = len(sig.outputs)
        output_param_count = max(0, len(_positional_params(fn)) - n_inputs)
        kernel_expects_outputs = output_param_count > 0

        if output_args and len(output_args) != n_out:
            raise TypeError(
                f"gufunc expected {n_out} output argument(s), "
                f"got {len(output_args)}"
            )
        if kernel_expects_outputs and output_param_count != n_out:
            raise TypeError(
                f"gufunc kernel has {output_param_count} output parameter(s), "
                f"but signature declares {n_out} output(s)"
            )

        if not input_arrays:
            return fn(*args)

        n_core_in = [len(d) for d in sig.inputs]
        arrays = [np.asarray(a) for a in input_arrays]

        # Determine outer (batch) dimensions by stripping core dims, and infer
        # named core dimension sizes from the inputs.
        core_sizes: dict[str, int] = {}
        batch_shapes = []
        for arr, dims, nc in zip(arrays, sig.inputs, n_core_in):
            if arr.ndim < nc:
                raise ValueError(
                    f"gufunc input has {arr.ndim} dimensions, but signature "
                    f"requires {nc} core dimensions"
                )
            batch_shape = arr.shape[:-nc] if nc else arr.shape
            core_shape = arr.shape[-nc:] if nc else ()
            batch_shapes.append(batch_shape)
            for name, size in zip(dims, core_shape):
                known = core_sizes.setdefault(name, size)
                if known != size:
                    raise ValueError(
                        f"gufunc core dimension {name!r} has inconsistent "
                        f"sizes: {known} vs {size}"
                    )

        out_batch_shape = np.broadcast_shapes(*batch_shapes) if batch_shapes else ()

        if not out_batch_shape and not output_args and not kernel_expects_outputs:
            # No batch dims — single call.
            return fn(*args)

        n_core_out = [len(dims) for dims in sig.outputs]
        output_core_shapes = []
        for dims in sig.outputs:
            try:
                output_core_shapes.append(tuple(core_sizes[name] for name in dims))
            except KeyError as exc:
                raise ValueError(
                    f"cannot infer gufunc output core dimension {exc.args[0]!r}"
                ) from exc

        if kernel_expects_outputs and any(not shape for shape in output_core_shapes):
            raise TypeError(
                "interpreted @gufunc does not support scalar write-through "
                "output parameters; return scalar output values instead"
            )

        broadcast_inputs = [
            np.broadcast_to(arr, out_batch_shape + (arr.shape[-nc:] if nc else ()))
            for arr, nc in zip(arrays, n_core_in)
        ]

        output_dtypes = _output_dtypes(fn, sig, n_inputs, output_param_count, np)
        results = []
        for i, core_shape in enumerate(output_core_shapes):
            shape = out_batch_shape + core_shape
            if i < len(output_args):
                out = np.asarray(output_args[i])
                if out.shape != shape:
                    raise ValueError(
                        f"gufunc output {i} has shape {out.shape}, expected {shape}"
                    )
            else:
                dtype = output_dtypes[i]
                out = (
                    np.empty(shape, dtype=dtype)
                    if dtype is not None
                    else np.empty(shape)
                )
            results.append(out)

        # Iterate over batch indices and call the scalar/core Python kernel.
        core_slices = [((slice(None),) * nc) for nc in n_core_in]
        for idx in np.ndindex(out_batch_shape):
            call_inputs = tuple(
                arr[idx + core_slice] if nc else arr[idx]
                for arr, nc, core_slice in zip(broadcast_inputs, n_core_in, core_slices)
            )
            call_outputs = tuple(
                out[idx + ((slice(None),) * nc)] if nc else out[idx]
                for out, nc in zip(results, n_core_out)
            ) if kernel_expects_outputs else ()
            ret = fn(*(call_inputs + call_outputs))
            if kernel_expects_outputs:
                continue
            if n_out == 1:
                results[0][idx] = ret
            elif n_out > 1:
                for out, value in zip(results, ret):
                    out[idx] = value

        if n_out == 1:
            return results[0]
        return tuple(results)

    except ImportError:
        # NumPy not available — fall back to direct scalar call.
        return fn(*args[:n_inputs])


# ---------------------------------------------------------------------------
# GUFuncWrapper
# ---------------------------------------------------------------------------

class GUFuncWrapper:
    """Wraps a POST Python function decorated with @gufunc.

    Attributes
    ----------
    __wrapped__   : the original Python function
    __pp_sig__    : the raw signature string, e.g. "(n),(n)->()"
    __pp_gufunc_sig__ : parsed GUFuncSignature
    """

    def __init__(self, fn: Callable, sig_str: str) -> None:
        self.__wrapped__         = fn
        self.__pp_sig__          = sig_str
        self.__pp_gufunc_sig__   = parse_gufunc_sig(sig_str)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _broadcast_call(self.__wrapped__, self.__pp_gufunc_sig__, args)

    def __repr__(self) -> str:
        return f"<gufunc {self.__wrapped__.__name__!r} sig={self.__pp_sig__!r}>"

    # Provide a numpy gufunc if numpy is available.
    def as_numpy_gufunc(self) -> Any:
        """Return a numpy.vectorize wrapper with the gufunc signature."""
        try:
            import numpy as np
            return np.vectorize(self.__wrapped__, signature=self.__pp_sig__)
        except ImportError:
            raise ImportError("NumPy is required for as_numpy_gufunc()")


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def gufunc(signature: str) -> Callable[[Callable], GUFuncWrapper]:
    """Decorator that marks a function as a POST Python generalized ufunc.

    Parameters
    ----------
    signature : str
        NumPy-style gufunc signature, e.g. ``"(n),(n)->()"`` for a dot
        product, ``"(m,k),(k,n)->(m,n)"`` for matrix multiply.

    Returns
    -------
    GUFuncWrapper
        A callable that behaves like the original function for scalar inputs
        and broadcasts over array inputs (using NumPy if available).  The
        POST Python compiler recognises `GUFuncWrapper` and lowers it to the
        NumPy C ufunc ABI.

    Examples
    --------
    ::

        @gufunc("(n),(n)->()")
        def dot(a: Array[Float64], b: Array[Float64]) -> Float64:
            result: Float64 = 0.0
            for i in range(len(a)):
                result += a[i] * b[i]
            return result

        # Works as Python function (interpreted):
        dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])  # → 32.0
    """
    # Validate signature string at decoration time.
    try:
        parse_gufunc_sig(signature)
    except ValueError as exc:
        raise ValueError(f"@gufunc: invalid signature {signature!r}: {exc}") from exc

    def decorator(fn: Callable) -> GUFuncWrapper:
        return GUFuncWrapper(fn, signature)

    return decorator
