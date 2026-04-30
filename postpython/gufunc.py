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
import re
from typing import Any, Callable

from .compiler.ir import GUFuncSignature


# ---------------------------------------------------------------------------
# Signature parsing (re-exported from frontend for convenience)
# ---------------------------------------------------------------------------

def parse_gufunc_sig(sig: str) -> GUFuncSignature:
    """Parse a gufunc signature string into a GUFuncSignature."""
    from .compiler.frontend import parse_gufunc_sig as _parse
    return _parse(sig)


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

        if not out_batch_shape:
            # No batch dims — single call.
            return fn(*args)

        n_out = len(sig.outputs)
        n_core_out = [len(dims) for dims in sig.outputs]
        output_core_shapes = []
        for dims in sig.outputs:
            try:
                output_core_shapes.append(tuple(core_sizes[name] for name in dims))
            except KeyError as exc:
                raise ValueError(
                    f"cannot infer gufunc output core dimension {exc.args[0]!r}"
                ) from exc

        broadcast_inputs = [
            np.broadcast_to(arr, out_batch_shape + (arr.shape[-nc:] if nc else ()))
            for arr, nc in zip(arrays, n_core_in)
        ]

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
                out = np.empty(shape)
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
            ) if output_args else ()
            ret = fn(*(call_inputs + call_outputs))
            if n_out == 1 and not output_args:
                results[0][idx] = ret
            elif n_out > 1 and not output_args:
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
