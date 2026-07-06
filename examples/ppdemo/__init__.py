"""ppdemo — the reference POST Python package for docs/distribution.md.

A deliberately small package with the shape every pp* package shares:

  - a private kernels module (`ppdemo._kernels`) compiled as its own
    translation unit, holding the plain functions (`smoothstep`, `mix`)
    and a `@vectorize` ufunc (`logistic`) lowering through libm,
  - this ``__init__.py``, the package's namespace manifest (spec §9.1):
    its POST imports, the module-level alias (`lerp = mix`, exported as
    its own pp_* symbol), and ``__all__`` define the compiled namespace,
  - a dynamic native-preferring loader — CPython-boundary code the
    compiler neither checks nor lowers; it runs in interpreted mode only.

So the export manifest exercises every kind: `function` (`smoothstep`,
`mix`), `ufunc` (`logistic`), and `alias` (`lerp`).

The reference recipe in examples/recipe/recipe.yaml builds it as the
split `libppdemo` + `ppdemo` pair; tests/test_recipe_layout.py exercises
that build end to end.
"""

from importlib import import_module as _import_module

from ppdemo._kernels import logistic, mix, smoothstep

lerp = mix

__all__ = ["lerp", "logistic", "mix", "smoothstep"]

__native_available__ = False


def _prefer_native() -> None:
    global __native_available__

    try:
        native = _import_module("ppdemo_native")
    except ModuleNotFoundError:
        return

    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
    __native_available__ = True


_prefer_native()

del _prefer_native, _import_module
