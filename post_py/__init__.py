"""Post-Py — Performance Optimized Statically Typed Python."""

# Make the repo root importable so submodules can `from postyp import ...`.
# Done once here, before any submodules are imported.
import os as _os
import sys as _sys

_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_repo_root, _os.path.join(_repo_root, "postyp-dist")):
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)

from .ufunc import guvectorize, vectorize

__all__ = ["guvectorize", "vectorize"]
