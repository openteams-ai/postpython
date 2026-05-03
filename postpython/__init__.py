"""POST Python — Performance Optimized Statically Typed Python."""

# Make the repo root importable so submodules can `from postyp import ...`.
# Done once here, before any submodules are imported.
import os as _os
import sys as _sys

_repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from .ufunc import guvectorize, vectorize

__all__ = ["guvectorize", "vectorize"]
