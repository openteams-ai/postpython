"""post-py CLI: check and build subcommands (docs/distribution.md)."""

import ctypes
import json
import platform
import shutil

import pytest

from postpyc.cli import main

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")

_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"

GOOD = (
    "from postyp import Float64\n"
    "def triple(x: Float64) -> Float64:\n"
    "    return x * 3.0\n"
)


def test_check_clean_file_returns_zero(tmp_path, capsys):
    path = tmp_path / "good.py"
    path.write_text(GOOD)
    assert main(["check", str(path)]) == 0


def test_check_violations_return_nonzero(tmp_path, capsys):
    path = tmp_path / "bad.py"
    path.write_text("def f(x):\n    return eval('x')\n")
    assert main(["check", str(path)]) == 1
    out = capsys.readouterr().out
    assert "PP002" in out


@needs_cc
def test_build_produces_artifact(tmp_path, capsys):
    src = tmp_path / "kern.py"
    src.write_text(GOOD)
    out = tmp_path / "kern.so"
    assert main(["build", str(src), "--output", str(out)]) == 0
    assert out.exists()
    lib = ctypes.CDLL(str(out))
    fn = lib.pp_triple
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(2.0) == 6.0


@needs_cc
def test_build_error_reports_and_fails(tmp_path, capsys):
    src = tmp_path / "broken.py"
    src.write_text(
        "from postyp import Float64\n"
        "def f(x: Float64) -> Float64:\n"
        "    return unknown_fn(x)\n"
    )
    assert main(["build", str(src), "--output", str(tmp_path / "x.so")]) == 1
    assert "PP502" in capsys.readouterr().err


@needs_cc
def test_build_prefix_layout(tmp_path, capsys):
    pkg = tmp_path / "ppdemo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(GOOD)
    prefix = tmp_path / "prefix"

    assert main(["build", str(pkg / "__init__.py"), "--prefix", str(prefix)]) == 0

    lib = prefix / "lib" / f"libppdemo{_LIB_SUFFIX}"
    header = prefix / "include" / "ppdemo.h"
    manifest_path = prefix / "share" / "post-py" / "ppdemo.json"
    assert lib.exists() and header.exists() and manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "ppdemo"
    assert "double pp_triple(double x);" in header.read_text()

    handle = ctypes.CDLL(str(lib))
    fn = handle.pp_triple
    fn.argtypes = [ctypes.c_double]
    fn.restype = ctypes.c_double
    assert fn(1.5) == 4.5


def test_build_prefix_rejects_ext_module(tmp_path, capsys):
    src = tmp_path / "kern.py"
    src.write_text(GOOD)
    code = main([
        "build", str(src), "--prefix", str(tmp_path / "p"), "--ext-module",
    ])
    assert code == 2
    assert "separate build targets" in capsys.readouterr().err
