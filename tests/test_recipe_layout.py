"""Live test of the distribution layout and reference recipe (issue #14).

Runs the exact build commands examples/recipe/recipe.yaml gives the
package manager — `post-py build ... --prefix $PREFIX` for the
libppdemo output and `--ext-module --output $SP_DIR/...` for the ppdemo
output — against examples/ppdemo, then consumes every artifact the way
its downstream consumers do: ctypes through the manifest, a compiled C
program through the installed header and library, and a NumPy import of
the extension module.
"""

import ctypes
import json
import math
import platform
import shutil
import subprocess
import sys

import pytest

from postpyc.cli import main

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
needs_cc = pytest.mark.skipif(cc is None, reason="No C compiler available")

_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "examples" / "ppdemo"
RECIPE = ROOT / "examples" / "recipe" / "recipe.yaml"

# The commands the recipe scripts run, modulo $PREFIX / $SP_DIR.
LIB_BUILD_ARGS = ["build", str(PKG_DIR / "__init__.py"), "--prefix"]
EXT_BUILD_ARGS = [
    "build", str(PKG_DIR / "__init__.py"),
    "--ext-module", "--module-name", "ppdemo_native", "--output",
]


@pytest.fixture(scope="module")
def prefix(tmp_path_factory):
    """A $PREFIX populated exactly as the libppdemo recipe output does."""
    prefix = tmp_path_factory.mktemp("prefix")
    assert main([*LIB_BUILD_ARGS, str(prefix)]) == 0
    return prefix


# ---------------------------------------------------------------------------
# libppdemo: install layout
# ---------------------------------------------------------------------------

@needs_cc
def test_prefix_layout_is_exactly_as_documented(prefix):
    expected = {
        f"lib/libppdemo{_LIB_SUFFIX}",
        "include/ppdemo.h",
        "share/post-py/ppdemo.json",
    }
    produced = {
        str(p.relative_to(prefix)) for p in prefix.rglob("*") if p.is_file()
    }
    assert produced == expected


@needs_cc
def test_manifest_is_the_machine_readable_contract(prefix):
    manifest = json.loads(
        (prefix / "share" / "post-py" / "ppdemo.json").read_text()
    )
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "ppdemo"

    exports = {e["name"]: e for e in manifest["exports"]}
    assert set(exports) == {"smoothstep", "mix", "logistic", "lerp"}
    assert exports["smoothstep"]["kind"] == "function"
    assert exports["logistic"]["kind"] == "ufunc"
    assert exports["logistic"]["ufunc"]["loop_symbol"] == "logistic_ufunc_loop"
    assert exports["lerp"]["kind"] == "alias"
    assert exports["lerp"]["c_symbol"] == "pp_lerp"
    # The package's private helper is internal linkage, never exported.
    assert "_clip01" not in exports


@needs_cc
def test_header_declares_the_stable_abi(prefix):
    header = (prefix / "include" / "ppdemo.h").read_text()
    assert "double pp_smoothstep(double x);" in header
    assert "double pp_mix(double x, double y, double t);" in header
    assert "double pp_logistic(double x);" in header
    assert "double pp_lerp(double x, double y, double t);" in header
    assert "typedef struct __pp_array" in header
    assert "logistic_ufunc_loop" in header


@needs_cc
def test_library_consumed_via_manifest_and_ctypes(prefix):
    # Consume the manifest instead of guessing symbol names — the rule
    # docs/distribution.md sets for recipes and bindings.
    manifest = json.loads(
        (prefix / "share" / "post-py" / "ppdemo.json").read_text()
    )
    lib = ctypes.CDLL(str(prefix / "lib" / f"libppdemo{_LIB_SUFFIX}"))
    bound = {}
    for export in manifest["exports"]:
        if export["kind"] not in ("function", "alias", "ufunc"):
            continue
        fn = getattr(lib, export["c_symbol"])
        fn.argtypes = [ctypes.c_double] * len(export["params"])
        fn.restype = ctypes.c_double
        bound[export["name"]] = fn

    assert bound["smoothstep"](0.5) == 0.5
    assert bound["smoothstep"](-1.0) == 0.0
    assert bound["smoothstep"](2.0) == 1.0
    assert bound["mix"](0.0, 10.0, 0.25) == 2.5
    assert bound["lerp"](0.0, 10.0, 0.25) == 2.5   # alias delegates
    assert bound["logistic"](0.0) == 0.5


@needs_cc
def test_header_and_library_work_from_c(prefix, tmp_path):
    # libppdemo's whole point: a C consumer with no Python anywhere.
    consumer = tmp_path / "consumer.c"
    consumer.write_text(
        '#include <stdio.h>\n'
        '#include "ppdemo.h"\n'
        'int main(void) {\n'
        '    printf("%.6f %.6f\\n", pp_smoothstep(0.5), pp_lerp(0.0, 10.0, 0.25));\n'
        '    return 0;\n'
        '}\n'
    )
    exe = tmp_path / "consumer"
    subprocess.run(
        [cc, str(consumer), f"-I{prefix / 'include'}",
         f"-L{prefix / 'lib'}", "-lppdemo", "-o", str(exe)],
        check=True, capture_output=True, text=True,
    )
    env_var = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
    result = subprocess.run(
        [str(exe)], check=True, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", env_var: str(prefix / "lib")},
    )
    assert result.stdout.split() == ["0.500000", "2.500000"]


# ---------------------------------------------------------------------------
# ppdemo: the Python output (extension module into $SP_DIR)
# ---------------------------------------------------------------------------

@needs_cc
def test_ext_module_output_imports_and_matches(tmp_path):
    np = pytest.importorskip("numpy")
    site_dir = tmp_path / "site-packages"
    site_dir.mkdir()
    assert main([*EXT_BUILD_ARGS, str(site_dir / "ppdemo_native.so")]) == 0
    shutil.copytree(PKG_DIR, site_dir / "ppdemo")

    sys.path.insert(0, str(site_dir))
    try:
        import ppdemo_native
        assert isinstance(ppdemo_native.logistic, np.ufunc)
        x = np.linspace(-6.0, 6.0, 101)
        expected = 1.0 / (1.0 + np.exp(-x))
        assert np.allclose(ppdemo_native.logistic(x), expected, rtol=1e-15)
        assert ppdemo_native.logistic(0.0) == 0.5

        # The manifest's boundary-code loader prefers the native module
        # when it is importable (the shape real pp* packages ship).
        import ppdemo
        assert ppdemo.__native_available__ is True
        assert ppdemo.logistic(0.36) == 1.0 / (1.0 + math.exp(-0.36))
        assert ppdemo.smoothstep(0.5) == 0.5
        assert ppdemo.lerp(0.0, 10.0, 0.25) == 2.5

        # Interpreted mode is the package: the pure kernels must agree.
        from ppdemo import _kernels
        assert _kernels.logistic(0.36) == 1.0 / (1.0 + math.exp(-0.36))
        assert _kernels.smoothstep(0.5) == 0.5
        assert _kernels.mix(0.0, 10.0, 0.25) == 2.5
    finally:
        sys.path.remove(str(site_dir))
        sys.modules.pop("ppdemo_native", None)
        sys.modules.pop("ppdemo", None)
        sys.modules.pop("ppdemo._kernels", None)


# ---------------------------------------------------------------------------
# The recipe file itself
# ---------------------------------------------------------------------------

def test_recipe_runs_the_commands_this_suite_tested():
    text = RECIPE.read_text()
    # The libppdemo script is the command the prefix fixture ran.
    assert "post-py build ppdemo/__init__.py --prefix $PREFIX" in text
    # The ppdemo script is the command the ext-module test ran.
    assert "--module-name ppdemo_native" in text
    assert "--output $SP_DIR/ppdemo_native.so" in text
    # Split-package shape, per docs/distribution.md.
    assert "name: libppdemo" in text
    assert "name: ppdemo" in text


def test_recipe_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    recipe = yaml.safe_load(RECIPE.read_text())
    assert [o["package"]["name"] for o in recipe["outputs"]] == [
        "libppdemo", "ppdemo",
    ]


@pytest.mark.skipif(shutil.which("rattler-build") is None,
                    reason="rattler-build not installed")
def test_recipe_renders_with_rattler_build(tmp_path):
    # Until postpyc itself is published to a conda channel, the full
    # build cannot solve its build environment; what must already hold is
    # that the recipe renders (context, jinja, outputs) and reaches
    # dependency resolution rather than dying on a recipe error.
    result = subprocess.run(
        ["rattler-build", "build", "--recipe", str(RECIPE),
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        blob = result.stdout + result.stderr
        assert "No candidates were found for postpyc" in blob, blob[-4000:]
        pytest.skip("recipe renders; postpyc not on a conda channel yet")
