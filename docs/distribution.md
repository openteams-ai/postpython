# Distributing POST Python Packages

How compiled POST Python libraries (the `pp*` family) reach users. This
guide is policy for packages under the PostSciPy effort and a reference
for anyone shipping POST Python code.

## The policy, in one paragraph

POST packages publish **pure source** to PyPI — `py3-none-any` wheels,
interpreted mode, fully functional. **No binary wheels are published,
ever.** The recommended path to compiled performance is an environment
package manager that treats native code as a first-class dependency —
pixi/conda, nix, spack — where each package ships as a real system
library plus a Python binding. Building locally is always legitimate
too: anyone with a functional POST-compatible compiler chain (a
conforming POST compiler such as the reference `postpyc`, plus a C
toolchain) can compile the pure package they installed — explicitly,
never behind their back. Package documentation points to the
environment managers by default.

## Why no binary wheels

Binary wheels exist because most compiled Python packages have no
fallback: without a prebuilt binary there is no package at all. That
forces vendoring — private copies of native libraries grafted into every
wheel — which yields N incoherent copies of the same library per
environment and no solver that understands any of them.

POST Python removes the premise:

1. **The interpreted fallback is total.** A POST package is valid Python
   by design contract (spec §1.1). A pure wheel is not a degraded
   artifact; it is the package.
2. **Compilation is trivial.** A conforming POST compiler + `cc` + libm.
   No Fortran, no BLAS bootstrap, no build farm. A full package builds
   in about a second.
3. **The compiled artifact is a real system library.** The Package ABI
   (spec §9.1.1) gives every artifact stable `pp_*` symbols, a C header,
   and a machine-readable manifest — exactly the shape environment
   package managers know how to version, share, and solve for, and
   something a wheel structurally cannot share across packages or
   languages.

Publishing binary wheels would spend the ABI's entire point on the
distribution channel least able to use it.

## The default: environment package managers

The pixi/conda (and nix, spack) recipe for a POST package is split the
way system libraries have always been split:

| Package | Contents | Consumers |
|---|---|---|
| `libpp<name>` | shared library with `pp_*` symbols, C header, export manifest | C, C++, Rust, Zig, Julia, R, ctypes/cffi — no Python required |
| `pp<name>` | Python source package + NumPy ufunc extension module | Python users |

One copy of `libpp<name>` per environment, shared by every consumer,
versioned and solved coherently alongside its dependencies. This is
where package READMEs, error messages, and tutorials should send users
who want compiled performance:

```bash
pixi add ppspecial      # or: conda install ppspecial
```

### Install layout

`postpyc build --prefix $PREFIX` produces the `libpp<name>` piece in
the conventional layout recipes expect:

```
$PREFIX/lib/lib<artifact>.so          (.dylib on macOS)
$PREFIX/include/<artifact>.h          stable C ABI declarations
$PREFIX/share/postpyc/<artifact>.json   export manifest ("post_abi": 1)
```

The NumPy extension module (`postpyc build --ext-module`) is built by
the `pp<name>` recipe and installed into the environment's
`site-packages` like any extension; it links the same translation units.

### Recipe sketch (rattler-build)

```yaml
recipe:
  name: ppspecial-split
outputs:
  - package:
      name: libppspecial
    requirements:
      build: [postpyc, c-compiler]
    build:
      script: postpyc build ppspecial/__init__.py --prefix $PREFIX
  - package:
      name: ppspecial
    requirements:
      build: [postpyc, c-compiler, python, numpy]
      run: [python, numpy]
    build:
      script: |
        postpyc build ppspecial/__init__.py --ext-module \
            --module-name ppspecial_native --output $SP_DIR/ppspecial_native.so
        python -m pip install . --no-deps
```

## PyPI: pure source

`pp*` packages ship `py3-none-any` wheels and sdists containing only
Python source. `pip install ppspecial` gives working, interpreted
kernels everywhere, with zero build requirements.

pip-only users therefore get interpreted speed by default. That is the
intended trade; the package's documentation points at the environment
managers above, or at explicit local compilation below.

## Local compilation

Building locally is always a supported path — from a source checkout
*or* from the pure package installed off PyPI, whose `.py` files are
the POST sources. It requires a functional POST-compatible compiler
chain: a conforming POST compiler (today, the reference `postpyc`)
and a C toolchain.

From a checkout:

```bash
postpyc build ppspecial/__init__.py --emit-header --emit-manifest
postpyc build ppspecial/__init__.py --ext-module
```

From an installed pure wheel:

```bash
postpyc build "$(python -c 'import ppspecial; print(ppspecial.__file__)')" \
    --ext-module --output ./ppspecial_native.so
```

Two rules keep this path honest:

- **Explicit, always.** Compilation happens when the user invokes it —
  never as an install-time or import-time side effect. Packages may
  document (or provide) a build command; they must not run one behind
  the user's back.
- **Nothing prebuilt, nothing vendored.** Local builds compile source
  against the local system. The output stays on the user's machine; it
  is not something to upload to PyPI.

## Summary for pp* package agents

- Publish source-only to PyPI. Never binary wheels; never install-time
  or import-time compile hooks.
- Point users to pixi/conda/nix **by default** for compiled
  performance; document explicit local compilation as the alternative
  for users with a POST-compatible compiler chain.
- Provide `build-native` and `build-ext` tasks (the ppspecial layout).
- When conda-forge/nix packaging begins, use the `libpp<name>` +
  `pp<name>` split and the `--prefix` layout above.
- The export manifest is the machine-readable contract between the
  compiler and recipes; consume it rather than guessing symbol names.
