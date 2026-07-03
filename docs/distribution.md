# Distributing POST Python Packages

How compiled POST Python libraries (the `pp*` family) reach users. This
guide is policy for packages under the PostSciPy effort and a reference
for anyone shipping POST Python code.

## The policy, in one paragraph

POST packages publish **pure source** to PyPI — `py3-none-any` wheels,
interpreted mode, fully functional. **No binary wheels are published,
ever.** Compiled artifacts are distributed through package managers that
treat native code as a first-class dependency — conda/pixi, nix, spack —
as a split package: a real system library plus a Python binding. Users
who want native speed from a source checkout compile locally with
`postpython build` (a C compiler and libm are the entire toolchain, and
a full package builds in about a second).

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
2. **Compilation is trivial.** `postpython` + `cc` + libm. No Fortran,
   no BLAS bootstrap, no build farm.
3. **The compiled artifact is a real system library.** The Package ABI
   (spec §9.1.1) gives every artifact stable `pp_*` symbols, a C header,
   and a machine-readable manifest — exactly the shape environment
   package managers know how to version, share, and solve for, and
   something a wheel structurally cannot share across packages or
   languages.

Publishing binary wheels would spend the ABI's entire point on the
distribution channel least able to use it.

## Tier 1: PyPI (pure source)

`pp*` packages ship `py3-none-any` wheels and sdists containing only
Python source. `pip install ppspecial` gives working, interpreted
kernels everywhere, with zero build requirements.

This means pip-only users get interpreted speed by default. That is the
intended trade: the README of each package points at the environment
managers (Tier 2) or at explicit local compilation for native speed.
Packages must not add install-time or import-time compilation hooks.

## Tier 2: environment package managers (binaries done right)

The conda/pixi (and nix, spack) recipe for a POST package is split the
way system libraries have always been split:

| Package | Contents | Consumers |
|---|---|---|
| `libpp<name>` | shared library with `pp_*` symbols, C header, export manifest | C, C++, Rust, Zig, Julia, R, ctypes/cffi — no Python required |
| `pp<name>` | Python source package + NumPy ufunc extension module | Python users |

One copy of `libpp<name>` per environment, shared by every consumer,
versioned and solved coherently alongside its dependencies.

### Install layout

`postpython build --prefix $PREFIX` produces the `libpp<name>` piece in
the conventional layout recipes expect:

```
$PREFIX/lib/lib<artifact>.so          (.dylib on macOS)
$PREFIX/include/<artifact>.h          stable C ABI declarations
$PREFIX/share/postpython/<artifact>.json   export manifest ("post_abi": 1)
```

The NumPy extension module (`postpython build --ext-module`) is built by
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
      build: [postpython, c-compiler]
    build:
      script: postpython build ppspecial/__init__.py --prefix $PREFIX
  - package:
      name: ppspecial
    requirements:
      build: [postpython, c-compiler, python, numpy]
      run: [python, numpy]
    build:
      script: |
        postpython build ppspecial/__init__.py --ext-module \
            --module-name ppspecial_native --output $SP_DIR/ppspecial_native.so
        python -m pip install . --no-deps
```

## Local compilation (source checkouts)

Developers and users working from a checkout compile explicitly:

```bash
postpython build ppspecial/__init__.py --emit-header --emit-manifest
postpython build ppspecial/__init__.py --ext-module
```

or through the package's pixi tasks (`pixi run build-native`,
`pixi run build-ext`). This is source compilation against the local
system — nothing prebuilt, nothing vendored.

## Summary for pp* package agents

- Publish source-only to PyPI. Never binary wheels; never install-time
  or import-time compile hooks.
- Provide `build-native` and `build-ext` tasks (the ppspecial layout).
- When conda-forge/nix packaging begins, use the `libpp<name>` +
  `pp<name>` split and the `--prefix` layout above.
- The export manifest is the machine-readable contract between the
  compiler and recipes; consume it rather than guessing symbol names.
