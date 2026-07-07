# PostSciPy Roadmap

Rebuilding SciPy, one subpackage at a time, in pure POST Python is the
primary way this project tests and grows the reference compiler and the
language specification. Each SciPy subpackage becomes its own `pp*`
repository under [openteams-ai](https://github.com/openteams-ai), owned by
its own agent or contributor, written as clean POST Python with no
compiler-specific escape hatches.

The flow runs in both directions:

- **Library → compiler.** Real numerical code discovers what the language
  is missing. Gaps are filed as issues in this repository with minimal
  reproducers and become compiler/spec work.
- **Compiler → library.** Each landed feature is announced on its issue so
  library agents can adopt it and mark their roadmap items unblocked.

[ppspecial](https://github.com/openteams-ai/ppspecial) is the exemplar:
its kernels drove cross-module linking, the NumPy gufunc ABI, extension-
module output, module-level constants, and several correctness fixes.
Every package below is expected to generate the same kind of pressure.

## Package map

| Repository | Mirrors | Status | Primary compiler pressure |
|---|---|---|---|
| [ppspecial](https://github.com/openteams-ai/ppspecial) | `scipy.special` | Active | scalar kernels, ufunc ABI, constants, aliases |
| [ppconstants](https://github.com/openteams-ai/ppconstants) | `scipy.constants` | Planned | module constants at scale, cross-module constant imports |
| [ppstats](https://github.com/openteams-ai/ppstats) | `scipy.stats` | Planned | cross-**package** POST dependencies (→ ppspecial), reductions |
| [ppsignal](https://github.com/openteams-ai/ppsignal) | `scipy.signal` | Planned | recurrence gufuncs, window kernels, cross-package deps |
| [pplinalg](https://github.com/openteams-ai/pplinalg) | `scipy.linalg` | Planned | 2-D gufuncs, local array workspaces (RAII), foreign BLAS/LAPACK |
| [ppintegrate](https://github.com/openteams-ai/ppintegrate) | `scipy.integrate` | Planned | fixed-sample rules now; **callable parameters** for adaptive quad |
| [ppoptimize](https://github.com/openteams-ai/ppoptimize) | `scipy.optimize` | Planned | **callable parameters** (the lead driver), iteration/tolerance patterns |
| [ppfft](https://github.com/openteams-ai/ppfft) | `scipy.fft` | Planned | complex-array kernels, constant tables, workspaces |
| [ppinterpolate](https://github.com/openteams-ai/ppinterpolate) | `scipy.interpolate` | Planned | search loops, banded solvers, workspaces |
| [ppcluster](https://github.com/openteams-ai/ppcluster) | `scipy.cluster` | Planned | distance/argmin kernels; dynamic structures later |
| [ppspatial](https://github.com/openteams-ai/ppspatial) | `scipy.spatial` | Planned | pairwise-distance gufuncs, fixed-size quaternion ops; trees later |
| [ppndimage](https://github.com/openteams-ai/ppndimage) | `scipy.ndimage` | Planned | stencil kernels; dynamic-rank iteration later |
| [ppsparse](https://github.com/openteams-ai/ppsparse) | `scipy.sparse` | Planned | integer index arrays, CSR kernels; **structs** for formats |
| [ppdifferentiate](https://github.com/openteams-ai/ppdifferentiate) | `scipy.differentiate` | Planned | stencil kernels now; callable parameters for the full API |

Not planned: `scipy.io` and `scipy.datasets` (I/O crosses the CPython
boundary by design — spec §10), `scipy.fftpack` (legacy, superseded by
`scipy.fft`), `scipy.misc` (deprecated upstream), `scipy.odr` (niche;
revisit after foreign-function support exists).

## Start-now slices vs. blocked slices

Every package has work that compiles **today** and work that is blocked on
a named compiler capability. Packages should start with the feasible slice
immediately and file the blocked slice as a postpython issue with a
minimal reproducer — that filing *is* part of the work.

The current capability matrix, by feature:

| Compiler capability | Packages that drive it | Tracking |
|---|---|---|
| Callable parameters (functions as arguments) | ppoptimize, ppintegrate, ppdifferentiate | file on first need |
| Local array allocation + RAII (spec §7) | pplinalg, ppfft, ppinterpolate | file on first need |
| Constant tuples/arrays for coefficient and twiddle tables | ppspecial, ppfft | [#11 follow-up](https://github.com/openteams-ai/postpython/issues/11) |
| Structs (`@dataclass` → C struct) | ppsparse, ppspatial, ppintegrate (result objects) | file on first need |
| Stable C ABI: export manifest, headers, alias policy, module-qualified symbols | all | [#12](https://github.com/openteams-ai/postpython/issues/12) |
| Cross-module inlining / optimization | ppstats, ppsignal (hot inner calls into ppspecial) | [#13](https://github.com/openteams-ai/postpython/issues/13) |
| Cross-package dependency resolution + wheel story | ppstats, ppsignal (depend on ppspecial) | [#14](https://github.com/openteams-ai/postpython/issues/14) |
| Floating-point semantics in the spec | all numerical packages | [#15](https://github.com/openteams-ai/postpython/issues/15) |
| Complex-dtype array kernels end to end | ppfft, ppsignal | exercise, then file gaps |
| Dynamic-rank (`AnyShape`) iteration | ppndimage | file on first need |
| RNG / random-state model | ppstats (`rvs`) | design discussion first |

## Working rules for pp* packages

These mirror ppspecial's roadmap and keep the ecosystem coherent:

1. **Pure POST Python.** Every kernel runs interpreted under CPython and
   compiles with the reference compiler. No compiler-specific escape
   hatches in library source.
2. **Compiler gaps go upstream.** When valid POST Python fails to compile
   or a needed construct is missing, file a postpython issue with a
   minimal reproducer instead of working around it silently. Reference
   the issue from the package's roadmap.
3. **Verify against `main`.** Use an up-to-date postpython checkout or the
   git dependency; do not treat results from feature branches as
   canonical.
4. **scipy is the reference, not a dependency.** Runtime code must not
   import scipy. Tests may use scipy (optional dependency) to generate or
   check reference values; deterministic hardcoded references are
   preferred so the suite runs without it.
5. **Follow the ppspecial layout.** `pp<name>/` package, `tests/`,
   `scripts/build_native.py` and `scripts/build_ext.py`, a pixi workspace
   in `pyproject.toml` with `test` / `build-native` / `build-ext` tasks,
   a git dependency on postpyc, and a `ROADMAP.md` tracking targets
   and upstream requests.
6. **Accuracy is a deliverable.** Document per-function accuracy targets
   and reference sources; validate against published values.
7. **Small, reviewable landings.** Each function family lands as a PR with
   tests, both execution modes verified, and a roadmap status update.
8. **Cross-package dependencies are allowed and encouraged** where scipy
   has them (ppstats → ppspecial). They exercise the packaging story and
   should be declared as ordinary git dependencies plus POST
   `search_paths` at build time until #14 lands.
9. **No binary wheels, ever.** pp* packages publish pure source to PyPI
   (`py3-none-any`). Point users to environment package managers
   (pixi/conda, nix) by default, via `libpp<name>` + `pp<name>` split
   packages; explicit local compilation with a POST-compatible compiler
   chain (from a checkout or the installed pure wheel) is the supported
   alternative. No install- or import-time compilation hooks. See
   postpython's `docs/distribution.md` for the full policy and layout.

## Sequencing guidance

Waves balance two orders: SciPy's own historical growth (special →
integrate → optimize → linalg → signal → …) and what the compiler can
express today.

- **Wave 1 (start immediately):** ppconstants, ppstats (descriptive +
  ppspecial-backed distributions), ppsignal (windows, lfilter), ppcluster
  (vq), ppspatial (distance kernels). All expressible with today's
  compiler; ppstats/ppsignal also stress cross-package linking.
- **Wave 2 (start now, expect to file features):** pplinalg (fixed-size
  ops and substitution solvers now; factorizations push on workspaces),
  ppfft (naive DFT proves complex arrays; fast paths push on constant
  tables), ppinterpolate (linear/barycentric now; splines push on
  solvers), ppintegrate (fixed-sample rules now; `quad` pushes on
  callables), ppsparse (CSR matvec now; formats push on structs),
  ppndimage (fixed 2-D stencils now).
- **Wave 3 (feature-gated):** ppoptimize and ppdifferentiate lead the
  callable-parameters design; ppstats `rvs` leads the RNG design;
  tree-based ppspatial and linkage-based ppcluster follow structs.

## Definition of done, per package

A pp* package is "recreated" when its core public API (the subset that is
numerical-kernel work, not I/O or plotting glue):

1. runs interpreted with tests against reference values,
2. compiles to the plain C shared library,
3. builds as a NumPy-ufunc extension module where the API is array-valued,
4. and documents accuracy plus any intentional divergences from scipy.

The long-term measure for the whole effort: a user can
`import ppstats_native` (etc.) and reproduce a realistic scipy-based
workflow with compiled kernels end to end.
