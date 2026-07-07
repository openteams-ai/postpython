# POST Python: Future Directions

A brainstorm of where this project could go, ordered from **highly
likely** (concrete needs already visible from the pp* rebuild) down to
**kind of crazy** (on-theme moonshots). Items move up when a signal
appears — usually an issue filed by a pp* library agent hitting a real
wall, which has been the engine of every compiler milestone so far.

Companion documents: [postscipy-roadmap.md](postscipy-roadmap.md) (the
package map this feeds), [docs/spec.md](docs/spec.md) (what's promised),
[docs/distribution.md](docs/distribution.md) (how it ships).

---

## Tier 1 — Highly likely

The next compiler milestones. Most are already blocking, promised by the
spec, or inevitable the moment the next pp* package starts.

1. **Callable parameters.** `Callable[[f64], f64]` arguments lowered to
   C function pointers, so `minimize(f, x0)` and `quad(f, a, b)` are
   expressible. The known blocker for ppoptimize and ppintegrate — the
   two most-wanted packages after ppspecial. Interpreted mode is free;
   the work is annotation grammar, IR call-indirect, and ABI treatment.

2. **Dtype-generic kernels.** One kernel source instantiated for several
   dtypes (Numba-style multi-signature `@vectorize(["f32(f32)",
   "f64(f64)"])` or a `TypeVar`-over-`DType` spelling) with
   monomorphized C output and multi-loop ufunc registration. ppspecial
   will want `float32` variants the first time someone benchmarks
   against `scipy.special` on `f32` arrays.

3. **Complex number lowering.** `Complex64`/`Complex128` are in the
   vocabulary (now with `c64`/`c128` spellings) but don't lower. C99
   `_Complex` is a clean target. Needed for complex `erf`, and ppfft is
   a non-starter without it.

4. **Cross-package POST imports.** `from ppspecial import ndtr` inside
   ppstats source, compiled against ppspecial's *installed* manifest +
   header and linked against `libppspecial`. The JSON manifest already
   carries everything needed; this extends §9.1 cross-module linking
   from one package to the ecosystem — and it's the whole point of the
   pp* family being separate packages.

5. **ABI v2: module-qualified symbols.** The flat `pp_<name>` namespace
   collides the moment two packages export `mean` (today that's a PP501
   only within one program). `pp_<pkg>_<name>` kernel symbols with
   per-package manifest namespacing, keeping `pp_<name>` aliases inside
   a package for compatibility.

6. **Floating-point semantics in the spec (#15).** What a conforming
   compiler may and may not do: FMA contraction, reassociation,
   fast-math opt-in, NaN payload behavior, subnormals. Numerical library
   authors are writing against promises §4.1.1 currently only sketches.

7. **Struct / dataclass lowering.** Spec §4.4 declares aggregate types;
   nothing lowers yet. Plain-data dataclasses → C structs, by-value
   semantics. Needed for optimizer results (`OptimizeResult`), rich
   return values, and workspaces.

8. **Debug builds that keep §9.6's promises.** `--debug`: array bounds
   checks, integer-overflow diagnostics, uninitialized-read traps —
   with documented release-mode behavior. The single biggest
   quality-of-life gap for people porting real algorithms.

9. **A separable conformance suite.** Split the spec-behavior tests from
   postpyc's implementation tests into a compiler-agnostic suite (the
   array-api-tests model): any tool claiming a profile runs it. This is
   the difference between "a spec" and "a standard."

10. **Spec 0.3 cut.** Batch everything since May — namespace manifests,
    package ABI, short-hand spellings, PP500-series codes — into a
    versioned release with a revision-history row, so implementors can
    target a frozen document instead of a moving draft.

11. **conda-forge feedstocks.** `postyp`, `postpyc`, then the
    `libppspecial`/`ppspecial` split pair as the template recipe.
    distribution.md's manager-first story becomes real the day
    `pixi add ppspecial` resolves; until then the policy is prose.

12. **A tracked benchmark suite.** postpyc vs Numba vs Cython vs
    hand-written C on the ppspecial kernels, run in CI and published on
    the site. Honest numbers (including the known ~1% cross-TU cost
    from parked issue #13) are the credibility currency for everything
    else here.

13. **Whole-array expressions.** `c = a * b + 1.0` on `Array` operands
    lowering to fused loops — the first step of POST Array beyond
    per-element kernels, and the gateway to shape-checked array code.

14. **Foreign function declarations.** A typed way to declare external C
    symbols from POST source (compile-time ctypes, checked against the
    header). This is how pplinalg binds BLAS/LAPACK without leaving the
    language — and it forces a healthy spec question: which externals
    are portable.

15. **Windows/MSVC support.** `win-64` is already in the pixi platforms
    list; the backend emits C99 that MSVC mostly accepts. CI matrix
    entry, `cl.exe` flag handling, DLL export conventions.

---

## Tier 2 — Probable

Natural extensions once Tier 1 exists. No hard blockers, real payoff.

16. **Executable output.** Spec §9.3–9.4 already name it: a `main()`
    entry convention and `postpyc build --exe` producing a standalone
    binary with no Python runtime. The "smaller than Python" promise
    made tangible for CLI tools.

17. **Local array allocation and `postpyc.mem`.** Stack and heap arrays
    created *inside* kernels with RAII lifetimes (§7.2), plus the
    spec-listed `alloc`/`free`/`share`. Integrators and optimizers need
    scratch workspaces; today all arrays cross the boundary.

18. **Parallel ufunc execution.** The §7.4 single-writer model makes
    outer-loop parallelism safe by construction: `@vectorize(parallel=True)`
    over OpenMP or a small thread pool. Numba's most-loved flag,
    reproduced with spec-backed semantics.

19. **A diagnostics program.** Source-highlighted errors,
    "did-you-mean" for near-miss names, a docs URL per PP code. The
    checker is the first POST Python most people meet; it should feel
    like a great linter, not a compiler from 1989.

20. **postpyc-lsp.** The checker is a fast AST pass — surface it as a
    language server with inline PP diagnostics and dtype hovers in VS
    Code. Disproportionate adoption leverage for the effort.

21. **Free-threaded Python story.** Compiled kernels already do their
    work without touching the interpreter; declare `Py_mod_gil` support
    in ext modules, test under 3.14t, and market POST as the sane path
    to nogil throughput.

22. **Incremental builds.** Content-hash caching per translation unit so
    a 40-module pp package rebuilds in the time of one module. Matters
    the day ppstats has real breadth.

23. **Compiler Explorer integration.** postpyc as a Godbolt compiler
    with POST source → C → asm panes. Cheap to do, catnip for exactly
    the systems-curious Python audience the standard needs.

24. **Site growth: an implementations page.** Versioned spec pages, an
    examples gallery, and a table tracking which tools claim which
    conformance profiles — even while that table has one row.

---

## Tier 3 — Ambitious

Credible bets that change the project's category. Each deserves a
design doc before code.

25. **LLVM backend.** Spec §9.5 leaves the backend open; C99 stays the
    readable reference, LLVM adds real optimization control,
    cross-compilation, and an eventual JIT. Also the prerequisite for
    Enzyme-style autodiff (see below).

26. **Cross-compilation.** `postpyc build --target linux-aarch64` using
    `zig cc` or clang target triples. Pairs beautifully with the
    no-vendored-wheels policy: recipes build natively, power users
    cross-build, nobody ships mystery binaries.

27. **WebAssembly target.** `wasm32-wasi` kernels loadable by Pyodide —
    scientific Python in the browser *without* binary wheels, which is
    the distribution philosophy taken to its logical extreme. A POST
    playground on post-py.org (write a kernel, run it compiled,
    client-side) falls out of this.

28. **POST DataFrame v1 over Arrow.** Lower `Series[f64]` kernels onto
    Arrow C Data Interface buffers; narwhals remains the interpreted
    bridge. The spec's dataframe profile gets teeth without building a
    query engine — Arrow becomes to POST DataFrame what the ufunc ABI
    is to POST Array.

29. **Accelerator profile v0 (GPU).** `@vectorize(target="gpu")` for
    elementwise kernels, emitting CUDA C or SPIR-V, with the extension
    profile machinery (§11) exercised for real. Scoped deliberately
    small — elementwise only — to prove the profile model before anyone
    says "Triton."

30. **A second independent implementation.** Recruit one existing
    compiler — Numba post-mode, Cython `--post`, LPython, Codon — to
    pass POST Core conformance, and move the spec toward neutral
    governance (the data-apis consortium model). A standard with N=1
    implementations is a README; N=2 is a movement.

31. **Autodiff as a compiler pass.** Forward-mode first (dual numbers on
    the IR — tractable), reverse-mode later (or via Enzyme once the
    LLVM backend exists). Gradients for free would make ppoptimize
    qualitatively better than its SciPy ancestor rather than merely
    equal.

32. **Extension types at the boundary.** POST-defined classes exported
    as real CPython extension types — spec goal 3 ("replace C/Rust/Zig
    for extension modules") requires objects, not just functions.
    Probably the hardest language-design work on this list.

33. **Fixed-shape specialization.** `Shape[3, 3]` kernels with unrolled,
    SIMD-friendly codegen — small-matrix determinants, quaternion ops,
    3D geometry. The annotation grammar already carries the
    information; the backend just doesn't exploit it.

---

## Tier 4 — Kind of crazy

Say them out loud anyway. Each has a real version hiding inside it.

34. **PostNumPy.** After the SciPy rebuild proves out: reimplement
    NumPy's umath inner loops as POST kernels and swap them in as an
    optional backend. The stack's foundation rebuilt on the standard —
    and a certain poetry in NumPy's author replacing its C with
    compilable Python, twenty years on.

35. **Self-hosting slices.** Rewrite the compiler's hottest paths — the
    checker walk, the C emitter — in POST Python and compile them with
    the previous release. Every serious language eventually eats its own
    cooking; the subset is arguably already sufficient for these passes.

36. **Upstream to CPython.** POST annotations as *guarantees* for
    CPython's tier-2 JIT (a function known to be POST-conforming can
    skip guard chains), or a PEP standardizing the dtype-annotation
    vocabulary itself. High-risk, glacial, and the single largest
    possible impact: the subset stops being a dialect and becomes a
    gradient inside Python itself.

37. **Distributed profile.** gufunc dataflow graphs scheduled across
    nodes — manifests already describe kernels well enough to place
    them remotely. The spec's goal 6 ("base layer for distributed
    DSLs") taken literally: a compiled, typed dask-lite.

38. **A compiled query engine.** LazyFrame plans compiled whole —
    predicate pushdown, loop fusion, vectorized execution — instead of
    delegating to a backend. A polars-lite whose semantics come from a
    published spec rather than an implementation. Enormous; only
    sane after #28 succeeds and demands it.

39. **The methodology as a product.** This project is quietly a second
    experiment: a spec-anchored open team of humans and agents, with
    conformance tests as the reward signal and GitHub issues as the
    inter-agent protocol. Write it down, tool it, and offer it to other
    standards efforts — the OpenTeams thesis made concrete. The compiler
    might not be the most durable artifact here; the playbook might be.

40. **Silicon vendors ship POST.** A hardware vendor implements the
    Accelerator Extension as their kernel-language front end — POST
    Python as the portable answer to "what do we give Python
    programmers on our chip?" Crazy until you remember every vendor
    currently writes a bespoke Python DSL anyway.

---

## Parked (with reopening conditions)

- **Cross-TU inlining / LTO (issue #13).** Measured ~1% for libm-bound
  kernels; shipped nothing. Reopen when profiling shows a hot
  cross-module call chain that isn't libm-dominated — likely first in
  ppstats calling ppspecial primitives in tight loops (item 4 makes
  this measurable).
