# Design: Callable Parameters

Status: **draft for review** — PR 1 of the series. Spec amendments land
with the PRs that implement them (spec-as-implemented, per project
practice).

Callable parameters let a POST Python function receive another function
as an argument and call it — the missing capability behind ppintegrate
and ppoptimize:

```python
from typing import Callable
from postyp import Array, f64, i64

def quad_simpson(f: Callable[[f64], f64], a: f64, b: f64, n: i64) -> f64:
    ...

def minimize_nm(f: Callable[[Array[f64]], f64], x0: Array[f64]) -> Array[f64]:
    ...
```

---

## 1. Design principles

These govern every decision below and should eventually be normative in
the spec.

**P1 — Compiled code never re-enters the interpreter (POST Core).**
A callable argument, at every compiled call site, must resolve to a
function this program compiled (or was linked against). The compiler
emits a plain indirect call through a native function pointer; there is
no boxing, no trampoline, no interpreter state in the emitted code.

**P2 — Prefer compiling everything.** The answer to "how do I pass my
Python function to `quad`?" is: write it as POST Python and it *is*
compiled — there is nothing to bridge. The bridge problem largely
dissolves when the whole call graph is in the language.

**P3 — The Python-callback bridge is a future, explicitly named
extension — never an implicit fallback.** f2py got this right the hard
way: connecting a Python callable to a low-level callable, and handling
re-entrance when compiled code calls back into the interpreter
mid-flight, is where the dragons live (GIL acquisition from a thread
that entered native code, interpreter re-entrance while a compiled
section holds state, exception propagation across the native frame,
free-threaded builds changing the rules). v1 avoids that situation
*structurally*: there is no code path by which an interpreted callable
reaches a compiled call site. When a boundary-callback extension is
designed, it will be opt-in, own these hazards explicitly, and live at
the CPython boundary (§7.5) — not inside POST Core.

A useful consequence of P1: because the compiled artifact just calls an
opaque function pointer, an *embedder* (C, Rust, Julia — or Python via
`ctypes.CFUNCTYPE`) may pass any pointer it likes, including one that
wraps a Python function. Re-entrance safety is then the embedder's
explicit contract (ctypes owns the GIL dance), not the compiler's. The
layering is clean: we never emit interpreter-touching code; tools that
manufacture such pointers own the consequences. This is the
`scipy.LowLevelCallable` idea promoted to the default calling
convention — no wrapper object required.

---

## 2. Status quo

Today the reference compiler fails a callable-passing program at three
precise points (this is the v1 acceptance surface):

```text
PP100  unsupported annotation for parameter `f`          (the Callable annotation)
PP502  call to unknown function `f`                      (the indirect call site)
PP900  name `gaussian` is not a parameter, local, ...    (the function reference argument)
```

Interpreted mode already works — functions are first-class in Python —
so v1 is purely a compiler capability, with interpreted behavior as the
parity oracle.

## 3. Language surface (v1)

### 3.1 Annotation spelling

`typing.Callable[[T1, ..., Tn], R]` is the canonical spelling:

- It is ordinary Python — runs interpreted today, and mypy/pyright
  already understand it (POST source keeps type-checking as plain
  Python, which is the design contract).
- `postyp` will re-export `Callable` in its next release so
  `from postyp import Callable, f64` stays a one-stop import; both
  spellings resolve identically (same object).

v1 restricts Ti and R to **scalar dtypes**; a later PR in this series
adds `Array[...]` in callable signatures (ppoptimize's `minimize`
needs `Callable[[Array[f64]], f64]`). `Callable[..., R]` (unspecified
arity) is rejected: POST signatures are exact.

### 3.2 What a callable-typed value can be

At every compiled call site, an argument bound to a `Callable`
parameter must be one of:

1. a **module-level POST function** of exactly matching signature —
   defined in the entry module or imported from a POST module
   (resolution reuses the §9.1 machinery: aliases follow
   `resolve_function`, private cross-module references keep PP503
   parity);
2. a **callable parameter forwarded** onward (higher-order chains:
   `def outer(f: Callable[[f64], f64], ...) -> f64: return quad(f, ...)`).

That's the whole v1 value universe. Excluded, each with a clear
diagnostic (see §6): lambdas (already PP032), nested functions /
closures, callable locals and module constants, callable **return
types**, callable **default values**, callables in `@vectorize` /
`@guvectorize` kernels (a ufunc with a function-pointer operand is not
a ufunc; revisit only if a pp* package produces a real need),
`bool(f)` / comparison / any non-call, non-forward use of the value.

### 3.3 Typing rules

- **Exact-match invariance.** A function argument's signature must
  equal the parameter's `Callable` type exactly — no promotion through
  function types in v1 (`f64(f64)` does not accept `f32(f32)` or
  `f64(i64)`). Variance through function pointers is a classic
  soundness trap; exactness is also what the C ABI needs. Generic
  kernels (brainstorm Tier 1 item 2) will later generate the
  per-dtype instantiations that make this restriction painless.
- **Normal promotion at the indirect call.** `f(x)` where `f:
  Callable[[f64], f64]` promotes `x` per the standard scalar rules,
  exactly as a direct call to a `f64` parameter would.
- **No exceptions across the pointer (v1).** Compiled POST functions
  cannot raise across the native boundary; integrand/objective failure
  is expressed in-band (NaN out, or a status field once structs lower).
  ppintegrate/ppoptimize API design should assume this from day one.

## 4. Lowering

- A callable parameter lowers to a **C function pointer**:
  `Callable[[f64], f64]` → `double (*f)(double)`.
- One **typedef per distinct signature** in emitted C and in the ABI
  header, named canonically from the signature, e.g.
  `__pp_fn_f64__f64_t` (mangling: params, then return; exact scheme
  fixed in PR 3): `typedef double (*__pp_fn_f64__f64_t)(double);`
- `f(x)` lowers to an indirect call; the IR gains either an
  `is_indirect` flag on `Call` or a `CallIndirect` node (decided in
  PR 2 by whichever keeps the backend simplest).
- A function-reference argument lowers to the **kernel symbol**'s
  address (the mangled/internal name, e.g. `__pp_j0` — not the `pp_`
  wrapper), so intra-program indirect calls pay no double-wrapper cost.
  Same-TU private functions may be referenced (their `static` address
  is valid within the object); cross-TU private references are PP503,
  as for direct calls.

## 5. C ABI and manifest (additive; `post_abi` stays 1)

Exported functions with callable parameters are first-class ABI
citizens — this is deliberate and is the LowLevelCallable payoff:

- **Header**: the typedef block plus prototypes using it:

  ```c
  typedef double (*__pp_fn_f64__f64_t)(double);
  double pp_quad_simpson(__pp_fn_f64__f64_t f, double a, double b, int64_t n);
  ```

- **Manifest**: a callable param's entry gains a nested signature
  object (consumers that predate the field ignore it — additive JSON):

  ```json
  {"name": "f", "dtype": null, "is_array": false,
   "callable": {"params": [{"dtype": "Float64"}], "return_dtype": "Float64",
                 "c_typedef": "__pp_fn_f64__f64_t"}}
  ```

- **Cross-language test contract**: a C (or Rust/Julia) caller passes a
  native function; Python callers may pass a `ctypes.CFUNCTYPE`
  pointer through the same prototype. Both are tested; the second
  documents the embedder-owns-re-entrance contract from §1.

## 6. Diagnostics

PP100 stays the generic annotation failure; callable-specific codes get
the next 1xx block (currently unused):

| Code | Meaning |
|------|---------|
| PP101 | malformed/unsupported `Callable` annotation (non-scalar in v1, `Callable[..., R]`, bad arity) |
| PP102 | function argument signature does not match the `Callable` parameter type (shows both signatures) |
| PP103 | name passed where a callable is expected does not resolve to a POST function (or resolves to something non-callable) |
| PP104 | unsupported use of a callable value (stored, returned, compared, defaulted) |
| PP9xx | valid-but-not-lowered: callables in `@vectorize`/`@guvectorize` kernels; callable return types |

Every message states the v1 rule it enforces and what to do instead
(usually: "define a module-level POST function and pass it by name").

## 7. Interpreted mode

Nothing to build: CPython already passes and calls functions. The
decorators and checker need no changes (the structural checker is
untouched — no new banned constructs). Every compiled-path test in this
series has an interpreted twin asserting identical results, which is
the conformance story for free.

## 8. Spec amendments (land with implementing PRs)

- **§4.6 Function Types** (new): callable parameter types, the value
  universe (§3.2), exact-match rule, the no-re-entrance principle P1
  stated normatively for POST Core.
- **§6.2 grammar**:

  ```text
  annotation  ::= ... | callable_type
  callable_type ::= "Callable" "[" "[" [dtype ("," dtype)*] "]" "," dtype "]"
  ```

- **§9.1.1 Package ABI**: signature typedefs, manifest `callable`
  field, kernel-symbol (not wrapper) address semantics.
- **§11 Extension Model**: name the future "CPython boundary callback"
  extension and the hazards it must own (P3) so nobody implements it
  casually inside Core.

## 9. PR series

All PRs in the series target the integration branch
**`feature/callable-params`** (sub-branches named
`callable-params/<stage>`), so each stage gets focused review while
main stays releasable. The integration branch merges to `main` once
PR 5 lands, giving one place to review the assembled feature against
main at the end. CI runs its full matrix on PRs regardless of base
branch.

| PR | Scope | Proof |
|----|-------|-------|
| 1 *(this)* | Design doc | review |
| 2 | Frontend + typechecker + IR: `Callable` annotations (scalars), function-reference arguments, indirect-call IR, PP101–PP104 | IR-level tests + interpreted parity |
| 3 | C backend + linking: typedefs, fnptr params, indirect calls, cross-module + PP503 parity | native tests incl. `ctypes.CFUNCTYPE` cross-boundary callback |
| 4 | ABI surface: header typedefs, manifest schema; spec §4.6/§6.2/§9.1.1/§11; toolchain + getting-started docs | manifest/header tests, site builds strict |
| 5 | `Array[...]` in callable signatures (the `minimize` shape); worked example (`quad_simpson`); notify ppintegrate/ppoptimize with target signatures | end-to-end example test |

Each PR keeps the suite green and lands independently; nothing in 2–4
blocks on 5's design.

## 10. Open questions for review

1. **postyp `Callable` re-export**: fold into a postyp 0.4.0 alongside
   the next vocabulary need, or ship immediately? (Nothing in PRs 2–4
   depends on it; `typing.Callable` works with postyp 0.3.0.)
2. **Kernel-symbol vs wrapper address** for function-reference
   arguments (§4): kernel symbol proposed for zero-cost intra-program
   calls; the `pp_` wrapper would be marginally more uniform with the
   external ABI. Any reason to prefer uniformity?
3. **PP104's storage restriction**: callable locals (`g = f`) are cheap
   in C — excluded from v1 only to keep the inference surface small.
   Fine to defer, or is there a near-term ppoptimize pattern (e.g.
   selecting a line-search function) that wants it in PR 2?
