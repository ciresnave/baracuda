# Baracuda — Profile v1 round-trip confirmation (2026-06-20)

Cover note to Fuel, per *Kernel-Seam Interop Contract* §8.4 ("Baracuda publishes
a conforming crate version") and our ratification reply
([`fuel-reply-seam-profile-v1-ratify-2026-06-20.md`](fuel-reply-seam-profile-v1-ratify-2026-06-20.md)).
**Profile v1 is ratified; this confirms Baracuda's conforming surface is built.**

TL;DR — the four publish items (plus the commutative-operand sort) are
implemented, tested, and on branch `feat/kernel-specialization` (PR #2). With the
full **rev-4 `fkc-fusion-patterns.md`** now in hand, the two leaf items we flagged
are **resolved** (§B), and an adversarial multi-agent conformance audit of our
emitters against the full spec found and fixed **14 divergences — including one
real misroute bug** (the GELU flavor). We report them here in full (§B'); none
remains open.

---

## A. Built — Baracuda's Profile v1 surface

| Surface | State | Where |
|---|---|---|
| **`baracuda_seam_hello()`** (§3.1/§3.5) | **Built**, to the ratified ABI exactly: a fixed 56-byte `#[repr(C)] SeamHello` POD (`SEAM_MAX_PROFILES = 16` + `profiles_len`, no variable-length member), out-param `int baracuda_seam_hello(SeamHello*)`, size/align asserted at compile time. Advertises profile `[1]`. | new `baracuda-seam` crate |
| **Full FKC contract emitter** (§4.3 + `ImplId` §4.11) | **Built**: front-matter (`fkc_version`, `provider.{…}`, `seam_profiles:[1]`) + per-kernel `op_kind`\|`fused_op`, the `ImplId` 5 fields, `accept` (carrying the StructureKey token as the admissibility predicate), `op_params`, `return`, `caps`, `cost`, `precision`, `determinism`, and `pattern:` for fusions. | `kernelgen::contract` |
| **`link_registry`** (§4.3 / FKC §12.6) | **Built**: generated, dependency-free Rust roster (`entry_point`, structure-key token, revision hash); emitted source compiles clean. | `kernelgen::link` |
| **Commutative-operand canonicalization** (§3a.2a) | **Built** (mechanism): `derive_pattern` canonicalizes `Add`/`Mul` operands so `a*b + c` and `c + a*b` emit one identical pattern. Sort *key* provisional — see §B. | `kernelgen::pattern` |
| **`structure_key` callable** (§4 / telemetry) | **Built**: `structure_key_token(op, &[OperandDesc], arch) -> token` — the one-call entry point your trampoline invokes after projecting each `FdxOperandDesc → OperandDesc`. | `kernels-types::structure_key` |

All of the above are unit-tested green; the kernels (and the 2.03× go/no-go) are
nvcc-built + run on sm_89 as before. A worked bundle (primitive `add`, fused
`relu_add`, scalar-param `affine_silu`) + its `link_registry` is committed as a
golden reference (`crates/baracuda-kernelgen/examples/sample_bundle.md`).

## B. The two flagged leaf items — both resolved against rev 4

1. **The §3a.2a commutative sort *key* — resolved, and better than we feared.**
   Rev-4 §3a.2a settles this as **"both sides canonicalize identically"**: Fuel
   canonicalizes the *imported pattern* as well as the user graph before matching,
   so "a provider emits operands in any one ordering and it matches regardless."
   So our emission order was never load-bearing — there is **no key to match**.
   We kept our `canonicalize` pass purely as an internal-determinism nicety (two
   authorings → byte-identical YAML) and corrected the docstrings that wrongly
   framed it as a matching obligation. Nothing to reconcile.
2. **The dtype codes (E5) — resolved by §5.** §5's logical-DType list + the B5/E5
   resolution pinned the spellings: `Bool → U8` (Fuel has no Bool dtype), signed-8
   `→ I8`, sub-byte `S4/U4/Bin` ride the FDX sidecar (not base dtypes), and
   `Fp8E5M2`/complex have no §5 slot. We reconciled `fkc_dtype` accordingly (§B').
   The FFI `structure_key` trampoline remains **optional** (the Rust callable path
   needs nothing more); if you ever want it over the C ABI like `seam_hello`, the
   numeric-code table is the only remaining input.

## B'. Conformance audit — 14 fixes we made against rev 4 (one real bug)

We ran an adversarial multi-agent audit of `pattern.rs` + `contract.rs` against the
full rev-4 spec (5 dimensions, each finding independently verified). It confirmed
14 divergences, now all fixed on PR #2 (commit `aec3bf7`):

- **GELU flavor (must-fix — a real misroute bug, reported in good faith).** Our
  CUDA kernel computes **exact erf** GELU (`0.5·x·(1+erf(x/√2))`), but the emitter
  named it bare `Gelu` — which §4.1 (B6/E2) defines as the **tanh** approximation.
  Fuel would have matched our exact-erf kernel against tanh-GELU subgraphs (~1e-3
  error). Now emits **`GeluErf`**. (Thank you for pinning the flavor in §4.1 —
  that's exactly what surfaced it.)
- **Dtype channel (must/should-fix).** `fkc_dtype` is now fallible: it emits only
  §5 base spellings (`Bool→U8`, `S8→I8`, …) and **skips** a cell whose dtype has
  no §5 base slot (sidecar payloads, `Fp8E5M2`, complex) rather than emit an
  unbindable token — so the planner's miss signal stays honest. `contract()`
  returns `Option` and drops such cells.
- **`op_kind` fallback (should-fix).** A body we can't express as a pattern
  (a `Const`, non-elementwise, bind-mismatch) no longer fakes `op_kind: <op.name>`
  (not an OpKind dispatch key) — it's skipped via the same fallible path.
- **Commutativity docs (should-fix + notes).** Reframed per §3a.2a (item B.1).

Net: our emitter output is now rev-4-conformant on the dimensions the spec covers
(op vocabulary, dtype spellings, pattern grammar/semantics, extract paths,
commutativity). The `accept`/`return`/`cost`/`precision`/`caps` *sub-field* values
remain reconstructed from the §4.3 matrix and will reconcile against the base
`kernel-contract-format.md` when we wire it.

## C. Capability bits Baracuda advertises at first connect

`SeamHello.capabilities` (the `local & remote` intersection applies, §3.4):

| Bit | Token | Advertised |
|---|---|---|
| FDX | `DlpackExtV1` (sub-byte S4/U4/Bin) | **on** |
| FDX | `DlpackExtMx` (FP8 microscaling) | **on** |
| FDX | `DlpackExtGgml` | **on** |
| FDX | `DlpackExtAffine` (incl. AFFINE_BLOCK / NF4-QLoRA) | **on** |
| FDX | `DlpackExtSymbolic` (attention KV) | **on** |
| FDX | `DlpackExtGather` (paged KV) | **on** |
| seam | `SeamCapJitOnRequest` (§5) | **off** (design-only until both sides build it) |

One reconciliation note: the FDX-token **bit positions** are pinned in
`baracuda-seam` to FDX §12's order; please confirm that order so our `capabilities`
low bits line up with your `BackendProbe` reader.

## D. Net

Baracuda's Profile v1 surface is **conforming, built, and rev-4-audited**, ready
for the §8.4 lockstep crate bump. With the full rev-4 spec received, the two leaf
items are closed (§B) and our emitters reconciled (§B'). The only inputs we still
want from your side are confirmations, not blockers: the **FDX §12 token-bit
order** (§C), and — when you wire the base `kernel-contract-format.md` — a pass
over our `accept`/`return`/`cost`/`precision` sub-field spellings. From ours: the
crate-version bump on the next release tag. Stamp it — *Profile v1, ratified,
implementation-true on both halves.*
