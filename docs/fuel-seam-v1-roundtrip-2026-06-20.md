# Baracuda — Profile v1 round-trip confirmation (2026-06-20)

Cover note to Fuel, per *Kernel-Seam Interop Contract* §8.4 ("Baracuda publishes
a conforming crate version") and our ratification reply
([`fuel-reply-seam-profile-v1-ratify-2026-06-20.md`](fuel-reply-seam-profile-v1-ratify-2026-06-20.md)).
**Profile v1 is ratified; this confirms Baracuda's conforming surface is built.**

TL;DR — the four publish items we listed (plus the commutative-operand sort) are
implemented, tested, and on branch `feat/kernel-specialization` (PR #2). Two leaf
items remain gated on the full rev-4 / FDX annex text and are flagged below; both
are localized.

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

## B. Two leaf items still gated on annex text (both localized)

1. **The §3a.2a commutative sort *key*.** Our canonicalization mechanism is in
   place but uses a **provisional** order (graph ops before leaves, then by op
   name / leaf index). To match your matcher bit-for-bit we need §3a.2a's exact
   key ("the same canonicalization `structure_key` uses") from the **full rev-4
   `fkc-fusion-patterns.md`** — which, per your own §8 circulation rule, travels
   in the Profile v1 bundle but didn't reach us. Send it and reconciling the key
   is a one-function change (`pattern::sig`), then we re-verify.
2. **An FFI `structure_key` trampoline (only if you want one).** Two Rust
   projects integrate via the Rust callable above directly (you write
   `FdxOperandDesc → OperandDesc`). A C-ABI trampoline would additionally need the
   **FDX numeric dtype codes** (review item E5) — `ElementKind` is not
   `#[repr]`-stable — so we deferred it rather than invent codes. Tell us if the
   seam expects structure-key computation over the C ABI like `seam_hello`, and
   we'll add it once E5's code table is pinned.

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

Baracuda's Profile v1 surface is **conforming and built**, ready for the §8.4
lockstep crate bump. Outstanding from your side: the full rev-4
`fkc-fusion-patterns.md` (for §B.1) and the FDX §12 token-bit order confirmation
(§C); from ours, the crate-version bump on the next release tag. Stamp it —
*Profile v1, ratified, implementation-true on both halves.*
