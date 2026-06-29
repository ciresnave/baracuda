# Baracuda reply — telemetry + FDX + FKC boundary (2026-06-19)

Reply to Fuel's counter-proposal of 2026-06-19, *"the kernel boundary as a
two-way contract (FDX + FKC + telemetry)"*. Companion to our
[`fuel-ask-telemetry-2026-06-17.md`](fuel-ask-telemetry-2026-06-17.md) and the
design doc [`design/kernel-specialization.md`](design/kernel-specialization.md),
which remains the source of truth for the structure key.

TL;DR — we accept the join-token model and the substance of all six asks. One
counter on *how our kernels get advertised*, three conditions, and a narrowed
pre-freeze dtype ask:

- **Accepted wholesale:** the join-token model (no parallel identity scheme),
  DLPack as the ecosystem boundary, the nullable `const FDXSidecar*` on the Fuel
  ABI, `ImplId` on your basis tuple, miss = "best admissible match = generic",
  `flipped` = signed stride, the telemetry wire schema (§4.5), and consuming the
  feed coverage-agnostically.
- **Open Question 1 = YES** acknowledged; the F32/square-matmul coverage caveat
  is fine — our critical path is the miss histogram, which doesn't depend on
  Judge timings (§7).
- **Confirmed (shared) — Baracuda generates FKC; it is not hand-authored (§3).**
  FKC contracts + the `link_registry` are a *projection* of Baracuda's existing
  `KernelSku` / `PrecisionGuarantee` / OP-MATRIX, emitted mechanically — bespoke
  kernels from the SKU, specialization-matrix kernels from the structure key,
  alongside the `.cu`. Hand-authoring stays *available* to other providers, never
  required. No change to your importer (B.9); the value is that the FKC
  admissibility predicates *equal* the structure key by construction, which is
  what makes your miss signal honest.
- **Adoption posture (shared):** everything beyond the DLPack structs stays
  entirely optional, for advanced consumers only — `FDXSidecar` a vendored
  optional header read opportunistically, FKC + `link_registry` generated
  artifacts. Baracuda stays standard-DLPack to the whole ecosystem and is
  *additionally* rich to Fuel; and because the optional layer carries no
  provider-specific semantics, another ML ecosystem can adopt it incrementally
  later — the extensions are designed to be standardizable, not Fuel-proprietary.
- **Three items to settle before freeze:** (1) `structure_key` takes a minimal
  operand-description projection (with `From<FDX>` / `From<DLTensor>` /
  `From<TensorRef>`), not the whole sidecar; (2) the `ImplId` wire encoding keeps
  its five fields **separable** (not hashed into one opaque id); (3) pre-freeze
  dtype reconciliation (§4).

---

## 1. The join-token model — accepted

Your §1 proof holds: every token the feed needs is already a fact the format
carries — structure key from the FDX operand description, `ImplId` from the
kernel identity tuple, the miss from planner matching, timings from the Judge.
We adopt it wholesale. The payoff you name is the right one: **no new identity
surface anywhere**, and a record captured on one build re-resolves on another by
construction. This is a better answer than the standalone feed we asked for.

## 2. The six asks — point by point

1. **Adopt FDX** — accepted with the posture above: standard DLPack (v1.3,
   versioned, 256-byte-aligned, explicit signed strides) on the ecosystem
   boundary; a nullable `const FDXSidecar*` on the Fuel-facing ABI, read
   opportunistically from a vendored header. The honesty invariant (base
   `DLTensor` never lies) is what makes a general kernel library able to accept
   a framework sidecar without becoming framework-specific — it's the load-
   bearing property and we're building on it. It is also what lets the optional
   layer travel: because the sidecar adds capability without altering the base
   tensor's meaning, another ecosystem can adopt FDX/FKC incrementally — the
   standard-track outcome we'd all like, rather than a Fuel-only dialect.
2. **`structure_key` over FDX** — accepted, with **item 1**: the callable
   takes a *minimal* operand-description projection (the five facts your §3.1
   table actually reads — strides, dtype, alignment, quant, symbolic extent),
   not the whole `FDXSidecar`. We ship `From<FDX>` / `From<DLTensor>` /
   `From<TensorRef>` adapters so the key is computable from a Fuel call, a bare
   DLPack handle, *or* Baracuda's own views — keeping it usable outside Fuel and
   the shared dependency small.
3. **Co-freeze the `ImplId` wire encoding** — accepted on the basis tuple
   `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`. B.7's
   *separate* `kernel_revision_hash` field already gives us what we need
   (group by `(BackendId, op, dtypes, kernel_source)` for matrix ranking; use
   the full tuple for exact re-resolution) — **item 2** is only that the
   wire bytes serialize the five fields independently, never as a single opaque
   hash. Ready to freeze on that basis.
4. **`flipped` = signed stride** — accepted, trivially. Our canonicalization
   (design doc §4) **preserves** flip as a key axis and never normalizes it away
   — a flipped-but-contiguous operand is its own cell, because a flip-
   specialized kernel is a distinct kernel. Your 2026-06-17 negative-strides-
   first-class reversal is precisely what keeps the axis observable end-to-end;
   without it our Ask 2 could never see flip demand. Noted and depended upon.
5. **Miss = "best admissible match = generic contract"** — accepted; no bolt-on
   detector either side. Its fidelity rests on our specialized kernels
   registering with *exactly* correct tight predicates — which the §3 counter
   guarantees by construction (the predicate ladder in B.4 was built axis-for-
   axis onto our `OperandKey`, so a generated contract's admissibility predicate
   *is* its structure key).
6. **Review telemetry schema + FKC** — wire schema (§4.5) accepted as-is:
   JSONL, `Off`/`Coarse`/`Detailed`, `schema`-versioned, v1 ⊂ v2. FKC accepted
   via the §3 counter; B.8 confirms the format expresses our hardest kernel
   (flash-attn) — see §5.

## 3. The shared model — Baracuda *generates* FKC

We're aligned here: you and we already expect Baracuda to emit FKC automatically
(hand-authoring stays available to other providers, never required). Recording
the mechanism and the guarantee it buys. Baracuda already holds structured
per-kernel truth:
`KernelSku` (category / op / element / layout / backend), `PrecisionGuarantee`
(math precision / accumulator / determinism), and OP-MATRIX. **FKC is a
projection of that, not a second source of truth.** Two kernel classes:

- **Bespoke hand-written kernels** (flash-attn, the existing op families, vendor
  facades) — a small, bounded set. Their FKC contract is generated *from the
  `KernelSku` + `PrecisionGuarantee`*, not typed by hand. The B.8 flash-attn
  contract is exactly what that generator emits.
- **Specialization-matrix kernels** (the AOT structure cells — potentially
  hundreds per op family). Their FKC contract is generated *alongside the `.cu`*,
  from the structure key, during codegen. One generator, four outputs per cell:
  the `.cu`, the dispatch-key code, the FKC contract, and the `link_registry`
  entry.

Consequences, all of which serve Fuel:

- **Your miss signal is honest** — generated predicates can't under- or over-
  declare admissibility, because they *are* the structure key (closes ask 5).
- **Zero drift** between our SKU/OP-MATRIX and the advertised contracts.
- **Zero hand-maintenance tax** across a matrix that grows over releases.
- **Your importer (B.9) is unchanged** — you parse the same ` ```fkc ` blocks
  and resolve the same `link_registry`; we just author them mechanically.

This is the same generator the design doc already specifies for the `.cu`; we're
adding two emit targets to it.

## 4. Pre-freeze dtype reconciliation (narrowed)

Correcting our opening note, which was too broad. Baracuda's `Element` set
(verified in
[`element.rs`](../crates/baracuda-kernels-types/src/element.rs)): `f16`, `bf16`,
`f32` / `F32Strict`, `f64`, `i32`, `i64`, `Bool`, `Complex32` (2×f32),
`Complex64` (2×f64), `S8`, `U8`, `S4`, `U4`, `Fp8E4M3`, `Fp8E5M2`, `Bin`.

Because the base `DLTensor.dtype` is standard `DLDataType` and FDX pins DLPack
v1.3, the types DLPack v1.3 can name ride the **base** honestly and need **no
FDX code**:

- `Fp8E5M2` → `kDLFloat8_e5m2`; `Complex32`/`Complex64` → `kDLComplex` (bits 64
  / 128); `Bool` → `kDLBool`; unpacked 4-bit int → `kDLInt`/`kDLUInt` bits 4.
- **Please confirm FDX honors these base v1.3 codes** rather than shadowing them
  with its `FDX_DTYPE_*` enum. If it honors them, there is no gap here.

The **genuine** gaps are in the sidecar logical-dtype namespace
(`FDXDTypeExt.logical_dtype`, which today lists only `F4`=13 among sub-byte):

1. **A 4-bit *integer* logical code** for packed `S4`/`U4` (stored two-per-byte;
   the base appears as opaque `u8`, so the meaning rides the sidecar sub-byte
   path — which cannot currently name a 4-bit int). Propose `I4` / `U4` logical
   codes.
2. **A 1-bit code** for `Bin` (bitpacked binary — no standard DLPack
   representation at all). Propose a `B1` logical code.

Naming caution for the boundary map: our `Complex32` = 2×f32 = numpy
`complex64`; our `Complex64` = 2×f64 = numpy `complex128` (we count per-component
bits, DLPack counts total) — map these unambiguously so a `complex64` handle
isn't misread.

## 5. Precision / determinism (B.6) ↔ `PrecisionGuarantee`

B.6 maps cleanly onto our existing `PrecisionGuarantee`. The generator emits the
FKC `precision` block from it; `audited` reflects our real audit state (no
`UNAUDITED` kernel ships silently — your CI flag and our audit discipline
agree), and `determinism` projects from our determinism flags. B.8's contract
(`audited: true`, `max_relative: 0.005`, `determinism: nondeterministic` for
online-softmax + warp-reduction order) is exactly how our attention kernels
declare — confirming the format is expressive enough for our hardest case, not
just elementwise.

One mapping note: `F32Strict` is a precision *mode* over f32 storage, not a
distinct `DLDataType`. It surfaces in the `precision` block (bit-stability /
accumulate width), never as a wire dtype — so it never reaches the FDX dtype
namespace.

## 6. Commitments and what we need from you

**Baracuda commits:**

- DLPack (v1.3) as the ecosystem boundary; nullable `const FDXSidecar*` on the
  Fuel-facing ABI via a vendored optional header.
- Ship `structure_key(op_class, operands, arch)` as the single canonical
  callable — minimal operand-description projection + `From` adapters; versioned
  independently of FDX / FKC / telemetry / DLPack.
- Co-freeze the `ImplId` wire encoding on the basis tuple, fields separable.
- Generate FKC contracts + `link_registry` for all Baracuda kernels (bespoke
  from the SKU, specialized from the structure key); map `PrecisionGuarantee` →
  FKC `precision`; declare `audited` honestly.
- Consume the telemetry feed coverage-agnostically; rank the AOT matrix by miss
  `count` first, layer in vendor-exclusion as Judge coverage densifies.

**We need from you:**

- Add `I4` / `U4` and `B1` sidecar logical codes; confirm base-DLPack
  passthrough for `fp8e5m2` / complex / bool / unpacked-4-bit (§4).
- Keep the `ImplId` wire fields separable (item 2).
- Confirm the minimal `structure_key` input projection is what you'll feed
  (item 1).
- Confirm our FKC being a *generated* projection is fine — it changes nothing in
  your importer, only our authoring model (§3).

## 7. Coverage-gating — acknowledged

Q1 = YES accepted. Coverage is F32 / square-matmul today, so Ask 1's payoff —
the vendor-exclusion gate from `candidates[]` — is **deferred** until Judge
coverage grows, while Ask 2's miss histogram is coverage-independent and on our
critical path. We rank the matrix by miss `count` first; `candidates[]`
densifies automatically with no format change as your matrix fills. Aligned with
your §4.3.

## 8. Process / next steps

Propose-first; nothing frozen on either side. The dependency that gated your
reply (Judge retention) is closed; the dependency that gates *ours* is the
`structure_key` callable — which is the same dispatch-key code the design-doc
pilot builds first, and which now does triple duty: the telemetry tag, the FKC
predicate generation, and runtime dispatch. So our blocking deliverable and the
specialization pilot are the same first step.

Next:

1. **You:** add the two sidecar dtype codes; confirm base-DLPack passthrough.
2. **Both:** freeze the `ImplId` wire encoding (basis settled; fields separable).
3. **You:** build the emission layer over the confirmed retention.
4. **Us:** ship `structure_key` + the FKC/`link_registry` generator as part of
   the elementwise-pilot codegen, and stand up the consumer that turns the v1
   miss histogram into a build matrix.
