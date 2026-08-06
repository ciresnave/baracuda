# kernelgen IR — layout/shape frontier: design-session kickoff brief

**Purpose.** Zero-context grounding for the design session on **layout/shape nodes** —
the IR ramp's most-foundational pending piece. Read this first, then brainstorm the
design collaboratively with Eric. Verified against code on 2026-07-31; where this brief
and an older doc disagree, trust the code (see *Staleness* below).

---

## Headline — corrects the stale "layout/shape gates MatMul" framing

The older framing (`docs/planning/foundational/12-ir-expansion-roadmap.md`) reads as if
MatMul doesn't exist in the IR yet. **It does.** `Access::Contraction` (`ir.rs:1165`) is a
shipped node emitting real rank-2 / rank-3 batched **dense row-major** matmul kernels,
and layout/shape `View` nodes (`ir.rs:1414`) are shipped and lowered (Permute / Broadcast).

The real gap is a **scope boundary, not a missing node**:

> **A non-Identity `View` is `Access::Elementwise`-ONLY in v1** — hard-asserted at
> `plan.rs:1043` ("a non-Identity View is Access::Elementwise-only … a {class}-op has its
> own axis machinery"), backstopped at `cuda.rs:1757`.

So a `View` (Permute / Broadcast / stride) **cannot currently ride a Contraction
operand.** That is the literal gate. Extending views to compose with `Access::Contraction`
(and the other non-Elementwise access classes) — folding a transpose/stride into the
M/N/K address math instead of materializing a `contiguize` copy — is what this session
designs. That unblocks transposed / column-major / mixed-layout / batched matmul, i.e. the
**SDPA-decomposition path** (QKᵀ as a batched Contraction + softmax RowReduce),
FlashDecoding++ flat-GEMM, sparse24 GEMM B-transpose, rope fw/bw, pixel_shuffle, fftshift,
repeat_bw.

---

## What is SHIPPED (do not redesign these)

- **8 Access classes** (`ir.rs:1103`): `Elementwise`, `Reduction{axes,keepdim,post}`,
  `RowReduce{stages,epilogue}`, `Contraction{axes,accum,epilogue}`, `Scan`, `Window`,
  `RowSort`, `Im2Col`. (Doc 12 §2's "four arms" is stale.)
- **Contraction** (`ir.rs:1165`, emitter `cuda.rs:3253`, schedule `plan.rs:383`): rank-2 +
  rank-3 batched, dense **row-major** `[M,K]·[K,N]→[M,N]`, `AccumSpec::WideFloat` only,
  `Reduced(0)` epilogue + optional fused bias `Input(2)`. Scope pinned at `plan.rs:360-371`.
  Axis vocabulary: `AxisRole{Batch,FreeM,FreeN,ContractedK}` (`ir.rs:1359`),
  `ContractionAxes::{matmul,batched_matmul}` (`ir.rs:1375`).
- **View** (`ir.rs:1429`): `Identity | Permute{perm} | Broadcast{bcast} | Reshape{producer_rank}`.
  Permute/Broadcast **fully lowered** (`cuda.rs:6340-6355`, `offset_expr`/`perm_of`), gated by
  `plan::assert_valid_views` (`plan.rs:1027`). Reshape is **recognition-only** (same-rank,
  `ir.rs:1449`). Views are **INPUT-only** — there is no output view (`cuda.rs:1105`: "the
  OUTPUT is never viewed"). Design rule (`ir.rs:1418`): a View is a per-operand read-through
  kept OUT of `ScalarExpr` (value math) and OUT of `Access` (loop-nest shape) — the
  value-math walkers never see it. **Preserve this separation.**
- **§6.20 shape-oracle** (`shape.rs:63`, `output_shape`): already derives the output shape
  for *every* Access variant including Contraction (role-derived `batch ++ [M,N]`). **The
  session does NOT invent matmul output-shape math — it reads the frame from `output_shape`.**
  Wire codec in `baracuda-kernel-vocab/src/shape_expr.rs`.
- **CPU differential oracle** (`oracle.rs:1053`, `oracle::evaluate`): validates any new
  lowering bit-exactly against a reference. Every new layout/shape lowering must pass it.
- Also shipped: multi-output elementwise, hetero-dtype outputs, gather (`ReadIndex`),
  scatter (`WriteIndex`+`WriteCombine`), runtime base-offset slice (`BaseOffset`), full
  scalar vocab (~40 unary + the binary/cmp/bitwise/logical set + `Coord` iota + int
  reductions incl. the S8/U8 work just shipped).

---

## The FRONTIER — what this session designs

**Most-foundational pending piece: layout/shape nodes that COMPOSE with non-Elementwise
access classes, primarily Contraction operands.** Concretely, the pending set:

1. **View × Contraction composition.** Fold `Permute`/`Broadcast`/stride on an lhs/rhs
   operand into the M/N/K address math. Blocked by `plan.rs:1043`. The direct unblock for
   transposed/mixed-layout/batched matmul + the SDPA path.
2. **Output views (`write_slice`).** `View` is input-only today; sliced / in-place output
   writes need an output-side view or an extension of `out_base_offset`
   (`ir.rs:1888`/`:1893`). Interacts with `WriteIndex` scatter (already output-side).
3. **Genuine rank-change `Reshape`.** Move from recognition-only to emitting real
   rank-change address math — needed for the im2col→GEMM→reshape Conv2D fusion (`ir.rs:1341`).
4. **General `ContractionAxes` role vectors** vs. the two hardwired `matmul()` /
   `batched_matmul()` constructors (`plan.rs:366`) — the einsum growth path (`ir.rs:1370`).
5. **Negative / stride-2 strides folded into views**; TF32 / tensor-core `AccumSpec`
   variants (reserved, `ir.rs:1410`).

## Open design questions — the session's agenda

1. **View × Contraction semantics.** Does a view fold into the operand stride *before*
   M/N/K role assignment, or become a distinct layout class inside `ContractionKey`
   (`structure_key.rs:239`)? Today they're mutually exclusive (`plan.rs:1043`).
2. **Output views.** What is the type shape for an output-side view, and how does it
   interact with `WriteIndex` scatter (already output-side) and `out_base_offset`?
3. **Rank-change Reshape.** Does a genuine reshape re-key the `StructureKey` rank, or stay
   a per-operand read-through?
4. **General einsum roles.** Type/validation for arbitrary `ContractionAxes` role vectors.
5. **Extent-agreement trust tier.** Views declare no concrete extents — agreement is a
   caller precondition, not plan-validated (`ir.rs:1461`, `structure_key.rs:302`: the key
   carries size *classes*, never extents). Does the same trust tier hold when a view rides a
   contraction operand, or does contraction need stronger validation?
6. **Shape-oracle reuse + the FKC boundary.** Confirm the lowering reads its frame from
   `shape.rs::output_shape` rather than re-deriving; decide the
   `NeedsReservedConstructor` / Fuel-#80 boundary for any new multi-axis layout arithmetic
   (the shape-oracle deferred `contract.rs shape_rule` emit + the `Dims`/`WithDim` encoder,
   both Fuel-evaluator-gated — see the shape-oracle design doc §"Out of scope for v1").

## Design constraints / invariants to preserve

- **"CUDA-shaped IR node is residue" rule** (`docs/design/ir-translation-hub.md:185`): a
  layout node that only makes sense on one backend is not IR. Layout/shape nodes must be
  backend-neutral — the IR is the translation hub.
- **View stays out of `ScalarExpr` and out of `Access`** (`ir.rs:1418`) — a per-operand
  read-through, invisible to value-math walkers. Don't leak layout into either.
- **Structure key carries size CLASSES, never concrete extents** (`structure_key.rs:302`);
  extent agreement is a caller precondition (`ir.rs:1461`).
- **KISS alignment.** The IR is the KISS-Ops-aligned neutral representation; layout/shape
  semantics should track KISS's shape/layout model — coordinate with the KISS/Fuel peers
  (Baracuda is the network POC). The recent `cuda:` namespace + shape-oracle work is the
  live seam.
- **Determinism / exact-byte.** Any new lowering is validated by the CPU differential oracle
  (`oracle.rs`) and must match the reference bit-for-bit.

## Staleness map — trust the code, not these

- `docs/planning/foundational/12-ir-expansion-roadmap.md` — the ramp plan. §2 "four Access
  arms" is stale (8 exist); lists layout/shape (#3) + contraction + gather/scatter/scan/
  window/sort as *pending* — all shipped (scoped). **Content accurate, framing stale.**
  The frontier bullets at `:60`/`:69`/`:75` are still the right target list.
- `docs/design/axis-role-vocabulary.md` — cites the OLD crate path
  (`baracuda-kernels-types` → renamed `baracuda-kernel-vocab`); `ContractionKey` now at
  `structure_key.rs:239`. Live insight (`:37`): "the roles keystone collapsed to
  recognition-only" — layout/shape shipped in a *reduced* form (this session's start point).
- **CURRENT (trust):** `docs/superpowers/specs/2026-07-23-shape-oracle-design.md` (+ its
  plan), `docs/design/ir-translation-hub.md`, `docs/design/oracle.md`.

## File cheat-sheet

| what | where |
|------|-------|
| IR core (types, `Access`, `View`, `ContractionAxes`) | `crates/baracuda-kernelgen/src/ir.rs` |
| planner + **the `plan.rs:1043` Elementwise-only view gate** | `crates/baracuda-kernelgen/src/plan.rs` |
| CUDA emitter + view lowering (`offset_expr` ~6340, `emit_contraction` 3253) | `crates/baracuda-kernelgen/src/cuda.rs` |
| shape oracle (`output_shape`) | `crates/baracuda-kernelgen/src/shape.rs` |
| CPU differential oracle | `crates/baracuda-kernelgen/src/oracle.rs` |
| keys / `OperandDesc` / `ContractionKey` | `crates/baracuda-kernel-vocab/src/structure_key.rs` |
| shape wire codec | `crates/baracuda-kernel-vocab/src/shape_expr.rs` |
| device `TensorRef` bridge | `crates/baracuda-kernels-types/src/operand_desc_ext.rs` |

## How to run the session

This is a **design** session → open with `superpowers:brainstorming` (explore context →
propose 2-3 approaches with trade-offs → present the design in sections, one question at a
time). The design calls — View×Contraction composition, output-view type shape, reshape
rank-change, einsum roles — are **Eric's architecture**; drive them collaboratively, don't
pre-decide. Then `superpowers:writing-plans` → `superpowers:subagent-driven-development`,
validating every new lowering against `oracle.rs`. The just-shipped §6.20 shape-oracle
(shape derivation) and the CPU oracle (value validation) are the two rails the design sits
between.
