# IR layout/shape for non-Elementwise access classes — parent design

**Status:** design approved (brainstorm, 2026-08-01). Parent of sub-specs A (detailed,
this cycle) and B/C/D (scoped stubs, each earns its own brainstorm when started).
**Grounded against code 2026-07-31/08-01.** Companion kickoff brief:
`docs/design/2026-07-31-ir-layout-shape-design-brief.md`.

---

## 1. Purpose & reframe

The kernelgen IR ramp's most-foundational pending piece is **layout/shape nodes that
compose with non-Elementwise access classes** — primarily `Access::Contraction` operands.
This is *not* a missing node. `Access::Contraction` (rank-2 + rank-3 batched dense
row-major matmul) and layout/shape `View` nodes (Permute/Broadcast, lowered) are both
**shipped**. The gap is a **scope boundary** enforced by **two gates**:

1. `plan.rs:1043` — a non-Identity `View` is `Access::Elementwise`-only (a view cannot
   ride a Contraction operand).
2. `structure_key.rs:616` — `derive_contraction`'s `dense()` predicate **rejects** any
   non-row-major operand (returns `None` → the cell is not a contraction).

Even lifting gate (1) leaves gate (2): the vocab crate refuses to produce a
`ContractionKey` for a transposed operand. This parent design lifts **both**,
consistently, so a transpose/permute/broadcast folds into the M/N/K address math instead
of materializing a `contiguize` copy — unblocking transposed / column-major / mixed-layout
/ batched matmul and the **SDPA-decomposition path** (QKᵀ as a batched Contraction).

## 2. The spine (shared representation)

A contraction operand's layout is a **discrete class in `ContractionKey`** (so it
**re-keys** → distinct, coalescing-legible, independently-schedulable cells), lowered
through **one stride-binding emitter**. The two vectors the IR *already has* carry it:

- `ContractionAxes.roles` (`ir.rs:1375`) — **semantic**: which axis is `FreeM` / `FreeN` /
  `ContractedK` / `Batch`.
- `View::Permute` per operand (`ir.rs:1436`) — **physical**: storage order.

The key stores only their **class projection**. The emitter derives each role's stride as
an **extent-product** from storage order — **no runtime stride launch-args** in v1 (extents
`m/n/k/batch` stay launch args exactly as today; arbitrary non-packed strides that need
real stride args are sub-spec D).

> **Stride binding.** `stride(role R on operand O)` = ∏ extents of O's axes **storage-inner**
> to R's axis. Address = `Σ_roles coord(role)·stride(role)`. Today's `mm*k + kk` is the
> Canonical special case; a permuted operand yields the swapped binding, still all
> extent-products. Broadcast axes drop their term (stride 0), exactly as the elementwise
> `offset_expr` already does.

## 3. The three-place change pattern (every sub-spec)

1. **`baracuda-kernel-vocab`** (`structure_key.rs`): generalize the `dense()` bool
   (`:616`) into `classify_mat_layout` returning a layout **class**; stop rejecting
   non-row-major; carry the class in `ContractionKey`.
2. **`plan.rs`**: lift the gates (`:1043` Elementwise-only view gate; `:366`
   `matmul()`/`batched_matmul()` role pin); validate `View` structural legality +
   role-vector legality.
3. **`emit_contraction`** (`cuda.rs:3269`): read the layout class → stride binding
   (one binding replaces the three hardcoded offsets).

## 4. Invariants preserved (verbatim from the brief)

- **View stays out of `ScalarExpr` and `Access`** (`ir.rs:1418`) — consumed into key +
  binding at plan time; value-math walkers never see it.
- **Structure key carries size/layout CLASSES, never concrete extents** (`structure_key.rs:302`).
- **Extent/stride agreement is a caller precondition** (`ir.rs:1461`) — the plan gate
  validates only structural legality (a `Permute` is a true permutation; a role vector is
  well-formed). Same trust tier as today (answers open-question Q5).
- **Backend-neutral IR** — a node meaningful on only one backend is residue
  (`docs/design/ir-translation-hub.md:185`).
- **Determinism / exact-byte** — every new lowering is CPU-oracle bit-validated
  (`oracle.rs:1053`) and, for headline cells, confirmed on the RTX 4070 through the
  `gpu-run` lock (see §7).

## 5. Sub-spec map & dependency order

| | Sub-spec | Frontier items | Depends on | This cycle |
|---|---|---|---|---|
| **A** | General contraction operand **roles + layout** (transpose · broadcast · rank-3 role order) | 1 + 4 (merged) | — *keystone* | **detailed** → `…-subspec-a-roles-layout-design.md` |
| **B** | **Output views** / `write_slice` | 2 | A | scoped stub (§6.1) |
| **C** | Genuine **rank-change Reshape** | 3 | A | scoped stub (§6.2) |
| **D** | **Arbitrary / negative / stride-2 strides** (runtime stride args) | 5 (layout half) | A | scoped stub (§6.3) |

**A is the keystone.** B/C/D each extend the stride-binding machinery A introduces; each
earns its own detailed brainstorm when started (do not pre-design cold).

## 6. Scoped stubs (B/C/D)

### 6.1 Sub-spec B — Output views / `write_slice`
Today `View` is input-only (`cuda.rs:1105` "the OUTPUT is never viewed"); the output offset
is `out_base_offset` (`ir.rs:1888`). **Boundary:** an output view is the symmetric dual of
the input `View` — a per-output *write*-through riding A's stride-binding. **v1 scope:**
Permute / identity-Reshape on the output (sliced/transposed writes). **Open call B resolves
(Q2):** reuse the `View` enum on the store side vs. extend `out_base_offset` into a richer
output descriptor; and whether an output View and a `WriteIndex` scatter are mutually
exclusive in v1 (scatter is already a data-dependent output remap).

### 6.2 Sub-spec C — Genuine rank-change Reshape
Move `Reshape` from recognition-only (same-rank, `ir.rs:1449`) to real rank-change address
math — the missing piece of the im2col→GEMM→reshape Conv2D fusion. **Boundary:** a
rank-change reshape of a **contiguous** producer is a pure linear-index remap (same element
order, different rank). **v1 scope:** contiguous producers only (non-contiguous reshape
needs materialization → decline). **Open call C resolves (Q3):** does it re-key
`StructureKey.rank`, or stay a per-operand read-through where the emitter unravels at
`View::Reshape{producer_rank}` and re-ravels at the consumer rank (keeping View out of
`Access`)?

### 6.3 Sub-spec D — Arbitrary / negative / stride-2 strides
The point where A's binding gains **runtime stride launch-args** (extent-product → runtime
arg, keyed by a "runtime-strided" contiguity class); negative strides fold via the existing
`OperandKey.flipped` (`structure_key.rs:285`) + a signed term. **v1 scope:** the general
non-packed operand. A's natural extension, not a new mechanism.

## 7. Validation discipline

Every new lowering passes the **CPU differential oracle** (`oracle::evaluate`, `oracle.rs:1053`)
bit-exactly against a reference before it is considered correct. Headline cells (transposed-rhs
QKᵀ, batched, GQA broadcast-KV) are additionally confirmed on the **RTX 4070** (sm_89) — every
on-device run routed through the machine-wide **`gpu-run` lock** (`pwsh scripts/gpu-run.ps1
-Project baracuda -- <cmd>`) per the 2026-07-31 host-aperture-crash coordination.

## 8. Roadmap — out of this frontier but TRACKED

- **TF32 / tensor-core `AccumSpec` variants** (`ir.rs:1410`, reserved). This is a
  **compute-precision** axis (accumulator + MMA path, honest KISS-Contract §6.8 contract
  flips), NOT a layout/shape node — so it is deliberately pulled OUT of this parent to keep
  the layout/shape charter clean. **It remains on the roadmap and must be done** as a
  separate **precision-variants initiative** (its own brainstorm). Recorded here so it is
  not lost when item 5's layout half (sub-spec D) ships without it.
- **Multi-K-group einsum** (contracting several axis groups) — deferred past sub-spec A's
  single-K v1; a genuine accumulator-structure change.

## 9. Shape-oracle reuse & the FKC boundary (Q6)

The lowering reads its output frame from `shape.rs::output_shape` (already role-derived
`batch ++ [M,N]`, `shape.rs:150`) — it does **not** re-derive matmul output-shape math. Any
NEW multi-axis layout arithmetic beyond what `output_shape` provides defers to the
Fuel-evaluator-gated reserved constructor (`NeedsReservedConstructor` / Fuel-#80), per the
shape-oracle design doc's "Out of scope for v1."

## 10. References

- **Kickoff brief:** `docs/design/2026-07-31-ir-layout-shape-design-brief.md`
- **CURRENT (trust):** `docs/superpowers/specs/2026-07-23-shape-oracle-design.md`,
  `docs/design/ir-translation-hub.md`, `docs/design/oracle.md`
- **Sub-spec A detail:** `docs/superpowers/specs/2026-08-01-ir-layout-shape-subspec-a-roles-layout-design.md`
