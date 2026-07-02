# 10 — MatMul/contraction node DESIGN SPIKE — implementation brief

> **This is a DESIGN SPIKE, not an implementation.** The deliverable is a design document
> (grammar, IR variant, schedule axes, StructureKey facts, FKC contract shape, vendor-exclusion
> gate) plus the precise set of things items **01 (layout/shape nodes)** and **02 (DAG-with-
> consumer-counts)** must provide so they are shaped correctly *now*. No `.cu` emitter, no
> `Access::Contraction` merged to `main`. A prototype single-cell kernel is optional and
> gated behind a feature/experiment dir — see §6 step 7.

---

## 1. Objective

Design the terminal **ORDER-3 contraction node** for `baracuda-kernelgen`: the IR grammar, a new
`Access` variant, the schedule axes (K-tiling, register/shared blocking, Tensor-Core fragment
selection, double buffering), the `StructureKey` facts (M/N/K structure classes, alignment, arch),
the FKC/contract shape, and the **§7 per-cell vendor-exclusion gate**. This is foundational because
it is the *last* algorithm class the generator cannot express (ir.rs today is elementwise +
last-axis reduction + fused RowReduce, a pure `ScalarExpr` **tree** — no contraction fits). The §1
thesis (kernel-specialization.md:27–32) scopes this node **explicitly to the fused long tail** —
small/skinny/irregular GEMM and especially fusions across a layout change (the FlashDecoding++
flat-GEMM cell is the canonical "generate" target) — **not** large aligned GEMM, which already
routes to the mature `baracuda-cutlass`/`baracuda-cublas` head (see §4). Getting the design right
now lets items 01 and 02 land with contraction-shaped hooks instead of being reshaped later.

## 2. Status & blockers

- **Design-open (this brief's core).** The contraction node has no agreed shape. This is the
  spike that produces one. **Baracuda can produce the entire design now** — it depends only on the
  *designs* of 01 and 02, not their landed code (§3).
- **Blocked on 01 + 02 for any implementation.** A contraction needs (a) shape/axis facts that
  today's `StructureKey` deliberately abstracts away (extents are bucketed to `DivBucket`, not
  carried numerically) and (b) a multi-consumer DAG (a contraction result is almost always consumed
  by an epilogue and often re-read). Neither exists: `ScalarExpr` is a tree (ir.rs:16–47), and
  `Access` has no shape/axis node (ir.rs:279–305). **Implementation cannot start until 01 and 02
  land; the design must be finished first so they land correctly.**
- **Fuel-adjacent, not Fuel-blocked for the design.** The FKC/seam wiring for a generated
  contraction (a new `OpTag`/region shape) will need a cross-repo answer (§10), but the design half
  — the Baracuda-side IR/schedule/StructureKey/gate — proceeds with no Fuel dependency.
- **Not blocked by the stale doc.** `docs/design/kernel-specialization.md` lists MatMul under
  "ORDER 3 — pending" (line ~432) and its open-questions (~443–453) still frame the algorithm-IR
  choice as open; treat that as context, not current status. It is stale on shipped items (it still
  calls `Param`/`AddScalar`/`MulScalar` not-emittable at ~423–426 though they shipped).

## 3. Dependencies & sequencing

**Must land BEFORE any implementation of this node (design must precede them so they are shaped right):**

- **01 — layout/shape nodes (keystone).** Provides the axis/shape machinery a contraction is
  *defined over*: named contraction axes (K), batch axes, free axes (M, N), and the per-operand
  layout facts (row/col-major, leading dimension, transpose) that pick the Tensor-Core fragment and
  the vendor-vs-generate gate. Without shape facts the emitter cannot form the M/N/K loop nest or
  decide alignment. **01 must expose the exact facts enumerated in §5.5.**
- **02 — DAG-with-consumer-counts (keystone).** A contraction is a **producer with a distinct
  value identity** feeding an epilogue (bias/activation) and, in fusion, re-read by other ops. The
  current `ScalarExpr` tree cannot represent "the matmul result, consumed twice." 02 must give
  contraction results a node identity with a consumer count (feeds accurate FKC `consumers:>1` and
  CSE). **02 must expose the facts in §5.5.**

**What this ENABLES / relates to downstream:**

- Consumes **03 (strided/multi-axis reductions)** conceptually — a contraction *is* a
  reduction over the K axis fused with two free axes; 03's non-last-axis reduction axis machinery
  (built on 01's shape facts) is the same axis vocabulary the K-loop uses. Coordinate the reduction-
  axis representation so a K-contraction and a 03 reduction name axes the same way.
- Feeds **07 (per-arch dispatch table + bench-gate harness)** — the §7 vendor-exclusion gate for
  contraction cells is decided by 07's benchmark gate (large-aligned → cuBLAS/CUTLASS; small/skinny
  → generate). This node *defines the cells*; 07 *measures the winner*.
- Feeds **08 (telemetry variant-selection)** — the FlashDecoding++ flat-GEMM demand signal is
  exactly the kind of `miss_record` 08 ingests; the contraction StructureKey is that record's join
  token.
- Coordinates with **09 (f16/bf16 half2 packed-SIMD)** and the fragment-selection schedule axis —
  Tensor-Core MMA fragment types are the half-precision emit path 09 refines.
- Independent of **04, 05, 06** (integer-accum reductions, RowReduce seam, residual-add LayerNorm).

## 4. Current code — what exists today

**No contraction fits the IR.** `ScalarExpr` (ir.rs:16–47) is a pure per-output-coordinate **tree**
of `Input/Const/Param/Reduced/Add/Sub/Mul/Div/Unary/Binary`. The doc comment (ir.rs:1–8) is explicit
that an op is "the *pure function* computed at each output coordinate" — a contraction is not a
pointwise function of one output coordinate; it sums a product over a shared K axis. `Access`
(ir.rs:279–305) has exactly three variants — `Elementwise`, `Reduction { op }` (last-axis, contiguous,
float-only), `RowReduce { stages, epilogue }` — none of which carries a *second* free axis or a
*shared contraction* axis. `#[non_exhaustive]` (ir.rs:280) already anticipates growth.

**The live §5 seam is elementwise-only and rejects MatMul as an honest miss:**

- `region_to_op` hardcodes `access: Access::Elementwise` (jit.rs:419) — every synthesized region is
  elementwise by construction.
- `synth_op`'s infix table falls through to `UnsupportedOp` for any name it doesn't know
  (jit.rs:457–463); `MatMul` hits that arm.
- The seam `optag_name` comment enumerates `MatMul` among the tags "not synthesized" (jit.rs:693–694),
  returning `None` → `JitError::UnsupportedOp`.
- The regression test `unsupported_op_is_rejected` pins this: a `MatMul` region asserts
  `JitError::UnsupportedOp("MatMul")` (jit.rs:1100–1107). **This test is the honest-miss contract we
  must preserve until the node ships** (it flips to a positive synthesis test only when the node lands).
- `derive_pattern` rejects any non-`Elementwise` access with `PatternError::NotElementwise`
  (pattern.rs:79–82), so even a hand-built `Access::Contraction` op would honest-miss the contract
  path until pattern.rs learns the shape.
- `build_plan` (plan.rs:85–117) has no contraction arm; `Schedule` (plan.rs:18–51) has no
  contraction schedule.
- `cuda.rs` `lower` dispatches on `plan.schedule` (cuda.rs:37–50) across `emit_vectorized`/`emit_scalar`/
  `emit_strided`/`emit_reduction`/`emit_row_reduce` (cuda.rs:122,159,187,265,397) — no contraction emitter.

**StructureKey already reserves the GEMM slot but carries none of the facts a contraction needs:**

- `OpCategory::Gemm` exists (sku.rs:31–35) with token code `"gem"` (structure_key.rs:862) —
  the category is *namable* today, so `structure_key(OpCategory::Gemm, …)` already produces a valid
  token. The head's GEMM surface tags under it (OP-MATRIX.md:32–43).
- But `StructureKey` (structure_key.rs:187–216) carries **no** M/N/K classes, **no** transpose/leading-
  dimension fact, and `reduce_axes` is documented "always empty in v1" (structure_key.rs:213–215,
  423). Extents are bucketed to `DivBucket` (structure_key.rs:99–112) and squeezed — the key
  deliberately "specializes on structure, not extents" (structure_key.rs comment; kernel-specialization.md
  §1 non-negotiable framing). A contraction needs a *few* extent-derived structure classes (§5.4) that
  the current key does not compute.

**The head already owns large GEMM (this is why §7 scopes us to the long tail):**

- `baracuda-cutlass` (README): `GemmPlan`/`BatchedGemmPlan`/`GroupedGemmPlan` over
  `{f16,bf16,f32(TF32),F32Strict(SIMT),f64(DGEMM)}` × `{Rcr,Rrr}` layouts, epilogues
  `{Identity,Bias,BiasRelu,BiasGelu,BiasSilu}`, `IntGemmPlan`, all plan-based and capture-safe.
- `baracuda-cublas` (README): full L3 GEMM, `GemmEx`, batched/strided-batched, cuBLASLt heuristics.
- OP-MATRIX.md:32–43: `GemmPlan`, `BatchedGemmPlan`, `GroupedGemmPlan`, `IntGemmPlan`,
  `Fp8GemmPlan`, `Int4GemmPlan`, `BinGemmPlan` — a mature, tuned surface. **The generated node must
  NOT reimplement any cell this surface already wins.** Its job is the cells the head leaves generic.

## 5. Design / delta

### 5.1 The grammar — how a contraction is expressed (given 01 + 02)

A contraction is an **Einstein-summation-shaped** node: it names, per operand, which axes are
**batch** (b), **free** (M for lhs, N for rhs), and **contracted** (K, shared, summed). The output
frees over {b, M, N}; K is summed. Keep it single-contraction-axis-group in v1 (a batched matmul:
`out[b,m,n] = Σ_k lhs[b,m,k] · rhs[b,k,n]`), because that is the FlashDecoding++/skinny-GEMM cell
and the only one the long-tail thesis targets. Higher-order tensor contractions (multi-K einsum) are
a deliberate follow-up.

The node **references 01's shape/axis facts** rather than re-encoding them: axes are 01's axis
handles, layout (row/col-major, leading dim, transpose) is 01's per-operand layout fact. The
**epilogue** (bias add + activation) is expressed with the *existing* `ScalarExpr` vocabulary,
reusing the `RowReduce { epilogue }` precedent (ir.rs:299–304): the contraction produces a
`Reduced`-like scalar-per-output-cell that an epilogue `ScalarExpr` consumes — so bias/relu/gelu
epilogues cost **zero new emitter vocabulary**, exactly as CUTLASS's `Bias*` epilogues do
(cutlass README:27–36).

### 5.2 New IR — `Access::Contraction`

Add a non-exhaustive variant to `Access` (ir.rs:279–305). Sketch (names illustrative; final axis
representation is 01's to define — see §5.5):

```rust
/// Batched contraction: out[b, m, n] = epilogue( Σ_k lhs[b,m,k] · rhs[b,k,n] , inputs… ).
/// One contracted axis group (K), free axes (M for input 0, N for input 1), shared batch axes.
/// The reduction over K is fused with the two free axes — NOT expressible as Reduction/RowReduce.
Contraction {
    /// Which operand axes are batch / free-lhs (M) / free-rhs (N) / contracted (K),
    /// expressed in item-01's axis-handle vocabulary (§5.5). Not raw indices — 01 owns this.
    axes: ContractionAxes,
    /// The K-accumulation combine + accumulator dtype policy (see §5.4 acc-precision).
    accum: AccumSpec,
    /// Per-element output epilogue over the K-reduced scalar (`Reduced(0)`) and the
    /// pointwise inputs (bias = Input(2) column-broadcast, etc.). Reuses ScalarExpr.
    epilogue: ScalarExpr,
}
```

`OpDef::body` continues to hold the epilogue (mirrors `row_reduce`'s `body = epilogue`,
ir.rs:378) so existing body-walkers (`params_used`/`count_flops`/`ulp_bound` in contract.rs) operate
unchanged. Add an `OpDef::contraction(...)` constructor beside `elementwise`/`reduction`/`row_reduce`
(ir.rs:326–386).

### 5.3 Schedule — the axes a contraction needs

Add `Schedule::Contraction { .. }` to plan.rs:18–51 (it is `Copy`, so the axis `Vec`s and the
epilogue ride on `KernelPlan::access` exactly as RowReduce's stages do, plan.rs:70–75). The schedule
axes (the "schedule half" of the algorithm/schedule split, kernel-specialization.md §10):

1. **K-tiling** — tile size along the contraction axis; multi-pass when K exceeds a shared-memory
   tile (the same shared-mem ceiling that sets predicate #10, kernel-specialization.md §3 / structure_key
   arch table).
2. **Register + shared blocking** — the (Bm × Bn) threadblock tile and the per-thread register
   tile; the double-buffered shared-memory staging of lhs/rhs K-slabs.
3. **Tensor-Core fragment selection** — MMA shape (`m16n8k8`/`m16n8k16` for f16/bf16;
   `m16n8k4` DGEMM; `m16n8k32` int8 — the head's bespoke int path uses exactly these, OP-MATRIX.md:40–42)
   vs SIMT/FMA fallback (for `F32Strict` bit-stability, matching CUTLASS's SIMT path, cutlass README:18).
   This axis is arch-gated and coordinates with **item 09** (half2/packed emit).
4. **Double buffering / software pipelining** — depth of the cp.async / prefetch pipeline
   (arch-gated: `cp.async` needs sm_80+).
5. **Epilogue schedule** — fuse the bias/activation in-register before the store (no extra global
   traffic), the CUTLASS `LinearCombinationBiasElementwise` model (cutlass README:30–36).

Determinism note (house discipline): Tensor-Core warp-level reductions are **not** bit-reproducible
across launches; the SIMT/`F32Strict` schedule is the deterministic fallback. The schedule must
record which it chose so the FKC `determinism`/`precision` block is honest (§5.6) — mirroring
CUTLASS's `F32Strict` "no tensor-core warp-reduction nondeterminism" guarantee (cutlass README:18).

### 5.4 StructureKey facts the node needs

The contraction gate and schedule need a *small* set of **structure classes** (not literal extents —
honor the §1 non-negotiable). Extend `StructureKey` (structure_key.rs:187–216) behind a version bump
(`STRUCTURE_KEY_VERSION`, structure_key.rs:48) with contraction-only facts, empty/default for every
non-`Gemm` op so existing tokens are unchanged:

- **M/N/K size classes** — each bucketed to `{Tiny, Skinny, Small, Large}` (e.g. `Skinny` = one of
  M/N is 1–8, the FlashDecoding++ flat-GEMM / GEMV-adjacent decode cell; `Large` = all ≥ a
  vendor-win threshold → route out). This is the axis that *drives the vendor gate*.
- **K-alignment / M-N-alignment class** — reuse the `DivBucket` ladder (structure_key.rs:99–112)
  per axis; Tensor-Core fragments need K aligned to the MMA-k (8/16/32).
- **Transpose / layout class per operand** — `{RowMajor, ColMajor}` (→ CUTLASS `Rrr`/`Rcr` analog,
  cutlass README:24–26); comes from 01's layout fact.
- **Batch class** — `{None, Uniform, Grouped}` (grouped = MoE variable-M, the `GroupedGemmPlan`
  cell — which the head owns, so this class mostly *routes out*).
- **Accumulator-precision policy** — `{TF32, F32, F32Strict, F64, S32}` — folds into `dtype`+a
  precision flag; picks the fragment and sets the determinism/ulp contract.

`reduce_axes` (structure_key.rs:213) can carry the K axis set (it is documented as reduction-class-
only and currently always empty — the contraction is its first real user). Add a `contraction:
Option<ContractionKey>` field rather than overloading operand keys, so non-GEMM keys serialize
identically (token codec, structure_key.rs:616–648, gets one optional trailing field).

### 5.5 What 01 and 02 must provide — shape these NOW

**Item 01 (layout/shape nodes) must expose:**

- **Named axis handles** with per-operand **role** (batch / free-M / free-N / contracted-K) — the
  contraction's `axes: ContractionAxes` is built from these. If 01 models axes only as anonymous
  positions, a contraction cannot name its K axis. **01 must support axis *roles*, not just extents.**
- **Per-operand layout facts**: row/col-major, leading dimension, transpose, contiguity of the
  *inner two* axes (the tile-load unit). The Tensor-Core fragment + the vendor gate read these.
- **Alignment per axis** (feeds the MMA-k alignment class, §5.4) — reuse/extend `DivBucket`.
- **A shape-fact API that both the generator (build) and dispatcher (runtime) call** — the
  single-source-of-truth invariant (kernel-specialization.md §4) must hold for contraction axes too.

**Item 02 (DAG-with-consumer-counts) must expose:**

- **A value-node identity for a contraction result** with a **consumer count** — so an epilogue
  reading it, and a downstream op re-reading it, are representable (impossible in today's tree). This
  feeds the accurate FKC `consumers:>1` (pattern.rs emits `consumers: 1` for sole-consumer interiors
  today, pattern.rs:16) and CSE in optimize.rs.
- **The ability to place a `Contraction` node as a DAG node whose result is a leaf (`Reduced`-like)
  in a consuming `ScalarExpr` epilogue** — the bridge between the contraction (structural) and the
  pointwise (tree) worlds.

Enumerate these in the spike output as explicit "01 must / 02 must" requirement lists so the two
keystone items land contraction-ready.

### 5.6 FKC / contract implications

- `contract()` (contract.rs:58–154) hardcodes `cost.class: elementwise` (contract.rs:137) and
  `count_flops` counts pointwise nodes only (contract.rs:204–214). A contraction needs a **`class:
  contraction`** cost with a `flops_per_output = 2·K` (MAC) term and a `bytes` model reflecting the
  tiled traffic, not `(n_inputs+1)·dtype_size` (contract.rs:299–303).
- `precision`/`determinism` must reflect the fragment choice (§5.3): a TF32/Tensor-Core cell is
  `approximate` + non-bitwise; the SIMT/`F32Strict` cell is `correctly_rounded` + `bitwise`. The
  `precision_of`/`ulp_bound` machinery (contract.rs:223–263) covers the *epilogue* but must be
  extended with the contraction-core precision.
- `derive_pattern` (pattern.rs:79–82) must learn a `Contraction` pattern shape (a `MatMul`/`Contract`
  graph-Op node with the epilogue subtree) OR the node advertises as a **primitive `op_kind`** (no
  `pattern:`) when it is a bare matmul — the honest-miss path (contract.rs:79 returns `None`) stays
  the default until the pattern shape is agreed with Fuel (§10).

### 5.7 The §7 vendor-exclusion gate (per-cell, measured)

The gate is **per-cell and measured, never an op-level blocklist** (kernel-specialization.md §7).
Seed it with hand-knowledge, make it durable via 07's benchmark gate:

- **Route OUT to cuBLAS/CUTLASS (do NOT generate):** `Large`-class M∧N∧K, aligned, uniform/grouped
  batch, standard `{f16,bf16,f32,f64}` — the head wins decisively (cutlass/cublas READMEs). Grouped/
  MoE → `GroupedGemmPlan`. Int8/fp8/int4/bin → the bespoke head GEMMs (OP-MATRIX.md:40–43).
- **GENERATE (the long tail):** `Skinny`/`Tiny` M or N (FlashDecoding++ flat-GEMM, decode-time
  GEMV-adjacent), irregular K-alignment, and — the biggest win — **contraction fused across a layout
  change or into a bias/activation/norm epilogue in one pass** (kernel-specialization.md §1: "op +
  transform + epilogue in one pass, skipping the contiguize round-trip"), which the vendor path
  cannot fuse without a round-trip.
- The gate lives in the per-arch dispatch table (item 07's build artifact). This node **emits the
  candidate cells and the seed verdicts**; 07 measures and records the winner per `(op, structure-key,
  dtype, arch)`.

## 6. Implementation steps (ordered; each names the file it edits)

> Steps 1–6 are the **design deliverables** (write them into this brief / a follow-up design doc).
> Step 7 is an optional gated prototype. The *merged-to-main* implementation waits for 01 + 02.

1. **IR design** — specify `Access::Contraction { axes, accum, epilogue }` + `OpDef::contraction`
   against `crates/baracuda-kernelgen/src/ir.rs` (variant on ir.rs:279–305; ctor beside
   ir.rs:326–386). Define `ContractionAxes`/`AccumSpec` in terms of 01's axis handles (do NOT invent
   a parallel axis system).
2. **StructureKey design** — specify the `ContractionKey` field + M/N/K/align/layout/batch/acc
   classes on `crates/baracuda-kernels-types/src/structure_key.rs` (struct at :187–216), the version
   bump (:48), and the token codec's optional trailing field (:616–703). Verify `OpCategory::Gemm`
   (sku.rs:31) + `"gem"` code (structure_key.rs:862) already round-trip.
3. **Schedule design** — specify `Schedule::Contraction` + the 5 schedule axes on
   `crates/baracuda-kernelgen/src/plan.rs` (enum :18–51; a `build_plan` arm :85–117 keyed off the
   contraction StructureKey classes → tile/fragment/pipeline choices).
4. **Emitter design (cuda.rs)** — specify `emit_contraction` beside cuda.rs:265/397: the tiled
   MMA/SIMT kernel skeleton (shared-mem staging, `wmma`/`mma.sync` or FMA fallback, double buffer,
   in-register epilogue). Enumerate which fragment shapes per dtype/arch. **Reuse `baracuda::coord`
   unravel helpers on non-const-folded paths** (kernel-specialization.md §10), do not reinvent.
5. **Pattern/contract design** — specify how `derive_pattern` (pattern.rs:79–82) and `contract`
   (contract.rs:58–154) grow: primitive `op_kind` path first (honest, no `pattern:`); `class:
   contraction` cost model; fragment-driven precision/determinism.
6. **Vendor-gate + FFI/build wiring design** — specify the seed verdict table (§5.7) and how it
   feeds item 07's per-arch dispatch table; specify the `optag_name`/`region_to_op` change (jit.rs:419,
   693–694) that flips `MatMul` from honest-miss to synthesis **only once the pattern shape is agreed
   with Fuel** (§10). Specify the OP-MATRIX.md row (a new "generated contraction — long tail" entry
   under `## OpCategory: Gemm`) and the kernel-specialization.md status update (unstale ~432).
7. **(Optional) gated single-cell prototype** — behind a `--features contraction-spike` flag / an
   `experiments/` dir (do NOT merge to main; mirrors `experiments/elementwise_specialization.cu`,
   kernel-specialization.md §11 Result): one skinny-GEMM f32 cell, on-device numeric-validated
   against a cuBLAS oracle, to prove the schedule axes and *measure* the long-tail win (the §11
   go/no-go gate for this node). This is what justifies the machinery before 01/02 land.

## 7. Test & on-device validation plan

House discipline: on-device validation is mandatory for any kernel that gets prototyped (§6 step 7).

- **Unit tests (Rust, no device):**
  - `Access::Contraction` round-trips through `OpDef::contraction`; `body == epilogue` (parity with
    `row_reduce`, ir.rs:378).
  - `StructureKey` with a `ContractionKey` **`to_token`/`from_token` round-trips** (extend the
    structure_key.rs:992 `token_round_trips` test) and a non-GEMM key's token is **byte-identical**
    to the pre-change token (the version bump is the only diff) — guards the "empty for non-GEMM"
    invariant.
  - `build_plan` maps each M/N/K class to the expected tile/fragment (plan.rs test).
  - `contract()` emits `class: contraction`, the `2·K` flop term, and the correct
    precision/determinism per fragment (extend contract.rs:443 tests).
  - **Honest-miss preserved:** until the node ships, `unsupported_op_is_rejected` (jit.rs:1100–1107)
    stays green (MatMul → `UnsupportedOp`); the flip to a positive test is the *last* wiring step.
- **nvrtc headerless compile cases** (feature `nvrtc`, joining the existing ignored on-device tests
  jit.rs:1246–1373): the generated contraction `.cu` compiles headerless for f32 (SIMT), f16/bf16
  (needs `cuda_fp16.h`/`cuda_bf16.h` — the `-I` path is already wired, jit.rs:235), and the
  `mma`/`wmma` intrinsics compile under the target `--gpu-architecture`. Guards the same
  headerless-portability property the `cstdint` regression taught (jit.rs:1244).
- **nvcc numeric on sm_89 (RTX 4070):** diff the prototype cell against a **cuBLAS `Sgemm` oracle**
  (`baracuda-cublas::gemm`) and the **generic strided kernel** (the standing oracle, kernel-
  specialization.md §10) — bit-for-bit for `F32Strict`/SIMT, within declared ulp for TF32. Cover:
  skinny (M=1 GEMV-adjacent), unaligned-K, and a fused bias+relu epilogue vs a two-kernel reference.
- **compute-sanitizer** (mandatory — this kernel has shared memory + cross-thread reduction):
  `synccheck` (the `__syncthreads` between K-tiles), `racecheck` (shared-mem double-buffer), and
  `initcheck` on the accumulator and the K-remainder / M-N-remainder tail.
- **Numeric oracle to diff against:** cuBLAS for the plain GEMM core; for the fused-epilogue cell,
  a reference = (cuBLAS GEMM) → (elementwise bias/activation) composed, since the fused kernel's
  whole point is to match that at lower traffic.

## 8. Adversarial-verify checklist

Run the multi-agent find → dedup → skeptic-refute pass (house discipline) probing THIS change's
failure modes:

- **K-remainder correctness:** a K not divisible by the MMA-k or tile size — does the tail loop
  drop or double-count the last partial tile? (The single most likely codegen bug.)
- **M/N-remainder + predication:** a skinny cell where Bm/Bn exceed M/N — out-of-bounds tile loads
  or stores (initcheck/racecheck target). The extent-abstraction hazard the RowReduce work already
  hit (plan.rs:151–173: key carries no numeric extents → a mis-sized operand keys identically) —
  **is the K/M/N extent a caller pre-condition here too, and is it documented?**
- **Accumulator precision misroute:** does an f16 cell silently accumulate in f16 (catastrophic
  error) instead of f32? Verify the acc-dtype policy (§5.4) is honored and the FKC `precision`
  block *matches* what the kernel actually does — an honest-contract violation is the worst outcome.
- **Determinism claim vs reality:** a Tensor-Core cell advertising `determinism: bitwise` would be a
  lie (warp-reduction nondeterminism). Probe every fragment/precision combo against its emitted
  contract (the same class of bug the NaN-misroute / UAF passes caught).
- **Vendor-gate leakage:** a `Large` aligned cell that should route OUT but the generated kernel
  gets selected anyway (a perf regression vs cuBLAS, not a correctness bug — but the whole thesis
  says we must not lose here). Verify the seed verdicts actually exclude the head-owned cells.
- **Transpose/layout confusion:** row-major vs col-major operand mixed up → transposed output that
  passes a *symmetric*-input test but fails on asymmetric M≠N. Use non-square, non-symmetric oracles.
- **NaN/Inf propagation** through the epilogue (the `fmaxf` NaN-misroute class): a bias/relu epilogue
  on a NaN accumulator.
- **Stale-doc trust:** confirm the design does not inherit a false "not-emittable" claim from
  kernel-specialization.md (~423–432) — verify against the live code, not the doc.

## 9. Definition of done

- **The design document exists** with: the `Access::Contraction` grammar + `ContractionAxes`/`AccumSpec`;
  the `Schedule::Contraction` axes (K-tile, register/shared block, fragment, double-buffer, epilogue);
  the `ContractionKey` StructureKey facts (M/N/K/align/layout/batch/acc classes) + version-bump plan;
  the FKC `class: contraction` cost/precision/determinism shape; the §7 seed vendor-gate verdicts.
- **The "01 must / 02 must" requirement lists are explicit and actionable** (§5.5) so the two
  keystone items land contraction-ready — this is the spike's primary downstream contract.
- **Vendor-exclusion boundary is stated per-cell** (generate the long tail; route large-aligned/int/
  fp8/grouped to the existing head) and traceable to the head's actual surface (cutlass/cublas READMEs,
  OP-MATRIX.md).
- **Honest-miss preserved:** `unsupported_op_is_rejected` (jit.rs:1100–1107) and `derive_pattern`'s
  `NotElementwise` (pattern.rs:79–82) stay green; no half-wired contraction path can emit an
  unbindable/dishonest FKC contract.
- **(If the optional prototype is built)** one skinny-GEMM cell is nvcc-numeric-validated on sm_89
  against a cuBLAS oracle, compute-sanitizer-clean, and the **long-tail win is measured** (the §11
  go/no-go number that justifies the node) — with the standing adversarial-verify pass run on it.
- `docs/design/kernel-specialization.md` ORDER-3/MatMul status un-staled; OP-MATRIX.md notes the
  generated-contraction long-tail cell under `## OpCategory: Gemm`.

## 10. Open questions / Fuel asks

- **Region grammar for a generated contraction (cross-repo, Fuel-owned).** Fuel's frozen
  `fuel-kernel-seam-types` `OpTag` currently has no synthesizable `MatMul`/`Contract` region shape
  (jit.rs:693–694 lists it as not-synthesized). For a generated contraction to be **adoptable via
  the §5 seam** (region in → kernel + recipe out), Fuel must define how a contraction region — and
  its fused epilogue — is spelled and matched, exactly the pattern established for fused-reduce ops
  in `docs/fuel-ask-fused-reduce-seam-2026-06-25.md`. **Ask (a): how is a fused matmul+epilogue
  region encoded in the frozen grammar?** Until answered, the node ships **AOT-only** (like the fused
  norms did before their seam ask), with the honest-miss preserved on the seam path.
- **Vendor-gate authority (relates to item 07 + 08).** Should the generate-vs-route-out threshold be
  a Baracuda build-time seed only, or driven live by Fuel's `dispatch_record`/`miss_record`
  (kernel-specialization.md §8)? The FlashDecoding++ flat-GEMM cell is the archetypal `miss_record`
  demand signal — coordinate the threshold's home with 07/08.
- **Accumulator/precision default per dtype.** Does Baracuda pick TF32 (fast, approximate) or
  `F32Strict`/SIMT (bit-stable) by default for an f32 contraction, or is that a per-request precision
  contract from Fuel (mirroring `PrecisionGuarantee`, cutlass README:239–245)? This decides the
  default determinism claim.
- **Scope of the contraction axis vocabulary (relates to 01 + 03).** v1 is single-K-group batched
  matmul. Do we commit to general multi-K einsum in the grammar now (shape 01/03's axis
  representation to allow it) or explicitly defer and keep 01's axis roles matmul-shaped? Recommend:
  design the axis vocabulary general, implement single-K first.
