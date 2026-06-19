# Kernel specialization — structure-class codegen

**Status:** design / not yet implemented. This captures the agreed shape of
an ahead-of-time (AOT) kernel-specialization system: a generator that takes
the *logical math* of an op plus a matrix of *structural predicates* over its
inputs/outputs and emits a specialized `.cu` kernel per cell, built with
`nvcc` and ready to use at runtime.

It sits on top of, and does not replace, the existing strided-kernel layer
(`TensorRef`/`TensorMut` + the `baracuda::coord::unravel_*` device helpers in
[`baracuda_coord_unravel.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_coord_unravel.cuh)).
That generic strided kernel becomes the **correctness oracle** and the
**runtime floor**; the specialized kernels are fast paths above it.

---

## 1. Thesis

Static knowledge of an input's *structure* — contiguity, broadcast pattern,
alignment, reduction-axis position — is the single biggest performance lever
after the algorithm itself. The generic strided kernel pays a `divmod`
coordinate-unravel and a dynamic stride dot-product per element and cannot
vectorize, hoist broadcasts, or drop remainder loops, because none of those
facts are known to the compiler. A kernel specialized to a known structure
folds all of it into constants.

We will **not** beat cuBLAS/cuDNN/CUTLASS at the ~20 ops they hand-tune. The
win is the **long tail** — fused elementwise + epilogue, normalization, shape
ops, attention variants, and especially *fusions across a layout change* (op +
transform + epilogue in one pass, skipping the contiguize round-trip) — which
every vendor leaves generic. Making the tail as fast as the head is the
defensible advantage.

### Non-negotiable framing

- **Specialize on structure, not on literal shapes.** A kernel hard-coded to
  `M=4096` buys almost nothing over one that knows "inner extent is a multiple
  of 8 and ≥ threshold" and costs unbounded cardinality. The matrix axes are
  *structural predicates* (closed enums), never numeric extents.
- **AOT is the primary delivery.** Kernels are built once with `nvcc` and
  shipped ready. JIT (NVRTC) is a *fallback* for cells we did not prebuild —
  and, importantly, **not a quality compromise** (see §9).
- **The generic strided kernel never dies.** It is the oracle every generated
  cell is differential-tested against (§10), and the dispatch floor when no
  specialized cell matches and JIT is unavailable.

---

## 2. Structural predicate catalog

These are the predicates that *actually change the emitted SASS*. That is the
filter: a predicate that does not alter codegen is matrix bloat and is
excluded. Each is a closed enum; the per-cell coordinate ("structure key") is
the tuple of all of them. Predicates marked **per-operand** are carried
independently by every input *and* the output.

| # | Predicate | Representation | Why it changes the SASS |
|---|---|---|---|
| 1 | Index width | `{Idx32, Idx64}` | int32 offset math is fewer registers + tighter loops; flips at 2³¹ elements. Arch-independent (PyTorch's `canUse32BitIndexMath`). |
| 2 | **Operand** contiguity | `{Contig, InnerContig, Strided, Broadcast}` | Contig → linear addressing, no unravel; InnerContig → vectorize the inner loop only; Broadcast → hoist the load out of the loop. |
| 3 | **Operand** broadcast axis set | bitmask over canonical axes (or `{None, Inner, Outer, Arbitrary}`) | A hoisted broadcast operand can become a register constant instead of a per-element load. |
| 4 | **Operand** vector width | `{Scalar, V2, V4, V8}`, *derived* from (base-ptr alignment, inner stride == 1, inner extent % w == 0, dtype size) | `ld.128`/`st.128` vs scalar is 2–4× on bandwidth-bound ops. |
| 5 | **Operand** inner-extent divisibility | `{%16, %8, %4, %2, Any}` | Eliminates the remainder/predication tail; enables full unroll of the inner loop. |
| 6 | Effective (collapsed) rank | small int, *after* merging adjacent contiguous axes (§4) | Fewer loop levels and less unravel; a rank-4 contiguous tensor collapses to rank-1 for addressing. |
| 7 | **Operand** negative stride (flip) | `bool` | Descending addresses break coalescing assumptions and gate whether vectorization is legal. |
| 8 | Work-size class | `{OneWarp, OneBlock, GridStride}` | Replaces the discarded "stepped max dims" axis. Tiny work wants a single-block / no-grid-stride kernel; everything larger is one grid-stride kernel. |
| 9 | *(reductions)* reduce-axis position | `{Inner, Outer, Middle, Multi}` | Inner → warp-shuffle reduction, coalesced; Outer → thread-per-column — a structurally different kernel. |
| 10 | *(reductions)* reduce-extent vs block | `{Warp, Block, MultiPass}` — the `Block` threshold is **derived from the target CC's shared-memory ceiling** (§3) | Shuffle-only vs shared-memory tree vs two-pass. This is the one place build-time CC knowledge feeds the matrix. |

`dtype` and `arch` (`sm_XX`) stay the axes they already are in the SKU system.

Note that #4 (vector width) is **derived**, not free: it is a function of #2,
#5, #7, and dtype. Most `(contiguity, vector-width)` combinations are
impossible (a `Strided` or `Broadcast` operand cannot be `V4` on that operand),
so the apparent cardinality of the cross-product overstates the real cell
count by a large factor (§6).

---

## 3. Hardware inputs at build time

The generator runs against a set of target compute capabilities (the
`-arch=sm_XX` flags). What that buys, and what it does not:

**CC gives you the per-block resource ceilings** — encode them once as a static
per-arch table (there is no build-time CUDA API; the values are fixed and
documented in the CUDA C Programming Guide's compute-capability table):

| CC | Max shared mem / block (opt-in) | Notes |
|---|---|---|
| 8.0 (A100) | ~163 KB | |
| 8.6 / 8.9 (consumer Ampere, Ada — e.g. RTX 4070, RTX 6000 Ada) | ~99 KB | opt-in above 48 KB via `cudaFuncAttributeMaxDynamicSharedMemorySize` |
| 9.0 (Hopper) | ~227 KB | |
| 10.x / 12.x (Blackwell) | ≥227 KB | pull exact from the programming guide |

Plus max threads/block (1024 everywhere modern), max registers/thread (255),
warp size (32). The shared-memory number is the one that matters: it sets the
`Block`-vs-`MultiPass` threshold for predicate #10.

**CC does NOT give you the memory ceiling, and you do not want it.** Total VRAM
is a *per-device* property, not per-CC: an A100 is 40 or 80 GB, an H100 is 80,
an H200 is 141 — all `sm_90`. You can recover a *current* max-VRAM-per-CC from
the (finite, known) device list (e.g. `sm_89` → 48 GB on the RTX 6000 Ada /
L40S; `sm_90` → 141 GB on the H200), but **it is codegen-irrelevant**: the only
size boundary that changes the kernel is int32 → int64 indexing, and that flips
at ~8 GB (fp32) / ~4 GB (fp16) — far *below* any modern card's VRAM. A 48 GB
ceiling and a 141 GB ceiling both sit entirely in int64 territory and change
nothing. The VRAM ceiling would only prune anything (skip int64 variants) on a
sub-4-GB device, which is rare enough to ignore.

**Per-device targeting (beyond CC) is deferred.** Knowing the exact device adds
SM count, L2 size, bandwidth — none of which change the emitted SASS (SM count
affects *launch configuration*, computed at runtime from
`cudaGetDeviceProperties`, not codegen). Low ROI; revisit only if a specific
kernel family proves L2-tile-size sensitive.

### The size axis, resolved

The discarded "stepped ladder of max dims" collapses to exactly three
size-derived facts, all already in the catalog:

1. **int32 → int64 indexing** (predicate #1) — the only large-end boundary.
2. **reduce-extent vs block** (predicate #10) — uses the CC shared-mem number.
3. **the tiny end** (predicate #8) — a short ladder of ~2–3 rungs
   (`OneWarp` → `OneBlock` → `GridStride`), specialized at the *small* end
   only. The large end is one grid-stride kernel modulo #1.

---

## 4. Canonicalization

The lever that keeps the matrix small **and** maximizes runtime hit-rate is
canonicalizing a layout *before* computing its structure key, so that many
distinct raw layouts map to the same cell. Canonicalization must be **legal for
the op class** — only transforms the op is invariant to are applied.

Algorithm (op-class-parameterized):

1. **Squeeze size-1 axes.** They contribute nothing to addressing; their stride
   is irrelevant.
2. **Merge adjacent contiguous axes.** If `stride[i] == stride[i+1] *
   shape[i+1]`, collapse the pair into one axis. This reduces effective rank
   (predicate #6); a fully contiguous rank-N tensor collapses to rank-1.
3. **Canonicalize axis order** where the op is permutation-invariant
   (elementwise is fully invariant; reductions are invariant *within* the kept
   and reduced axis groups but not across them). E.g. sort by descending
   stride.
4. **Factor broadcasts.** A `stride == 0` axis becomes a bit in the broadcast
   mask (predicate #3), not a distinct extent.
5. **Quantize** alignment and inner-extent divisibility to their buckets
   (predicates #4, #5).

The result is the canonical **structure key**. Because canonicalization is
shared code between the generator (build time) and the dispatcher (runtime),
there is exactly one definition of the key — the build matrix and the runtime
lookup speak the same language by construction.

---

## 5. Dispatch-key computation (runtime)

At dispatch, the plan layer computes the structure key from the live
`TensorRef`s and looks up the prebuilt kernel. Sketch:

```rust
// Single source of truth, shared with the generator.
fn structure_key<T, const N: usize>(
    op_class: OpClass,
    operands: &[OperandView<'_, T, N>],   // inputs + output
    arch: Sm,
) -> StructureKey {
    let canon: Vec<CanonOperand> = operands.iter()
        .map(|o| canonicalize(op_class, o))     // §4
        .collect();

    StructureKey {
        idx:   if max_offset(&canon) < (1 << 31) { Idx32 } else { Idx64 },
        rank:  effective_rank(&canon),          // post-collapse
        work:  work_size_class(&canon, arch),   // OneWarp | OneBlock | GridStride
        ops:   canon.iter().map(|o| OperandKey {
                   contig:    o.contiguity,      // Contig | InnerContig | Strided | Broadcast
                   bcast:     o.broadcast_mask,
                   vec_width: vector_width(o.align, o.inner_stride, o.inner_extent, size_of::<T>()),
                   inner_div: divisibility_bucket(o.inner_extent),
                   flipped:   o.has_negative_stride,
               }).collect(),
        reduce: op_class.reduction_axes(&canon), // None for non-reductions
        dtype:  T::DTYPE,
        arch,
    }
}
```

The key must be a compact, hashable, **versioned** value (e.g. a packed struct
or `u64` bag) so lookup is O(1) and so it doubles as a stable telemetry token
(§8). Version it — the predicate set will grow.

Lookup → hit: launch the specialized kernel. Miss: §9.

---

## 6. The build matrix — sizing and pruning

The full cross-product is large. Worked example, elementwise binary
`z = a ⊙ b`, before pruning:

```
idx(2) × a-contig(4) × b-contig(4) × z-contig(2) × vec(4) × dtype(8) × arch(4)
  = 16,384 cells
```

That is the number that makes pruning non-optional. It collapses hard:

- **Derived axes are not independent.** Vector width (#4) is a function of
  contiguity/alignment/dtype; most `(contiguity, vec)` pairs are impossible.
  This alone removes the majority of the cross-product.
- **Canonicalization** (§4) folds whole families of raw layouts onto one cell,
  so the *reachable* key set is far smaller than the nominal product.
- **Vendor-owned cells are excluded** (§7).
- **Demand-driven generation** (§8): only cells that occur in — or would unlock
  — real Fuel workloads are built. Realistically a few hundred cells per op
  family, not tens of thousands.

**Any cap on the matrix must be logged**, not silent. If a build skips a cell
class (e.g. "no V8 variants for f64"), emit it to the build report so a missing
fast path reads as a deliberate decision, not an oversight.

---

## 7. Excluding vendor-owned cells

Where a masterfully tuned vendor kernel already wins, route to it instead of
generating our own. But the exclusion is **per-cell and measured, not an
op-level blocklist**: "cuBLAS wins" is true for large aligned GEMM and *false*
for small/skinny/irregular GEMM. So the winner for each `(op, structure-key,
dtype, arch)` is decided by a benchmark gate and recorded in a per-arch
dispatch table (a build artifact). Hand-knowledge *seeds* the gate (e.g. "don't
even generate large-aligned GEMM — route to cuBLAS") to save build time, but
the durable mechanism is measurement, because the winner moves with arch and
dtype. See §8 — Fuel can supply this measurement from real workloads instead of
us synthesizing it.

---

## 8. The Fuel feedback loop

Fuel already times and compares every kernel implementation available to it and
picks the best. That makes Fuel the ideal source of two datasets Baracuda
otherwise has to manufacture:

- **(a) Comparative performance** per `(structure-key, dtype, arch)`: which
  implementation won and by how much. This *is* the §7 benchmark gate, measured
  on real shapes in the real pipeline instead of synthetically.
- **(b) Missing-kernel demand**: "a specialized kernel for structure-key *K*
  would have been an exact fit here, but it did not exist, so we fell back to
  *F* at cost Δ." This is the demand signal for what to generate next — and it
  is what defeats the chicken-and-egg trap (Fuel reports the cell it *wanted*,
  even though it had to route around it today).

**The structure key (§5) is the telemetry schema.** Because Baracuda defines
the key and Fuel reports against it, the two sides join on the same token by
construction. This is one artifact serving codegen, dispatch, *and* the Fuel
channel.

Proposed contract (slots into the existing `fuel/docs/baracuda-ask-*.md` ↔
`docs/fuel-reply-*.md` channel; since we own both sides this is a contract to
*design*, not a favor to request):

```
// opt-in, aggregated, no tensor data — the key is already an abstraction
// over raw shapes (divisibility buckets, not literal dims), so it is
// privacy-friendlier by construction.

dispatch_record {                 // dataset (a)
    structure_key, dtype, arch,
    chosen_impl, time_ns, candidates_considered[]
}

miss_record {                     // dataset (b)
    wanted_structure_key, dtype, arch,
    fallback_impl, est_speedup_if_available
}
```

### Rollout: v1 batch → v2 live

- **v1 — batch (build/release time).** Fuel publishes an aggregated, opt-in
  report; Baracuda's maintainers consume it to choose the next release's AOT
  matrix. No runtime coupling, privacy-friendly, low risk. Do this first.
- **v2 — live (in-process).** Fuel passes a `miss_record` to Baracuda during a
  run; Baracuda generates the ideal kernel in the background via NVRTC, caches
  it, and notifies Fuel that the available-kernel set changed so Fuel
  re-benchmarks. This is the self-optimizing dream — but it has real hazards to
  design against: background compilation contending for SMs/PCIe during a
  training run; cache coherence; and **kernels appearing mid-run shifting the
  ground under Fuel's own timing comparisons** (Fuel must treat the kernel set
  as versioned within a run). Desirable, but a careful v2.

---

## 9. AOT → JIT → AOT: the kernel lifecycle

These are not two systems. A cell flows through one continuum keyed by the same
structure key:

```
miss (no prebuilt cell)
  → JIT-generate via NVRTC + nvJitLink (this run, cached on disk by key+arch)
  → telemetry (aggregate the miss as demand, §8)
  → AOT-build into the next release (the miss becomes a permanent prebuilt cell)
```

The JIT step is **quality-equivalent**, not a compromise: nvcc and nvrtc share
the same `NVVM`/`ptxas` backend, so the same generated source + offline-grade
`ptxas -O3` (via `baracuda-nvjitlink` LTO) yields essentially the same SASS. So
a background-generated kernel is a legitimate citizen to promote into Fuel's
rotation, and the *same source* later rolls into the AOT build. AOT is
preferred only for zero warmup, determinism, and ship-once artifacts — not for
codegen quality. (See [`docs/guides/nvrtc.md`](../guides/nvrtc.md).)

---

## 10. Generator design and correctness

- **Separate algorithm from schedule (the Halide insight).** The "logical math"
  of the op is the algorithm; the `(structure-key → tile/vectorize/unroll)`
  choice is the schedule. Keep them separate — the same algorithm specializes
  across many schedules.
- **Emit readable `.cu`, not proc-macro spaghetti.** The generator is a
  build-time tool that writes inspectable `.cu` files (which `baracuda-forge`
  then compiles), *not* a proc-macro that emits opaque device code. Generated
  source you can open, read, and diff is a decisive debugging and trust
  advantage. A macro wrapper can come later.
- **Reuse the existing device helpers.** Specialized kernels still call
  `baracuda::coord::unravel_*` and friends on their non-constant-folded paths;
  the generator's job is to *eliminate* the unravel where structure makes it
  constant, not to reinvent it.
- **Differential-test every generated cell against the generic strided oracle.**
  One harness covers the whole matrix: for random inputs in each structure
  class, the specialized cell must match the generic kernel bit-for-bit (or
  within the op's declared tolerance). This is the non-negotiable safety net for
  codegen — a generator bug's blast radius is every cell, so the test must be
  automatic and exhaustive over the built matrix.

---

## 11. Go / no-go pilot

Do **not** build the general compiler first. Validate the thesis on one op
family that already has a contiguous + strided sibling (the oracle):
**elementwise + fused epilogue**, or **rms/layernorm**.

1. Define the minimal algorithm IR for just that family.
2. Generate a *small* matrix — ~6–12 cells (contiguous / last-axis-broadcast /
   transposed-2D × aligned/unaligned × scalar/V4).
3. Differential-test every cell against the generic kernel (§10).
4. **Benchmark whether the specialized cells beat the generic sibling by enough
   to justify the machinery.** This is the whole decision: if broadcast/
   transpose specialization is 2–3×, the thesis holds and we generalize with
   confidence; if it is 5%, we just avoided building an exponential edifice for
   nothing.

---

## Open questions

- Algorithm IR: adopt/adapt an existing tensor-expression IR (Triton-like,
  TVM TE, MLIR Linalg, tinygrad UOps) or roll a minimal bespoke one for the
  pilot family?
- Where does the dispatch table live — generated Rust `match`, a loaded data
  table, or both? (The SKU system today favors exhaustive `match`.)
- Autotuning: does the generator emit one heuristic schedule per cell (fast
  build, leaves perf on the table) or N candidates gated by §8 measurement
  (best-in-class, multiplies build/bench cost)?
- Structure-key versioning + migration when predicates are added.
```