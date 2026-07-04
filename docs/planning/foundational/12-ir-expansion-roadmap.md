# 12 — IR expansion roadmap: covering the full bespoke surface

> Charter (user directive, 2026-07-03): "expand the IR so it can express the
> functionality in all of the bespoke kernels we already have" — the generator
> must eventually cover the whole bespoke surface (not call it). Produced by a
> 13-agent inventory workflow: 11 family readers over all 23 kernel dirs
> (~415 .cu files + 77 headers), one IR ground-truth reader, one synthesis.
> The synthesis below is the committed plan of record; increments land under
> the house discipline (validate-reject honest misses, adversarial verify,
> on-device oracle, audit vs the bespoke sibling, extract-the-delta on losses).

# IR expansion synthesis

## 1. Coverage snapshot

**395 distinct ops inventoried across 11 family reports; 225 (57%) expressible today** (including chain decompositions the inventory marks expressible); 170 gapped. Three additional gemm+linalg ops (extract_v, materialize_r, materialize_identity) are ELEMENTWISE-shaped but blocked solely on a missing iota/coord node — counted in the gap table as their own row.

| Gap | Count | Dominant families |
|---|---|---|
| GATHER_SCATTER | 34 | indexing/embedding/segment (12), loss (8), spatial (5), gemm+moe (6), attention (2), quantize (1) |
| MULTI_OUTPUT | 27 | elementwise-bw (21), attention (2), norm/loss/quantize/misc (4) |
| ATOMIC_HISTOGRAM | 16 | indexing/embedding/segment (11), spatial (4), moe (1) |
| TILED_TC | 16 | attention (11), gemm (2), quantize (3) |
| SUBBYTE_DTYPE | 15 | quantize+gguf (8), gemm (4), misc (2), moe (1) |
| WINDOW_STENCIL | 14 | spatial (12), attention/rope (2) |
| BACKWARD_COMPOUND | 13 | norm (6), softmax (3), elementwise-bw (2), attention (1), segment (1) |
| SORT_PERM | 9 | sort (5), elementwise-bw (2), sparsemax (1), fftshift (1) |
| SCAN | 7 | scan-family bw (4), ctc (2), moe prefix (1) |
| CROSS_ROW_STATE | 7 | ssd/mamba (4), ormqr (2), ring attention (1) |
| RNG | 4 | sampling (2), gumbel (1), uniform/normal (1) |
| DATA_DEP_CONTROL | 3 | nonzero, unique_consecutive, nms |
| HOST_LOGIC | 2 | fft (cuFFT), mhc (vendored) |
| iota/coord only | 3 | gemm+linalg near-misses |

## 2. Cheap wins (no new Access pattern)

All are vocabulary, dtype wiring, or constructor work on the existing four Access arms:

- **UnaryOp additions** (~20): `Erf, Erfc, Trunc, Exp2, Expm1, Log2, Log10, Log1p, Sinh, Cosh, Tan, Asin, Acos, Atan, Asinh, Acosh, Atanh, Cbrt, Lgamma`. Everything else in elementwise-core rows 2/5 composes (`tanhshrink = x−Tanh(x)`, `relu6 = Min(Relu,6)`, `frac = x−Trunc(x)`, `reciprocal = Div(1,x)`, `softplus/mish/hardsigmoid/hardswish/selu/leaky_relu` with baked Consts, `gelu_tanh`, `hypot = Sqrt(a²+b²)`, `floor_divide = Floor(Div)`). Unlocks ~44 fwd kinds **plus** the 46 unary backwards (all derivative expressions compose from this set).
- **BinaryOp additions** (4 irreducible): `Atan2`, `Copysign`, `Nextafter` (bit-level), `FmaxIeee/FminIeee` (NaN-suppressing — must stay distinct from NaN-propagating Max/Min), `RemTrunc` (C fmod — distinct from floored Rem).
- **Cmp + bool output**: `Eq/Ne/Lt/Le/Gt/Ge` scalar fns with u8(0/1) output dtype differing from input dtype. Unlocks cmp x6, logical x3, any/all predicates, reduce_max/min_backward eq-mask, hinge_embedding (i64 eq-const), flce count_non_ignore.
- **Int/bool dtype completion** in `scalar_ctype`: i8/u8/bool lowering + bitwise `And/Or/Xor/Shl/Shr` fns. Unlocks bitwise x5, logical x3, casts, quantize/dequantize i8 stores (with saturating narrow), masked_fill (u8 mask cast), dropout bw, token_penalty.
- **Iota/coord leaf node** (`Coord(axis)` — backend-neutral, evaluates to the loop coordinate): alibi fw, one_hot, triu/tril direct form, ormqr extract_v, qr materialize_r/identity, affine_grid_2d. ~7 ops, trivial emitter cost.
- **`Param` beyond f32 + Param-in-RowReduce**: lifts cuda.rs:47 restriction; needed by eps/margin/correction params everywhere (currently baked as Const per-JIT — acceptable but recompiles).
- **Reduction upgrades, no new Access**: `Prod` combiner (trivial monoid); fused pre-expr/post-expr on Reduction (norm2 = Sqr→Sum→Sqrt in one kernel, dot, dγ chains); integer accumulation (already item 04); hetero output dtype (any/all→u8, count→i64).
- **RowReduce per-row-scalar epilogue output + arbitrary axis**: unlocks logsumexp fwd, var/std fwd, cross_entropy LSE half.
- **In-place aliasing + base-offset/strided output views**: affine_inplace, scale_inplace, write_slice, pad_constant_backward, reduce_sum/mean_backward (0-stride bcast operands already work).

**Estimated cheap-win coverage: ~110 of the 225 "expressible today" become *single-kernel* rather than chains, plus the 3 iota near-misses and several per-row caveat ops become fully covered.**

## 3. Capability increments, dependency-ordered

### Ramp

| # | Increment | Unlocks | Effort | Depends on |
|---|---|---|---|---|
| 0 | Vocabulary batch (§2) | ~50 ops single-kernel; foundations for everything | S–M | — |
| 1 | **MULTI_OUTPUT** | 27 direct; **unblocks BACKWARD coverage broadly** | M | §2 partially |
| 2 | RowReduce generalization (2nd row-streamed input, per-row-scalar operands/outputs) | ~13 BACKWARD_COMPOUND: softmax bw, log_softmax bw, layer/rms/group-norm bw dx, sparsemax bw, cross_entropy_soft fused | M | #1 (norm bw emits dx+stats consumers) |
| 3 | **Layout/shape nodes** (wire `OpDef::views`) | rope fw/bw, pixel_shuffle, concat2_bw, fftshift, repeat_bw, sparse24 gemm B-transpose; **gates Contraction growth: batching, transposes, mixed layouts (SDPA decomposition path)** | M–L | — (parallel to #1/#2) |
| 4 | GATHER (read-side index-tensor operand) | ~16: embedding, index_select, gather, nll fw/bw-gather-half, cross_entropy fw/bw, multi_margin fw, segment bw gathers, sort/topk bw, kv_cache-style reads | M | #3 (operand descriptor extension) |
| 5 | SCATTER + ATOMIC_HISTOGRAM | ~18: scatter/scatter_add, embedding/gather/index bw, histogram, bincount, col2im, unsorted segment reduce | M–L | #4; FKC nondet flip |
| 6 | SCAN | ~7: cumsum/cumprod/cummax bw, logcumsumexp, moe prefix; feeds future softmax-state work | M | smem_scan primitive exists |
| 7 | WINDOW_STENCIL | ~14: im2col, causal_conv1d, pooling family, interpolate | L | #3 (offset views), Coord |
| 8 | SORT_PERM | ~9: sort/argsort/topk/msort, sparsemax | L | #1 (val+idx outputs) |

**Increment 1 (MULTI_OUTPUT) is the broad-backward unblock**: the inventory states explicitly that all 21 elementwise-bw MULTI_OUTPUT ops decompose into already-expressible maps and "a multi-output store in the IR clears all 21 at once"; #2 then clears the compound norm/softmax backwards. Together they cover every backward outside indexing/spatial/scan territory.

**Layout/shape nodes (#3)** keep their previously-flagged most-foundational status for the *Contraction* growth axis (batching/transposes → the SDPA-decomposition and sparse24 story) but are not on the critical path for backward coverage — hence they run parallel at slot 3, not first.

### Per-increment detail

- **#1 MULTI_OUTPUT** — see §5.
- **#2 RowReduce roles**: allow `RrRole::RowStreamed` for inputs beyond input 0 (stage `pre` may read them, same contiguity rules); add `RowScalar` role (rank-aligned, last axis broadcast — inverse of ColBroadcast) for saved stats (μ, rstd, lse); allow per-row-scalar *output* shape. Validate: extend `validate_row_reduce` role classification (bcast pattern → RowStreamed | ColBroadcast | RowScalar, ambiguous rank-1 still rejected). Risk M — emitter change is another streamed load in the stage loop. Oracle: CPU fold; audit sibling: bespoke `softmax bw`, `layer_norm bw dx` (both deterministic single-kernel — a genuine fusion-vs-chain rematch, and the smemrow variant applies).
- **#3 Layout/shape**: consume `OpDef::views` in `build_plan` (fold Permute/Broadcast into operand strides; Reshape re-keys), add element `base_offset` and stride-2/negative strides to the operand descriptor, add output views (write_slice). Backend-neutral: pure index arithmetic, matches `coord_unravel` primitive. Risk M; audit sibling: rope_apply (pure map given even/odd views), pixel_shuffle vs bespoke.
- **#4 GATHER**: new operand role `Indexed{index_operand: u8, axis: u8}` — read address for one axis comes from an i32/i64 index tensor operand; OOB policy field (`Skip | Clamp | ZeroFill`). Deterministic, value-preserving → unconditional cells. Risk M; oracle CPU gather; compute-sanitizer memcheck for OOB policy; audit siblings: `embedding`, `index_select` (bespoke are plain gathers — tie expected at memory wall, coverage is the point).
- **#5 SCATTER/ATOMIC**: output role `ScatterIndexed{...}` + `WriteCombine::{Assign, AtomicAdd, AtomicMaxMin, CasMul}`. FP atomic accumulation is order-nondeterministic ⇒ **must ship as variant with honest FKC determinism flip, never silent** (house rule); integer atomics (histogram/bincount) are order-independent ⇒ unconditional. Deterministic alternatives (segment-sorted binary-search sweep, one-thread-per-output gather-sum) are the *default*, atomics the gated fast variant — this mirrors the audit lesson: no structural cliffs. Risk L; sanitizer racecheck mandatory; audit siblings: `scatter_add`, `embedding_backward`, sorted `segment_reduce`.
- **#6 SCAN**: `Access::Scan{op, axis, reverse, exclusive}` with monoid combine (Sum/Prod/Max/Min + index-carrying pair for cummax bw later); lowers onto the already-written `smem_scan` primitive. Reassociation-sensitive for FP ⇒ block-scan variant is ReassociatedDeterministic vs serial fold default. Risk M; audit: bespoke scan_* kernels.
- **#7 WINDOW_STENCIL**: operand view `Window{axis, size, stride, dilation, pad_lo, zero_fill}` + Reduction-over-window. Covers im2col, pools, causal_conv1d, interpolate (bilinear = 2×2 window with computed weights — needs Coord). Overlap-backward stays on #5 atomics or gather-sum reformulation. Risk L.
- **#8 SORT_PERM**: `Access::RowSort{order, stable}` emitting bitonic ≤1024 + a CUB-style fallback; needs #1 for (values, indices). Risk L; deterministic; audit: bespoke sort/topk.

## 4. Honest out-of-scope

- **TILED_TC (16)**: flash attention, FA2, flashinfer, int8/fp8 tensor-core GEMM. These are vendored/bespoke competitive kernels where the memory-wall lesson cuts the other way — the win is arithmetic-intensity scheduling, not coverage. The generator's role is the *decomposed* path (batched Contraction + softmax RowReduce) as the no-cliff fallback once #3 lands; absorbing tile/MMA scheduling is a separate bench-gated program (`AccumSpec` reservation already exists).
- **SUBBYTE_DTYPE (15)**: GGUF block-quant AoS structs, nibble packing, fp8 storage. These are storage-format codecs, not op semantics; IR-neutral modeling would leak layout minutiae. Keep bespoke; revisit only if a packed-dtype operand descriptor earns its keep via mmvq.
- **CROSS_ROW_STATE (7)**: mamba scans, ring attention, ormqr recurrences — kernel-launch-spanning or matrix-state recurrences; wrong altitude for this IR.
- **DATA_DEP_CONTROL (3)** (nonzero, unique, nms), **HOST_LOGIC (2)** (cuFFT, mhc), **RNG kernels**: RNG stays externalized (host cuRAND fills a noise operand — already the bespoke convention), making bernoulli/dropout/gumbel ordinary maps; philox-in-kernel sampling stays vendored.

## 5. Increment 1 spec: MULTI_OUTPUT elementwise

**IR changes** (`ir.rs`):
- `OpDef.body: ScalarExpr` → `bodies: Vec<ScalarExpr>` (len = `n_outputs`, new field `n_outputs: u8`, default 1; existing constructors unchanged, delegate).
- New constructor `OpDef::elementwise_multi(name, n_inputs, n_outputs, dtypes, bodies)`.
- `ExprDag::from_exprs(&[ScalarExpr])`: hash-cons **across** bodies so shared subexpressions (`dy*b` / `dy*a` both loading `dy`) CSE into one `tmp` — this is the fusion value.
- `StructureKey.n_operands = n_inputs + n_outputs`; token derivation includes `n_outputs` (variant identity stays `(token, entry_point)`).
- v1 scope: `Access::Elementwise` only; uniform dtype across all outputs (hetero u8-bool output is the follow-up that unlocks dropout fw mask); no aliasing between outputs.

**Validate rules** (panic, AOT):
- `1 ≤ n_outputs < MAX_OPERANDS − n_inputs`; `bodies.len() == n_outputs`.
- Each body: `Input(i) < n_inputs`; no `Reduced`; `Const` finite; `Param` under existing f32 rule.
- All outputs same shape; each output operand independently Contig-or-strided (same rules as today's single output); outputs must not alias inputs (in-place variant deferred).

**Emitter** (`cuda.rs`): one kernel; existing schedule selection with vec-width = min over *all* operands including every output. Body: evaluate the shared DAG once (hoisted tmps), then N stores. Packed f16/bf16 pairs: `body_packs` must hold for *every* body (all leaves `Input`), else scalar fallback — same rule as today, applied per-DAG. Scalar/strided paths reuse the operand-offset machinery verbatim; only the store loop grows.

**FKC contract**: value-preserving vs the 2–3-kernel decomposition and strictly fewer global loads + one launch ⇒ **unconditional** per contract discipline (never-worse; register pressure with ≤3 small bodies is not credible as a cliff — confirm in audit before declaring). `count_unit` mirrors `effective_count_width` as today; one contract covering N output buffers (contract schema gains an outputs list; determinism unchanged: BitIdentical per output vs the split form since each store computes the identical scalar DAG).

**Golden tests**:
1. `mul_backward` (da=dy·b, db=dy·a) — shared-load CSE visible in PTX/SASS-level load count.
2. `div_backward` (db = −dy·a/b² — nontrivial DAG sharing).
3. `where_backward` (cond u8 mixed-dtype input; bit-exact mask routing).
4. `clamp_backward` (3 outputs, NaN → all-zero convention).
5. `fma_backward` (3 outputs, one is a plain copy of dy).
6. `glu/swiglu backward` (sigmoid/exp stable-branch composition).
7. `margin_ranking bw` (dx2 = −dx1 — CSE across negation).
8. Strided + broadcast operand mix; packed-f16 pass and forced scalar-fallback (Const in one body).
9. Validate-rejection tests: output aliasing, mismatched shapes, n_outputs overflow.

**On-device validation**: CPU oracle per output (f64 reference, per-dtype ULP bounds as existing harness); compute-sanitizer memcheck + racecheck on strided multi-store; determinism check (two runs bit-compare, must be BitIdentical). **Audit siblings**: bespoke `mul/div/pow_backward` and `glu family backward` kernels (single-kernel dual-store, contig) — extract-the-delta rule applies; expected outcome per the measured lesson: tie at the memory wall on the fast path, generator wins on strided/broadcast shapes where bespoke is contig-only (its fallback is a materialization pass — the catastrophe case).

**Estimated new coverage**: 21 elementwise-bw ops single-kernel; +dropout fw, alibi bw, dynamic-range quantize, cascade pairwise merge, sort val+idx once their other prerequisites land (~27 total attributable); removes the 2-kernel fusion-loss caveat from another ~8 chain-decomposed loss backwards.
