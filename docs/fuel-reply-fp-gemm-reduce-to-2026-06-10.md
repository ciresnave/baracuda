# Baracuda reply — dense FP GEMM + broadcast-reverse reductions (2026-06-10)

Reply to Fuel's coordinated ask of 2026-06-10
(`fuel/docs/baracuda-ask-fp-gemm-reduce-to-2026-06-10.md`). Work
landed as **Phase 74**; ships in the next alpha after 0.0.1-alpha.66.

TL;DR — both structural gaps are closed, and one of them was closed
before you asked:

- **Ask 1 (dense FP GEMM): shipped in Phase 74.** 12 new FFI symbols
  `baracuda_kernels_gemm_dense_{f32, f64, f16, bf16}_{run,
  can_implement, workspace_size}`, cuBLAS-backed, with RRR/RCR/**CRR**
  layouts, flexible `lda/ldb/ldd`, and strided-batch folded into the
  base symbol. `matmul_via_cublas` can retire against these.
- **Ask 2 (reduce-to-shape): already shipped — bind the symbols.**
  Your preferred option 1 contract has existed since **alpha.46**
  (Phase 31, sum/max) and **alpha.52** (Phase 37, min/prod), at full
  `{f32, f64, f16, bf16}` coverage. Your own
  `byte_kernels::reduce_to_f32` wrapper already calls the f32 pair —
  the f64/f16/bf16 siblings have identical signatures. Zero baracuda
  kernel work was needed; details below on why the audit missed them.

---

## Ask 1 — dense FP GEMM family

### What shipped

`baracuda-kernels-sys` (module `gemm_dense_cublas_facade`) — one
symbol family per dtype, batch folded in per your
"(m, n, k, batch) dims" binding-table shape:

```c
int32_t baracuda_kernels_gemm_dense_f32_run(
    int32_t m, int32_t n, int32_t k, int32_t batch,
    int32_t layout,            // 0 = RRR, 1 = RCR, 2 = CRR
    float   alpha, float beta, // double for the _f64 symbol
    const void* a, int64_t lda, int64_t stride_a,
    const void* b, int64_t ldb, int64_t stride_b,
    void*       d, int64_t ldd, int64_t stride_d,
    void* workspace, size_t workspace_bytes,  // reserved, ignored
    void* stream);
```

plus `_can_implement(m, n, k, batch, layout, lda, ldb, ldd, stride_a,
stride_b, stride_d)` (pure host validation) and `_workspace_size(m,
n, k, batch, layout)` (always 0 — cuBLAS manages its workspace
internally per handle). A typed Rust plan (`DenseGemmPlan<T>` in
`baracuda-kernels`) wraps the same symbols.

### Semantics

Row-major problem, `D[g] = α·A[g]·B[g] + β·D[g]` per batch slot.
No separate C operand — `β ≠ 0` accumulates into `D` in place
(your call sites use `β = 0`).

| `layout` | A storage | B storage | ld minimums |
|---|---|---|---|
| 0 RRR | row-major `[M,K]` | row-major `[K,N]` | `lda≥K, ldb≥N, ldd≥N` |
| 1 RCR | row-major `[M,K]` | col-major `[K,N]` | `lda≥K, ldb≥K, ldd≥N` |
| 2 CRR | col-major `[M,K]` | row-major `[K,N]` | `lda≥M, ldb≥N, ldd≥N` |

- **Leading dims are free beyond the minimum** — your
  `project_cuda_matmul_noncontig_gap` cases (BERT / SD CLIP /
  Qwen2-MoE row-slice views) pass straight through, no contiguize.
- **CRR shipped in v1** (you'd marked transpose flags acceptable as a
  narrowing; cuBLAS makes CRR a transa/transb mapping, so you also
  get the `xᵀ·dy` grad-weight shape without materializing a
  transpose).
- **Batch**: element strides, `ptr + g·stride`. `stride_a`/`stride_b`
  may be 0 (broadcast — covers the uniform-stride part of your GQA
  shapes; the non-uniform repeat-interleave part stays a loop of
  `batch = 1` calls on your side, same as today). `stride_d` must be
  non-zero at `batch > 1`. `batch == 1` routes to `cublasGemmEx`,
  `batch > 1` to `cublasGemmStridedBatchedEx`.
- **Precision**: f16/bf16 store half, accumulate f32
  (`CUBLAS_COMPUTE_32F`) — matches the reduce family's convention.
  f32 is **true IEEE binary32** (cuBLAS default math mode; we do NOT
  enable TF32 — note this differs from `GemmPlan<f32>`'s CUTLASS
  TF32 SKU; caveat: the process-wide `NVIDIA_TF32_OVERRIDE=1` env
  var would force TF32 inside cuBLAS — don't set it). f64 is
  `CUBLAS_COMPUTE_64F`. Run-to-run bitwise reproducibility is
  cuBLAS's guarantee *with its condition*: same toolkit / arch /
  SM count / shape AND a **single active CUDA stream** — concurrent
  multi-stream GEMMs may pick different internal implementations.
- **Status codes**: 0 / 2 (invalid) / 5 (internal), house convention.
  Empty problems (`m`, `n`, or `batch` = 0) return 0 without
  launching; `k = 0` launches (BLAS: `D = β·D`).

### Implementation notes you may care about

- cuBLAS-backed per your implementation-latitude note; if a CUTLASS
  path later wins some SKU it slots in behind the same symbols.
- The facade keeps a small lock-free pool of cuBLAS handles keyed by
  the calling thread's current CUDA context (NOT the
  transient-handle-per-call pattern of the cuSOLVER/cuFFT facades) —
  per-call `cublasCreate/Destroy` costs hundreds of µs and hides a
  device-syncing `cudaFree`, which would have made every Fuel matmul
  slower than your current cached-handle path. Steady state is one
  pooled handle, stream re-bound per call. Two documented hazards to
  be aware of (both irrelevant to Fuel's
  one-context-for-process-lifetime usage): (a) pool slots are keyed
  by context address and never re-keyed, so destroying a context and
  later allocating a NEW one at the same address could revive a
  stale handle — don't churn contexts mid-process; (b) the pool has
  8 slots — workloads with more than 8 distinct (or concurrently
  GEMM-ing) contexts degrade gracefully to transient create/destroy
  for the overflow, never to an error.
- Mapping to your current `matmul_f32`: your
  `gemm_strided_batched_ex(Op::N, Op::N, lda=n, ldb=k, ldc=n, …)`
  call is exactly `layout = 0` (RRR) here with
  `(lda, ldb, ldd) = (k, n, n)` in OUR naming — note our `lda` names
  A's leading dim (the facade does the column-major swap internally,
  same trick your code and our Phase 30 backend already use).
- Stream-capture behavior follows cuBLAS's own rules (no special
  casing at this layer).

---

## Ask 2 — reduce-to-shape: already shipped, bind and delete

Your option 1 (`reduce_to_shape` — exact consumer contract) has been
the shipped contract since alpha.46. The full symbol set:

```
baracuda_kernels_reduce_{sum,max}_to_{f32,f64,f16,bf16}_run   (alpha.46, Phase 31)
baracuda_kernels_reduce_{min,prod}_to_{f32,f64,f16,bf16}_run  (alpha.52, Phase 37)
```

(+ `_can_implement` companions.) All 16 share the signature your
`byte_kernels.rs` `ReduceToF32Run` typedef already declares for the
f32 pair:

```c
int32_t baracuda_kernels_reduce_sum_to_f64_run(
    const void* src, void* dst,
    const int32_t* input_shape, const int64_t* input_stride,
    int32_t rank, const int32_t* output_shape,
    void* workspace, size_t workspace_bytes,   // unused, pass null/0
    void* stream);
```

Semantics are exactly your ask: per-dim `out[d] ∈ {1, in[d]}`
(left-pad output with 1s to input rank, which your wrapper already
does), arbitrary input strides, contiguous output, deterministic
sequential per-cell accumulation, f16/bf16 accumulating in f32,
rank ≤ 8. Empty reduce sets write the identity: `0` for sum, `1`
for prod, `∓FLT_MAX`/`∓DBL_MAX` for max/min — most-extreme *finite*
value on the f32/f64 symbols; on the f16/bf16 symbols the f32
identity overflows the storage dtype on the final narrowing store
and lands as `∓inf`. So: bind the 6 missing FP symbols and the
f16/bf16/f64 CPU fallback at every broadcast gradient edge goes
away; your two byte_kernels wrappers generalize to one generic one.

### Why the audit missed it (and what we changed)

The symbols were sys-only — no `baracuda-kernels` plan facade
existed, so an audit of the plan surface (`reduce/axis.rs`) couldn't
see them. That's the same gap class as your `unary_step` facade
note. Phase 74 closes the class for both:

- New `ReduceToPlan<T, N>` plan facade
  (`{Sum, Max, Min, Prod} × {f32, f64, f16, bf16}`) + OP-MATRIX row,
  so the capability is visible at every layer.
- `UnaryKind::Step` now dispatches through `UnaryPlan` (contig +
  strided, 4 dtypes). No FFI change — your direct sys binding is
  unaffected.

---

## Info items

1. **Gelu flavor naming** — doc comments added in Phase 74 on all
   three sys families and on `UnaryKind::Gelu`/`GeluTanh`:
   `unary_gelu_*` = ERF-EXACT (`0.5·x·(1+erf(x/√2))`);
   `unary_gelu_erf_*` = bit-identical alias (kept so consumers can
   bind the flavor unambiguously by name); `unary_gelu_tanh_*` = tanh
   approximation (~1e-4 divergence). Neither erf twin is deprecated
   in the alphas — `unary_gelu_erf_*` is the recommended binding for
   new consumers precisely because it's self-describing; a 1.0-freeze
   pass will revisit whether plain `unary_gelu_*` survives.
2. **`unary_step` facade gap** — closed (see above). Cosmetic for
   you, as you noted.
3. **Memory-pressure notifications** — agreed: no native CUDA
   pressure-event API exists; polling `mem_get_info` via `would_fit`
   is the right call. Dropped on our side too.

## Test coverage (RTX 4070, sm_89)

- `dense_gemm_smoke` — 15 tests: all 3 layouts vs f64 CPU reference,
  padded leading dims, `β ≠ 0` accumulate, strided batch +
  `stride_a = 0` broadcast, f64/f16/bf16, one direct-FFI launch
  shaped exactly like your binding-table call, `_can_implement` +
  plan-level `BufferTooSmall` rejection matrices, and a 12-thread
  handle-pool concurrency test.
- `reduce_to_plan_smoke` — 8 plan-level tests: 4 ops on f32 (two
  reduced dims), f16 sum (accumulate-in-f32 tolerance), strided
  (transposed-view) input, stride-0 broadcast input,
  empty-reduce-set identity.
- `unary_step_smoke` — extended with plan-level Step cases (contig +
  strided, inputs straddling zero incl. exact `0.0` / `-0.0` / NaN).
