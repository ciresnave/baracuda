# Lessons Learned

A field manual of mistakes made and caught in the baracuda kernel library,
with debugging fingerprints and root causes. Aimed at future contributors
(human or AI sub-agents) so they don't relitigate solved problems.

This file is curated from the project's commit history (Phases 0-10) and
the memory entries under
`~/.claude/projects/c--Users-cires-OneDrive-Documents-projects-baracuda/memory/`.
Where a lesson has a deeper-dive memory file, the lesson links to it.

## Table of Contents

1. NVCC silent-codegen and compile-time gotchas
2. CUDA library FFI conventions
3. SMEM and tensor-core constraints
4. Numerical correctness patterns
5. Build infrastructure
6. Workspace and library-convention drift
7. CUTLASS vendoring stop-rules
8. Test-flake patterns
9. CUDA 13 / CCCL / cuDNN evolution gotchas (Phases 12-14)

---

## 1. NVCC silent-codegen and compile-time gotchas

### 1.1 `__device__`-only helpers called from `__host__ inline` launchers

**Mistake.** Declared templated helpers like `seg_zero<T>()`,
`seg_max_init<T>()`, and `seg_min_init<T>()` as `__device__ inline` and
then called them from a `__host__ inline` launcher.

**Fingerprint.** Test process exits with code 1 — no Rust panic, no stderr
output, no CUDA error code surfaced. `compute-sanitizer` reports zero GPU
errors. The kernel never actually launches; the host-side `_run` function
aborts before it gets there.

**Fix.** Mark every helper reachable from both host launchers and device
kernels as `__host__ __device__ inline`.

**Why.** NVCC silently produces a host-side stub that calls into a
device-only symbol. Linker doesn't flag it because the symbol exists for
the device path. The process aborts at runtime with no diagnostic when
the host code tries to invoke the broken function.

**First caught.** Milestone 7.6 (segment ops, commit `c8bec6f`). Same
shape recurred in Milestone 8.5 (MoE-on-GGUF vendor, commit `1149f8a`) —
the commit body explicitly notes "Phase 6.6 Flash-Attention's earlier
discipline carried forward: every helper called from both `__host__`
launchers and `__device__` kernels is `__host__ __device__` (otherwise
NVCC silently emits broken code)."

---

### 1.2 `__builtin_memcpy` / `__builtin_memset` under MSVC nvcc

**Mistake.** Used `__builtin_memcpy` / `__builtin_memset` in indexing
kernels — these compile fine under Linux/gcc nvcc but not MSVC nvcc.

**Fingerprint.** `.cu` file fails to compile on Windows-MSVC host with
an "undefined identifier `__builtin_memcpy`" error.

**Fix.** Use `std::memcpy` / `std::memset` (and `#include <cstring>`).
These are portable across the toolchains nvcc supports.

**First caught.** Phase 7 (commit `c8bec6f`) during indexing-kernel
integration on Windows.

---

### 1.3 `QI6_K` macro collision with a local `const int`

**Mistake.** Vendored llama.cpp MoE code declared `const int QI6_K = ...`
inside a function body, but the surrounding GGUF header `#define`s `QI6_K`
as a macro. The macro substitution turns the variable declaration into
gibberish.

**Fingerprint.** A cryptic nvcc compile error around the declaration line
inside the MoE kernel — typically "expected an identifier" or "expression
must have integral type" pointing at what looks like a perfectly valid
`const int` line.

**Fix.** Remove the local `const int` and let the code use the GGUF
macro directly. The preprocessor wins; do not fight it.

**First caught.** Milestone 8.5 (MoE-on-GGUF vendor, commit `1149f8a`).

---

### 1.4 NVCC auto-fuses `a*b + c` into IEEE FMA at `-O2`

**Mistake.** Naive `FmaFunctor<float>` implemented as `a * b + c`. NVCC
at `-O2` fuses the mul+add into a single hardware FMA (one rounding),
while the host reference does two roundings (`a*b` then `+ c`). Output
diverges by 1 ULP on f32/f64.

**Fingerprint.** Test passes for f16/bf16 (which detour through f32 with
explicit conversion roundings) but f32/f64 mismatches by exactly 1 ULP
on dense fixtures, and the mismatch is systematic — not noise.

**Fix.** Use the explicit round-to-nearest intrinsics that nvcc is
forbidden from fusing:

```cuda
// f32: PyTorch's plain mul+add convention
out = __fadd_rn(__fmul_rn(a, b), c);

// f64
out = __dadd_rn(__dmul_rn(a, b), c);
```

f16/bf16 paths f32-detour through the same unfused intrinsics. Used in
`ternary_fma_fp.cu`, `ternary_addcmul_fp.cu`, and `ternary_addcdiv_fp.cu`.

**Why.** PyTorch's Python-level `*` then `+` matches the two-rounding
host reference. Hardware FMA matches a future-PyTorch-with-FMA reference
but not today's. Choose unfused intrinsics if you want bit-parity with
PyTorch's mul+add convention.

**Universal workaround.** Whenever a kernel implements something PyTorch
defines as plain mul+add, reach for `__fmul_rn` + `__fadd_rn` (f32) or
`__dmul_rn` + `__dadd_rn` (f64). See the unary BW negation pattern in §4.3
for a sibling case.

---

### 1.5 Stale `.o` lingers in rlib when a header change makes a file redundant

**Mistake.** A header refactor moved a kernel definition out of one `.cu`
file. `baracuda-forge` noticed nothing changed in the now-redundant `.cu`
file (because nothing did) and reported "All library kernels up-to-date,
skipping compilation." The stale `.o` stayed inside the rlib, defining
the same `__global__` kernel as the new file.

**Fingerprint.** Link-time duplicate-symbol error on a kernel that
compiles cleanly. The kernel name appears in two object files inside the
rlib.

**Fix.** Nuke `target/release/build/baracuda-kernels-sys-*` and the
corresponding rlib in `target/release/deps/` to force forge to re-emit
the build set.

**Caught.** Phase 2 bin (B1) RRR Identity work during int4/bin GEMM
fanout. Documented in the `baracuda-kernels-direction` memory entry.

---

## 2. CUDA library FFI conventions

### 2.1 `cublasXXgeqrfBatched` lives in cuBLAS, NOT cuSOLVER

**Mistake.** Hunted for `cusolverDnSgeqrfBatched` and friends to wire
`BatchedQrPlan`. cuSOLVER-Dn has no batched-geqrf entry point.

**Fingerprint.** Symbol lookup in cuSOLVER headers turns up empty;
cuSOLVER only ships `cusolverDn{S,D,C,Z}geqrf` (single-matrix).

**Fix.** Use `cublas{S,D,C,Z}geqrfBatched` from cuBLAS. The link line for
the batched-QR plan adds `cublas`. Real variants take `*mut *mut T`
host-resident pointer-tables to device matrices; complex variants take
`*mut *mut cuComplex` / `*mut *mut cuDoubleComplex`. Returns a single
host-side info int (not a per-batch array).

**Caught.** Milestone 6.11 (commit `cd6845a`).

---

### 2.2 cuSOLVER `Xgeev` follows LAPACK's packed-real convention strictly

**Mistake.** Set `dataTypeW = CUDA_C_*` (complex) for a real-input call
to `cusolverDnXgeev`, expecting complex-output eigenvalues as in NumPy /
PyTorch.

**Fingerprint.** Call returns `CUSOLVER_STATUS_INVALID_VALUE` (numeric
code 5). cuSOLVER docs don't currently spell out the dtype-matching
requirement, making this initially look like a CUDA 12→13 API drift.

**Fix.** `dataTypeW` must equal `dataTypeA`. The supported configurations
are:

| dtypeA       | dtypeW       | dtypeVL/VR   | computeType  |
|--------------|--------------|--------------|--------------|
| CUDA_R_32F   | CUDA_R_32F   | CUDA_R_32F   | CUDA_R_32F   |
| CUDA_R_64F   | CUDA_R_64F   | CUDA_R_64F   | CUDA_R_64F   |
| CUDA_C_32F   | CUDA_C_32F   | CUDA_C_32F   | CUDA_C_32F   |
| CUDA_C_64F   | CUDA_C_64F   | CUDA_C_64F   | CUDA_C_64F   |

For real input, `W` is sized `[2 * N]` real: first N elements are `wr`
(real parts), last N are `wi` (imag parts). Complex eigenvalue pairs
appear at adjacent indices in the packed layout. The NVIDIA samples
confirm this; JAX hit the same convention (see `jax-ml/jax#27265`).

**Caught.** Milestone 6.12+ (commit `cd6845a`).

---

### 2.3 cuFFT plan handle is `i32`, not a pointer

**Mistake.** Assumed cuFFT followed the cuBLAS / cuSOLVER convention of
an opaque handle pointer (`*mut struct`). Got compilation errors when
trying to store the handle as `Cell<*mut ...>`.

**Fingerprint.** cuFFT headers define `cufftHandle` as `int`, not a
pointer type. `-1` is the conventional "not yet created" sentinel.

**Fix.** Store the cuFFT handle as `Cell<i32>` with `-1` as "uninitialized."

**Caught.** Milestone 6.4 (commit `cd6845a`).

---

### 2.4 cuSOLVER `gesvdaStridedBatched` reports `lwork` in elements

**Mistake.** Sized the workspace for `cusolverDn{S,D}gesvdaStridedBatched`
in bytes (the convention for the rest of cuSOLVER).

**Fingerprint.** Workspace ~4× larger than required for f32; subtle
heap corruption / out-of-bounds writes downstream.

**Fix.** `lwork` from `gesvdaStridedBatched_bufferSize` is in **elements**,
not bytes. Multiply by `sizeof(T)` at the workspace boundary.

Also note: `gesvdaStridedBatched` takes element-strides between batch
slots (not pointer arrays like `gesvdjBatched`), has an `econ` flag for
thin SVD, and `h_R_nrmF` is a host array of per-slot residual Frobenius
norms.

**Caught.** Milestone 6.15 (commit `cd6845a`).

---

### 2.5 cuSOLVER complex apply-Q is `unmqr`, NOT `ormqr`

**Mistake.** Tried to wire `cusolverDnCormqr` / `cusolverDnZormqr` to
apply the Q factor for complex Householder reflectors.

**Fingerprint.** Symbols don't exist in cuSOLVER. The complex apply-Q
entry points are `cusolverDn{C,Z}unmqr` — unitary mqr, not orthogonal mqr.

**Fix.** Use `unmqr` for complex, `ormqr` for real. Mirror cuBLAS's
op-token: `CUBLAS_OP_T` for real transpose, `CUBLAS_OP_C` for complex
conjugate-transpose. The kernel template's per-side / per-op iteration
table also differs (see Milestone 6.18 commit body for the full table).

**Caught.** Milestone 6.18 (commit `cd6845a`).

---

### 2.6 cuDNN CTC uses f32 internal scratch even for f64 input

**Mistake.** Tightened the cuDNN-CTC test tolerance for f64 to match the
bespoke f64 path's `64 × f64::EPSILON`.

**Fingerprint.** Tolerance failures around `~1e-7` on otherwise sound
f64 fixtures. f32 tolerance passes.

**Fix.** Use a `~1e-7` tolerance floor for cuDNN CTC regardless of input
dtype. cuDNN's f64 CTC accumulates in f32 internally — that's a library
behavior, not a kernel bug.

**Caught.** Milestone 7.4 (commit `c8bec6f`).

---

### 2.7 `gels` mixed-precision iterative refinement and the QR fallback

**Mistake.** Built a test fixture meant to force `cusolverDn{SS,DD}gels`
into the non-convergence branch (so the QR fallback could be exercised).
Even Hilbert-style ill-conditioned matrices converged at f32 precision.

**Fingerprint.** The "expect Unsupported on non-convergence" assertion
test was unreliable — pass/fail depended on input fixture luck.

**Fix.** The original brittle assertion test was removed. A forced-fallback
path would require an API knob (deferred). Callers wanting the fallback
should size their workspace as
`max(plan.workspace_size(), plan.qr_fallback_workspace_size())` and supply
a pristine `a_backup` (because `_gels` destroys A in place).

**Caught.** Milestone 6.16 (commit `cd6845a`).

---

### 2.8 Column-major vs row-major flag flip for Cholesky

**Mistake.** Passed the caller-facing `lower` flag through to cuSOLVER
Cholesky unchanged.

**Fingerprint.** Cholesky returned the wrong triangle and the
A = L·L^T (or U^T·U) reconstruction tests failed.

**Fix.** cuSOLVER reads matrices column-major. Baracuda's caller passes
row-major. The `lower` flag must be flipped to match cuSOLVER's
column-major view of the same physical bytes. (For factorizations like
LU, SVD, QR, Solve the column-major pass-through happens to be
self-consistent — only Cholesky needs the explicit flip because its
result depends on which triangle is read.)

**Caught.** Milestone 6.3 (commit `cd6845a`).

---

## 3. SMEM and tensor-core constraints

### 3.1 Flash SDPA SMEM allocated by `kMaxD` instead of runtime `d_k`

**Mistake.** A sub-agent's first cut allocated Flash Attention SMEM by
the compile-time `kMaxD = 128` unconditionally. Tests use `d_k = 32`.

**Fingerprint.** Kernel launch fails with
`CUDA_ERROR_INVALID_VALUE` / "out of resources" even on sm_89, which has
the largest SMEM budget. Per-block cap exceeded at trivial test sizes.

**Fix.** Stride SMEM by the runtime `d_k` / `d_v`, not compile-time
`kMaxD`. Each `sX += sizeof(T) * Br * d_k` etc. uses the actual runtime
stride. `kMaxD` is only the upper bound for the kernel's max-supported
head dim, not a fixed allocation size.

**Caught.** Milestone 6.6 (commit `cd6845a`).

---

### 3.2 `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in past 48 KiB

**Mistake.** Set per-block SMEM ~72 KiB (Flash Attention BW at d_k=32).
Kernel launch fails.

**Fingerprint.** Launch error "out of resources" even though the block
cap on sm_89 is 99 KiB.

**Fix.** sm_80+ requires explicit opt-in to dynamic SMEM past the default
48 KiB carveout:

```cuda
cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    requested_bytes);
```

Required for f32 Flash FW (49 KiB) and all BW dQ paths (~72 KiB at d_k=32).
Per-launch — not a global setting.

**Caught.** Milestone 6.6 (commit `cd6845a`).

---

### 3.3 f64 Flash BW SMEM exceeded 99 KiB at Br=Bc=64

**Mistake.** Initial f64 BW kernel kept `Br = Bc = 64` (matching the FP32
path). f64 doubles every per-element byte count; SMEM usage blew past
sm_89's 99 KiB cap. Original commit guarded f64 BW with
`if (sizeof(T) >= 8) return 3;` Unsupported.

**Fingerprint.** Launch fails on f64 BW at d_k = 32, even with the
opt-in carveout maxed out.

**Fix.** Per-dtype `TileShape<T>` trait specialization. f64 specialization
drops to `Br = Bc = 32`:

```cuda
template <typename T> struct TileShape { static constexpr int Br = 64, Bc = 64; };
template <> struct TileShape<double> { static constexpr int Br = 32, Bc = 32; };
```

Per-instantiation aliases land at the top of each kernel-template body
(FW + dQ BW + dKdV BW) and inside each `__host__` launcher and `smem_bytes`
helper. f64 BW at d_k = 32 now uses ~56 KiB SMEM (vs ~145 KiB at 64×64) —
fits under sm_89's 99 KiB cap with the opt-in carveout.

**Caught.** Milestone 6.13 (commit `cd6845a`).

---

### 3.4 Precision-preserving load/store: `load_ct<T>` vs `load_as_f32`

**Mistake.** Original Flash SDPA used `load_as_f32(...)` helpers that
unconditionally narrow to f32. For f64 inputs this silently dropped
~1000× precision before the math even started.

**Fingerprint.** f64 Flash FW failed even at coarse tolerances. The
ratio of expected vs actual matched what truncation-to-f32 would produce.

**Fix.** Introduced `load_ct<T>` and `store_ct<T>` (compute-type)
helpers returning / taking `ComputeType<T>::type` directly:

- `ComputeType<float>::type   = float`
- `ComputeType<double>::type  = double`
- `ComputeType<__half>::type  = float`         (f16 detours through f32)
- `ComputeType<__nv_bfloat16>::type = float`   (bf16 detours through f32)

After fix, f64 FW passes at `32 × f64::EPSILON` tolerance.

**Caught.** Milestone 6.6 (commit `cd6845a`).

---

### 3.5 sm_89 SMEM headroom — spend it on cp.async stages, not bigger tiles

**Mistake.** Considered bumping Flash SDPA tiles to `Br = Bc = 128` on the
sm_89 sibling plan to take advantage of Ada's bigger SMEM.

**Fingerprint.** 128/128 with d=128 blew past the 99 KiB SMEM cap.

**Fix.** Kept `Br = Bc = 64` and spent the SMEM headroom on a second
`cp.async.cg.shared.global` stage (double-buffered K/V loads) plus
bumping threads/block 128 → 256. Net result: K/V-load latency masked
behind the inner GEMM, 99 KiB opt-in carveout still fits.

**Caught.** Milestone 10.3 (commit `495db1e`).

---

### 3.6 RRR layout for low-precision GEMM needs gather B-loads

**Pattern (not a mistake, just a fingerprint).** Int4 RRR and Bin (B1)
RRR don't have natural smem layouts that match what `mma.sync` consumes.

**Fingerprint.** RRR B-load costs 2-8× the gmem byte reads of the RCR
sibling because the data has to be re-gathered into K-contig packing.

**Resolution.**
- **Int4 RRR**: B is N-pair-packed in gmem; kernel needs K-pair-packed
  smem. Load gathers two nibbles from two K-row gmem bytes (low/high
  nibble picked by `(col_g & 1)`) and packs into one K-pair smem byte.
  2 gmem byte reads per smem byte.
- **Bin (B1) RRR**: B is bit-packed along N in gmem; MMA wants
  bit-packed along K. Load gathers the bit at `(col_g & 7)` from each of
  8 K-row gmem bytes and OR's into one K-pair smem byte. 8 gmem byte
  reads per smem byte — bandwidth-heavy but correct.

Bin (B1) RRR additionally requires N divisible by 8 (gmem byte boundary
along N). RCR variants stay the canonical fast path.

**Caught.** Phase 2 GEMM fanout (commit `c471106`).

---

## 4. Numerical correctness patterns

### 4.1 Rust `.round()` (half-away-from-zero) vs GPU `__float2int_rn` (half-to-even)

**Mistake.** Wrote an int-GEMM CPU reference using Rust's default `.round()`
(IEEE half-away-from-zero), expecting it to match GPU `__float2int_rn` /
`cvt.rni.sat` (IEEE half-to-even, banker's rounding).

**Fingerprint.** First int8 RRR smoke run: 3 of 4 tile-aligned shapes
fail with consistent off-by-one errors (`got=12 expected=13`,
`got=4 expected=5`, etc.). Failure rate is high for tile-aligned cases
(many threads do valid work) and zero for the ragged
`100×70×50` case — looks like a layout bug but isn't. With `alpha = 0.125`,
the disagreement happens at every `acc % 16 == 4` (i.e. every `acc + 0.5`
that lands on an even integer-below — `12.5 → 13` for Rust vs `12` for
GPU).

**Fix.** Rust 1.77+ has `f32::round_ties_even()` /
`f64::round_ties_even()`. Use it whenever the GPU side uses
`__float2int_rn` or `cvt.rni.sat`. For `__float2int_rd` / `_ru` /
`_rz`, use `.floor()` / `.ceil()` / `as i32` cast respectively.

**Why.** The two rounding modes disagree exactly at half-integers where
the integer-below is even. With bounded inputs (e.g. A ∈ [-7, 7], B ∈
[-6, 6]) and small alpha, this case is frequent enough to fail dense
tile-aligned tests but sparse enough to skip ragged tests by luck. The
load-bearing diagnostic was a hand-rolled scalar accumulator with the
same register layout — it produced identical mismatch counts to the MMA
version, ruling out the suspected register-pack bug and pointing at the
reference.

**Memory entry.** `project_s8_rrr_bespoke_tile_aligned_bug.md` —
RESOLVED — has the full debugging story.

**Caught.** Phase 1 (commit `b2230a7` body + first follow-up).

---

### 4.2 Cancellation-weighted tolerance for sum-of-summands BW formulas

**Mistake.** Tested GELU / GELU-tanh / Mish / SiLU backward at flat
`K · eps · |expected|`. Tests failed sporadically for `x` in the
cancellation-sensitive ranges.

**Fingerprint.** Test passes for most `x` values but fails for
`x ∈ [-1.5, -0.3]` (GELU), `x ≈ -1.5` (Mish / SiLU). The numerator-vs-
summand ratio shows the formula has cancellation, not noise.

**Fix.** Two-part:
1. **Rewrite the kernel** to dodge cancellation when an algebraic
   identity allows:
   - GELU: replace `0.5 · (1 + erf(x/√2))` with `0.5 · erfc(-x/√2)`
     for `x < 0`. Mathematically identical; numerically clean.
   - GELU-tanh: rewrite via `s = sigmoid(2u) = 0.5 · (1 + tanh(u))`;
     use `1 - tanh²(u) = 4 · s · (1 - s)`. Final form
     `dy · s · (1 + 2 · x · (1 - s) · u')` has only a small residual
     cancellation (~7×) near `x ≈ -1.5`.
2. **Use weighted tolerance** for the irreducible residual:
   `K · eps · |dy| · (|a| + |b|)` rather than `K · eps · |dy · (a + b)|`.
   When there's no cancellation `|a| + |b| ≈ |a + b|` and the bound
   collapses to flat-eps. When there is, it scales correctly.
   Default `K = 16` for one cancellation; raise to ~32 only if there's
   a second multiplicative chain.

**Memory entry.** `project_activation_bw_cancellation_tolerance.md`.

**Caught.** Phase 3 activation BW fanout (within commit `cd6845a`).

---

### 4.3 IEEE `0 - x` for negation across f16 / bf16 toolkit drift

**Mistake.** Used unary `-x` for negation inside a templated kernel
hoping it works for all four FP dtypes.

**Fingerprint.** Compile error on some toolkit versions — unary minus
isn't uniformly provided for `__half` / `__nv_bfloat16` across CUDA
toolkit releases.

**Fix.** `T(0) - x` instead of `-x`. IEEE 754 makes `0 - x` exact on
finite values (sign-bit flip), so this introduces no numerical error.
Same trick works for both `Sub` BW (`(da, db) = (dy, 0 - dy)`) and
`Rsqrt` BW (`T(0) - T(0.5) * dy * y * y * y`).

**Caught.** Binary-BW fanout (within commit `cd6845a`).

---

### 4.4 Max / Min BW dual-save plan-shape vs Mul / Div BW

**Pattern, not a mistake.** Max / Min reduction BW was initially scoped as
"a new dual-save plan shape." On inspection the dual-save shape (both
`x` and `y` as `Option<TensorRef>`) had already shipped for Mul / Div
binary BW. Only the *tie-break convention* was new.

**Resolution.** Reuse the existing `ReduceBackwardArgsWithSaves`-style
struct. The semantic difference is that for Mul / Div the saves are
*multipliers in the formula*; for Max / Min they're *references for a
comparison*. Same kernel ABI.

PyTorch's tie convention for Max / Min BW:
- `a > b`: `(da, db) = (dy, 0)`
- `a < b`: `(da, db) = (0, dy)`
- `a == b`: `(da, db) = (dy/2, dy/2)` (split)
- NaN: all comparisons false → `(da, db) = (dy, dy)` (falls naturally
  out of the PyTorch `where(==, dy/2, dy).masked_fill(<, 0)` chain).

For Max / Min *reductions* the trailblazer used JAX's split-across-ties
convention (every tied position gets the full gradient), avoiding the
need for a saved argmax/argmin index tensor.

**Memory entry.** `project_max_min_bw_no_new_shape.md`.

---

### 4.5 Shape-op BWs reuse the forward kernel — but only for the involutive ones

**Pattern, not a mistake.** When adding shape-op BWs (Category N), the
shortcut of "BW = FW with mutated params" works for some ops and not
others:

| Op       | BW formula                          | New kernel? |
|----------|-------------------------------------|-------------|
| Flip     | `flip(dy, axes)` (involutive)       | No          |
| Roll     | `roll(dy, -shifts)` (negated)       | No          |
| Permute  | `permute(dy, inv_dims)` (inverse)   | No          |
| Pad      | slice (`dy[pad_low : pad_low + N]`) | Yes         |
| Repeat   | sum-reduce over repeats grid        | Yes         |
| Concat2  | split into 2 sources                | Yes         |

For Flip / Roll / Permute the BW Plan is a pure Rust wrapper calling
the forward FFI with `dy → x_in, dx → y_out` and the inverted param.
Saves a full CU + FFI + build.rs cycle per op.

When BW changes the shape relation (output shape ≠ input shape, as for
Pad / Repeat / Concat) you need a new kernel. Don't twist the forward
signature to fit — the symmetry breaks.

**Memory entry.** `project_shape_bw_reuse_pattern.md`.

---

### 4.6 CTC BW γ-accumulation sign-flip

**Mistake.** Per-k γ scatter computed
`arg = α + β − logp − fw_loss` (sign-flipped `fw_loss`). Since
`fw_loss` stores positive NLL and `1/P = exp(+fw_loss)`, the correct
factor is `exp(α + β − logp + fw_loss)`. The `−fw_loss` form scaled γ
by `P²` instead of `1`.

**Fingerprint.** BW kernel produces finite output (smoke passes) but
the analytic gradient matches `exp(log_probs)` without subtracting the
posterior γ. Finite-difference cross-check at `t=0, c=1` of a
`T=2, target=[1]` fixture: expected `≈ -0.63`, kernel emits `≈ +0.27`
(matches `exp(log_probs)` only).

**Fix.** Flip the sign on `fw_loss` in
`crates/baracuda-kernels-sys/kernels/include/baracuda_ctc.cuh`
(f32 + f64 BW kernels).

**Diagnostic methodology.** The original finite-difference test confused
matters: FD measures `∂L/∂log_probs = -γ`, but the kernel returns
PyTorch's `exp(log_probs) - γ` (gradient w.r.t. implicit logits under
log_softmax). The two differ by an `exp(log_probs)` term. The correct
invariant for the PyTorch convention is
`Σ_c dlog_probs[t, n, c] = 0` (both `exp(log_probs)` and γ sum to 1
along the class axis). After the fix, 14/14 CTC tests pass.

**Caught.** Phase 6 (CTC BW closure within commit `cd6845a`). Phase 5
(commit `cd6845a` body marks the same bug as
"smoke-only, finite-difference helper retained").

---

### 4.7 Welford accumulator dtype must outrun input dtype

**Pattern.** A naive Var / Std kernel using the input dtype for the
running variance accumulator loses precision on long axes.

**Fix.** `WelfordAcc<T>` trait — `Acc = double` for `f64`, `Acc = float`
for everything else. Variance accumulator stays at higher precision
throughout the one-pass loop:

```text
M2 = 0; mean = 0;
for k in 0..n:
    delta  = x[k] - mean
    mean  += delta / (k + 1)
    delta2 = x[k] - mean
    M2    += delta * delta2
variance = M2 / (n - correction)   // correction=1 → sample (PyTorch default)
```

**Caught.** Phase 4 Welford non-f32 generalization (commit `cd6845a`).

---

### 4.8 cuSOLVER `syevd` returns eigenvalues in ascending order

**Pattern.** Test fixtures had to sort their reference eigenvalues
ascending to match cuSOLVER's output convention. Future LAPACK-wrap
work: check the ordering convention against the docs before authoring
tests.

**Caught.** Milestone 6.12 (commit `cd6845a`).

---

### 4.9 cuSOLVER row-major-vs-column-major flag flips for orthogonal-Q ops

**Pattern.** For `ormqr`, iteration direction and conjugation flip by
the (Side, op) combination:

| Side  | op | Iteration order | conj(τ)? | conj(v) where? |
|-------|----|-----------------|----------|----------------|
| Left  | N  | K-1, …, 0       | no       | reduce side    |
| Left  | T  | 0, …, K-1       | no       | reduce side    |
| Left  | C  | 0, …, K-1       | yes      | reduce side    |
| Right | N  | 0, …, K-1       | no       | update side    |
| Right | T  | K-1, …, 0       | no       | update side    |
| Right | C  | K-1, …, 0       | yes      | update side    |

Real T forbids op=C; complex T forbids op=T. Side=Right also flips the
shape of `a_packed` from `[B, M, K]` to `[B, N, N]` (K=N for Right).

**Memory entry.** Milestone 6.18 in `project_phase6_foundation.md`.

---

### 4.10 LAPACK DLARFT — `T[i,i] = +τ_i`, not `-τ_i`

**Mistake.** WY-blocked `ormqr` kernel was authored with `T[i,i] = -τ_i`
on the assumption that the negative form fell out of a `Q = I - V·T·V^T`
identity.

**Fingerprint.** Single-reflector check `Q = I - τ · v · v^T` produces
the wrong sign. Cross-check against the reflector-by-reflector reference
kernel disagrees.

**Fix.** LAPACK DLARFT specifies `T[i,i] = +τ_i`. The off-diagonal
`-τ_k` scaling in the build loop's Step 2 is correct; only the diagonal
seed flipped.

**Also caught.** Implicit-1 in T-build dot product — at
`r = block_start + k` the kernel read `v_k[r]` from packed-A, but that
slot holds `R[k,k]` (the R diagonal), not the implicit Householder 1.
Special-case in the inner loop.

**Caught.** Milestone 6.17 (commit `cd6845a`).

---

## 5. Build infrastructure

### 5.1 Windows 32 KiB argv limit when linking many `.o` files

**Mistake.** `baracuda-forge` builder called `nvcc --lib ...` with 200+
`.o` paths in the argv. Hit Windows's 32 KiB command-line cap.

**Fingerprint.** Linker invocation fails with a truncation error
mid-argument-list on Windows hosts. Less than ~100 `.o` files works;
more breaks.

**Fix.** On MSVC hosts, builder now uses `lib.exe` driven via a
response file (`@response.txt`) rather than direct argv. Captured during
Phase 3 / 4 elementwise fanout when the kernel count crossed ~200 `.o`
files. Without this, the whole crate couldn't link.

**Caught.** Within the Phase 3/4 sweep that landed in commit `cd6845a`.

---

### 5.2 `lld-link.exe` missing from PATH

**Pattern.** On a fresh Windows + LLVM install, `lld-link.exe` ships
under `C:\Program Files\LLVM\bin\` but isn't on PATH by default.
Rust uses `lld-link.exe` as the linker via the user's
`~/.cargo/config.toml` (`linker = "lld-link.exe"`); every cargo
build, test, and publish fails at the link step with `error: linker
'lld-link.exe' not found` until the LLVM bin dir is reachable.

**Fingerprint.**

```text
error: linker `lld-link.exe` not found
  |
  = note: program not found

note: the msvc targets depend on the msvc linker but `link.exe` was
not found
```

**Fix.** Add `C:\Program Files\LLVM\bin` to the user or system `PATH`
once (Windows Settings → Environment Variables). After that, every
shell — bash, PowerShell, cargo invocations from any tool — resolves
`lld-link.exe` automatically.

**Interim workaround.** Before the PATH fix landed, every cargo
invocation in this project prepended `$env:PATH = "C:\Program
Files\LLVM\bin;$env:PATH"` (PowerShell) or `export PATH="/c/Program
Files/LLVM/bin:$PATH"` (bash). Hundreds of commits, regressions, and
publishes ran through that prefix. It's no longer required — the
PATH was permanently set on 2026-05-18, late in the alpha.26 publish
sweep.

**Why baracuda pins lld-link rather than MSVC `link.exe`.** lld-link
gives faster incremental link times for the heavy `baracuda-cutlass`
+ `baracuda-kernels-sys` static libraries (hundreds of CUDA object
files). On a fresh install of CUDA + MSVC + LLVM, it's the right
default — the documentation note here is for the install-time gap,
not a long-term workaround.

---

### 5.3 cuDNN auto-discovery convention

**Pattern.** No single official cuDNN install location — PyTorch /
TensorFlow / ONNX Runtime / CMake's `FindCUDNN.cmake` all probe a
roughly similar list. Baracuda's `crates/baracuda-kernels-sys/build.rs`
encodes the ecosystem consensus:

1. `CUDNN_PATH` — canonical env override (points at the install root,
   the directory containing `lib/` and `bin/`).
2. `CUDNN_ROOT` / `CUDNN_HOME` — historical alternates.
3. **Windows installer layout** (cuDNN 9+):
   `C:\Program Files\NVIDIA\CUDNN\v<X.Y>\lib\<cuda_ver>\x64\` —
   versioned by both cuDNN release and target CUDA toolkit.
4. **Legacy "drop into CUDA toolkit" layout**: `$CUDA_PATH\lib\x64\`
   (Windows) or `$CUDA_PATH/lib64/` (Linux). Pre-cuDNN-9 convention.
5. **Linux distro paths**: `/usr/lib/x86_64-linux-gnu/`,
   `/usr/local/cuda/lib64/`, `/usr/lib64/`.

When wrapping a new NVIDIA library that ships outside the CUDA toolkit
proper, apply the same probe order.

**Caught.** Milestone 7.1 (cuDNN integration, commit `c8bec6f`) and an
earlier "match Windows cuDNN per-CUDA-major subdir" fix (commit `6a6f92d`).

---

### 5.4 cuDNN gated behind a cargo feature

**Pattern.** cuDNN-dependent modules (Conv2d, Pool, CTC-cuDNN sibling)
are gated behind a `cudnn` cargo feature so the crate builds cleanly on
machines without cuDNN installed.

**How to apply.** Any new NVIDIA-library wrap with a non-trivial install
footprint gets its own cargo feature. Don't force everyone to install
cuDNN to build the workspace.

**Caught.** Phase 7 (commit `c8bec6f`).

---

### 5.5 KernelBuilder must watch header files for cache invalidation

**Mistake.** `KernelBuilder` watched only the `.cu` source files for
cargo rebuild signaling. A pure-header change (`baracuda_*.cuh`) didn't
trigger rebuilds.

**Fingerprint.** Edit a kernel header, run tests, results don't change —
because the stale `.o` from the previous header state is still in the
rlib.

**Fix.** `KernelBuilder::watch(collect_header_files())` now monitors
`kernels/include/*.cuh`. Pre-existing rebuild bug; benefits every
kernel family.

**Caught.** Phase 4 / 5 (within commit `cd6845a`).

---

### 5.6 Workspace dependency budget: build-deps liberal, runtime-deps minimal

**Convention.** For baracuda crates:

- **Build-time deps** (`[build-dependencies]`, proc-macros, codegen):
  unconstrained. Pull `bindgen`, `cc`, `rayon`, `sha2`, `walkdir`,
  `serde_json`, etc. as needed.
- **Runtime deps** (`[dependencies]` of crates users link into binaries):
  minimize aggressively. Prefer `thiserror` (proc-macro, build-only at
  the call site) over `anyhow` (runtime). Avoid pulling `serde` into
  runtime crates unless the type is genuinely user-serialized.

**Why.** Runtime deps bloat every downstream binary. Build deps run
once on the developer's machine and don't ship.

**Memory entry.** `feedback_dep_budget.md`.

---

## 6. Workspace and library-convention drift

### 6.1 Plan + Descriptor + Args triple convention

**Pattern.** Every op family in `baracuda-kernels` follows the same
trio:

- `<Op>Plan<T, N>` — owns lazy handles (cuBLAS / cuSOLVER / cuRAND /
  cuFFT) and per-instance state; the `select` constructor validates that
  the (kind, dtype, layout) combo is supported.
- `<Op>Descriptor<N>` — shape and immutable config (axes masks, padding
  amounts, eigenvalue side, etc.).
- `<Op>Args<'a, T, N>` — borrowed device views (`TensorRef`,
  `TensorMut`) and per-launch knobs (`reduction_mode`, `correction`).

`run` is `&self`; `select` is `Self::select(descriptor) -> Result<Self,
Error>`; `can_implement` validates that args match the descriptor
(rank, shape, contig where required).

**Apply.** When adding a new op family, follow this shape exactly.
Don't fork to a 2-struct or 4-struct variant unless the op genuinely
needs it.

---

### 6.2 Sibling-plan pattern for arch-specific tuning

**Pattern.** Arch-specific kernels live as **sibling plan types**, not
hidden inside one plan with an arch switch:

- `FlashSdpaPlan<T>` — sm_80 baseline, all FP dtypes, forward-compat on
  Ada.
- `FlashSdpaSm89Plan<T>` — sm_89-tuned, `cp.async` double-buffering,
  256 threads/block, 99 KiB SMEM opt-in. f16 / bf16 only.

`KernelSku::arch = ArchSku::Sm89` marks the new variant. A future
dispatcher (or Fuel's judge) can race them per shape and dtype at
runtime.

**Apply.** When adding sm_89-tuned int8 or fp8 GEMM, or Conv2d
WINOGRAD_NONFUSED for 3×3 stride-1, follow the same sibling pattern.

**Caught.** Milestone 10.3 (commit `495db1e`).

---

### 6.3 `bit_stable_on_same_hardware` precision-guarantee flagging

**Convention.** Every Plan publishes a `PrecisionGuarantee` via
`fn precision_guarantee(&self) -> PrecisionGuarantee` carrying:
- `math_precision: MathPrecision` (which intrinsic does the math)
- `accumulator: ElementKind`
- `bit_stable_on_same_hardware: bool`
- `deterministic: bool`

Atomic-add accumulators are flagged non-deterministic (e.g. `roi_align`
BW, `histogram` family, `index_select` BW). Tensor-core MMA is bit-stable
for integer ops (integer reductions have no rounding nondeterminism) and
not bit-stable for floats (warp-shuffle ordering can vary).

**Apply.** Mark every new plan honestly. Don't hide nondeterminism
behind silence; downstream callers need to know.

---

### 6.4 Vendor + diverge with attribution (vs wrap-as-dep)

**Convention.** When integrating an MIT / Apache-2.0 crate that's
diverging from baracuda's design, vendor the source rather than
wrap-as-dep. Add a `NOTICE` file in the consuming crate listing the
upstream copyright, both LICENSE files, and the original commit hash.
Surface attribution prominently in the new crate's README and the
workspace root README.

**Why.** Wrap-as-dep creates ongoing version-coupling pain when upstream
drifts; vendor + attribute gives engineering freedom while staying
generous to the original author.

**Memory entry.** `feedback_vendor_with_attribution.md`.

**Applied to.** llama.cpp lineage for GGUF quantization and MoE-on-GGUF
in Phase 8 (commit `1149f8a`); Fuel's `sort.cu` bitonic ladder in Phase 9
(commit `1f0bf74`, adapted to i32 index dtype + added `STABLE` template
int + emit both sorted values and indices in one launch).

---

### 6.5 Verify on the user's real hardware

**Convention.** The user has an RTX 4070 + CUDA toolkit installed. Don't
treat the machine as a CUDA-less CI runner. `nvidia-smi` should be the
first command run before declaring a hardware-dependent task blocked.

**Why.** `#[ignore]` is a CI guard, not a permission slip to skip
locally. Running the real integration tests is the fastest way to catch
driver-API signature mismatches and PTX compatibility issues.

**First caught.** Early baracuda session — driver-API ABI versioning
bug (`cuCtxCreate_v3` returned when `_v2` was wanted) only surfaced
after the first real-hardware run.

**Memory entry.** `feedback_verify_on_real_hw.md`.

---

### 6.6 Test-only fix vs kernel fix discipline

**Pattern.** When a smoke test fails, the bug can be in the kernel, in
the CPU reference, in the tolerance, or in the test fixture. The load-
bearing diagnostic for ruling out the kernel is a hand-rolled scalar
accumulator using the same register layout — if it produces identical
mismatches to the MMA version, the bug is upstream of the MMA. See the
s8 RRR rounding-mode bug (§4.1) for the canonical example.

---

### 6.7 Reject-test "wired-vs-unwired" footgun

**Mistake.** Smoke files include a `select_rejects_non_<op>_today` test
asserting an unwired discriminant is rejected. Fanout sessions wire that
discriminant — the test breaks.

**Fingerprint.** Reject test fails because the variant it claimed was
unsupported is now supported.

**Fix.** Fanout sub-agents must explicitly check + repoint this reject
test to a still-unwired discriminant whenever their work overlaps the
asserted one. Caught at least 4 times across unary / binary / reduction
BW fanouts.

**Memory entry note.** Recurring; mentioned in
`project_baracuda_kernels_direction.md` (Agent A → Sqrt, Agent C left
broken until follow-up swapped Sqrt → Cbrt).

---

## 7. CUTLASS vendoring stop-rules

### 7.1 Vendor partial specs for **routing-layer** gaps only

**Pattern.** CUTLASS sometimes defines a building block but doesn't wire
it through the canonical template chain. Example: `DefaultEpilogueWithBroadcastSimt`
exists for years, but `DefaultGemmWithBroadcast` unconditionally routes
to the TensorOp variant regardless of `OperatorClass`.

**Fix that works.** A vendored partial specialization in
`crates/baracuda-cutlass-kernels-sys/kernels/include/baracuda_simt_broadcast_epilogue.h`
adds the missing `OperatorClass = OpClassSimt` routing. ~50 lines, BSD-3
attribution, uses CUTLASS primitives untouched.

**Memory entry.** `project_simt_broadcast_vendor.md`.

---

### 7.2 Vendoring stops working when iterator **bodies** need to differ

**Mistake (twice).** Tried to vendor int8 RRR support into CUTLASS via:

- **Phase 2b** — vendored `MmaTensorOpMultiplicandTileIterator<Congruous<8, _>>`.
  Compiled clean. u8 passed; s8 failed with consistent ±2 LSB errors.
  Root cause: `RowMajorTensorOpMultiplicandCongruous<8, 128>` smem
  layout has N-contig bytes within b16 chunks. After `ldmatrix.x4.trans`,
  each thread's b32 holds 4 bytes arranged as 2 K × 2 N — but
  `mma.sync.m16n8k32.s8` expects 4 K-adjacent bytes per b32. Mismatch.
  u8 hid the bug because test data had mean ~7 (DC term dominated); s8
  with mean ~0 surfaced it.

- **Phase 2b-v2** — vendored `DefaultMmaCore` for
  `{int8_t, uint8_t} × RowMajor × {same} × RowMajor × OpClassTensorOp × Sm80`
  to swap `SmemLayoutB` to `ColumnMajorTensorOpMultiplicandCrosswise<8, K>`.
  Compiled clean. s8 numerics failed with random-pairing noise
  (16010/16384 cells mismatched). Root cause:
  `TransposePitchLinearThreadMap` only **relabels** thread coords; the
  16-byte gmem load lands as 16 consecutive smem bytes — physically
  N-contig regardless of relabeling. Warp iterator reads K-contig and
  gets garbage.

**Resolution.** Both attempts reverted (commit `6a1a4dd`). Int8 RRR moved
to a bespoke kernel in `baracuda-kernels-sys` (commit `b2230a7` and
subsequent Phase 1 finalization in `c471106`).

**Rule.** Before vendoring a partial spec, mentally simulate one warp's
gmem-load → smem-store → ldmatrix-load → mma cycle. If any leg requires
a different access pattern than what existing CUTLASS iterators provide,
vendor the iterator body too — or write a bespoke kernel.

**Stop rule.** When the third vendoring step in a chain doesn't unblock
the path, the bespoke kernel is the cheaper path. The vendoring cost has
exceeded what writing the kernel from PTX intrinsics would take.

**Memory entries.** `project_int8_rrr_cutlass_gap.md` (terminal),
`project_simt_broadcast_vendor.md` (works for routing, fails for body).

---

### 7.3 `RowMajorTensorOpMultiplicandCongruous<bits, K>` specialization coverage

**Reference.** CUTLASS 4.2.0 ships generic `<sizeof_bits, 64>` and
specifics for `<16, 32>` / `<16, 16>` / `<32, 32>`. 8-bit Congruous
isn't covered for the layouts low-precision RRR needs. Don't be surprised
to hit "incomplete type" errors when reaching for it.

---

## 8. Test-flake patterns

### 8.1 Bespoke CTC under parallel test execution

**Symptom.** Bespoke CTC tests intermittently fail with
`CutlassInternal(5)` when the test runner schedules them concurrently
with other tests. Pass on rerun.

**Likely cause.** Workspace / stream isolation issue, or cuDNN handle
contention spilling over. Not chased; logged for follow-up.

**Workaround.** Rerun, or run CTC tests in their own pass with
`-- --test-threads=1` for that test file.

**Status.** Logged in Phase 10's commit body as "known flaky"
(commit `495db1e`).

---

### 8.2 cuDNN handle contention under parallel test execution

**Symptom.** Single cuDNN-CTC test fails sporadically under the
parallel runner; passes on rerun.

**Status.** Phase 8 (commit `1149f8a`) noted "1 flaky cuDNN CTC test
under parallel test execution (passes on rerun; cuDNN handle contention;
not chased)."

---

### 8.3 MoE-on-GGUF CPU reference math mismatch

**Status.** The vendored llama.cpp MoE kernel's `topk_weight` +
accumulation semantics don't match the obvious CPU reference. Smoke
path retained (kernel launches, no crash, plausible output), assertion
disabled, proper reference rederivation deferred — tagged
`moe-cpu-reference-mismatch` TODO.

**Apply.** When vendoring code whose math is non-trivial, do not
hand-roll the CPU reference under time pressure. Use the original
project's reference (or a captured trace) and a smoke-only `let _ = ...`
placeholder until a real reference is justified.

**Caught.** Milestone 8.5 (commit `1149f8a`).

---

### 8.4 `roi_align` tolerance bump

**Pattern.** `roi_align` test tolerance bumped from 2.0 to 4.0 because
PyTorch's `align_corners=false` + adaptive bilinear sampling diverges
from a naive quadrant-mean by up to ~3 units on the fixture. The kernel
matches PyTorch; the test's approximation was over-tight.

**Apply.** When the kernel agrees with the upstream library at coarse
inspection but a "simple-formula" reference disagrees, the reference is
usually wrong, not the kernel.

**Caught.** Phase 9 (commit `1f0bf74`).

---

### 8.5 Tolerance-bump warning: do it for the right reason

**Anti-pattern.** Bumping `K` in `K · eps · |expected|` to make
cancellation-sensitive BW tests pass.

**Why it's wrong.** Hides the diagnostic that the formula has a
cancellation problem. The weighted bound
`K · eps · |dy| · (|a| + |b|)` is honest about what propagates;
flat-eps with a fudged K is not.

See §4.2 for the principled fix.

---

## 9. CUDA 13 / CCCL / cuDNN evolution gotchas (Phases 12-14)

Lessons added during the Fuel-driven Phase 11-14 work. These post-date
Sections 1-8 and reflect the state of CUDA 13 + CCCL + cuDNN 9+ as
shipped in the user's RTX 4070 + CUDA toolkit environment.

### 9.1 `cub::Max` / `cub::Min` are gone in CUDA 13's bundled CCCL

**Mistake.** Used `cub::Max` / `cub::Min` as the reduction operator in
`cub::BlockReduce::Reduce(value, cub::Max{})` while implementing the
Phase 11.6 block-cooperative Sparsemax rewrite.

**Fingerprint.** NVCC compile error: `cub::Max is undefined` or
`'Max' is not a member of namespace 'cub'`.

**Fix.** Replace with `::cuda::maximum<T>{}` / `::cuda::minimum<T>{}`
from `<cuda/functional>`. These are functors (object-constructed at the
call site), not constants. Include `<cuda/functional>` at global scope.

**Why.** CCCL (the unified CUB + Thrust + libcudacxx repo NVIDIA ships
with CUDA 13) consolidated the reduction-operator vocabulary into
`::cuda::*` and dropped the CUB-side aliases.

**Where first hit.** Phase 11.6 Sparsemax block-cooperative sort
rewrite.

### 9.2 CUB headers must be `#include`d at global scope

**Mistake.** Included `<cub/block/block_radix_sort.cuh>` inside a
`namespace baracuda::softmax { ... }` block in the new Sparsemax
kernel file.

**Fingerprint.** Cascade of `cuda::std::*` symbol resolution errors —
`cuda::std::pair`, `cuda::std::tuple`, internal CUB types — fail to
qualify even though their definitions look correct. Often manifests as
an error pointing inside a CUB header at a line that compiled fine
elsewhere.

**Fix.** Move every CUB `#include` above the file's outer namespace
declaration. CUB internals expect `cuda::std::*` at the unqualified
global namespace; nesting a CUB include re-anchors those symbols under
the enclosing namespace and they no longer match their qualified
references.

**Where first hit.** Phase 11.6 Sparsemax (same agent run as §9.1).

### 9.3 `__bfloat16_as_ushort` is not universally available

**Mistake.** Wrote a bf16 atomicAdd-via-CAS helper using
`__bfloat16_as_ushort(val)` to extract the 16-bit bit pattern for the
CAS-slot manipulation.

**Fingerprint.** NVCC error: `'__bfloat16_as_ushort' was not declared
in this scope` on some CUDA / `cuda_bf16.h` revisions. Works on others
(making this look like a transient that re-appears across CUDA bumps).

**Fix.** Use `memcpy` byte-bit-casts in both directions:
`memcpy(&bits, &val, sizeof(bits));` and `memcpy(&val, &bits,
sizeof(val));`. Always include `<cstring>` not `<string.h>` and never
write `__builtin_memcpy` (see §1.2 — MSVC nvcc rejects it).

**Why.** The `__half_as_ushort` / `__ushort_as_half` pair has been
stable for years; the bf16 counterparts arrived later and are still
intermittently shipped/dropped across `cuda_bf16.h` revisions.

**Where first hit.** Phase 11.3 bf16 / f16 `atomicAdd_via_cas`
helper.

### 9.4 cuDNN's NdDescriptor APIs reject `nb_dims < 4`

**Mistake.** Phase 11.7 Conv1D / ConvTranspose1D first implementation
used `cudnnSetTensorNdDescriptor(desc, dt, /*nb_dims=*/3, dims,
strides)` and `cudnnSetConvolutionNdDescriptor(cd, /*array_length=*/1,
…)` — the natural 1-D shape.

**Fingerprint.** Workspace-size query (`cudnnGetConvolutionForwardWorkspaceSize`
or `BackwardData` / `BackwardFilter`) returns
`CUDNN_STATUS_BAD_PARAM` (-3000). The descriptor-set call itself does
**not** fail — the error surfaces later when the descriptor is used.

**Fix.** Internally pad the rank-3 NCL shape to rank-4 NCLW with
`W = 1` (singleton trailing spatial dim) for tensor + filter
descriptors; set `array_length = 2` on the convolution descriptor with
the trailing axis zero-padded / unit-strided / unit-dilated. The dummy
axis is transparent to callers — output stays logically rank-3 NCL.

**Why.** cuDNN treats anything below 4-D as "not a valid spatial
convolution layout" and refuses to compute workspace bounds for it.
The error is delayed because cuDNN does minimal validation at
`SetNdDescriptor` time — full validation happens at the first
`GetWorkspaceSize` / `Convolution{Forward,Backward}` call that
references the descriptor.

**3D doesn't need this** — NCDHW is rank-5, comfortably above the
threshold. Only 1D triggers it.

**Where first hit.** Phase 11.7 Conv1D smoke test. Phase 11.8 Pool
agent independently discovered the same constraint for
`cudnnSetPoolingNdDescriptor` and padded its 1D pool plans similarly.

### 9.5 `f64::powi(small_n)` is not bit-exact to repeated multiplication

**Mistake.** Phase 12.1 PowI smoke test for `n=2` over f64 asserted
bit-exact equality between the GPU kernel output (which collapses
power-by-squaring to a single `x * x` for n=2) and Rust's `f64::powi(2)`
host reference.

**Fingerprint.** `assertion failed: g.to_bits() == e.to_bits()` with
the two bit patterns differing by exactly 1 ULP at random elements.
Reproduces deterministically per element but the *set* of failing
elements depends on the input distribution.

**Fix.** Use `x * x` directly in the host reference, matching exactly
what the kernel does. The test was implicitly testing two different
math paths and expecting them to agree.

**Why.** Rust's `f64::powi` lowers to the LLVM `llvm.powi.f64`
intrinsic. On the target the user runs on (Windows x86_64), that
intrinsic may route through libm's `pow(x, n_as_double)` rather than
a literal `x * x` reduction. The two paths differ by ≤ 1 ULP because
`pow` internally goes through `exp(n * log(x))` with higher
intermediate precision.

**Lesson.** For kernel-test references that need bit-exact agreement,
compute the kernel's actual operation directly (`x * x` for n=2,
`x * x * x` for n=3, etc.). Don't lean on `.powi(n)` or `.pow(n as f32)`
or anything that gives you "the math result" — you want "the same
floating-point sequence the kernel evaluates."

**Where first hit.** Phase 12.1 PowI smoke test.

### 9.6 Strided plans need buffer-size checks relaxed

**Mistake.** Phase 14 strided FFI siblings reused the existing
contig path's `data.len() >= numel` guard in `can_implement` /
`run`.

**Fingerprint.** `Error::BufferTooSmall { needed: N, got: M }`
returned at `run()` time on legitimately-correct strided views. The
"needed" value is the logical numel, the "got" is the underlying
buffer length — but for a transposed view of a `[B, S, H, D]`
allocation viewed as `[B, H, S, D]`, the buffer is much larger than
the logical numel.

**Fix.** Apply the `data.len() >= numel` check only when the tensor
is canonical-contiguous. For strided views, the buffer behind the
view is opaque to baracuda-kernels (the caller owns the contract that
the view's strides + start-offset are within the underlying buffer).

**Where first hit.** Phase 14.4 SDPA strided FFI sibling. The same
relaxation was needed for every other Phase 14 strided sibling (and
became part of the standard "strided-sibling" template in
`crates/baracuda-kernels/src/shape_layout/contiguize.rs`, which is
the canonical reference for the pattern).

### 9.7 Nibble-packed dtypes count differently in shape vs. storage

**Mistake.** Phase 13.2 Contiguize's first `can_implement` check
compared `args.dest.numel()` (logical element count from shape
product) against `args.dest.data.len()` (storage element count for
the typed `DeviceBuffer<T>` slice). For S4 / U4 where 2 nibbles pack
into 1 storage byte, the storage requirement is `ceil(numel / 2)`,
not `numel`.

**Fingerprint.** `Error::BufferTooSmall { needed: 8, got: 4 }` on a
legitimately-correct S4 buffer (rank-1 shape `[8]` = 8 nibbles = 4
bytes of storage in a `DeviceBuffer<S4>` of length 4).

**Fix.** Dispatch on `desc.element` in the size check:

```rust
let needed_storage = match self.desc.element {
    ElementKind::S4 | ElementKind::U4 => (numel + 1) / 2,
    _ => numel,
};
if dest_len < needed_storage { return Err(BufferTooSmall { … }); }
```

**Why.** baracuda's `S4(pub u8)` / `U4(pub u8)` are 1-byte storage
newtypes that hold one nibble in the low half. The shape vector
counts logical elements; the storage vector counts bytes. They
diverge by factor 2 for nibble-packed dtypes.

**Where first hit.** Phase 13.2 Contiguize. Phase 13.1 WriteSlice
had the same check correctly from the start — the WriteSlice agent
saw the nibble distinction; the Contiguize agent didn't. Caught by
Phase 13 regression and fixed in the same release.

### 9.8 SDPA backward + GQA broadcast needs atomicAdd

**Mistake.** Phase 14.4 SDPA strided FFI assumed the BW path could
handle GQA-broadcast K/V the same way the FW path does (treating
`stride_k[head_axis] == 0` as "read the same K value for every Q
head in the group").

**Fingerprint.** Correct results in FW; silent wrong results in BW
when the broadcast pattern is active (`dK` / `dV` accumulate only the
last Q-head group's contribution, dropping the rest).

**Fix.** Reject `stride_k[head_axis] == 0` and
`stride_v[head_axis] == 0` at the BW plan layer with
`Error::Unsupported("SDPA backward with GQA broadcast requires
atomicAdd-based dK/dV accumulation, deferred")`. FW continues to
accept the broadcast normally.

**Why.** FW reads K / V — broadcast just means multiple Q heads see
the same K / V. BW writes dK / dV — broadcast means multiple Q heads
contribute gradients to the same K / V slot. Without atomicAdd,
those concurrent writes race and the kernel keeps only one. The fix
is mechanically a kernel rewrite using `atomicAdd_via_cas` (per §9.3)
for the head-broadcast axis only; tracked in [`ROADMAP.md`](ROADMAP.md).

**Where first hit.** Phase 14.4 SDPA strided FFI sibling, caught by
the agent during implementation rather than at test time.

### 9.9 crates.io rewrites README image paths from the **crate** dir, not the README's dir

**Mistake.** The workspace `README.md` lives at the repo root and
contains `![hero](assets/barracuda.png)` (relative path). The
`baracuda` crate at `crates/baracuda/Cargo.toml` exposes it via
`readme = "../../README.md"`. GitHub renders the image correctly
(resolves relative to the README's location); crates.io shows a
broken-image icon.

**Fingerprint.** Broken image on `crates.io/crates/baracuda` but
fine on `github.com/<user>/baracuda`. Crates.io tries to load from
`https://github.com/<user>/baracuda/raw/HEAD/crates/baracuda/assets/barracuda.png`
(the crate's dir + the relative path) instead of from the repo root
where the README actually lives.

**Fix.** Use absolute `raw.githubusercontent.com` URLs for images in
workspace READMEs re-exported via `readme = "../../README.md"`:
`![hero](https://raw.githubusercontent.com/<user>/<repo>/refs/heads/main/assets/barracuda.png)`.
Works on both GitHub and crates.io.

**Why.** crates.io anchors relative URLs in rendered READMEs to the
**crate's directory location** in the source tree, regardless of where
the README file physically lives. There's no setting that changes
this; absolute URLs are the workaround.

**Where first hit.** alpha.30 release prep, after Fuel team flagged
the broken image on crates.io. The README badge-update memory
entry was created in the same session.

---

## Appendix: source map

This document is curated from the project's commits and memory entries
listed below.

**Commits mined.**
- `7f4f09a` Phase 0 — scaffolding + type migration.
- `b2230a7` Phase 1 WIP — first bespoke kernel, rounding-mode bug.
- `c471106` Phases 1-2 — full sm_89 GEMM dtype matrix.
- `cd6845a` Phases 3-6 — full ML op surface.
- `c8bec6f` Phase 7 — conv + pool + indexing + embedding + segment ops + CTC-cuDNN sibling.
- `1149f8a` Phase 8 — quantization + GGUF + MoE.
- `1f0bf74` Phase 9 — sort + topk + image transforms.
- `495db1e` Phase 10 — sm_89 tuning + bench harness.
- `6a1a4dd` revert of Phase 2b / 2b-v2 int8 RRR CUTLASS attempts.
- `6a6f92d` Windows cuDNN per-CUDA-major subdir match.

**Memory entries mined** (at
`~/.claude/projects/c--Users-cires-OneDrive-Documents-projects-baracuda/memory/`).

- `MEMORY.md` (index)
- `project_phase6_foundation.md` (detailed; CTC γ-bug, ormqr WY, Eig
  packed-real, Flash f64 tiling, cuFFT, KV-cache, LstSq).
- `project_phase5_complete.md` (norm / loss / softmax + CTC bug status).
- `project_phases_3_4_complete.md` (Phase 3+4 closure, Welford acc,
  binary BW dual-save, cuRAND).
- `project_phase3_completion_status.md`, `project_phase5_softmax_start.md`.
- `project_baracuda_kernels_direction.md` (longest entry; complete
  trailblazer + fanout chronology).
- `project_int8_rrr_cutlass_gap.md` (Phase 2b / 2b-v2 post-mortem).
- `project_simt_broadcast_vendor.md` (SIMT broadcast vendor trick).
- `project_s8_rrr_bespoke_tile_aligned_bug.md` (round_ties_even resolution).
- `project_max_min_bw_no_new_shape.md` (dual-save plan reuse).
- `project_activation_bw_cancellation_tolerance.md` (cancellation
  rewrites + weighted tolerance).
- `project_shape_bw_reuse_pattern.md` (Flip / Roll / Permute BW reuse).
- `feedback_dep_budget.md`, `feedback_vendor_with_attribution.md`,
  `feedback_verify_on_real_hw.md`.
- `project_baracuda.md`, `project_fuel_cutlass_asks.md`,
  `user_rust_ffi_background.md`.

**Known gaps in this document.** A handful of debugging fingerprints in
older commit bodies refer to events without preserved literal error
strings (e.g. the early `cuCtxCreate_v3` vs `_v2` ABI mismatch
mentioned in `feedback_verify_on_real_hw.md`). Where the literal stderr
wasn't recorded, the lesson is omitted rather than fabricated.
