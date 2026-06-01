# Session prompt — Build `baracuda_dtype_promote.cuh` helper

You are working on baracuda, a Rust/CUDA stack at
`c:\Users\cires\OneDrive\Documents\projects\baracuda`. Your job is to
build one specific CUDA kernel helper header that other sessions are
building in parallel — see [`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md)
for the coordination protocol BEFORE starting.

## Context

Baracuda's kernels frequently promote half-precision (f16/bf16) storage
to f32 for compute, then narrow back at write time. This pattern is
currently duplicated across many `.cuh` files. The canonical pair of
functions is `load_as_acc<T>(x)` (T → f32) and `store_from_acc<T>(v)`
(f32 → T), defined locally inside `baracuda_norm.cuh` (lines 66-80 or
so — search the file).

Your job: lift those into a shared header so every kernel can use
them, and extend the coverage to int dtypes (i8, u8, i32, etc.) where
the promotion is to i32 or i64 accumulator.

## File layout

Create the file at
`crates/baracuda-kernels-sys/kernels/include/baracuda_dtype_promote.cuh`.

## Conventions (REQUIRED — match `baracuda_smem_row_stager.cuh` style)

- `#ifndef BARACUDA_DTYPE_PROMOTE_CUH` include guard
- File-top docstring (~15 lines) explaining what the helper provides + when to use it
- `namespace baracuda { ... }` around all functions
- Every function `__device__ __forceinline__`
- Templated where appropriate; specializations for `__half` and `__nv_bfloat16`
- No host-callable functions in this file (unlike the row stager — pure device-side)

## Scope (what to provide)

**Floating-point promotion to f32 (compute) and back:**

```cuda
template <typename T> __device__ __forceinline__ float load_as_f32(T x);
template <typename T> __device__ __forceinline__ T     store_from_f32(float v);
```

Specializations for: `float` (identity), `double` (round to f32 — note precision loss), `__half`, `__nv_bfloat16`. Maybe also `cutlass::half_t` / `cutlass::bfloat16_t` if you can include their headers without cost.

**Double promotion path (when accumulating in f64):**

```cuda
template <typename T> __device__ __forceinline__ double load_as_f64(T x);
template <typename T> __device__ __forceinline__ T      store_from_f64(double v);
```

Specializations for f32, f64, half, bf16.

**Integer accumulator promotion (for `reduce_sum_int` patterns):**

```cuda
template <typename T> __device__ __forceinline__ int64_t load_as_i64(T x);
template <typename T> __device__ __forceinline__ T       store_from_i64(int64_t v);
```

Specializations for i8, u8, i16, u16, i32, u32, i64, u64. Note that storing back from i64 to a narrower type uses C++20 two's-complement modular semantics (matches baracuda's existing int reduce/affine behavior; see commit history for [`baracuda_reduce_int.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_reduce_int.cuh) for the convention).

## Deliverables

1. The new `.cuh` header (~150 LOC).
2. Update `docs/internals/kernel-helpers.md` — move this row from "planned" to "existing" with the commit hash (write the row, fill in hash after commit).
3. Brief commit message explaining what was lifted from where + what's new.

## Tests

This header is pure templates — no standalone test. Verification happens when the next phase that uses it (e.g. Phase 65b normalizer retrofit) calls these functions and confirms numerical equivalence. **Don't write tests for this header itself.**

## Out of scope

- Don't modify any existing `.cu` or `.cuh` files. Leave `baracuda_norm.cuh`'s local copy of `load_as_acc` alone — future kernel retrofits will switch to the shared version one kernel at a time.
- Don't add casts for sub-byte dtypes (S4, U4, Bin, FP8) — those have bespoke pack/unpack patterns; not a fit for this helper.
- Don't add SIMT-vectorized variants (float2/float4 etc.) — that's the `baracuda_vec_load.cuh` helper's job, separate session.

## Coordination

- **Working directory**: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- **Branch**: `phase67a-dtype-promote`
- **Read first**: [`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md)
- **No version bump, no publish** — accumulating for next release.
- **When done**: commit on branch, push, update the index, stop. Don't merge to main — Eric will review parallel branches and merge in order.

## Stop conditions

- If you find the `load_as_acc` functions already lifted into a different shared header (someone else got there first): stop, report what's there, no action needed.
- If the integer promotion semantics don't match an existing baracuda kernel's behavior: stop, ask Eric before proceeding.
- If the file naming convention conflicts with an existing helper: stop, ask Eric.
