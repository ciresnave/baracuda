# Session prompt — Build `baracuda_smem_scan.cuh` helper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Multiple parallel kernel-helper sessions running — read
[`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md) FIRST.

## Context

baracuda has scan kernels (cumsum, cumprod, cummax, cummin, logcumsumexp) that currently use bespoke per-row sequential loops. Modern CUDA does block-level scans via the **Brent-Kung / Hillis-Steele** patterns, which are O(log N) instead of O(N) sequential. CUB has `cub::BlockScan` for this; baracuda doesn't use it directly today.

Beyond the scan family, several other patterns benefit from block-level scans:
- Online softmax (running max + running sum updates)
- Cumulative LSE in attention
- Prefix-sum-driven indexing patterns

This helper consolidates block scans into reusable primitives.

## File layout

Create `crates/baracuda-kernels-sys/kernels/include/baracuda_smem_scan.cuh`.

## Conventions

Same as Phase 65a helpers — include guard, namespace, `__device__ __forceinline__` everywhere, file-top docstring explaining what's provided + when to use.

## Scope

```cuda
namespace baracuda { namespace scan {

// Block-level inclusive scan: each thread's `value` becomes the
// running sum/max/min of all values at-or-before its threadIdx.x
// position. Cooperative — all threads in the block must call.
// `warp_buf` is __shared__ float[32] scratch.
__device__ __forceinline__ float block_scan_inclusive_sum_f32(
    float value, float* __restrict__ warp_buf);

__device__ __forceinline__ float block_scan_inclusive_max_f32(
    float value, float* __restrict__ warp_buf);

__device__ __forceinline__ float block_scan_inclusive_min_f32(
    float value, float* __restrict__ warp_buf);

// Exclusive variants (each thread sees the sum of all PRIOR threads,
// not including its own value; the first thread gets the identity).
__device__ __forceinline__ float block_scan_exclusive_sum_f32(
    float value, float* __restrict__ warp_buf);

// (Optional) f64 variants if needed by f64 scan kernels.
__device__ __forceinline__ double block_scan_inclusive_sum_f64(
    double value, double* __restrict__ warp_buf);

} }  // namespace baracuda::scan
```

## Implementation pattern

Use warp-level scan via `__shfl_up_sync` (Kogge-Stone or similar), then aggregate per-warp results in the cross-warp SMEM buffer + scan those, then broadcast the per-warp offsets back. Standard CUB-style approach. Mirror the `block_reduce_sum_f32` pattern in `baracuda_smem_reduce.cuh` for the cross-warp aggregation.

A reference implementation pattern (warp inclusive scan):

```cuda
__device__ __forceinline__ float warp_scan_inclusive_sum_f32(float v) {
    // Kogge-Stone style: log2(32) = 5 rounds
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, v, offset, 32);
        if ((threadIdx.x & 31) >= offset) v += n;
    }
    return v;
}
```

Then for block scan: warp-scan within each warp, lane 31 of each warp writes the warp-total to SMEM, warp 0 scans the per-warp totals, broadcasts the warp-offset back, and each thread adds the offset to its in-warp result.

## Coordination with `baracuda_smem_reduce.cuh`

This header depends on patterns in `baracuda_smem_reduce.cuh` (already shipped as Phase 65a). You don't need to `#include` reduce.cuh; just write standalone primitives following the same cross-warp aggregation pattern.

## Deliverables

1. The new `.cuh` (~200 LOC).
2. Update the index doc.
3. Commit message lists which existing baracuda kernels are candidates to migrate (cumsum, cumprod, cummax, cummin, logcumsumexp — these are in `kernels/include/baracuda_scan.cuh` or similar).

## Tests

Pure templates — no standalone test. Verification when first kernel migrates.

Optional: a small standalone kernel test that scans `[1,1,1,...,1]` and checks the output is `[1,2,3,...,N]`. Not required.

## Out of scope

- Don't migrate existing scan kernels to use these helpers in this session. Migration is a future phase.
- Don't add segmented scans (per-row scans within a 2D tensor) — those are kernel-specific dispatching that doesn't generalize well into a helper.
- Don't add device-wide scans (`cub::DeviceScan` style) — that's CUB's domain; just use CUB directly when needed.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase67d-smem-scan`
- Read first: `docs/internals/kernel-helpers.md`
- No version bump, no publish.
- Commit on branch + push + update index + stop.

## Stop conditions

- If baracuda already has a block-scan helper anywhere (grep `__shfl_up_sync` across `kernels/include/`): stop, report what's there.
- If the existing scan kernels use a fundamentally different algorithm that wouldn't benefit from this helper: stop, document why in the commit message but still ship the helper for future kernels.
