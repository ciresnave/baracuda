# Session prompt — Build `baracuda_hmath.cuh` helper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Multiple parallel kernel-helper sessions running — read
[`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md) FIRST.

## Context

Half-precision math on CUDA has scattered support across architectures
+ intrinsics. Some operations have native hardware instructions
(`__hadd`, `__hmul`, `__hfma2` etc.) on Pascal+, but the API
fragmentation between f16, bf16, and f16x2/bf16x2 packed forms is
notable. Baracuda's existing kernels often promote to f32 for math
(via `load_as_acc<T>`) which works but loses some performance.

This helper provides a uniform half-precision math facade:

- Consistent function names across f16 / bf16
- Auto-fallback to f32 promotion when no native intrinsic exists
- Pack/unpack helpers for f16x2 / bf16x2 vectorized math

## File layout

Create `crates/baracuda-kernels-sys/kernels/include/baracuda_hmath.cuh`.

## Conventions

Same as Phase 65a — include guard `BARACUDA_HMATH_CUH`, namespace
`baracuda::hmath`, file-top docstring, `__device__ __forceinline__`.

Include `<cuda_fp16.h>` and `<cuda_bf16.h>`.

## Scope

```cuda
namespace baracuda { namespace hmath {

// Scalar half-precision math — uniform names across f16 / bf16.
__device__ __forceinline__ __half        hadd(__half a, __half b);
__device__ __forceinline__ __nv_bfloat16 hadd(__nv_bfloat16 a, __nv_bfloat16 b);

__device__ __forceinline__ __half        hmul(__half a, __half b);
__device__ __forceinline__ __nv_bfloat16 hmul(__nv_bfloat16 a, __nv_bfloat16 b);

__device__ __forceinline__ __half        hsub(__half a, __half b);
__device__ __forceinline__ __nv_bfloat16 hsub(__nv_bfloat16 a, __nv_bfloat16 b);

// Fused-multiply-add: a*b + c. Hardware-accelerated on Pascal+ for f16,
// Ampere+ for bf16. Falls back to f32 promotion when not native.
__device__ __forceinline__ __half        hfma(__half a, __half b, __half c);
__device__ __forceinline__ __nv_bfloat16 hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c);

// Transcendental — almost always implemented via f32 promotion since
// CUDA's `hexp` / `hlog` / `hsin` etc. have limited precision +
// inconsistent support across arches.
__device__ __forceinline__ __half        hexp(__half x);
__device__ __forceinline__ __nv_bfloat16 hexp(__nv_bfloat16 x);
__device__ __forceinline__ __half        hlog(__half x);
__device__ __forceinline__ __nv_bfloat16 hlog(__nv_bfloat16 x);
__device__ __forceinline__ __half        htanh(__half x);
__device__ __forceinline__ __nv_bfloat16 htanh(__nv_bfloat16 x);
__device__ __forceinline__ __half        hsigmoid(__half x);
__device__ __forceinline__ __nv_bfloat16 hsigmoid(__nv_bfloat16 x);
__device__ __forceinline__ __half        hsqrt(__half x);
__device__ __forceinline__ __nv_bfloat16 hsqrt(__nv_bfloat16 x);

// Packed (vectorized) variants — operate on a pair of half-precision
// values in a single 32-bit register. Significant perf win for
// elementwise kernels.
__device__ __forceinline__ __half2          hadd2(__half2 a, __half2 b);
__device__ __forceinline__ __nv_bfloat162   hadd2(__nv_bfloat162 a, __nv_bfloat162 b);
__device__ __forceinline__ __half2          hmul2(__half2 a, __half2 b);
__device__ __forceinline__ __nv_bfloat162   hmul2(__nv_bfloat162 a, __nv_bfloat162 b);
__device__ __forceinline__ __half2          hfma2(__half2 a, __half2 b, __half2 c);
__device__ __forceinline__ __nv_bfloat162   hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c);

} }  // namespace baracuda::hmath
```

## Implementation guidance

For f16 + bf16 native ops on supported archs:
- `__hadd`, `__hmul`, `__hsub`, `__hfma` for f16 → wrap directly
- `__hadd` (bf16 overload), `__hmul` (bf16), etc. for bf16 → wrap directly when available; fallback to f32 promotion on older arches

For transcendentals:
- `hexp` etc. → always promote to f32, compute `expf(...)`, narrow back. Document this as the stable path.

For packed:
- f16x2: `__hadd2`, `__hmul2`, `__hfma2` are native
- bf16x2: similar set on Ampere+, fallback via component-wise on older arches

Use `__CUDA_ARCH__` macro to dispatch where arch-specific. Pattern:

```cuda
__device__ __forceinline__ __half hadd(__half a, __half b) {
#if __CUDA_ARCH__ >= 530
    return __hadd(a, b);
#else
    return __float2half(__half2float(a) + __half2float(b));
#endif
}
```

## Coordination with `baracuda_dtype_promote.cuh`

If that helper is already shipped, use its `load_as_f32` / `store_from_f32`
for the fallback paths. If not: write self-contained inline conversions; future merge can factor them out.

## Deliverables

1. The new `.cuh` (~200 LOC).
2. Update the index doc.
3. Commit message lists which existing kernels currently use f32-promotion-only and could benefit from native half-precision math.

## Tests

Optional standalone kernel test that exercises each function against
an f32 reference. Not required for first ship.

## Out of scope

- f64 math wrappers (unrelated; f64 already has native scalar instructions everywhere).
- Quaternion / complex variants.
- FP8 (e4m3 / e5m2) math — those have their own headers and conversion idioms.
- Integer math wrappers.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase67f-hmath`
- Read first: `docs/internals/kernel-helpers.md`
- No version bump, no publish.
- Commit on branch + push + update index + stop.

## Stop conditions

- If you find an existing baracuda half-precision math wrapper anywhere
  (grep `__hadd` across `kernels/include/`): stop, report.
