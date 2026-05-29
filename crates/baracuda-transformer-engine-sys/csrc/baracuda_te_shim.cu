// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda_te_shim.cu — flat C-ABI bridge for the
// `baracuda-transformer-engine-sys` crate.
//
// Phase 55 — wraps the NVIDIA TransformerEngine (Apache-2.0) FP8
// cast/transpose + delayed-scaling recipe primitives in a Rust-
// friendly C surface. NO pybind11 (Rust talks raw C ABI), NO cuDNN
// dep (cast/recipe paths only — the recipe machinery is pure math
// over a small amax history ring buffer, and the cast intrinsics
// live in `<cuda_fp8.h>` which ships with the CUDA toolkit).
//
// The shim implements the published TE delayed-scaling algorithm
// directly rather than vendoring TE's C++ template machinery:
//
//   * `scale = max_representable / max_amax_in_history` per
//     `transformer_engine/common/recipe/delayed_scaling.cu`.
//   * "fmax" amax-history reduction (the TE default; "most_recent"
//     is the other supported reduction).
//   * Sliding-window history with wrap-around index.
//   * `amax = max(|x|)` reduced inside the cast kernel (saves an
//     extra D2H sync over the separate-pass implementation).
//
// This matches what TE's `delayed_scaling.cu` does at the algorithm
// level — TE's value-add over a hand-roll is correctness + the
// integration with their fused MMA pipeline. The recipe state /
// cast surface is small enough that we don't need to vendor the
// template heart of TE to get the bandwidth-saving FP8 paths on
// sm_89 (where the FP8 MMA throughput == BF16, so we don't get
// anything by depending on TE's MMA pipeline either).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ============================================================================
// FP8 format helpers
// ============================================================================
//
// TE format ids match `transformer_engine/common/include/transformer_engine/transformer_engine.h`:
//   NVTE_DType::kFloat8E4M3 = 4
//   NVTE_DType::kFloat8E5M2 = 5
//
// `max_representable` per format:
//   E4M3: 448.0  (per IEEE-style 4-bit exp / 3-bit mantissa; finite max)
//   E5M2: 57344.0
//
// `min_normal` (used by the optional `compute_scaling_factor` margin
// — not exposed in Phase 55):
//   E4M3: 2^-6
//   E5M2: 2^-14
//
// Source: NVIDIA TE `delayed_scaling.cu` plus the OFP8 spec.

namespace baracuda_te {

enum Fp8Format : int32_t {
    FP8_E4M3 = 0,  // NVTE_DType::kFloat8E4M3 in upstream
    FP8_E5M2 = 1,  // NVTE_DType::kFloat8E5M2 in upstream
};

__host__ __device__ __forceinline__ float fp8_max_representable(int32_t fmt) {
    switch (fmt) {
        case FP8_E4M3: return 448.0f;
        case FP8_E5M2: return 57344.0f;
        default:       return 1.0f;
    }
}

// Per-thread saturating cast to FP8 — mirrors TE's `t2x` helper. We
// route through f32 via the CUDA-toolkit intrinsics (same convention
// as baracuda Phase 13.3's `cast_subbyte_fp8.cu`).
__device__ __forceinline__ uint8_t cvt_f32_to_e4m3(float x) {
    return static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
__device__ __forceinline__ uint8_t cvt_f32_to_e5m2(float x) {
    return static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2));
}
__device__ __forceinline__ float cvt_e4m3_to_f32(uint8_t b) {
    __half h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(h);
}
__device__ __forceinline__ float cvt_e5m2_to_f32(uint8_t b) {
    __half h = __nv_cvt_fp8_to_halfraw(b, __NV_E5M2);
    return __half2float(h);
}

// ============================================================================
// Fused cast + amax reduction
// ============================================================================
//
// For each input element, the kernel:
//   1. Loads x_in (TIn — float / __half / __nv_bfloat16).
//   2. Multiplies by the current `scale` to map the value into FP8's
//      representable range. `scale` is a device-resident scalar
//      (single-element f32 buffer); the recipe machinery updates it
//      before the next forward pass.
//   3. Saturating-casts the scaled value to FP8 via NVIDIA's
//      `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, fmt)`.
//   4. Block-reduces `|x_in|` (un-scaled — the amax tracks the raw
//      tensor's dynamic range, not the post-scale value) into a
//      `__shared__` scratch, then `atomicMax`-bit-reinterpret to the
//      device-resident `amax_current` scalar.
//
// One block per `BLOCK_SIZE` elements. Bias toward big blocks
// (BLOCK_SIZE = 512) since the amax reduction is the load-bearing
// op — small blocks mean more `atomicMax` contention.

constexpr int BARACUDA_TE_BLOCK_SIZE = 512;

// `atomicMax` on f32 needs a bit-reinterpret since CUDA doesn't
// expose a built-in float `atomicMax`. The trick: f32 ordering
// agrees with int32 ordering for non-negative finite values, and
// `|x|` is always non-negative. So we treat the bits as int32 and
// `atomicMax` them. Source: standard CUDA idiom (e.g. used in
// TE's `transformer_engine/common/recipe/delayed_scaling.cu`).
__device__ __forceinline__ void atomic_max_abs_f32(float* dst, float val) {
    if (val < 0.0f) val = -val;
    int32_t* dst_int = reinterpret_cast<int32_t*>(dst);
    int32_t  val_int;
    std::memcpy(&val_int, &val, sizeof(int32_t));
    // Non-negative float -> int reinterpret preserves order.
    (void)atomicMax(dst_int, val_int);
}

template <typename TIn>
__device__ __forceinline__ float load_as_float(const TIn* x, int64_t i);

template <>
__device__ __forceinline__ float load_as_float<float>(const float* x, int64_t i) {
    return x[i];
}
template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* x, int64_t i) {
    return __half2float(x[i]);
}
template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(
    const __nv_bfloat16* x, int64_t i) {
    return __bfloat162float(x[i]);
}

// Fused cast + amax. `fmt` selects E4M3 vs E5M2. The kernel runs
// over `numel` elements with a grid-stride loop so it scales to
// any input size without changing the launch geometry.
template <typename TIn, int FMT>
__global__ void fused_cast_amax_kernel(
    const TIn* __restrict__ x_in,
    uint8_t* __restrict__ x_out,
    const float* __restrict__ scale,      // device scalar
    float* __restrict__ amax_current,     // device scalar
    int64_t numel)
{
    __shared__ float smem_max[BARACUDA_TE_BLOCK_SIZE];

    const float s = scale[0];
    float tmax = 0.0f;

    const int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x
                       + (int64_t)threadIdx.x;
    const int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;

    for (int64_t i = tid; i < numel; i += step) {
        const float v   = load_as_float<TIn>(x_in, i);
        const float av  = fabsf(v);
        if (av > tmax) tmax = av;

        const float vs = v * s;
        if (FMT == FP8_E4M3) {
            x_out[i] = cvt_f32_to_e4m3(vs);
        } else {
            x_out[i] = cvt_f32_to_e5m2(vs);
        }
    }

    // Block-reduce `tmax`. Standard tree-reduction in shared memory.
    smem_max[threadIdx.x] = tmax;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const float a = smem_max[threadIdx.x];
            const float b = smem_max[threadIdx.x + offset];
            smem_max[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomic_max_abs_f32(amax_current, smem_max[0]);
    }
}

template <typename TOut>
__device__ __forceinline__ void store_from_float(TOut* y, int64_t i, float v);

template <>
__device__ __forceinline__ void store_from_float<float>(float* y, int64_t i, float v) {
    y[i] = v;
}
template <>
__device__ __forceinline__ void store_from_float<__half>(__half* y, int64_t i, float v) {
    y[i] = __float2half(v);
}
template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(
    __nv_bfloat16* y, int64_t i, float v) {
    y[i] = __float2bfloat16(v);
}

// Dequantize: y = (fp8_decode(x_in_byte) * scale_inv) cast to TOut.
//
// `scale_inv` is the device-resident reciprocal that the recipe
// machinery publishes alongside `scale` — keeping it precomputed
// saves a division per element on the dequant hot path.
template <typename TOut, int FMT>
__global__ void dequant_kernel(
    const uint8_t* __restrict__ x_in,
    TOut* __restrict__ y_out,
    const float* __restrict__ scale_inv,  // device scalar
    int64_t numel)
{
    const float si = scale_inv[0];

    const int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x
                       + (int64_t)threadIdx.x;
    const int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;

    for (int64_t i = tid; i < numel; i += step) {
        float v;
        if (FMT == FP8_E4M3) {
            v = cvt_e4m3_to_f32(x_in[i]);
        } else {
            v = cvt_e5m2_to_f32(x_in[i]);
        }
        store_from_float<TOut>(y_out, i, v * si);
    }
}

// ============================================================================
// Recipe state update
// ============================================================================
//
// Mirrors TE's `transformer_engine/common/recipe/delayed_scaling.cu`
// at the algorithm level:
//
//   1. The just-finished forward pass populated `amax_history[wp]`
//      via the fused cast above. The recipe-update kernel then:
//   2. Reduces the amax_history ring with `fmax` (the TE default).
//   3. Computes `new_scale = max_repr / max_amax_in_history` (clamped
//      to avoid division by zero on a freshly-initialized ring).
//   4. Writes `new_scale` and its reciprocal `1.0f / new_scale` into
//      `scale[0]` / `scale_inv[0]`.
//   5. Resets `amax_history[wp]` to 0 and advances the write
//      pointer (caller passes the new `wp` next round).
//
// The kernel is launched with `<<<1, BARACUDA_TE_BLOCK_SIZE>>>` —
// the per-tensor recipe state is tiny (history len ~1024 max
// typical) and there's no benefit to multi-block reduction here.

constexpr int BARACUDA_TE_MAX_HISTORY = 8192;  // upper guard

__global__ void recipe_update_kernel(
    float* __restrict__ amax_history,   // length = hist_len
    float* __restrict__ scale,          // single element
    float* __restrict__ scale_inv,      // single element
    int32_t hist_len,
    int32_t write_pos,                  // index just written
    float   max_representable)
{
    __shared__ float smem[BARACUDA_TE_BLOCK_SIZE];

    float tmax = 0.0f;
    for (int i = threadIdx.x; i < hist_len; i += blockDim.x) {
        const float v = amax_history[i];
        if (v > tmax) tmax = v;
    }

    smem[threadIdx.x] = tmax;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const float a = smem[threadIdx.x];
            const float b = smem[threadIdx.x + offset];
            smem[threadIdx.x] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float amax = smem[0];
        // Floor at 1.0 if the history is all-zero (freshly created
        // recipe) — produces an identity scale and keeps the first
        // cast from emitting all-zero output.
        if (!(amax > 0.0f)) amax = 1.0f;
        // Guard against NaN/Inf propagation (a single bad input
        // should not poison the recipe permanently).
        if (!isfinite(amax)) amax = max_representable;

        const float new_scale     = max_representable / amax;
        const float new_scale_inv = amax / max_representable;
        scale[0]     = new_scale;
        scale_inv[0] = new_scale_inv;

        // Reset the just-written slot; the next FW pass will overwrite
        // it via the fused-cast `atomicMax`.
        if (write_pos >= 0 && write_pos < hist_len) {
            amax_history[write_pos] = 0.0f;
        }
    }
}

}  // namespace baracuda_te

// ============================================================================
// C ABI surface — what the Rust `baracuda-transformer-engine-sys` crate sees.
// ============================================================================
//
// Status codes:
//   0 — success
//   1 — invalid argument (length / format / null pointer)
//   5 — launch failure (`cudaGetLastError` returned non-zero)

extern "C" {

// Format ids that the Rust side passes (mirror the in-namespace enum).
int32_t baracuda_te_fp8_format_e4m3() { return baracuda_te::FP8_E4M3; }
int32_t baracuda_te_fp8_format_e5m2() { return baracuda_te::FP8_E5M2; }
float   baracuda_te_fp8_max_representable(int32_t fmt) {
    return baracuda_te::fp8_max_representable(fmt);
}

// Fused cast TIn -> FP8 with amax reduction into the recipe's
// `amax_history[write_pos]` slot.
//
//   x_in           — device pointer to TIn elements (TIn discriminated
//                    by `in_dtype`: 0=f32, 1=f16, 2=bf16)
//   x_out          — device pointer to `numel` bytes of FP8 output
//   scale          — device scalar f32 (single element)
//   amax_history   — device array f32 length >= write_pos+1
//   write_pos      — index into amax_history that this call writes
//   numel          — number of elements to cast
//   fmt            — 0=E4M3, 1=E5M2
//   in_dtype       — 0=f32, 1=f16, 2=bf16
//   stream         — cudaStream_t (opaque pointer)
//
// Returns 0/1/5 per the status-code convention.
int32_t baracuda_te_fused_cast_amax_run(
    const void*  x_in,
    void*        x_out,
    const float* scale,
    float*       amax_history,
    int32_t      write_pos,
    int64_t      numel,
    int32_t      fmt,
    int32_t      in_dtype,
    void*        stream)
{
    if (!x_in || !x_out || !scale || !amax_history) return 1;
    if (numel <= 0) return 1;
    if (fmt != baracuda_te::FP8_E4M3 && fmt != baracuda_te::FP8_E5M2) return 1;
    if (in_dtype < 0 || in_dtype > 2) return 1;
    if (write_pos < 0) return 1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    // Slot-into-history is the `amax_current` argument from the
    // kernel's perspective. The caller is responsible for zeroing
    // the slot before the first cast of the pass — the recipe-update
    // kernel does that automatically on every prior update.
    float* slot = amax_history + write_pos;

    const int   block = baracuda_te::BARACUDA_TE_BLOCK_SIZE;
    const int64_t pre = (numel + block - 1) / block;
    // Cap grid dim at 4096 to keep `atomicMax` contention reasonable;
    // the grid-stride loop in the kernel handles the rest.
    const int   grid  = static_cast<int>(pre < 4096 ? pre : 4096);

    if (fmt == baracuda_te::FP8_E4M3) {
        if (in_dtype == 0) {
            baracuda_te::fused_cast_amax_kernel<float, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const float*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        } else if (in_dtype == 1) {
            baracuda_te::fused_cast_amax_kernel<__half, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const __half*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        } else {
            baracuda_te::fused_cast_amax_kernel<__nv_bfloat16, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        }
    } else {
        if (in_dtype == 0) {
            baracuda_te::fused_cast_amax_kernel<float, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const float*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        } else if (in_dtype == 1) {
            baracuda_te::fused_cast_amax_kernel<__half, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const __half*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        } else {
            baracuda_te::fused_cast_amax_kernel<__nv_bfloat16, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const __nv_bfloat16*>(x_in),
                    reinterpret_cast<uint8_t*>(x_out),
                    scale, slot, numel);
        }
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// FP8 -> TOut dequantize. Symmetric to the fused-cast above; no
// amax reduction (dequant doesn't update recipe state).
//
//   out_dtype — 0=f32, 1=f16, 2=bf16
int32_t baracuda_te_dequant_run(
    const void*  x_in,
    void*        y_out,
    const float* scale_inv,
    int64_t      numel,
    int32_t      fmt,
    int32_t      out_dtype,
    void*        stream)
{
    if (!x_in || !y_out || !scale_inv) return 1;
    if (numel <= 0) return 1;
    if (fmt != baracuda_te::FP8_E4M3 && fmt != baracuda_te::FP8_E5M2) return 1;
    if (out_dtype < 0 || out_dtype > 2) return 1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    const int   block = baracuda_te::BARACUDA_TE_BLOCK_SIZE;
    const int64_t pre = (numel + block - 1) / block;
    const int   grid  = static_cast<int>(pre < 4096 ? pre : 4096);

    if (fmt == baracuda_te::FP8_E4M3) {
        if (out_dtype == 0) {
            baracuda_te::dequant_kernel<float, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<float*>(y_out),
                    scale_inv, numel);
        } else if (out_dtype == 1) {
            baracuda_te::dequant_kernel<__half, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<__half*>(y_out),
                    scale_inv, numel);
        } else {
            baracuda_te::dequant_kernel<__nv_bfloat16, baracuda_te::FP8_E4M3>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<__nv_bfloat16*>(y_out),
                    scale_inv, numel);
        }
    } else {
        if (out_dtype == 0) {
            baracuda_te::dequant_kernel<float, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<float*>(y_out),
                    scale_inv, numel);
        } else if (out_dtype == 1) {
            baracuda_te::dequant_kernel<__half, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<__half*>(y_out),
                    scale_inv, numel);
        } else {
            baracuda_te::dequant_kernel<__nv_bfloat16, baracuda_te::FP8_E5M2>
                <<<grid, block, 0, s>>>(
                    reinterpret_cast<const uint8_t*>(x_in),
                    reinterpret_cast<__nv_bfloat16*>(y_out),
                    scale_inv, numel);
        }
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Recipe update: reduces the amax_history ring with `fmax`, computes
// `scale = max_repr / max_amax`, publishes `scale` / `scale_inv` to
// their device-resident scalars, resets `amax_history[write_pos]` to
// 0 so the next FW pass's fused-cast atomicMax starts from a clean
// slate.
//
//   hist_len  — length of the amax_history ring (typical 1024)
//   write_pos — slot the just-finished FW pass wrote into
//   fmt       — 0=E4M3, 1=E5M2 (determines `max_representable`)
int32_t baracuda_te_recipe_update_run(
    float*  amax_history,
    float*  scale,
    float*  scale_inv,
    int32_t hist_len,
    int32_t write_pos,
    int32_t fmt,
    void*   stream)
{
    if (!amax_history || !scale || !scale_inv) return 1;
    if (hist_len <= 0 || hist_len > baracuda_te::BARACUDA_TE_MAX_HISTORY) return 1;
    if (write_pos < 0 || write_pos >= hist_len) return 1;
    if (fmt != baracuda_te::FP8_E4M3 && fmt != baracuda_te::FP8_E5M2) return 1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);

    const float max_repr = baracuda_te::fp8_max_representable(fmt);

    baracuda_te::recipe_update_kernel<<<1, baracuda_te::BARACUDA_TE_BLOCK_SIZE, 0, s>>>(
        amax_history, scale, scale_inv, hist_len, write_pos, max_repr);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Initialize a recipe's device-resident state to defaults:
//   scale     = 1.0f
//   scale_inv = 1.0f
//   amax_history[..] = 0.0f
//
// Synchronous w.r.t. the caller's stream — the recipe is typically
// created outside the hot path so an extra stream sync is fine.
int32_t baracuda_te_recipe_init_run(
    float*  amax_history,
    float*  scale,
    float*  scale_inv,
    int32_t hist_len,
    void*   stream)
{
    if (!amax_history || !scale || !scale_inv) return 1;
    if (hist_len <= 0 || hist_len > baracuda_te::BARACUDA_TE_MAX_HISTORY) return 1;

    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    const float ones[2] = {1.0f, 1.0f};
    cudaError_t e;

    e = cudaMemcpyAsync(scale,     &ones[0], sizeof(float), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return 5;
    e = cudaMemcpyAsync(scale_inv, &ones[1], sizeof(float), cudaMemcpyHostToDevice, s);
    if (e != cudaSuccess) return 5;
    e = cudaMemsetAsync(amax_history, 0, sizeof(float) * hist_len, s);
    if (e != cudaSuccess) return 5;

    return 0;
}

}  // extern "C"
