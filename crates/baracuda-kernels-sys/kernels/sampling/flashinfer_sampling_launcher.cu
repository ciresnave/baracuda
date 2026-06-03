// baracuda-kernels Phase 46 — FlashInfer sort-free sampling launcher.
//
// Bridges baracuda's `extern "C"` FFI surface to FlashInfer's
// `TopKTopPSamplingFromProb` / `TopPSamplingFromProb` /
// `TopKSamplingFromProb` / `MinPSamplingFromProb` templated host APIs.
//
// FlashInfer's sampling kernels accept a probability tensor `probs`
// (shape `[batch_size, vocab_size]`, already softmaxed; values must be
// non-negative and rows must sum to ~1) and produce one sampled index
// per batch row. The implementation is sort-free: it draws a uniform
// random `u ~ U(0, 1)` per row, then uses a single block-level inclusive
// scan over the row probabilities to find the cell whose cumulative
// mass exceeds `u`. Top-K / Top-P / Min-P filtering is fused into the
// same scan via rejection (re-draw if the chosen cell doesn't pass the
// filter threshold) so no separate sort is needed.
//
// Scope (Phase 46 Tier 1):
//   - f32 probabilities, i32 output indices only.
//   - Combined Top-K + Top-P + Min-P via dedicated launchers per
//     filter combo (Top-K only, Top-P only, Min-P only, Top-K + Top-P).
//   - Deterministic mode wired through (FlashInfer's `deterministic`
//     bool toggle picks a sort-based tiebreaker on the rare ambiguous-
//     cell case).
//   - Scalar `top_k_val` / `top_p_val` / `min_p_val` (one value per
//     batch). Per-row arrays (the `top_k_arr` / `top_p_arr` /
//     `min_p_arr` pointers) are wired as nullptr — caller passes the
//     scalar through the `*_val` argument. Per-row arrays are a
//     mechanical extension.
//
// Caller contract:
//   - `probs`     : `[batch, vocab]` row-major f32, must be non-negative
//                   and sum to ~1 per row (callers typically chain after
//                   a softmax / log_softmax + exp).
//   - `output`    : `[batch]` i32, written.
//   - `valid`     : `[batch]` u8 bool, written (1 if the sample was
//                   accepted; 0 means rejection sampling timed out, in
//                   which case the caller should re-draw with a fresh
//                   seed). May be nullptr if the caller doesn't care.
//   - `indices`   : `[batch]` i32, optional — if non-null, FlashInfer
//                   uses it to perform indirect index mapping (output
//                   becomes `indices[chosen_cell]`). Wire nullptr for
//                   the straight sampling case.
//   - `seed_val`  : RNG seed. Same seed → same output (given identical
//                   probs + filter args + deterministic flag).
//   - `offset_val`: philox offset (typically the running token-counter
//                   from the caller's RNG state).
//
// Status codes (same convention as the rest of the attention family):
//   0 = ok, 2 = invalid_problem, 3 = unsupported.

#include <cstdint>
#include <cuda_runtime.h>

#include "../../vendor/flashinfer/include/flashinfer/sampling.cuh"

namespace {
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

inline int translate(cudaError_t e) {
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

// The vendored FlashInfer sampling kernels dereference the per-row
// success ("output emitted") pointer UNCONDITIONALLY — a null caller
// pointer is an illegal access (CUDA 700), not a no-op. When the caller
// doesn't want the flag (`valid == nullptr`) we hand the kernel a
// stream-ordered scratch buffer and release it after the launch.
// Returns false only if the scratch allocation fails.
inline bool acquire_success(void* valid, int32_t batch, cudaStream_t stream,
                            bool** success_out, void** scratch_out) {
    if (valid) {
        *success_out = reinterpret_cast<bool*>(valid);
        *scratch_out = nullptr;
        return true;
    }
    void* scratch = nullptr;
    if (cudaMallocAsync(&scratch, static_cast<size_t>(batch) * sizeof(bool), stream)
        != cudaSuccess) {
        return false;
    }
    *success_out = reinterpret_cast<bool*>(scratch);
    *scratch_out = scratch;
    return true;
}

inline void release_success(void* scratch, cudaStream_t stream) {
    if (scratch) cudaFreeAsync(scratch, stream);
}

// Standalone `TopKSamplingFromProb` types its per-row top_k array as `T*`
// (float), unlike the combined sampler which uses `IdType*` (int32).
// baracuda keeps a uniform int32 per-row top_k API and converts here.
__global__ void convert_i32_to_f32_kernel(const int32_t* __restrict__ src,
                                          float* __restrict__ dst, int32_t n) {
    int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<float>(src[i]);
}
}  // namespace

extern "C" {

// =====================================================================
// Top-K only sampling
// =====================================================================
//
// `top_k_val` selects the K largest-probability cells per row; any cell
// outside the top-K is masked to zero before normalising and sampling.
//
// Setting `top_k_val == 0` returns failure (FlashInfer treats 0 as
// "all cells allowed" but baracuda standardises on "must be > 0"; pass
// a different sampler if you want unfiltered sampling).

int baracuda_kernels_flashinfer_top_k_sampling_f32_run(
    int32_t batch, int32_t vocab, int32_t top_k_val,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0 || top_k_val <= 0) return STATUS_INVALID_ARG;
    if (top_k_val > vocab) return STATUS_INVALID_ARG;
    if (!probs || !output) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<int32_t*>(output),
        success,
        /*indices=*/nullptr,
        /*top_k_arr=*/nullptr,
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(top_k_val),
        static_cast<uint32_t>(vocab),
        deterministic != 0,
        /*seed_arr=*/nullptr,
        seed_val,
        /*offset_arr=*/nullptr,
        offset_val,
        st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_top_k_sampling_f32_can_implement(
    int32_t batch, int32_t vocab, int32_t top_k_val)
{
    if (batch <= 0 || vocab <= 0 || top_k_val <= 0) return STATUS_INVALID_ARG;
    if (top_k_val > vocab) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

// =====================================================================
// Top-P only sampling (nucleus sampling)
// =====================================================================
//
// `top_p_val` ∈ (0, 1]. The smallest set of largest-probability cells
// whose cumulative mass exceeds `top_p_val` is kept; the rest are masked.

int baracuda_kernels_flashinfer_top_p_sampling_f32_run(
    int32_t batch, int32_t vocab, float top_p_val,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!(top_p_val > 0.0f && top_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    if (!probs || !output) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<int32_t*>(output),
        success,
        /*indices=*/nullptr,
        /*top_p_arr=*/nullptr,
        static_cast<uint32_t>(batch),
        top_p_val,
        static_cast<uint32_t>(vocab),
        deterministic != 0,
        /*seed_arr=*/nullptr,
        seed_val,
        /*offset_arr=*/nullptr,
        offset_val,
        st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_top_p_sampling_f32_can_implement(
    int32_t batch, int32_t vocab, float top_p_val)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!(top_p_val > 0.0f && top_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

// =====================================================================
// Min-P only sampling
// =====================================================================
//
// `min_p_val` ∈ (0, 1]. Cells whose probability `< min_p_val *
// max_prob_in_row` are masked. Compared to Top-P this adapts the cut-off
// to each row's spread — flatter distributions retain more cells.

int baracuda_kernels_flashinfer_min_p_sampling_f32_run(
    int32_t batch, int32_t vocab, float min_p_val,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!(min_p_val > 0.0f && min_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    if (!probs || !output) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::MinPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        /*min_p_arr=*/nullptr,
        reinterpret_cast<int32_t*>(output),
        success,
        /*indices=*/nullptr,
        static_cast<uint32_t>(batch),
        min_p_val,
        static_cast<uint32_t>(vocab),
        deterministic != 0,
        /*seed_arr=*/nullptr,
        seed_val,
        /*offset_arr=*/nullptr,
        offset_val,
        st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_min_p_sampling_f32_can_implement(
    int32_t batch, int32_t vocab, float min_p_val)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!(min_p_val > 0.0f && min_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

// =====================================================================
// Top-K + Top-P combined sampling (the canonical decode hot path)
// =====================================================================
//
// Both filters are applied; `top_k_val` and `top_p_val` are scalars
// shared across the batch. Per-row scalars (`top_k_arr` / `top_p_arr`)
// are wired as nullptr — caller passes scalars only. Per-row support
// is a mechanical extension for a future phase.

int baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_run(
    int32_t batch, int32_t vocab, int32_t top_k_val, float top_p_val,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0 || top_k_val <= 0) return STATUS_INVALID_ARG;
    if (top_k_val > vocab) return STATUS_INVALID_ARG;
    if (!(top_p_val > 0.0f && top_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    if (!probs || !output) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        /*top_k_arr=*/nullptr,
        /*top_p_arr=*/nullptr,
        reinterpret_cast<int32_t*>(output),
        success,
        /*indices=*/nullptr,
        static_cast<uint32_t>(batch),
        static_cast<int32_t>(top_k_val),
        top_p_val,
        static_cast<uint32_t>(vocab),
        deterministic != 0,
        /*seed_arr=*/nullptr,
        seed_val,
        /*offset_arr=*/nullptr,
        offset_val,
        st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_can_implement(
    int32_t batch, int32_t vocab, int32_t top_k_val, float top_p_val)
{
    if (batch <= 0 || vocab <= 0 || top_k_val <= 0) return STATUS_INVALID_ARG;
    if (top_k_val > vocab) return STATUS_INVALID_ARG;
    if (!(top_p_val > 0.0f && top_p_val <= 1.0f)) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

// =====================================================================
// Per-row parameter variants (Phase 66 Tier 2)
// =====================================================================
//
// Identical to the scalar samplers above but the filter threshold is a
// device array `[batch]` — one value per request. The vendored kernels
// already support this (`top_k_arr == nullptr ? top_k_val : top_k_arr[bx]`);
// these entry points just forward a non-null array pointer. The scalar
// fallback value is passed too but ignored by the kernel when the array
// is present.

// Standalone per-row Top-K. `top_k_arr` is int32 `[batch]` (baracuda's
// uniform convention); we convert to the float array FlashInfer's
// standalone sampler expects via a stream-ordered scratch buffer.
int baracuda_kernels_flashinfer_top_k_sampling_f32_arr_run(
    int32_t batch, int32_t vocab, const void* top_k_arr,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!probs || !output || !top_k_arr) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    // i32 -> f32 scratch for the float-typed top_k_arr.
    float* top_k_f32 = nullptr;
    if (cudaMallocAsync(&top_k_f32, static_cast<size_t>(batch) * sizeof(float), st) != cudaSuccess)
        return STATUS_INVALID_ARG;
    {
        int blocks = (batch + 255) / 256;
        convert_i32_to_f32_kernel<<<blocks, 256, 0, st>>>(
            reinterpret_cast<const int32_t*>(top_k_arr), top_k_f32, batch);
    }
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) {
        cudaFreeAsync(top_k_f32, st);
        return STATUS_INVALID_ARG;
    }
    cudaError_t e = flashinfer::sampling::TopKSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<int32_t*>(output), success, /*indices=*/nullptr,
        /*top_k_arr (T*=float)=*/top_k_f32,
        static_cast<uint32_t>(batch), /*top_k_val=*/0u, static_cast<uint32_t>(vocab),
        deterministic != 0, /*seed_arr=*/nullptr, seed_val, /*offset_arr=*/nullptr, offset_val, st);
    release_success(scratch, st);
    cudaFreeAsync(top_k_f32, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_top_p_sampling_f32_arr_run(
    int32_t batch, int32_t vocab, const void* top_p_arr,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!probs || !output || !top_p_arr) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::TopPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<int32_t*>(output), success, /*indices=*/nullptr,
        reinterpret_cast<float*>(const_cast<void*>(top_p_arr)),
        static_cast<uint32_t>(batch), /*top_p_val=*/0.0f, static_cast<uint32_t>(vocab),
        deterministic != 0, /*seed_arr=*/nullptr, seed_val, /*offset_arr=*/nullptr, offset_val, st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_min_p_sampling_f32_arr_run(
    int32_t batch, int32_t vocab, const void* min_p_arr,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!probs || !output || !min_p_arr) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::MinPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<float*>(const_cast<void*>(min_p_arr)),
        reinterpret_cast<int32_t*>(output), success, /*indices=*/nullptr,
        static_cast<uint32_t>(batch), /*min_p_val=*/0.0f, static_cast<uint32_t>(vocab),
        deterministic != 0, /*seed_arr=*/nullptr, seed_val, /*offset_arr=*/nullptr, offset_val, st);
    release_success(scratch, st);
    return translate(e);
}

int baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_arr_run(
    int32_t batch, int32_t vocab, const void* top_k_arr, const void* top_p_arr,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* probs, void* output, void* valid, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!probs || !output || !top_k_arr || !top_p_arr) return STATUS_INVALID_ARG;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
    bool* success = nullptr; void* scratch = nullptr;
    if (!acquire_success(valid, batch, st, &success, &scratch)) return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::TopKTopPSamplingFromProb<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(probs)),
        reinterpret_cast<int32_t*>(const_cast<void*>(top_k_arr)),
        reinterpret_cast<float*>(const_cast<void*>(top_p_arr)),
        reinterpret_cast<int32_t*>(output), success, /*indices=*/nullptr,
        static_cast<uint32_t>(batch), /*top_k_val=*/0, /*top_p_val=*/0.0f,
        static_cast<uint32_t>(vocab), deterministic != 0,
        /*seed_arr=*/nullptr, seed_val, /*offset_arr=*/nullptr, offset_val, st);
    release_success(scratch, st);
    return translate(e);
}


// Per-symbol _can_implement companions for the per-row (_arr) sampler variants.
// Threshold pointer is opaque (validated at call time); we validate batch/vocab here.
extern "C" int32_t baracuda_kernels_flashinfer_top_k_sampling_f32_arr_can_implement(
    int32_t batch, int32_t vocab)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

extern "C" int32_t baracuda_kernels_flashinfer_top_p_sampling_f32_arr_can_implement(
    int32_t batch, int32_t vocab)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

extern "C" int32_t baracuda_kernels_flashinfer_min_p_sampling_f32_arr_can_implement(
    int32_t batch, int32_t vocab)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

extern "C" int32_t baracuda_kernels_flashinfer_top_k_top_p_sampling_f32_arr_can_implement(
    int32_t batch, int32_t vocab)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

}  // extern "C"
