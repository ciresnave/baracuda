// baracuda_sort.cuh
//
// Bitonic-sort + argsort + msort + sort-backward kernels for the
// Phase 9 Category O sorting family.
//
// Trailblazer algorithm — block-bitonic sort:
//   * One CUDA block per "row" of the input.
//   * `row_len` must be <= MAX_ROW (1024 today; equal to the block-size
//     cap and to the max bitonic-stage we keep in shared memory).
//   * Indices are kept in shared memory and sorted in lockstep with the
//     comparator on `x_row[indices[k]]`. The output values + output
//     indices are then written sequentially per the sorted index array.
//
// Lineage: the inner bitonic-compare ladder is adapted from
// llama.cpp / Fuel's `argsort.cu` (ggml). Key adaptations from the Fuel
// vendor:
//   * Indices are i32 (Fuel used uint32). baracuda's plan layer keys
//     gradients and gather kernels off i32 index throughout.
//   * Both ascending AND descending variants emitted; Fuel emits both
//     too but routes through a template int constant — we keep the
//     same template-int dispatch but plumb the order parameter through
//     the launcher signature.
//   * Stability tie-breaker (msort): when comparator says equal, prefer
//     the smaller index (i.e. the earlier element in the input array)
//     to preserve input order — turns the bitonic sort into a stable
//     sort. We branch on a `STABLE` template parameter.
//   * Sort BW: pure scatter `dx[indices[i]] = dy[i]` per row. Lives in
//     this header next to FW because both share the indices array.
//   * sort FW emits BOTH values + indices in one launch (Fuel only
//     emits indices since downstream code re-gathered). baracuda's
//     plan layer needs indices saved for BW anyway, so we colocate.
//
// Status codes returned by the launchers mirror the rest of the kernel
// family:
//   0 success
//   1 misaligned operand
//   2 invalid problem (e.g. row_len > MAX_ROW or negative)
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error (launch failure)
//
// Trailblazer dtype coverage:
//   * sort / argsort / msort: f32, f64, i32, i64.
//   * sort BW / msort BW: f32, f64 (gradients are FP-only).
//
// Trailblazer limits:
//   * `row_len <= 1024` per block — bitonic stages fit in `shared mem`
//     scaled by `row_len_pad = next_pow2(row_len)`, max 2048 i32 cells.
//   * Larger arrays should route through a future tile-radix kernel —
//     reserved follow-up. The launcher returns status 3 (unsupported)
//     for `row_len > 1024`.

#ifndef BARACUDA_SORT_CUH
#define BARACUDA_SORT_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace baracuda { namespace sort {

// Phase 36 (Fuel ask Gap 6a) — FP8 E4M3 wrapper for the bitonic
// comparator. FP8 E4M3 storage is byte-identical to `uint8_t`, but a
// raw byte-compare does NOT match numerical order (sign bit at the
// top, varying exponent bias). The wrapper carries the raw storage
// byte and decodes to `float` on every `operator<` / `operator>`
// invocation — slightly more expensive than a primitive compare but
// only matters at the comparator (the load + store paths still see a
// flat `uint8_t` byte). Naming mirrors NVIDIA's other FP8 type
// wrappers (`__nv_fp8_e4m3`).
struct Fp8E4M3Sort {
    uint8_t bits;
    __device__ __host__ inline float as_float() const {
#if defined(__CUDA_ARCH__)
        // Decode via NVIDIA's f8→f16 intrinsic then promote to f32.
        // `__nv_cvt_fp8_to_halfraw` returns `__half_raw`; wrap as
        // `__half` for `__half2float` (matches the cast-subbyte
        // family's pattern in `e4m3_to_f32`).
        __half_raw raw = __nv_cvt_fp8_to_halfraw(
            static_cast<__nv_fp8_storage_t>(bits), __NV_E4M3);
        return __half2float(__half(raw));
#else
        // Host-side fallback — same path as the device version but the
        // host CRT doesn't ship the intrinsic. Approximate by
        // unpacking the E4M3 fields manually. The sort kernel never
        // runs on the host (this is just for `__host__` compatibility
        // of `__host__ __device__` paths); the conversion below is
        // bit-accurate for finite values.
        uint8_t b = bits;
        uint32_t sign = (b >> 7) & 0x1;
        uint32_t exp4 = (b >> 3) & 0xF;
        uint32_t mant3 = b & 0x7;
        if (exp4 == 0xF && mant3 == 0x7) {
            // E4M3 NaN encoding (all 1s ex sign).
            uint32_t nan_bits = 0x7FC00000u | (sign << 31);
            float f;
            memcpy(&f, &nan_bits, 4);
            return f;
        }
        int32_t exp32;
        uint32_t mant32;
        if (exp4 == 0) {
            if (mant3 == 0) {
                return sign ? -0.0f : 0.0f;
            }
            // Subnormal — normalize.
            int shift = 0;
            while ((mant3 & 0x4) == 0) { mant3 <<= 1; shift++; }
            mant3 &= 0x3;
            exp32 = -6 - shift + 127;
            mant32 = (uint32_t)mant3 << 21;
        } else {
            exp32 = (int32_t)exp4 - 7 + 127;
            mant32 = (uint32_t)mant3 << 20;
        }
        uint32_t bits32 = (sign << 31) | ((uint32_t)exp32 << 23) | mant32;
        float f;
        memcpy(&f, &bits32, 4);
        return f;
#endif
    }
    __device__ __host__ inline bool operator<(const Fp8E4M3Sort& other) const {
        return as_float() < other.as_float();
    }
    __device__ __host__ inline bool operator>(const Fp8E4M3Sort& other) const {
        return as_float() > other.as_float();
    }
    __device__ __host__ inline bool operator==(const Fp8E4M3Sort& other) const {
        return as_float() == other.as_float();
    }
};

inline constexpr int MAX_ROW = 1024;

// Next-power-of-two — used to pad `row_len` up to the bitonic-network
// power-of-two requirement. Called from both host (launcher) and
// device (kernel inner loop bound), so `__host__ __device__` is
// load-bearing here.
__host__ __device__ inline int32_t next_pow2_i32(int32_t v) {
    if (v <= 1) return 1;
    int32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Strict-less comparator for the bitonic ladder. `STABLE == 1` adds the
// tie-break-on-index so equal-keys preserve input order (stable sort
// / msort). `ORDER == 1` is ascending; `0` is descending.
//
// Both `a_idx` / `b_idx` are the ORIGINAL element indices (what
// indices[col] / indices[ixj] hold). The comparator decides which one
// should appear FIRST in the sorted output.
template <typename T, int ORDER, int STABLE>
__device__ inline bool cmp_swap_needed(
    T a, T b, int32_t a_idx, int32_t b_idx, bool ascending_block)
{
    // For STABLE: equal keys → prefer smaller index in ascending block
    // (so smaller-idx comes first), prefer larger index in descending
    // block (so larger-idx stays at higher position → equivalent of
    // preserving original order from the back). For ORDER == descending
    // sort, the macro-level "ascending_block" parameter flips meaning;
    // the cmp_swap is computed as if for ascending then negated.

    bool a_first;
    if (ORDER == 1) {  // ascending sort
        if (a < b) a_first = true;
        else if (a > b) a_first = false;
        else { // equal keys
            if (STABLE == 0) a_first = true; // arbitrary
            else a_first = ascending_block ? (a_idx < b_idx) : (a_idx > b_idx);
        }
    } else {           // descending sort
        if (a > b) a_first = true;
        else if (a < b) a_first = false;
        else {
            if (STABLE == 0) a_first = true;
            else a_first = ascending_block ? (a_idx < b_idx) : (a_idx > b_idx);
        }
    }
    // In an ascending bitonic block we want `a_first` (smaller-sort-key)
    // at position `col`; if that's already the case, no swap. In a
    // descending bitonic block, swap iff `a_first`.
    return ascending_block ? (!a_first) : a_first;
}

// Block-bitonic sort: one block per row, threads cooperate to sort the
// row's indices array in shared memory, then write values + indices to
// gmem.
//
// Padding contract: positions in `[row_len, row_len_pad)` are filled
// with sentinel `INT32_MAX` indices so the comparator (which reads
// `x_row[indices[col]]`) is never given an OOB-index. We instead make
// the comparator treat sentinel-indexed slots as "greater than any
// real value" by short-circuiting on `idx >= row_len`.
template <typename T, int ORDER, int STABLE>
__global__ void sort_block_kernel(
    const T*  __restrict__ x,           // [batch, row_len]
    T*        __restrict__ y_vals,      // [batch, row_len]
    int32_t*  __restrict__ y_idx,       // [batch, row_len]
    int32_t   row_len,
    int32_t   row_len_pad)
{
    int row = blockIdx.x;
    const T* x_row = x + (int64_t)row * (int64_t)row_len;

    extern __shared__ int32_t s_idx[];

    for (int col = threadIdx.x; col < row_len_pad; col += blockDim.x) {
        s_idx[col] = col;  // including padding slots (cols >= row_len)
    }
    __syncthreads();

    for (int32_t k = 2; k <= row_len_pad; k <<= 1) {
        for (int32_t j = k >> 1; j > 0; j >>= 1) {
            for (int col = threadIdx.x; col < row_len_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    bool ascending_block = ((col & k) == 0);

                    int32_t a_idx = s_idx[col];
                    int32_t b_idx = s_idx[ixj];

                    // Treat padding slots (idx >= row_len) as "greater than
                    // any real value" so they bubble to the end in an
                    // ascending block (and to the start in a descending one,
                    // which we still drop after the sort by reading only
                    // the first row_len cells).
                    bool a_is_pad = (a_idx >= row_len);
                    bool b_is_pad = (b_idx >= row_len);

                    bool swap;
                    if (a_is_pad && b_is_pad) {
                        swap = false;
                    } else if (a_is_pad) {
                        // 'a' is sentinel-greater; swap when ascending block
                        // wants smaller-first.
                        swap = ascending_block;
                    } else if (b_is_pad) {
                        swap = !ascending_block;
                    } else {
                        T a = x_row[a_idx];
                        T b = x_row[b_idx];
                        swap = cmp_swap_needed<T, ORDER, STABLE>(
                            a, b, a_idx, b_idx, ascending_block);
                    }
                    if (swap) {
                        s_idx[col] = b_idx;
                        s_idx[ixj] = a_idx;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write out the first `row_len` cells of the sorted indices and
    // the corresponding values.
    int64_t row_off = (int64_t)row * (int64_t)row_len;
    for (int col = threadIdx.x; col < row_len; col += blockDim.x) {
        int32_t src = s_idx[col];
        if (y_idx)  y_idx [row_off + col] = src;
        if (y_vals) y_vals[row_off + col] = x_row[src];
    }
}

// Sort BW — scatter the upstream grad back to the original positions:
//   dx[row, indices[row, i]] = dy[row, i]
//
// One thread per (row, col). dx must be zeroed by the caller before
// the launch (the launcher does `cudaMemsetAsync` of dx).
template <typename T>
__global__ void sort_backward_kernel(
    const T*       __restrict__ dy,         // [batch, row_len]
    const int32_t* __restrict__ indices,    // [batch, row_len]
    T*             __restrict__ dx,         // [batch, row_len]
    int32_t        batch,
    int32_t        row_len)
{
    int64_t total = (int64_t)batch * (int64_t)row_len;
    int64_t tid   = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step  = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t row = (int32_t)(i / (int64_t)row_len);
        int32_t col = (int32_t)(i - (int64_t)row * (int64_t)row_len);
        int32_t src = indices[i];
        if (src >= 0 && src < row_len) {
            dx[(int64_t)row * (int64_t)row_len + (int64_t)src] = dy[i];
        }
    }
}

template <typename T, int ORDER, int STABLE>
__host__ inline int32_t launch_sort_block(
    const T* x, T* y_vals, int32_t* y_idx,
    int32_t batch, int32_t row_len,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0) return 2;
    if (row_len > MAX_ROW) return 3;
    if (batch == 0 || row_len == 0) return 0;
    if (x == nullptr) return 2;
    int32_t row_pad = next_pow2_i32(row_len);
    int threads = row_pad;
    if (threads > 1024) threads = 1024;
    if (threads < 32) threads = 32;
    size_t smem = (size_t)row_pad * sizeof(int32_t);
    sort_block_kernel<T, ORDER, STABLE><<<batch, threads, smem, stream>>>(
        x, y_vals, y_idx, row_len, row_pad);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_sort_backward(
    const T* dy, const int32_t* indices, T* dx,
    int32_t batch, int32_t row_len,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0) return 2;
    int64_t total = (int64_t)batch * (int64_t)row_len;
    if (total == 0) return 0;
    if (dy == nullptr || indices == nullptr || dx == nullptr) return 2;
    // Zero dx — input positions not referenced by any sorted index keep
    // 0 gradient. (For a true permutation this is a no-op; we zero
    // defensively in case the caller hands us a partially-initialized
    // buffer or the index array contains OOB sentinels.)
    cudaError_t merr = cudaMemsetAsync(dx, 0, (size_t)total * sizeof(T), stream);
    if (merr != cudaSuccess) return 5;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    sort_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, indices, dx, batch, row_len);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::sort

// ============================================================================
// INSTANTIATE macros — sort FW (values + indices), argsort FW (indices
// only), msort FW (stable), sort BW (gradient).
//
// FFI signature for FW:
//   (batch, row_len, descending, x, y_vals, y_idx, ws, ws_bytes, stream)
// FFI signature for argsort FW (no values):
//   (batch, row_len, descending, x, y_idx, ws, ws_bytes, stream)
// FFI signature for BW:
//   (batch, row_len, dy, indices, dx, ws, ws_bytes, stream)
// ============================================================================

#define BARACUDA_KERNELS_SORT_INSTANTIATE(NAME, T)                                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t descending,                                       \
        const void* x, void* y_vals, void* y_idx,                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        if (descending == 0) {                                                                    \
            return baracuda::sort::launch_sort_block<T, 1, 0>(                                    \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        } else {                                                                                  \
            return baracuda::sort::launch_sort_block<T, 0, 0>(                                    \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        }                                                                                          \
    }

#define BARACUDA_KERNELS_MSORT_INSTANTIATE(NAME, T)                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t descending,                                       \
        const void* x, void* y_vals, void* y_idx,                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        if (descending == 0) {                                                                    \
            return baracuda::sort::launch_sort_block<T, 1, 1>(                                    \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        } else {                                                                                  \
            return baracuda::sort::launch_sort_block<T, 0, 1>(                                    \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        }                                                                                          \
    }

#define BARACUDA_KERNELS_ARGSORT_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t descending,                                       \
        const void* x, void* y_idx,                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        if (descending == 0) {                                                                    \
            return baracuda::sort::launch_sort_block<T, 1, 0>(                                    \
                static_cast<const T*>(x), nullptr,                                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        } else {                                                                                  \
            return baracuda::sort::launch_sort_block<T, 0, 0>(                                    \
                static_cast<const T*>(x), nullptr,                                                \
                static_cast<int32_t*>(y_idx), batch, row_len, stream);                            \
        }                                                                                          \
    }                                                                                              \
    /* Phase 36 (Fuel ask Gap 6a): host-side `_can_implement` companion. */                       \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t batch, int32_t row_len)                                                           \
    {                                                                                              \
        if (batch < 0 || row_len < 0) return 2;                                                   \
        if (row_len > baracuda::sort::MAX_ROW) return 3;                                          \
        return 0;                                                                                 \
    }

#define BARACUDA_KERNELS_SORT_BACKWARD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len,                                                           \
        const void* dy, const void* indices, void* dx,                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::sort::launch_sort_backward<T>(                                           \
            static_cast<const T*>(dy),                                                            \
            static_cast<const int32_t*>(indices),                                                 \
            static_cast<T*>(dx),                                                                  \
            batch, row_len, stream);                                                              \
    }

#endif // BARACUDA_SORT_CUH
