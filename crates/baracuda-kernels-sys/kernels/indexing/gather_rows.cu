// SPDX-FileCopyrightText: 2026 baracuda project contributors  (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gather_rows.cu — capture-safe decode-shaped row gather (embedding lookup).
//
//   dest[n, H] row r  =  table[V, H] row idx[r]      (idx: rank-1 U32, device)
//
// Exists to unblock Fuel's CUDA-graph `CapturedRun` for the embedding
// lookup. The general `index_select` kernel is correct warm but a
// captured graph mis-computes its FIRST output element on replay (element
// 0 goes 1 -> 0 on cuGraphLaunch replay — a replay-time mis-write, not a
// dropped store; the audit found NO CUDA-C source path that skips it, so
// it sits in the driver/graph-node-replay layer). The leading suspect is
// index_select_kernel's heavy by-value param block (a DimsI32 + two
// DimsI64 = 160 bytes of POD-struct kernel args). gather_rows removes
// that entirely:
//
//   - Every output element is written by exactly ONE thread via a plain
//     store; grid-stride with a STRICT `e < n*H` bound so no element is
//     ever skipped (even when n*H exceeds grid capacity), and an OOB index
//     writes a deterministic zero rather than skipping — so an
//     "unwritten/mis-written element" is unrepresentable by construction.
//   - All shape metadata (V, H, n) is passed BY VALUE as int64 scalars —
//     no host-pointer arrays, no by-value struct, no memset, no atomics,
//     no thread-0 path. A captured launch replays bit-identically.
//   - Native U32 index (no bitcast-to-i32 needed, unlike index_select).
//
// Kernel / launcher mechanics mirror the NF4 GEMV family
// (kernels/quantize/nf4_launcher.cu); the all-metadata-by-value shape
// mirrors the `_doff` WriteSlice pattern.
//
// Status codes: 0 success, 2 invalid problem, 5 internal launch failure.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

// Bit-zero for the out-of-range fallback. Explicit per-dtype so we never
// depend on `T{}` value-init semantics for __half / __nv_bfloat16.
template <typename T> __device__ __forceinline__ T gather_zero();
template <> __device__ __forceinline__ float gather_zero<float>() { return 0.0f; }
template <> __device__ __forceinline__ __half gather_zero<__half>() {
    return __float2half(0.0f);
}
template <> __device__ __forceinline__ __nv_bfloat16 gather_zero<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

// -----------------------------------------------------------------------------
// gather_rows kernel. One thread owns one output element `e ∈ [0, n*H)`;
// grid-stride keeps the grid bounded while writing EVERY element.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void gather_rows_kernel(
    T * __restrict__ dest,
    const T * __restrict__ table,
    const uint32_t * __restrict__ idx,
    int64_t V,
    int64_t H,
    int64_t n)
{
    const int64_t total = n * H;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t e = tid; e < total; e += step) { // strict `< total` — covers all
        const int64_t row = e / H;
        const int64_t col = e - row * H;
        const uint32_t v = idx[row];
        // Data-dependent branch on the index VALUE (not on tid): both arms
        // are a plain idempotent store, and every element is always written.
        if (static_cast<int64_t>(v) < V) {
            dest[e] = table[static_cast<int64_t>(v) * H + col];
        } else {
            dest[e] = gather_zero<T>(); // OOB idx -> deterministic zero, never skip
        }
    }
}

template <typename T>
int32_t launch_gather_rows(
    void * dest, const void * table, const void * idx,
    int64_t V, int64_t H, int64_t n,
    cudaStream_t stream)
{
    if (V <= 0 || H <= 0 || n < 0) { return 2; }
    if (n == 0) { return 0; } // valid no-op
    if (!dest || !table || !idx) { return 2; }

    const int64_t total = n * H;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) { blocks = 1; }

    gather_rows_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<T *>(dest),
        static_cast<const T *>(table),
        static_cast<const uint32_t *>(idx),
        V, H, n);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// =============================================================================
// FFI surface — gather_rows {f16, bf16, f32} : _run / _can_implement / _workspace_size
// =============================================================================
//   baracuda_kernels_gather_rows_<T>_run(dest, table, idx, V, H, n, stream)
//     dest  [n, H] of T   (device)
//     table [V, H] of T   (device, row-major)
//     idx   [n]    U32     (device)
//   No workspace. Pure gather -> _workspace_size returns 0.

#define BARACUDA_GATHER_ROWS_LAUNCHER(T_TAG, T_TY)                                 \
    extern "C" int32_t baracuda_kernels_gather_rows_##T_TAG##_run(                 \
        void * dest, const void * table, const void * idx,                        \
        int64_t V, int64_t H, int64_t n,                                          \
        void * stream_ptr)                                                        \
    {                                                                             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);              \
        return launch_gather_rows<T_TY>(dest, table, idx, V, H, n, stream);       \
    }                                                                             \
    extern "C" int32_t baracuda_kernels_gather_rows_##T_TAG##_can_implement(       \
        int64_t V, int64_t H, int64_t n)                                          \
    {                                                                             \
        if (V <= 0 || H <= 0 || n < 0) { return 2; }                              \
        return 0;                                                                 \
    }                                                                             \
    extern "C" size_t baracuda_kernels_gather_rows_##T_TAG##_workspace_size(       \
        int64_t V, int64_t H, int64_t n)                                          \
    {                                                                             \
        (void)V;                                                                  \
        (void)H;                                                                  \
        (void)n;                                                                  \
        return 0;                                                                 \
    }

BARACUDA_GATHER_ROWS_LAUNCHER(f16, __half)
BARACUDA_GATHER_ROWS_LAUNCHER(bf16, __nv_bfloat16)
BARACUDA_GATHER_ROWS_LAUNCHER(f32, float)

#undef BARACUDA_GATHER_ROWS_LAUNCHER
