// baracuda_sort_big.cuh
//
// Phase 40 (Fuel ask Gap 6b) — multi-block argsort for `row_len > 1024`.
//
// The block-bitonic argsort kernels in `baracuda_sort.cuh` cap `row_len`
// at 1024 because each row is sorted in shared memory inside a single
// block. For LLM top-k of logits the typical row is `vocab_size`
// (32k-256k) so we need a path that scales beyond a single block.
//
// Trailblazer algorithm — CUB segmented radix sort:
//   * One CUDA segment per "row" of the input. Segments are concatenated
//     contiguously: row `r` occupies indices `[r * row_len, (r+1) * row_len)`.
//   * `cub::DeviceSegmentedRadixSort::SortPairs` (descending variant for
//     `descending == 1`) sorts (key=value, payload=index) pairs.
//   * The launcher pre-populates two device-side scratch buffers:
//       - `offsets_d[batch + 1] = {0, row_len, 2*row_len, ..., batch*row_len}`
//       - `indices_in_d[batch * row_len] = {0,1,..,row_len-1, 0,1,..,row_len-1, ...}`
//     via a tiny init kernel. CUB then radix-sorts the keys + payloads
//     into a pair of double-buffer outputs.
//   * The keys-out buffer is discarded (caller only wants indices); only
//     the payload-out buffer is copied to the caller's `indices_out`.
//
// Memory layout of the caller-supplied workspace blob:
//   [0)                    keys_in   : row_len * batch * sizeof(T)
//   [keys_in_bytes)        keys_out  : row_len * batch * sizeof(T)
//   [keys_out_bytes)       idx_in    : row_len * batch * sizeof(int32_t)
//   [idx_in_bytes)         offsets   : (batch + 1) * sizeof(int32_t)
//   [offsets_bytes)        cub_temp  : cub_temp_bytes (queried below)
//
// The launcher COPIES the input into `keys_in` (so the caller's input
// buffer is not aliased into CUB's double-buffer) and then runs the
// sort. The CUB temp size is queried by calling `SortPairs` with
// `d_temp_storage == nullptr`.
//
// Trailblazer dtype coverage: f32, f64, i32, i64. These are the dtypes
// CUB's radix sort supports natively (`cub::Traits<T>` has specs for
// each). f16 / bf16 / fp8 NOT included in this phase — they'd need a
// `cub::Traits` extension or a per-key cast to f32 (deferred).
//
// Status codes returned by the launchers mirror the rest of the kernel
// family:
//   0 success
//   1 misaligned operand
//   2 invalid problem (e.g. negative batch / row_len, null pointer)
//   3 unsupported (row_len <= 1024 — caller should use block-bitonic)
//   4 workspace too small
//   5 internal kernel error (launch failure or CUB error)
//
// **Determinism**: CUB's segmented radix sort is deterministic AND
// stable WITH RESPECT TO THE PAYLOAD when called with the appropriate
// flag. We pass `BEGIN_BIT = 0, END_BIT = sizeof(T) * 8` so the full
// key range is used. Within a row, equal keys retain their original
// payload (= original index) order = stable sort.

#ifndef BARACUDA_SORT_BIG_CUH
#define BARACUDA_SORT_BIG_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

namespace baracuda { namespace sort_big {

// Init kernel — populates `offsets[0..=batch]` with stride-`row_len`
// values and `indices_in[0..batch*row_len)` with the per-row identity
// permutation `{0,1,..,row_len-1, 0,1,..,row_len-1, ...}`.
//
// `static` gives the kernel internal linkage at every TU including
// this header (only `argsort_big.cu` today, but keeps ODR clean).
static __global__ void init_offsets_indices_kernel(
    int32_t* __restrict__ offsets,         // [batch + 1]
    int32_t* __restrict__ indices_in,      // [batch * row_len]
    int32_t batch,
    int32_t row_len)
{
    int64_t total = (int64_t)batch * (int64_t)row_len;
    int64_t tid   = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step  = (int64_t)gridDim.x  * (int64_t)blockDim.x;

    // Offsets: a small one-pass cooperative write. The first
    // `batch + 1` threads (across the whole grid) write the offsets.
    if (tid <= (int64_t)batch) {
        offsets[tid] = (int32_t)(tid * (int64_t)row_len);
    }

    // Identity-per-row payload.
    for (int64_t i = tid; i < total; i += step) {
        int32_t col = (int32_t)(i % (int64_t)row_len);
        indices_in[i] = col;
    }
}

// Compute the bytes needed for the caller-supplied workspace blob given
// (batch, row_len) and the key dtype `T`. Returns 0 on degenerate
// inputs (caller still passes a non-null pointer; CUB tolerates a
// zero-byte run but the offsets / index init still need their scratch
// — we always reserve those).
template <typename T>
__host__ inline size_t workspace_bytes(int32_t batch, int32_t row_len) {
    if (batch < 0 || row_len < 0) return 0;
    int64_t total = (int64_t)batch * (int64_t)row_len;
    size_t keys_in_bytes  = (size_t)total * sizeof(T);
    size_t keys_out_bytes = (size_t)total * sizeof(T);
    size_t idx_in_bytes   = (size_t)total * sizeof(int32_t);
    size_t offsets_bytes  = ((size_t)batch + 1) * sizeof(int32_t);

    // CUB temp size — query with d_temp_storage = nullptr.
    // NB: CUDA 13 / CCCL 3.x dropped the trailing `debug_synchronous`
    // bool parameter from `DeviceSegmentedRadixSort::SortPairs`.
    size_t cub_temp = 0;
    cub::DeviceSegmentedRadixSort::SortPairs<T, int32_t>(
        nullptr, cub_temp,
        /*d_keys_in*/   (const T*)nullptr,
        /*d_keys_out*/  (T*)nullptr,
        /*d_values_in*/ (const int32_t*)nullptr,
        /*d_values_out*/(int32_t*)nullptr,
        /*num_items*/   (int)total,
        /*num_segments*/(int)batch,
        /*d_begin_offsets*/(const int32_t*)nullptr,
        /*d_end_offsets*/  (const int32_t*)nullptr,
        /*begin_bit*/   0,
        /*end_bit*/     (int)(sizeof(T) * 8),
        /*stream*/      (cudaStream_t)0);

    // Round each section up to 256 bytes to keep CUB happy with
    // alignment on every architecture.
    auto align_up = [](size_t n) -> size_t { return (n + 255) & ~(size_t)255; };
    return align_up(keys_in_bytes) + align_up(keys_out_bytes)
         + align_up(idx_in_bytes)  + align_up(offsets_bytes) + align_up(cub_temp);
}

template <typename T>
__host__ inline int32_t launch_argsort_big(
    const T* x, int32_t* indices_out,
    int32_t batch, int32_t row_len, int32_t descending,
    void* workspace, size_t workspace_bytes_in,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0) return 2;
    if (batch == 0 || row_len == 0) return 0;
    if (x == nullptr || indices_out == nullptr) return 2;
    // Multi-block path is reserved for `row_len > 1024`; smaller rows
    // should use the block-bitonic kernel via `baracuda_sort.cuh`.
    if (row_len <= 1024) return 3;

    size_t needed = workspace_bytes<T>(batch, row_len);
    if (workspace == nullptr || workspace_bytes_in < needed) return 4;

    int64_t total = (int64_t)batch * (int64_t)row_len;

    auto align_up = [](size_t n) -> size_t { return (n + 255) & ~(size_t)255; };

    // Partition the caller's workspace blob.
    size_t keys_in_bytes  = align_up((size_t)total * sizeof(T));
    size_t keys_out_bytes = align_up((size_t)total * sizeof(T));
    size_t idx_in_bytes   = align_up((size_t)total * sizeof(int32_t));
    size_t offsets_bytes  = align_up(((size_t)batch + 1) * sizeof(int32_t));

    uint8_t* base = static_cast<uint8_t*>(workspace);
    T*       keys_in   = reinterpret_cast<T*>      (base + 0);
    T*       keys_out  = reinterpret_cast<T*>      (base + keys_in_bytes);
    int32_t* idx_in    = reinterpret_cast<int32_t*>(base + keys_in_bytes + keys_out_bytes);
    int32_t* offsets   = reinterpret_cast<int32_t*>(base + keys_in_bytes + keys_out_bytes + idx_in_bytes);
    void*    cub_temp  = static_cast<void*>(base + keys_in_bytes + keys_out_bytes + idx_in_bytes + offsets_bytes);
    size_t   cub_temp_bytes = workspace_bytes_in
                              - (keys_in_bytes + keys_out_bytes + idx_in_bytes + offsets_bytes);

    // (1) Stage the input into keys_in. CUB's SortPairs does NOT
    // tolerate aliasing between `d_keys_in` and the caller's `x`
    // pointer (it writes into the buffer via double-buffer ping-pong),
    // so we copy explicitly. `cudaMemcpyAsync` on the same stream
    // serializes correctly with the subsequent kernel + CUB launches.
    cudaError_t err = cudaMemcpyAsync(
        keys_in, x, (size_t)total * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) return 5;

    // (2) Populate offsets + identity payload.
    constexpr int kBlock = 256;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    if (blocks_i64 < 1) blocks_i64 = 1;
    if (blocks_i64 > 65535) blocks_i64 = 65535;
    int blocks = (int)blocks_i64;
    init_offsets_indices_kernel<<<blocks, kBlock, 0, stream>>>(
        offsets, idx_in, batch, row_len);
    err = cudaGetLastError();
    if (err != cudaSuccess) return 5;

    // (3) Run the segmented radix sort.
    cudaError_t cub_err;
    if (descending == 0) {
        cub_err = cub::DeviceSegmentedRadixSort::SortPairs<T, int32_t>(
            cub_temp, cub_temp_bytes,
            keys_in, keys_out,
            idx_in, indices_out,
            (int)total,
            (int)batch,
            offsets, offsets + 1,
            0, (int)(sizeof(T) * 8),
            stream);
    } else {
        cub_err = cub::DeviceSegmentedRadixSort::SortPairsDescending<T, int32_t>(
            cub_temp, cub_temp_bytes,
            keys_in, keys_out,
            idx_in, indices_out,
            (int)total,
            (int)batch,
            offsets, offsets + 1,
            0, (int)(sizeof(T) * 8),
            stream);
    }
    if (cub_err != cudaSuccess) return 5;

    return 0;
}

}} // namespace baracuda::sort_big

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher per dtype.
//
// FFI signature mirrors the block-bitonic `argsort` family (so the Rust
// plan can dispatch between bitonic / big-radix with the same call shape):
//
//   _run(batch, row_len, descending, x, y_idx, workspace, workspace_bytes, stream)
//
// Plus two host queries:
//
//   _can_implement(batch, row_len)                  -> i32 status
//   _workspace_size(batch, row_len)                 -> size_t bytes
// =============================================================================

#define BARACUDA_KERNELS_ARGSORT_BIG_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t descending,                                       \
        const void* x, void* y_idx,                                                               \
        void* workspace, size_t workspace_bytes_in,                                               \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::sort_big::launch_argsort_big<T>(                                         \
            static_cast<const T*>(x),                                                              \
            static_cast<int32_t*>(y_idx),                                                          \
            batch, row_len, descending,                                                            \
            workspace, workspace_bytes_in, stream);                                                \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t batch, int32_t row_len)                                                           \
    {                                                                                              \
        if (batch < 0 || row_len < 0) return 2;                                                   \
        if (row_len <= 1024) return 3;                                                            \
        return 0;                                                                                 \
    }                                                                                              \
    extern "C" size_t baracuda_kernels_##NAME##_workspace_size(                                   \
        int32_t batch, int32_t row_len)                                                           \
    {                                                                                              \
        if (batch < 0 || row_len < 0) return 0;                                                   \
        if (batch == 0 || row_len == 0) return 0;                                                 \
        return baracuda::sort_big::workspace_bytes<T>(batch, row_len);                            \
    }

#endif // BARACUDA_SORT_BIG_CUH
