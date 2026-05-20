// baracuda_write_slice.cuh
//
// Templated kernel + INSTANTIATE macros for the WriteSlice op
// (Phase 13.1 — Category N / ShapeLayoutKind::WriteSlice).
//
// Op semantics:
//   write_slice(dest, source, ranges) -> dest
//     dest[start_0..end_0, ..., start_{N-1}..end_{N-1}] = source
// Assign (not accumulate). `dest` is contiguous, zero-offset, mutated
// in place. `source` is contiguous, zero-offset, with shape per axis
// equal to (end_i - start_i). 1 <= rank <= 8.
//
// Driven by Fuel team's persistent KV-cache append workflow during
// autoregressive decoding — the host-side fast path is a single
// cuMemcpyDtoDAsync; the kernel here covers the generic strided case.
//
// Byte-width dispatch:
//   bN ∈ {b1, b2, b4, b8, b16} — one symbol per sizeof(T). Five total
//   cover all byte-aligned dtypes baracuda's element bank exposes.
//   Plus one nibble-packed symbol for S4/U4 (one byte = two elements).
//
// Status codes mirror the rest of baracuda-kernels-sys:
//   0 success
//   1 misaligned operand
//   2 invalid problem (negative dim, rank out of range, etc.)
//   3 unsupported (e.g. nibble write with odd-aligned innermost axis)
//   4 workspace too small
//   5 internal kernel error (launch failure)

#ifndef BARACUDA_WRITE_SLICE_CUH
#define BARACUDA_WRITE_SLICE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace write_slice {

inline constexpr int MAX_RANK = 8;

struct DimsI32 { int32_t v[MAX_RANK]; };

// Byte-width payload types — POD blobs the kernel just copies. The
// nibble-packed variant treats one byte as two logical elements but the
// memcpy semantics are byte-level too (since we constrain alignment so
// no read-modify-write straddles a byte boundary).
struct Blob1  { uint8_t  bytes[1];  };
struct Blob2  { uint16_t bytes;     };
struct Blob4  { uint32_t bytes;     };
struct Blob8  { uint64_t bytes;     };
struct Blob16 { uint64_t lo, hi;    };

// Generic per-element kernel for the byte-aligned case.
//
// One thread per source element. Maps a flat slab index to per-axis
// source coordinate, then to a dest coordinate by adding the per-axis
// range start. Source contiguous row-major; dest contiguous row-major.
template <typename Blob>
__global__ void write_slice_byte_kernel(
    Blob* __restrict__ dest,
    const Blob* __restrict__ source,
    int64_t source_numel,
    int32_t rank,
    DimsI32 dest_shape,
    DimsI32 source_shape,
    DimsI32 range_start)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < source_numel; i += step) {
        // Unravel `i` into the per-axis coord under `source_shape`.
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = source_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        // Compute dest linear offset (contig row-major, coord shifted by
        // range_start) and source linear offset (== i since contig).
        int64_t dest_off = 0;
        int64_t mul = 1;
        for (int d = rank - 1; d >= 0; --d) {
            dest_off += (coord[d] + (int64_t)range_start.v[d]) * mul;
            mul *= (int64_t)dest_shape.v[d];
        }
        dest[dest_off] = source[i];
    }
}

template <typename Blob>
__host__ inline int32_t launch_write_slice_byte(
    void* dest, const void* source,
    int64_t source_numel,
    int32_t rank,
    const int32_t* dest_shape_host,
    const int32_t* source_shape_host,
    const int32_t* range_start_host,
    cudaStream_t stream)
{
    if (rank < 1 || rank > MAX_RANK) return 2;
    DimsI32 ds = {}, ss = {}, rs = {};
    for (int i = 0; i < rank; ++i) {
        ds.v[i] = dest_shape_host[i];
        ss.v[i] = source_shape_host[i];
        rs.v[i] = range_start_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (source_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    write_slice_byte_kernel<Blob><<<blocks, kBlock, 0, stream>>>(
        static_cast<Blob*>(dest),
        static_cast<const Blob*>(source),
        source_numel, rank, ds, ss, rs);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// Nibble-packed kernel (S4 / U4).
//
// The Rust safe layer constrains `range_start[rank-1]` and
// `range_end[rank-1]` to both be even so no read-modify-write straddles
// a byte boundary on the innermost axis. With that constraint the
// nibble kernel is identical to the b1 (one-byte-per-thread) kernel
// except the innermost axis extent is in *bytes* (= half the elements)
// for both the source and dest. The caller's CUDA-side shape arrays
// already encode the byte-counted innermost axis (Rust side divides by
// two before passing).
__global__ void write_slice_nibble_kernel(
    uint8_t* __restrict__ dest,
    const uint8_t* __restrict__ source,
    int64_t source_byte_numel,
    int32_t rank,
    DimsI32 dest_byte_shape,
    DimsI32 source_byte_shape,
    DimsI32 range_start_bytes)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < source_byte_numel; i += step) {
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = source_byte_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        int64_t dest_off = 0;
        int64_t mul = 1;
        for (int d = rank - 1; d >= 0; --d) {
            dest_off += (coord[d] + (int64_t)range_start_bytes.v[d]) * mul;
            mul *= (int64_t)dest_byte_shape.v[d];
        }
        dest[dest_off] = source[i];
    }
}

__host__ inline int32_t launch_write_slice_nibble(
    void* dest, const void* source,
    int64_t source_byte_numel,
    int32_t rank,
    const int32_t* dest_byte_shape_host,
    const int32_t* source_byte_shape_host,
    const int32_t* range_start_bytes_host,
    cudaStream_t stream)
{
    if (rank < 1 || rank > MAX_RANK) return 2;
    DimsI32 ds = {}, ss = {}, rs = {};
    for (int i = 0; i < rank; ++i) {
        ds.v[i] = dest_byte_shape_host[i];
        ss.v[i] = source_byte_shape_host[i];
        rs.v[i] = range_start_bytes_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (source_byte_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    write_slice_nibble_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<uint8_t*>(dest),
        static_cast<const uint8_t*>(source),
        source_byte_numel, rank, ds, ss, rs);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::write_slice

// Emit one `_run` launcher per byte-width. The Rust side picks the
// matching symbol from `sizeof(T)`.
#define BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(SUFFIX, BLOB)                                       \
    extern "C" int32_t baracuda_kernels_write_slice_##SUFFIX##_run(                                   \
        void* dest, const void* source,                                                               \
        int64_t source_numel,                                                                          \
        int32_t rank,                                                                                  \
        const int32_t* dest_shape,                                                                     \
        const int32_t* source_shape,                                                                   \
        const int32_t* range_start,                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                              \
    {                                                                                                  \
        if (rank < 1 || rank > baracuda::write_slice::MAX_RANK) return 2;                              \
        if (source_numel < 0) return 2;                                                                \
        if (source_numel == 0) return 0;                                                               \
        if (dest == nullptr || source == nullptr) return 2;                                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                   \
        return baracuda::write_slice::launch_write_slice_byte<BLOB>(                                   \
            dest, source, source_numel, rank,                                                          \
            dest_shape, source_shape, range_start, stream);                                            \
    }

#endif // BARACUDA_WRITE_SLICE_CUH
