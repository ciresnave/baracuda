// baracuda_contiguize.cuh
//
// Phase 13.2 — Contiguize op (strided→contiguous copy).
//
// `contiguize(source, source_layout) -> dest (contiguous, zero offset)`.
// Materializes a contiguous tensor from an arbitrary strided view. The
// kernel is byte-level dtype-agnostic: a single sizeof(T)-templated
// kernel covers every byte-aligned dtype (f16, bf16, f32, f64,
// F32Strict, i32, i64, Bool, S8, U8, Fp8E4M3, Fp8E5M2, Complex32,
// Complex64). A separate nibble kernel handles S4 / U4 with the
// innermost-stride constraint documented at `launch_contiguize_nibble`.
//
// The source layout is described by (shape, source_strides,
// source_offset). Strides are signed `int64_t` — Flip ops produce
// negative strides, BroadcastTo ops produce zero strides. The kernel
// uses signed arithmetic throughout.
//
// Three host-side fast paths are selected by `launch_contiguize`:
//   (1) Source already contiguous + zero offset — one cudaMemcpyAsync of
//       the full element-count × sizeof(T) D2D copy.
//   (2) Innermost dim has stride 1 — per-outer-coord cudaMemcpyAsync of
//       `shape[N-1] * sizeof(T)` bytes. Halves instruction count for
//       transpose-friendly layouts like NCHW→NHWC where the C axis
//       remains contiguous.
//   (3) Generic per-element kernel — one thread per output element;
//       thread decomposes its output linear index into multi-index;
//       dots with `source_strides` (signed) to get the source element
//       offset; element-sized memcpy.
//
// Status codes mirror the rest of baracuda-kernels-sys:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch error.

#ifndef BARACUDA_CONTIGUIZE_CUH
#define BARACUDA_CONTIGUIZE_CUH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace baracuda { namespace contiguize {

inline constexpr int MAX_RANK = 8;

// Plain-data structs for fixed-rank-8 shape / stride pass-by-value.
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

// =============================================================================
// Byte-width-templated generic kernel — one thread per output element.
// =============================================================================
//
// `ElemBytes` is the element size in bytes (1, 2, 4, 8, or 16). The
// kernel does a memcpy of that size per output element so it is dtype-
// agnostic at the kernel level — every byte-aligned dtype routes to
// the same compiled body. The element-sized payload is moved as a
// small fixed-size struct so the compiler can issue an aligned LD/ST
// pair for the common 4 / 8 / 16-byte cases.
//
// `source_offset` is in ELEMENTS (not bytes) — matches the Rust-side
// `ContiguizeDescriptor::source_offset` semantics.

template <int ElemBytes>
__global__ void contiguize_generic_kernel(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ dest,
    int64_t numel,
    int32_t rank,
    DimsI32 shape,
    DimsI64 source_strides,
    int64_t source_offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        // Unravel output linear index into multi-index and accumulate
        // the signed source element offset.
        int64_t linear = i;
        int64_t src_elem_off = source_offset;
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            src_elem_off += c * source_strides.v[d];  // signed!
        }
        // Element-sized memcpy. `__builtin_memcpy` is lowered to an
        // aligned LD/ST pair for the natural sizes (1/2/4/8/16).
        const unsigned char* src_ptr = source + src_elem_off * (int64_t)ElemBytes;
        unsigned char* dst_ptr = dest + i * (int64_t)ElemBytes;
        // The compiler turns this into the right hardware instruction
        // for ElemBytes ∈ {1, 2, 4, 8, 16}.
        #pragma unroll
        for (int b = 0; b < ElemBytes; ++b) {
            dst_ptr[b] = src_ptr[b];
        }
    }
}

// =============================================================================
// Fast-path #2 kernel: innermost dim has stride 1.
// =============================================================================
//
// One thread per OUTER coord; each thread runs a strided per-element
// loop over the innermost axis. Reduces divmod cost from N → N-1 axes
// per element. We don't use cudaMemcpyAsync per outer coord because
// that requires N-1 host-issued launches (per-coord) and would serialize.
//
// The kernel produces the same result as `contiguize_generic_kernel`
// for the special case `source_strides[rank-1] == 1`. Numerically
// identical (pure copy, no math).

template <int ElemBytes>
__global__ void contiguize_inner_stride1_kernel(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ dest,
    int64_t outer_numel,
    int32_t rank,
    int32_t inner_extent,
    DimsI32 outer_shape,         // shape[0..rank-1]
    DimsI64 outer_source_strides, // source_strides[0..rank-1]
    int64_t source_offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t outer = tid; outer < outer_numel; outer += step) {
        // Decompose outer linear into multi-index over the first rank-1
        // axes; the innermost axis is the contiguous run.
        int64_t linear = outer;
        int64_t src_elem_off = source_offset;
        for (int d = rank - 2; d >= 0; --d) {
            int32_t s = outer_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            src_elem_off += c * outer_source_strides.v[d];
        }
        // Each outer coord copies a contiguous run of `inner_extent`
        // elements from source to dest.
        const unsigned char* src_run =
            source + src_elem_off * (int64_t)ElemBytes;
        unsigned char* dst_run =
            dest + outer * (int64_t)inner_extent * (int64_t)ElemBytes;
        const int64_t run_bytes = (int64_t)inner_extent * (int64_t)ElemBytes;
        for (int64_t b = 0; b < run_bytes; ++b) {
            dst_run[b] = src_run[b];
        }
    }
}

// =============================================================================
// Host launcher — picks among fast paths and the generic kernel.
// =============================================================================
//
// `ElemBytes` MUST match the on-device element size of the dtype the
// caller intended. The Rust dispatcher routes (dtype → ElemBytes)
// before calling the per-byte-width FFI symbol; this function does not
// check dtype.

template <int ElemBytes>
__host__ inline int32_t launch_contiguize(
    const void* source, void* dest,
    const int32_t* shape_host,
    const int64_t* source_strides_host,
    int64_t source_offset,
    int32_t rank,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (source == nullptr || dest == nullptr) return 2;

    // Compute numel and canonical (row-major contiguous) strides for
    // fast-path #1 detection.
    int64_t numel = 1;
    for (int d = 0; d < rank; ++d) {
        if (shape_host[d] < 0) return 2;
        numel *= (int64_t)shape_host[d];
    }
    if (numel == 0) return 0;

    // Fast path #1: source already contiguous + zero offset → one D2D copy.
    bool already_contig = (source_offset == 0);
    if (already_contig) {
        int64_t expected = 1;
        for (int d = rank - 1; d >= 0; --d) {
            if (source_strides_host[d] != expected) {
                already_contig = false;
                break;
            }
            expected *= (int64_t)shape_host[d];
        }
    }
    if (already_contig) {
        size_t total_bytes = (size_t)numel * (size_t)ElemBytes;
        cudaError_t err = cudaMemcpyAsync(
            dest, source, total_bytes,
            cudaMemcpyDeviceToDevice, stream);
        return (err == cudaSuccess) ? 0 : 5;
    }

    // Pack shape / strides into DimsI32 / DimsI64 (pass-by-value).
    DimsI32 shape{};
    DimsI64 sx{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d]  = shape_host[d];
        sx.v[d]     = source_strides_host[d];
    }

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;

    // Fast path #2: innermost stride == 1 → per-outer-coord runs.
    // Skip rank-0 (no inner axis) and rank-1 broadcast (stride 0 still
    // contiguous? — only stride 1 qualifies).
    if (rank >= 1 && source_strides_host[rank - 1] == 1) {
        int32_t inner_extent = shape_host[rank - 1];
        // For rank-1 contiguous case with non-zero offset, falling
        // through to this path with outer_numel == 1 works correctly.
        int64_t outer_numel = numel / (int64_t)(inner_extent ? inner_extent : 1);
        DimsI32 outer_shape{};
        DimsI64 outer_sx{};
        for (int d = 0; d < rank - 1; ++d) {
            outer_shape.v[d] = shape_host[d];
            outer_sx.v[d]    = source_strides_host[d];
        }
        int64_t blocks_i64 = (outer_numel + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        contiguize_inner_stride1_kernel<ElemBytes><<<blocks, kBlock, 0, stream>>>(
            static_cast<const unsigned char*>(source),
            static_cast<unsigned char*>(dest),
            outer_numel, rank, inner_extent,
            outer_shape, outer_sx, source_offset);
        cudaError_t err = cudaGetLastError();
        return (err == cudaSuccess) ? 0 : 5;
    }

    // Generic path: one thread per output element.
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    contiguize_generic_kernel<ElemBytes><<<blocks, kBlock, 0, stream>>>(
        static_cast<const unsigned char*>(source),
        static_cast<unsigned char*>(dest),
        numel, rank, shape, sx, source_offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Nibble (S4 / U4) contiguize kernel.
// =============================================================================
//
// S4 / U4 are stored as packed pairs of nibbles (two elements per byte,
// LSB = even index, MSB = odd index). Contiguize on packed nibbles is
// awkward in general because the source-side stride is expressed in
// ELEMENTS but the byte layout interleaves two elements per byte.
//
// Pragmatic constraint (documented on the Rust side too): the source's
// innermost stride MUST be one of {1, -1, 2}:
//   *  1 : source's innermost axis is contiguous (matching the dest
//          layout). The most common case — produced by Permute /
//          BroadcastTo where the innermost axis is preserved.
//   * -1 : source's innermost axis is reverse-contiguous (produced by
//          Flip on the innermost axis). Same nibble-alignment as +1,
//          read in reverse.
//   *  2 : skip every other nibble — touches the same parity of every
//          source byte, so still nibble-aligned. (Less common but
//          arises from slice-by-2 patterns.)
// Any other inner stride breaks nibble alignment (the source's pair-
// granularity doesn't divide the access pattern) and returns
// `Unsupported` (status 3) from the launcher.
//
// The kernel: one thread per pair of OUTPUT nibbles (i.e. one thread
// per OUTPUT BYTE). Each thread computes both source nibble offsets,
// reads each nibble, and writes the packed output byte.
//
// `total_nibbles` is the total number of OUTPUT nibbles == numel.
// `inner_extent` is the OUTPUT innermost extent in nibbles (== shape[rank-1]).

__global__ inline void contiguize_nibble_kernel(
    const unsigned char* __restrict__ source,
    unsigned char* __restrict__ dest,
    int64_t total_nibbles,
    int32_t rank,
    DimsI32 shape,
    DimsI64 source_strides,
    int64_t source_offset)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    // Each thread handles ONE output byte == two output nibbles
    // (`out_idx` and `out_idx + 1`). Total output bytes is
    // ceil(total_nibbles / 2).
    int64_t out_bytes = (total_nibbles + 1) / 2;
    for (int64_t byte_i = tid; byte_i < out_bytes; byte_i += step) {
        int64_t lo_nibble = byte_i * 2;
        int64_t hi_nibble = lo_nibble + 1;

        // Compute source element offset for the low nibble.
        auto src_off_for = [&](int64_t out_lin) -> int64_t {
            int64_t linear = out_lin;
            int64_t off = source_offset;
            for (int d = rank - 1; d >= 0; --d) {
                int32_t s = shape.v[d];
                int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
                if (s != 0) linear /= (int64_t)s;
                off += c * source_strides.v[d];
            }
            return off;
        };

        auto read_nibble = [&](int64_t src_elem_idx) -> unsigned char {
            int64_t byte_idx = src_elem_idx >> 1;
            int parity = (int)(src_elem_idx & 1);
            unsigned char b = source[byte_idx];
            return parity ? ((b >> 4) & 0x0F) : (b & 0x0F);
        };

        unsigned char lo = read_nibble(src_off_for(lo_nibble));
        unsigned char hi = 0;
        if (hi_nibble < total_nibbles) {
            hi = read_nibble(src_off_for(hi_nibble));
        }
        dest[byte_i] = (unsigned char)((hi << 4) | (lo & 0x0F));
    }
}

__host__ inline int32_t launch_contiguize_nibble(
    const void* source, void* dest,
    const int32_t* shape_host,
    const int64_t* source_strides_host,
    int64_t source_offset,
    int32_t rank,
    cudaStream_t stream)
{
    if (rank < 1 || rank > MAX_RANK) return 2;
    if (source == nullptr || dest == nullptr) return 2;

    // Validate innermost-stride nibble-alignment constraint.
    int64_t inner_s = source_strides_host[rank - 1];
    if (!(inner_s == 1 || inner_s == -1 || inner_s == 2)) {
        return 3;  // Unsupported — source breaks nibble alignment.
    }

    int64_t numel = 1;
    for (int d = 0; d < rank; ++d) {
        if (shape_host[d] < 0) return 2;
        numel *= (int64_t)shape_host[d];
    }
    if (numel == 0) return 0;

    DimsI32 shape{};
    DimsI64 sx{};
    for (int d = 0; d < rank; ++d) {
        shape.v[d]  = shape_host[d];
        sx.v[d]     = source_strides_host[d];
    }

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t out_bytes = (numel + 1) / 2;
    int64_t blocks_i64 = (out_bytes + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    contiguize_nibble_kernel<<<blocks, kBlock, 0, stream>>>(
        static_cast<const unsigned char*>(source),
        static_cast<unsigned char*>(dest),
        numel, rank, shape, sx, source_offset);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::contiguize

// =============================================================================
// extern "C" launcher INSTANTIATE — one per byte-width.
// =============================================================================
//
// NAME   : symbol suffix — `b1`, `b2`, `b4`, `b8`, `b16`.
// BYTES  : element byte-width — must match the template parameter.

#define BARACUDA_KERNELS_CONTIGUIZE_INSTANTIATE(NAME, BYTES)                                       \
    extern "C" int32_t baracuda_kernels_contiguize_##NAME##_run(                                   \
        void* dest, const void* source,                                                            \
        const int32_t* shape, const int64_t* source_strides, int64_t source_offset,                \
        int32_t rank,                                                                              \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (shape == nullptr || source_strides == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::contiguize::launch_contiguize<BYTES>(                                     \
            source, dest, shape, source_strides, source_offset, rank, stream);                     \
    }

#define BARACUDA_KERNELS_CONTIGUIZE_NIBBLE_INSTANTIATE(NAME)                                       \
    extern "C" int32_t baracuda_kernels_contiguize_##NAME##_run(                                   \
        void* dest, const void* source,                                                            \
        const int32_t* shape, const int64_t* source_strides, int64_t source_offset,                \
        int32_t rank,                                                                              \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (shape == nullptr || source_strides == nullptr) return 2;                               \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::contiguize::launch_contiguize_nibble(                                     \
            source, dest, shape, source_strides, source_offset, rank, stream);                     \
    }

#endif  // BARACUDA_CONTIGUIZE_CUH
