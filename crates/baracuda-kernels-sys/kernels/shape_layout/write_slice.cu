// baracuda-kernels Phase 13.1 — WriteSlice trailblazer.
//
// `dest[start_0..end_0, ..., start_{N-1}..end_{N-1}] = source` (assign,
// not accumulate). Drives Fuel team's persistent KV-cache append during
// autoregressive decoding.
//
// Byte-width dispatch: one extern symbol per `sizeof(T)` ∈ {1, 2, 4, 8,
// 16} covers all byte-aligned baracuda elements via a Blob-typed
// memcpy kernel. The nibble symbol (S4/U4) lives next to it, sharing
// the header but with its own kernel that operates on byte-coalesced
// element-pairs.

#include "../include/baracuda_write_slice.cuh"

BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b1,  baracuda::write_slice::Blob1)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b2,  baracuda::write_slice::Blob2)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b4,  baracuda::write_slice::Blob4)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b8,  baracuda::write_slice::Blob8)
BARACUDA_KERNELS_WRITE_SLICE_INSTANTIATE(b16, baracuda::write_slice::Blob16)

// Nibble-packed (S4 / U4) — bypasses the macro because its shape arrays
// are byte-counted on the innermost axis (Rust side halves them) and
// the launch helper is a separate symbol. Constraint: range_start and
// range_end on the innermost axis must both be even.
extern "C" int32_t baracuda_kernels_write_slice_nibble_run(
    void* dest, const void* source,
    int64_t source_byte_numel,
    int32_t rank,
    const int32_t* dest_byte_shape,
    const int32_t* source_byte_shape,
    const int32_t* range_start_bytes,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    void* stream_ptr)
{
    if (rank < 1 || rank > baracuda::write_slice::MAX_RANK) return 2;
    if (source_byte_numel < 0) return 2;
    if (source_byte_numel == 0) return 0;
    if (dest == nullptr || source == nullptr) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return baracuda::write_slice::launch_write_slice_nibble(
        dest, source, source_byte_numel, rank,
        dest_byte_shape, source_byte_shape, range_start_bytes, stream);
}

extern "C" int32_t baracuda_kernels_write_slice_nibble_can_implement(
    const void* /*dest*/, const void* /*source*/,
    int64_t source_byte_numel,
    int32_t rank,
    const int32_t* dest_byte_shape,
    const int32_t* source_byte_shape,
    const int32_t* range_start_bytes)
{
    if (source_byte_numel < 0) return 2;
    if (rank < 1 || rank > baracuda::write_slice::MAX_RANK) return 2;
    if (source_byte_numel > 0 && (dest_byte_shape == nullptr || source_byte_shape == nullptr ||
                                   range_start_bytes == nullptr)) return 2;
    return 0;
}
