// baracuda_coord_unravel.cuh — linear-index → multi-coordinate → byte-offset
// helpers for the "one thread per element, strided layout" kernel pattern.
// Phase 67b — consolidates the coordinate-unravel loop duplicated across
// flip, roll, permute, affine_strided, where_strided, ternary_clamp_strided,
// the rms_norm helpers, the indexing kernels, and ~20 other strided kernels.
//
// Every strided baracuda kernel decomposes a linear thread index `i` into a
// per-axis multi-coordinate over the tensor's logical shape, then dot-products
// that coordinate against one-or-more per-axis stride arrays to derive the
// element offset(s). The canonical loop (row-major / C-contiguous unravel —
// last axis is fastest-varying) looks like:
//
//     int64_t linear = i;
//     int64_t off_x = 0, off_y = 0;
//     for (int d = rank - 1; d >= 0; --d) {
//         int32_t s = shape.v[d];
//         int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
//         if (s != 0) linear /= (int64_t)s;
//         off_x += c * stride_x.v[d];
//         off_y += c * stride_y.v[d];
//     }
//
// These helpers capture exactly that loop — bit-for-bit identical arithmetic,
// including the `s == 0` empty-axis guard (coord pinned to 0, no divide) — so
// migrating an existing kernel is a behavior-preserving substitution.
//
// Broadcast is honored transparently: the stride-0-for-broadcast-axis
// convention already baked into baracuda's strided kernels needs nothing
// special here, since `c * 0 == 0` contributes nothing to the offset. Pass the
// broadcast operand's stride array as-is.
//
// Usage:
//
//     int64_t off = baracuda::coord::unravel_offset_1(i, rank, shape, stride);
//     y[off] = f(x[off]);                         // one stride array
//
//     int64_t ox, oy;
//     baracuda::coord::unravel_offsets_2(i, rank, shape, sx, sy, ox, oy);
//     y[oy] = f(x[ox]);                           // input + output strides
//
//     int64_t oc, oa, ob, oy;                     // where / ternary
//     baracuda::coord::unravel_offsets_4(i, rank, shape, sc, sa, sb, sy,
//                                        oc, oa, ob, oy);
//
// ---------------------------------------------------------------------------
// DimsI32 / DimsI64 and the ODR caveat
// ---------------------------------------------------------------------------
// `DimsI32` / `DimsI64` are currently defined (structurally identically) in
// 8+ kernel headers under their own per-subsystem namespaces — see the
// Phase 62 audit. To avoid a one-definition-rule violation in any `.cu` that
// includes BOTH this header AND one of those, this header does NOT redefine
// them at `baracuda::` scope. It declares its own copies under the strictly
// scoped `baracuda::coord::` sub-namespace (handy for standalone use), and the
// unravel functions are *templates* over the `Shape` / `Stride` types — the
// compiler deduces the concrete type at the call site, so a kernel can pass
// its own in-scope `baracuda::elementwise::DimsI32` / `baracuda::affine::DimsI64`
// / etc. directly. All those structs are layout-compatible PODs
// (`int32_t v[MAX_RANK]` / `int64_t v[MAX_RANK]`, MAX_RANK == 8), so deduction
// "just works" without any cross-header coupling. Migrating the 8 definitions
// to a single shared one is deliberately left as future work — one kernel at a
// time — and is NOT required to use these helpers.

#ifndef BARACUDA_COORD_UNRAVEL_CUH
#define BARACUDA_COORD_UNRAVEL_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda {
namespace coord {

// Maximum rank the strided kernels support, matching the value used by every
// `DimsI32` / `DimsI64`-bearing kernel header. The unravel functions are
// bounded by the runtime `rank` argument rather than this constant, so they
// stay correct for any `Shape` / `Stride` whose array length is >= `rank`.
inline constexpr int MAX_RANK = 8;

// Self-contained POD shape / stride descriptors. The unravel templates accept
// these OR any layout-compatible struct that exposes `.v[d]` (see the ODR note
// above). Provided so callers without one of the existing per-subsystem copies
// in scope can still use the helpers standalone.
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

// Unravel `linear` into a multi-coordinate over `shape` and return the dotted
// byte/element offset into a SINGLE stride array.
template <typename Shape, typename Stride>
__device__ __forceinline__ int64_t unravel_offset_1(
    int64_t linear,
    int32_t rank,
    const Shape& shape,
    const Stride& stride)
{
    int64_t off = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int32_t s = shape.v[d];
        int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
        if (s != 0) linear /= (int64_t)s;
        off += c * stride.v[d];
    }
    return off;
}

// Same coordinate, dotted into TWO independent stride arrays (e.g. input +
// output). Offsets returned via out-params. A single unravel pass feeds both,
// so this is strictly cheaper than two `unravel_offset_1` calls.
template <typename Shape, typename Stride>
__device__ __forceinline__ void unravel_offsets_2(
    int64_t linear,
    int32_t rank,
    const Shape& shape,
    const Stride& stride_a,
    const Stride& stride_b,
    int64_t& off_a,
    int64_t& off_b)
{
    off_a = 0;
    off_b = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int32_t s = shape.v[d];
        int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
        if (s != 0) linear /= (int64_t)s;
        off_a += c * stride_a.v[d];
        off_b += c * stride_b.v[d];
    }
}

// THREE stride arrays — binary-op-with-output (a, b, y) or where-without-output.
template <typename Shape, typename Stride>
__device__ __forceinline__ void unravel_offsets_3(
    int64_t linear,
    int32_t rank,
    const Shape& shape,
    const Stride& stride_a,
    const Stride& stride_b,
    const Stride& stride_c,
    int64_t& off_a,
    int64_t& off_b,
    int64_t& off_c)
{
    off_a = 0;
    off_b = 0;
    off_c = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int32_t s = shape.v[d];
        int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
        if (s != 0) linear /= (int64_t)s;
        off_a += c * stride_a.v[d];
        off_b += c * stride_b.v[d];
        off_c += c * stride_c.v[d];
    }
}

// FOUR stride arrays — ternary-op-with-output (a, b, c, y) and where-with-output
// (cond, a, b, y). Three current call sites justify this variant: the strided
// where kernel, the ternary pointwise kernel, and the ternary-clamp kernel.
template <typename Shape, typename Stride>
__device__ __forceinline__ void unravel_offsets_4(
    int64_t linear,
    int32_t rank,
    const Shape& shape,
    const Stride& stride_a,
    const Stride& stride_b,
    const Stride& stride_c,
    const Stride& stride_d,
    int64_t& off_a,
    int64_t& off_b,
    int64_t& off_c,
    int64_t& off_d)
{
    off_a = 0;
    off_b = 0;
    off_c = 0;
    off_d = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int32_t s = shape.v[d];
        int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
        if (s != 0) linear /= (int64_t)s;
        off_a += c * stride_a.v[d];
        off_b += c * stride_b.v[d];
        off_c += c * stride_c.v[d];
        off_d += c * stride_d.v[d];
    }
}

}  // namespace coord
}  // namespace baracuda

#endif  // BARACUDA_COORD_UNRAVEL_CUH
