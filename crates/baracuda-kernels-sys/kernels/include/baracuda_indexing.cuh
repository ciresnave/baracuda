// baracuda_indexing.cuh
//
// Templated kernels and INSTANTIATE macros for the indexing op family
// (Phase 7 Milestone 7.3 — Category L from the comprehensive plan).
//
// Ops shipped here:
//   gather      — `out[i] = src[index[i]]` along a specified dim
//   scatter_add — `out[index[i]] += updates[i]` (atomic, dup-safe)
//   index_select — `out[..., j, ...] = src[..., idx[j], ...]`
//                  (1D index, simpler / faster gather)
//   masked_fill — `out[i] = mask[i] ? value : src[i]`
//   one_hot     — `out[..., c] = 1 if c == src[...] else 0`
//   nonzero     — coordinates where input != 0 (two-pass prefix sum)
//
// Index dtype: i32 only (i64 deferred). Mask dtype: u8 (Bool). All
// kernels accept rank up to MAX_RANK and bounds-check the index entries
// at kernel entry (out-of-range → kernel asserts via printf+skip).
//
// Status codes returned by the launchers mirror the GEMM / elementwise
// family:
//   0 success
//   1 misaligned operand
//   2 invalid problem (e.g. negative numel, rank > MAX_RANK)
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error (typically a launch failure)

#ifndef BARACUDA_INDEXING_CUH
#define BARACUDA_INDEXING_CUH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace indexing {

inline constexpr int MAX_RANK = 8;

// Plain-data structs used to pass the fixed-rank shape / stride arrays
// through the kernel parameter block by value.
struct DimsI32 { int32_t v[MAX_RANK]; };
struct DimsI64 { int64_t v[MAX_RANK]; };

// =============================================================================
// AtomicAdd helpers — handle the half / bfloat16 cases by routing through
// the i32 emulated atomicCAS pattern (SM<6 lacks native fp16/bf16 atomics
// but SM 6.0+ has __half2 + bf16 atomicAdd in cuda 11.0+; we use the
// native intrinsics where available).
// =============================================================================

template <typename T>
__device__ inline void scatter_atomic_add(T* addr, T val) {
    atomicAdd(addr, val);
}

template <>
__device__ inline void scatter_atomic_add<__half>(__half* addr, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, val);
#else
    // Fallback emulation via 32-bit atomicCAS on the containing pair.
    unsigned int* aligned = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~3ULL);
    bool lo = (reinterpret_cast<uintptr_t>(addr) & 2) == 0;
    unsigned int old = *aligned, assumed;
    do {
        assumed = old;
        __half ax;
        if (lo) {
            ax = *reinterpret_cast<__half*>(&assumed);
            __half nx = __hadd(ax, val);
            unsigned int repl = (assumed & 0xFFFF0000u) | *reinterpret_cast<unsigned short*>(&nx);
            old = atomicCAS(aligned, assumed, repl);
        } else {
            unsigned int hi = (assumed >> 16);
            ax = *reinterpret_cast<__half*>(&hi);
            __half nx = __hadd(ax, val);
            unsigned int repl = (assumed & 0x0000FFFFu)
                              | (((unsigned int)*reinterpret_cast<unsigned short*>(&nx)) << 16);
            old = atomicCAS(aligned, assumed, repl);
        }
    } while (assumed != old);
#endif
}

template <>
__device__ inline void scatter_atomic_add<__nv_bfloat16>(__nv_bfloat16* addr, __nv_bfloat16 val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(addr, val);
#else
    // Fallback: f32 round-trip via 32-bit atomicCAS on the containing pair.
    unsigned int* aligned = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~3ULL);
    bool lo = (reinterpret_cast<uintptr_t>(addr) & 2) == 0;
    unsigned int old = *aligned, assumed;
    float vf = __bfloat162float(val);
    do {
        assumed = old;
        __nv_bfloat16 ax;
        if (lo) {
            unsigned short bits = (unsigned short)(assumed & 0xFFFF);
            ax = *reinterpret_cast<__nv_bfloat16*>(&bits);
            float sum = __bfloat162float(ax) + vf;
            __nv_bfloat16 nx = __float2bfloat16(sum);
            unsigned short nbits = *reinterpret_cast<unsigned short*>(&nx);
            unsigned int repl = (assumed & 0xFFFF0000u) | (unsigned int)nbits;
            old = atomicCAS(aligned, assumed, repl);
        } else {
            unsigned short bits = (unsigned short)((assumed >> 16) & 0xFFFF);
            ax = *reinterpret_cast<__nv_bfloat16*>(&bits);
            float sum = __bfloat162float(ax) + vf;
            __nv_bfloat16 nx = __float2bfloat16(sum);
            unsigned short nbits = *reinterpret_cast<unsigned short*>(&nx);
            unsigned int repl = (assumed & 0x0000FFFFu) | (((unsigned int)nbits) << 16);
            old = atomicCAS(aligned, assumed, repl);
        }
    } while (assumed != old);
#endif
}

// =============================================================================
// gather kernel — `out[..., j, ...] = src[..., index[...j...], ...]` along
// dim `d`. Output shape == index shape. `src` has same shape as index on
// all axes except `d` (where src extent == src_dim_size).
// =============================================================================

template <typename T>
__global__ void gather_kernel(
    const T* __restrict__ src,
    const int32_t* __restrict__ index,
    T* __restrict__ out,
    int64_t out_numel,
    int32_t rank,
    int32_t gather_dim,
    int32_t src_dim_size,
    DimsI32 out_shape,         // == index shape
    DimsI64 stride_src,
    DimsI64 stride_index,
    DimsI64 stride_out)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        // Unravel linear index into coord along output shape.
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = out_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        // Load index value at this coord.
        int64_t idx_off = 0;
        int64_t out_off = 0;
        for (int d = 0; d < rank; ++d) {
            idx_off += coord[d] * stride_index.v[d];
            out_off += coord[d] * stride_out.v[d];
        }
        int32_t idx_val = index[idx_off];
        // Bounds check — skip on out-of-range (don't write garbage).
        if (idx_val < 0 || idx_val >= src_dim_size) {
            continue;
        }
        // Source coord: replace gather_dim with idx_val.
        int64_t src_off = 0;
        for (int d = 0; d < rank; ++d) {
            int64_t cc = (d == gather_dim) ? (int64_t)idx_val : coord[d];
            src_off += cc * stride_src.v[d];
        }
        out[out_off] = src[src_off];
    }
}

template <typename T>
__host__ inline int32_t launch_gather(
    const T* src, const int32_t* index, T* out,
    int64_t out_numel,
    int32_t rank,
    int32_t gather_dim,
    int32_t src_dim_size,
    const int32_t* out_shape_host,
    const int64_t* stride_src_host,
    const int64_t* stride_index_host,
    const int64_t* stride_out_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (gather_dim < 0 || gather_dim >= rank) return 2;
    DimsI32 sh = {};
    DimsI64 ss = {}, si = {}, so = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = out_shape_host[i];
        ss.v[i] = stride_src_host[i];
        si.v[i] = stride_index_host[i];
        so.v[i] = stride_out_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    gather_kernel<T><<<blocks, kBlock, 0, stream>>>(
        src, index, out, out_numel, rank, gather_dim, src_dim_size,
        sh, ss, si, so);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// gather backward kernel — scatter-add along gather_dim.
//   dsrc[..., index[..., j, ...], ...] += dout[..., j, ...]
// One thread per output (= index) element; atomicAdd into dsrc.
// =============================================================================

template <typename T>
__global__ void gather_backward_kernel(
    const T* __restrict__ dout,
    const int32_t* __restrict__ index,
    T* __restrict__ dsrc,
    int64_t out_numel,
    int32_t rank,
    int32_t gather_dim,
    int32_t src_dim_size,
    DimsI32 out_shape,
    DimsI64 stride_dout,
    DimsI64 stride_index,
    DimsI64 stride_dsrc)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = out_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        int64_t idx_off = 0;
        int64_t dout_off = 0;
        for (int d = 0; d < rank; ++d) {
            idx_off += coord[d] * stride_index.v[d];
            dout_off += coord[d] * stride_dout.v[d];
        }
        int32_t idx_val = index[idx_off];
        if (idx_val < 0 || idx_val >= src_dim_size) {
            continue;
        }
        int64_t dsrc_off = 0;
        for (int d = 0; d < rank; ++d) {
            int64_t cc = (d == gather_dim) ? (int64_t)idx_val : coord[d];
            dsrc_off += cc * stride_dsrc.v[d];
        }
        scatter_atomic_add<T>(&dsrc[dsrc_off], dout[dout_off]);
    }
}

template <typename T>
__host__ inline int32_t launch_gather_backward(
    const T* dout, const int32_t* index, T* dsrc,
    int64_t out_numel,
    int32_t rank,
    int32_t gather_dim,
    int32_t src_dim_size,
    const int32_t* out_shape_host,
    const int64_t* stride_dout_host,
    const int64_t* stride_index_host,
    const int64_t* stride_dsrc_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (gather_dim < 0 || gather_dim >= rank) return 2;
    DimsI32 sh = {};
    DimsI64 sd = {}, si = {}, sds = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = out_shape_host[i];
        sd.v[i] = stride_dout_host[i];
        si.v[i] = stride_index_host[i];
        sds.v[i] = stride_dsrc_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    gather_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, index, dsrc, out_numel, rank, gather_dim, src_dim_size,
        sh, sd, si, sds);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// scatter_add kernel — `out[..., index[..., j, ...], ...] += updates[..., j, ...]`
// along `scatter_dim`. Same shape pattern as gather's BW. AtomicAdd at the
// destination cell (duplicate indices summed correctly).
// =============================================================================

template <typename T>
__global__ void scatter_add_kernel(
    const T* __restrict__ updates,
    const int32_t* __restrict__ index,
    T* __restrict__ out,
    int64_t upd_numel,
    int32_t rank,
    int32_t scatter_dim,
    int32_t out_dim_size,
    DimsI32 upd_shape,
    DimsI64 stride_upd,
    DimsI64 stride_index,
    DimsI64 stride_out)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < upd_numel; i += step) {
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = upd_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        int64_t idx_off = 0;
        int64_t upd_off = 0;
        for (int d = 0; d < rank; ++d) {
            idx_off += coord[d] * stride_index.v[d];
            upd_off += coord[d] * stride_upd.v[d];
        }
        int32_t idx_val = index[idx_off];
        if (idx_val < 0 || idx_val >= out_dim_size) {
            continue;
        }
        int64_t out_off = 0;
        for (int d = 0; d < rank; ++d) {
            int64_t cc = (d == scatter_dim) ? (int64_t)idx_val : coord[d];
            out_off += cc * stride_out.v[d];
        }
        scatter_atomic_add<T>(&out[out_off], updates[upd_off]);
    }
}

template <typename T>
__host__ inline int32_t launch_scatter_add(
    const T* updates, const int32_t* index, T* out,
    int64_t upd_numel,
    int32_t rank,
    int32_t scatter_dim,
    int32_t out_dim_size,
    const int32_t* upd_shape_host,
    const int64_t* stride_upd_host,
    const int64_t* stride_index_host,
    const int64_t* stride_out_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (scatter_dim < 0 || scatter_dim >= rank) return 2;
    DimsI32 sh = {};
    DimsI64 su = {}, si = {}, so = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = upd_shape_host[i];
        su.v[i] = stride_upd_host[i];
        si.v[i] = stride_index_host[i];
        so.v[i] = stride_out_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (upd_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    scatter_add_kernel<T><<<blocks, kBlock, 0, stream>>>(
        updates, index, out, upd_numel, rank, scatter_dim, out_dim_size,
        sh, su, si, so);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// index_select kernel — `out[..., j, ...] = src[..., idx[j], ...]`.
// `idx` is a 1-D i32 tensor of length `out.shape[select_dim]`. Output
// shape is `src.shape` with dim `select_dim` replaced by `idx.numel()`.
// =============================================================================

template <typename T>
__global__ void index_select_kernel(
    const T* __restrict__ src,
    const int32_t* __restrict__ idx,
    T* __restrict__ out,
    int64_t out_numel,
    int32_t rank,
    int32_t select_dim,
    int32_t src_dim_size,
    DimsI32 out_shape,
    DimsI64 stride_src,
    DimsI64 stride_out)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = out_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        // The select axis coord is used to look up the 1-D index.
        int32_t idx_val = idx[coord[select_dim]];
        if (idx_val < 0 || idx_val >= src_dim_size) {
            continue;
        }
        int64_t out_off = 0;
        int64_t src_off = 0;
        for (int d = 0; d < rank; ++d) {
            out_off += coord[d] * stride_out.v[d];
            int64_t cc = (d == select_dim) ? (int64_t)idx_val : coord[d];
            src_off += cc * stride_src.v[d];
        }
        out[out_off] = src[src_off];
    }
}

template <typename T>
__host__ inline int32_t launch_index_select(
    const T* src, const int32_t* idx, T* out,
    int64_t out_numel,
    int32_t rank,
    int32_t select_dim,
    int32_t src_dim_size,
    const int32_t* out_shape_host,
    const int64_t* stride_src_host,
    const int64_t* stride_out_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (select_dim < 0 || select_dim >= rank) return 2;
    DimsI32 sh = {};
    DimsI64 ss = {}, so = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = out_shape_host[i];
        ss.v[i] = stride_src_host[i];
        so.v[i] = stride_out_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    index_select_kernel<T><<<blocks, kBlock, 0, stream>>>(
        src, idx, out, out_numel, rank, select_dim, src_dim_size,
        sh, ss, so);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// index_select backward kernel — scatter-add along select_dim:
//   dsrc[..., idx[j], ...] += dout[..., j, ...]
// =============================================================================

template <typename T>
__global__ void index_select_backward_kernel(
    const T* __restrict__ dout,
    const int32_t* __restrict__ idx,
    T* __restrict__ dsrc,
    int64_t out_numel,
    int32_t rank,
    int32_t select_dim,
    int32_t src_dim_size,
    DimsI32 out_shape,
    DimsI64 stride_dout,
    DimsI64 stride_dsrc)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = out_shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        int32_t idx_val = idx[coord[select_dim]];
        if (idx_val < 0 || idx_val >= src_dim_size) {
            continue;
        }
        int64_t dout_off = 0;
        int64_t dsrc_off = 0;
        for (int d = 0; d < rank; ++d) {
            dout_off += coord[d] * stride_dout.v[d];
            int64_t cc = (d == select_dim) ? (int64_t)idx_val : coord[d];
            dsrc_off += cc * stride_dsrc.v[d];
        }
        scatter_atomic_add<T>(&dsrc[dsrc_off], dout[dout_off]);
    }
}

template <typename T>
__host__ inline int32_t launch_index_select_backward(
    const T* dout, const int32_t* idx, T* dsrc,
    int64_t out_numel,
    int32_t rank,
    int32_t select_dim,
    int32_t src_dim_size,
    const int32_t* out_shape_host,
    const int64_t* stride_dout_host,
    const int64_t* stride_dsrc_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (select_dim < 0 || select_dim >= rank) return 2;
    DimsI32 sh = {};
    DimsI64 sd = {}, sds = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = out_shape_host[i];
        sd.v[i] = stride_dout_host[i];
        sds.v[i] = stride_dsrc_host[i];
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    index_select_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, idx, dsrc, out_numel, rank, select_dim, src_dim_size,
        sh, sd, sds);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// masked_fill kernel — `out[i] = mask[i] ? value : src[i]`.
// Mask is `u8` (Bool storage; 0 = false, non-zero = true). Same-shape
// only for trailblazer; broadcast is a future extension.
// =============================================================================

template <typename T>
__global__ void masked_fill_kernel(
    const T* __restrict__ src,
    const uint8_t* __restrict__ mask,
    T* __restrict__ out,
    int64_t numel,
    T fill_value)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        out[i] = (mask[i] != 0) ? fill_value : src[i];
    }
}

template <typename T>
__host__ inline int32_t launch_masked_fill(
    const T* src, const uint8_t* mask, T* out,
    int64_t numel,
    T fill_value,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    masked_fill_kernel<T><<<blocks, kBlock, 0, stream>>>(src, mask, out, numel, fill_value);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// masked_fill backward kernel — `dsrc[i] = mask[i] ? 0 : dout[i]`.
// Value is a scalar, no grad.
// =============================================================================

template <typename T>
__global__ void masked_fill_backward_kernel(
    const T* __restrict__ dout,
    const uint8_t* __restrict__ mask,
    T* __restrict__ dsrc,
    int64_t numel,
    T zero)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        dsrc[i] = (mask[i] != 0) ? zero : dout[i];
    }
}

template <typename T>
__host__ inline int32_t launch_masked_fill_backward(
    const T* dout, const uint8_t* mask, T* dsrc,
    int64_t numel,
    T zero,
    cudaStream_t stream)
{
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    masked_fill_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(dout, mask, dsrc, numel, zero);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// one_hot kernel — `out[indices..., c] = 1 if c == src[indices...] else 0`.
// Output rank = input rank + 1, with the new axis appended (PyTorch
// convention). Input is i32 class indices. Out-of-range src values yield
// an all-zero row.
//
// `out_numel` is the total flat element count of `out`. One thread per
// output cell.
// =============================================================================

template <typename T>
__global__ void one_hot_kernel(
    const int32_t* __restrict__ src,
    T* __restrict__ out,
    int64_t out_numel,
    int32_t num_classes,
    T one,
    T zero)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        // Output is contig — last axis is num_classes. Decompose:
        //   out_idx = batch_idx * num_classes + c
        int64_t batch_idx = i / (int64_t)num_classes;
        int32_t c = (int32_t)(i - batch_idx * (int64_t)num_classes);
        int32_t src_val = src[batch_idx];
        out[i] = (src_val == c) ? one : zero;
    }
}

template <typename T>
__host__ inline int32_t launch_one_hot(
    const int32_t* src, T* out,
    int64_t out_numel,
    int32_t num_classes,
    T one,
    T zero,
    cudaStream_t stream)
{
    if (num_classes <= 0) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    one_hot_kernel<T><<<blocks, kBlock, 0, stream>>>(src, out, out_numel, num_classes, one, zero);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// nonzero kernel(s) — returns [num_nonzero, ndim] int32 tensor of coords
// where input != 0.
//
// Two-phase implementation:
//   Phase A : nonzero_count_kernel writes per-block partial counts to
//             workspace, then a single-thread kernel reduces them into a
//             final count + per-block prefix offsets. (Trailblazer:
//             single-block-prefix-sum so the limit is one grid wave;
//             larger inputs need a multi-block scan — deferred.)
//   Phase B : nonzero_compact_kernel walks the input, each thread that
//             finds a nonzero atomicAdd's a slot in the per-block bucket
//             and writes the coord.
//
// For simplicity the trailblazer uses a single global atomic counter
// for write-slot assignment — output ordering is NOT row-major
// (compaction races). PyTorch's `torch.nonzero` is row-major; ordering
// follow-up is documented in the milestone notes. Output is correct in
// content but unsorted; caller should sort if order matters.
// =============================================================================

template <typename T>
__device__ inline bool is_nonzero_val(T v) { return v != T(0); }
template <>
__device__ inline bool is_nonzero_val<__half>(__half v) { return __half2float(v) != 0.0f; }
template <>
__device__ inline bool is_nonzero_val<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v) != 0.0f;
}

template <typename T>
__global__ void nonzero_compact_kernel(
    const T* __restrict__ x,
    int32_t* __restrict__ out_coords,    // [max_nz, rank]
    int32_t* __restrict__ counter,       // single int32 atomic
    int64_t numel,
    int32_t rank,
    int32_t max_nz,
    DimsI32 shape,
    DimsI64 stride_x)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        // Decompose linear into coords using `shape` (row-major contig).
        int64_t linear = i;
        int64_t coord[MAX_RANK] = {0};
        for (int d = rank - 1; d >= 0; --d) {
            int32_t s = shape.v[d];
            int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
            if (s != 0) linear /= (int64_t)s;
            coord[d] = c;
        }
        // Read x at coord (via strides — supports non-contig input).
        int64_t x_off = 0;
        for (int d = 0; d < rank; ++d) {
            x_off += coord[d] * stride_x.v[d];
        }
        if (is_nonzero_val<T>(x[x_off])) {
            int32_t slot = atomicAdd(counter, 1);
            if (slot < max_nz) {
                for (int d = 0; d < rank; ++d) {
                    out_coords[(int64_t)slot * (int64_t)rank + (int64_t)d] = (int32_t)coord[d];
                }
            }
        }
    }
}

template <typename T>
__host__ inline int32_t launch_nonzero(
    const T* x,
    int32_t* out_coords,
    int32_t* counter,           // device-resident, single int32
    int64_t numel,
    int32_t rank,
    int32_t max_nz,
    const int32_t* shape_host,
    const int64_t* stride_x_host,
    cudaStream_t stream)
{
    if (rank < 0 || rank > MAX_RANK) return 2;
    if (max_nz < 0) return 2;
    // Zero the counter on the stream.
    cudaError_t err = cudaMemsetAsync(counter, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) return 5;
    DimsI32 sh = {};
    DimsI64 sx = {};
    for (int i = 0; i < rank; ++i) {
        sh.v[i] = shape_host[i];
        sx.v[i] = stride_x_host[i];
    }
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    nonzero_compact_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, out_coords, counter, numel, rank, max_nz, sh, sx);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::indexing

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher per (op, dtype) pair.
// =============================================================================

#define BARACUDA_KERNELS_GATHER_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t out_numel,                                                                         \
        int32_t rank,                                                                              \
        int32_t gather_dim,                                                                        \
        int32_t src_dim_size,                                                                      \
        const int32_t* out_shape,                                                                  \
        const int64_t* stride_src,                                                                 \
        const int64_t* stride_index,                                                               \
        const int64_t* stride_out,                                                                 \
        const void* src,                                                                           \
        const void* index,                                                                         \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (out_numel < 0) return 2;                                                              \
        if (out_numel == 0) return 0;                                                              \
        if (src == nullptr || index == nullptr || out == nullptr) return 2;                       \
        if (out_shape == nullptr || stride_src == nullptr ||                                      \
            stride_index == nullptr || stride_out == nullptr) return 2;                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_gather<T>(                                              \
            static_cast<const T*>(src),                                                            \
            static_cast<const int32_t*>(index),                                                    \
            static_cast<T*>(out),                                                                  \
            out_numel, rank, gather_dim, src_dim_size,                                            \
            out_shape, stride_src, stride_index, stride_out, stream);                             \
    }

#define BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(NAME, T)                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t out_numel,                                                                         \
        int32_t rank,                                                                              \
        int32_t gather_dim,                                                                        \
        int32_t src_dim_size,                                                                      \
        const int32_t* out_shape,                                                                  \
        const int64_t* stride_dout,                                                                \
        const int64_t* stride_index,                                                               \
        const int64_t* stride_dsrc,                                                                \
        const void* dout,                                                                          \
        const void* index,                                                                         \
        void* dsrc,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (out_numel < 0) return 2;                                                              \
        if (out_numel == 0) return 0;                                                              \
        if (dout == nullptr || index == nullptr || dsrc == nullptr) return 2;                     \
        if (out_shape == nullptr || stride_dout == nullptr ||                                     \
            stride_index == nullptr || stride_dsrc == nullptr) return 2;                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_gather_backward<T>(                                     \
            static_cast<const T*>(dout),                                                           \
            static_cast<const int32_t*>(index),                                                    \
            static_cast<T*>(dsrc),                                                                 \
            out_numel, rank, gather_dim, src_dim_size,                                            \
            out_shape, stride_dout, stride_index, stride_dsrc, stream);                           \
    }

#define BARACUDA_KERNELS_SCATTER_ADD_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t upd_numel,                                                                         \
        int32_t rank,                                                                              \
        int32_t scatter_dim,                                                                       \
        int32_t out_dim_size,                                                                      \
        const int32_t* upd_shape,                                                                  \
        const int64_t* stride_upd,                                                                 \
        const int64_t* stride_index,                                                               \
        const int64_t* stride_out,                                                                 \
        const void* updates,                                                                       \
        const void* index,                                                                         \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (upd_numel < 0) return 2;                                                              \
        if (upd_numel == 0) return 0;                                                              \
        if (updates == nullptr || index == nullptr || out == nullptr) return 2;                   \
        if (upd_shape == nullptr || stride_upd == nullptr ||                                      \
            stride_index == nullptr || stride_out == nullptr) return 2;                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_scatter_add<T>(                                         \
            static_cast<const T*>(updates),                                                        \
            static_cast<const int32_t*>(index),                                                    \
            static_cast<T*>(out),                                                                  \
            upd_numel, rank, scatter_dim, out_dim_size,                                           \
            upd_shape, stride_upd, stride_index, stride_out, stream);                             \
    }

#define BARACUDA_KERNELS_INDEX_SELECT_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t out_numel,                                                                         \
        int32_t rank,                                                                              \
        int32_t select_dim,                                                                        \
        int32_t src_dim_size,                                                                      \
        const int32_t* out_shape,                                                                  \
        const int64_t* stride_src,                                                                 \
        const int64_t* stride_out,                                                                 \
        const void* src,                                                                           \
        const void* idx,                                                                           \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (out_numel < 0) return 2;                                                              \
        if (out_numel == 0) return 0;                                                              \
        if (src == nullptr || idx == nullptr || out == nullptr) return 2;                         \
        if (out_shape == nullptr || stride_src == nullptr || stride_out == nullptr) return 2;     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_index_select<T>(                                        \
            static_cast<const T*>(src),                                                            \
            static_cast<const int32_t*>(idx),                                                      \
            static_cast<T*>(out),                                                                  \
            out_numel, rank, select_dim, src_dim_size,                                            \
            out_shape, stride_src, stride_out, stream);                                           \
    }

#define BARACUDA_KERNELS_INDEX_SELECT_BACKWARD_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t out_numel,                                                                         \
        int32_t rank,                                                                              \
        int32_t select_dim,                                                                        \
        int32_t src_dim_size,                                                                      \
        const int32_t* out_shape,                                                                  \
        const int64_t* stride_dout,                                                                \
        const int64_t* stride_dsrc,                                                                \
        const void* dout,                                                                          \
        const void* idx,                                                                           \
        void* dsrc,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (out_numel < 0) return 2;                                                              \
        if (out_numel == 0) return 0;                                                              \
        if (dout == nullptr || idx == nullptr || dsrc == nullptr) return 2;                       \
        if (out_shape == nullptr || stride_dout == nullptr || stride_dsrc == nullptr) return 2;   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_index_select_backward<T>(                               \
            static_cast<const T*>(dout),                                                           \
            static_cast<const int32_t*>(idx),                                                      \
            static_cast<T*>(dsrc),                                                                 \
            out_numel, rank, select_dim, src_dim_size,                                            \
            out_shape, stride_dout, stride_dsrc, stream);                                         \
    }

// For masked_fill we receive the fill value as raw bits (host->device via
// the kernel param block) so the FFI shape is uniform across dtypes.
// `T` may be a __half / __nv_bfloat16 / float / etc. — we reinterpret the
// caller-provided raw f64 or raw i64 bits into T at the launcher entry.
#define BARACUDA_KERNELS_MASKED_FILL_INSTANTIATE(NAME, T, BITS_T)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* src,                                                                           \
        const void* mask,                                                                          \
        void* out,                                                                                 \
        BITS_T fill_bits,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                  \
        if (src == nullptr || mask == nullptr || out == nullptr) return 2;                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        T fill_value;                                                                              \
        static_assert(sizeof(BITS_T) >= sizeof(T),                                                 \
                      "BITS_T must be at least as wide as T");                                     \
        /* Bitcast the low-order sizeof(T) bytes of fill_bits into T. */                          \
        BITS_T tmp = fill_bits;                                                                    \
        std::memcpy(&fill_value, &tmp, sizeof(T));                                                 \
        return baracuda::indexing::launch_masked_fill<T>(                                         \
            static_cast<const T*>(src),                                                            \
            static_cast<const uint8_t*>(mask),                                                     \
            static_cast<T*>(out),                                                                  \
            numel, fill_value, stream);                                                            \
    }

#define BARACUDA_KERNELS_MASKED_FILL_BACKWARD_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        const void* dout,                                                                          \
        const void* mask,                                                                          \
        void* dsrc,                                                                                \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (numel == 0) return 0;                                                                  \
        if (dout == nullptr || mask == nullptr || dsrc == nullptr) return 2;                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        T zero_val;                                                                                \
        std::memset(&zero_val, 0, sizeof(T));                                                      \
        return baracuda::indexing::launch_masked_fill_backward<T>(                                \
            static_cast<const T*>(dout),                                                           \
            static_cast<const uint8_t*>(mask),                                                     \
            static_cast<T*>(dsrc),                                                                 \
            numel, zero_val, stream);                                                              \
    }

#define BARACUDA_KERNELS_ONE_HOT_INSTANTIATE(NAME, T, ONE_EXPR, ZERO_EXPR)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t out_numel,                                                                         \
        int32_t num_classes,                                                                       \
        const void* src,                                                                           \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (out_numel < 0) return 2;                                                              \
        if (out_numel == 0) return 0;                                                              \
        if (src == nullptr || out == nullptr) return 2;                                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        T one_val  = (ONE_EXPR);                                                                   \
        T zero_val = (ZERO_EXPR);                                                                  \
        return baracuda::indexing::launch_one_hot<T>(                                             \
            static_cast<const int32_t*>(src),                                                      \
            static_cast<T*>(out),                                                                  \
            out_numel, num_classes, one_val, zero_val, stream);                                    \
    }

#define BARACUDA_KERNELS_NONZERO_INSTANTIATE(NAME, T)                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        int32_t rank,                                                                              \
        int32_t max_nz,                                                                            \
        const int32_t* shape,                                                                      \
        const int64_t* stride_x,                                                                   \
        const void* x,                                                                             \
        void* out_coords,                                                                          \
        void* counter,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (numel < 0) return 2;                                                                  \
        if (x == nullptr || out_coords == nullptr || counter == nullptr) return 2;                \
        if (shape == nullptr || stride_x == nullptr) return 2;                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::indexing::launch_nonzero<T>(                                             \
            static_cast<const T*>(x),                                                              \
            static_cast<int32_t*>(out_coords),                                                     \
            static_cast<int32_t*>(counter),                                                        \
            numel, rank, max_nz, shape, stride_x, stream);                                         \
    }

#endif // BARACUDA_INDEXING_CUH
