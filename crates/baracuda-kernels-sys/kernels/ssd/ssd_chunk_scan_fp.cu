// baracuda-kernels Phase 50 — Mamba-2 SSD chunk-scan FW (FP types).
//
// Hand-port of state-spaces/mamba SSD chunk-scan algorithm (Apache-2.0).
// See `vendor/mamba/VENDOR.md` for upstream attribution.
//
// State-Space Duality (SSD) — Mamba-2's reformulation that maps the
// selective-SSM recurrence
//
//     h[t] = A_t · h[t-1] + B[t] · x[t]    where A_t = exp(dt[t] · A)
//     y[t] = C[t]^T · h[t]
//
// into chunk-scan + GEMM. Each chunk is length `chunk_size` (typically
// 64 or 128). Within a chunk the recurrence is materialized as a
// semi-separable matrix (intra-chunk pass). Across chunks the chunk
// "summary" state is propagated sequentially (inter-chunk pass) and
// combined with the intra contribution.
//
// Shape convention (matches upstream mamba_ssm):
//   x   : [B, L, H, D]            — SSM input projections per head.
//   dt  : [B, L, H]               — per-time, per-head step sizes (>0).
//   A   : [H]                     — per-head SSM eigenvalue (scalar; <0).
//   B   : [B, L, H, N]            — input-side state projection per time.
//   C   : [B, L, H, N]            — output-side state readout per time.
//   y   : [B, L, H, D]            — output (same shape as x).
//
// Where:
//   B = batch
//   L = sequence length (must be divisible by chunk_size in the dense
//       trailblazer; ragged-batch / cu_seqlens deferred)
//   H = heads (typically nheads_ssm)
//   D = head dim (typically 64 or 128)
//   N = state dim (typically 64 or 128)
//
// **Mamba-2 simplification**: `A` is a scalar per head (not a full
// N×N matrix as in S4). This is what makes the chunk-scan algorithm
// efficient — the transition A_t is `exp(dt * A_scalar)`, a single
// scalar per (b, t, h), not an N×N matrix exponential.
//
// This trailblazer uses a single-block-per-(b, h) work decomposition
// that processes the chunks sequentially within the block (each block
// owns the full L-sequence for one (b, h) pair). State `h ∈ R^{D × N}`
// is kept in registers / SMEM across chunks. This is the
// straight-forward correctness-first port; the perf-tuned multi-block
// + tensor-core SSD lives in a follow-up.
//
// Threads per block: 256. Each block handles up to D*N/256 cells of
// the state matrix. For D=64, N=64 → 4096 cells / 256 threads = 16
// cells per thread.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace ssd {

// ---- dtype helpers ----

template <typename T>
__device__ __forceinline__ float load_as_f32(T x) { return (float)x; }
template <>
__device__ __forceinline__ float load_as_f32<__half>(__half x) { return __half2float(x); }
template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_f32(float v) { return (T)v; }
template <>
__device__ __forceinline__ __half store_from_f32<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// =========================================================================
// FW kernel — naive per-(b, h) sequential recurrence
// =========================================================================
//
// One block per (b, h). Block processes all L time steps sequentially,
// keeping the [D × N] state in SMEM. Each thread owns DN/blockDim cells.
//
// For correctness this is identical to the "decode" path of Mamba-2;
// the chunk-scan decomposition is a perf optimization that produces the
// same y given the same inputs.

template <typename T>
__global__ void ssd_fwd_kernel(
    const T* __restrict__ x,        // [B, L, H, D]
    const T* __restrict__ dt,       // [B, L, H]
    const T* __restrict__ A,        // [H]
    const T* __restrict__ B,        // [B, L, H, N]
    const T* __restrict__ C,        // [B, L, H, N]
    T* __restrict__ y,              // [B, L, H, D]
    int32_t batch,
    int32_t seqlen,
    int32_t heads,
    int32_t head_dim,               // D
    int32_t state_dim)              // N
{
    const int32_t b = blockIdx.x;
    const int32_t h = blockIdx.y;
    if (b >= batch || h >= heads) return;

    extern __shared__ float smem[];
    // state[D * N], B_now[N], C_now[N]
    float* state = smem;                                // D * N floats
    float* B_now = smem + (int64_t)head_dim * state_dim;
    float* C_now = B_now + state_dim;

    const int32_t dn = head_dim * state_dim;
    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    // Zero-init state.
    for (int32_t i = tid; i < dn; i += tcount) state[i] = 0.0f;
    __syncthreads();

    const float A_h = load_as_f32<T>(A[h]);

    // Per-batch / per-head offsets.
    const int64_t x_bh_off  = ((int64_t)b * seqlen) * (int64_t)heads * head_dim
                              + (int64_t)h * head_dim;  // step by H*D per t
    const int64_t y_bh_off  = x_bh_off;
    const int64_t bn_bh_off = ((int64_t)b * seqlen) * (int64_t)heads * state_dim
                              + (int64_t)h * state_dim;  // step by H*N per t
    const int64_t dt_bh_off = ((int64_t)b * seqlen) * (int64_t)heads + (int64_t)h;

    const int64_t x_step  = (int64_t)heads * head_dim;
    const int64_t bn_step = (int64_t)heads * state_dim;
    const int64_t dt_step = (int64_t)heads;

    // Iterate over time steps.
    for (int32_t t = 0; t < seqlen; ++t) {
        const float dt_t = load_as_f32<T>(dt[dt_bh_off + (int64_t)t * dt_step]);
        const float decay = __expf(dt_t * A_h);   // A_t scalar

        // Load B_now and C_now into SMEM (length N each).
        const int64_t bn_t_off = bn_bh_off + (int64_t)t * bn_step;
        for (int32_t n = tid; n < state_dim; n += tcount) {
            B_now[n] = load_as_f32<T>(B[bn_t_off + n]);
            C_now[n] = load_as_f32<T>(C[bn_t_off + n]);
        }
        __syncthreads();

        // x[t, h, :] — read once per thread per relevant d-slot.
        // Update state: h[d, n] = decay * h[d, n] + dt_t * B[n] * x[d]
        // Output: y[d] = sum_n C[n] * h[d, n]
        // The "* dt_t" on B*x is the discretization rule
        // (zero-order-hold-like). Matches mamba_ssm reference.
        const int64_t x_t_off = x_bh_off + (int64_t)t * x_step;

        // Each thread owns a strided subset of (d, n) cells.
        // For correctness simplicity we do two passes:
        //   pass 1: update state cells this thread owns
        //   pass 2: compute y[d] for each d (one thread per d, others idle)
        for (int32_t i = tid; i < dn; i += tcount) {
            const int32_t d = i / state_dim;
            const int32_t n = i % state_dim;
            const float xd = load_as_f32<T>(x[x_t_off + d]);
            float s = state[i];
            s = decay * s + dt_t * B_now[n] * xd;
            state[i] = s;
        }
        __syncthreads();

        // Compute y[d] = sum_n C[n] * state[d * N + n].
        // One thread per d (when D <= tcount); otherwise threads
        // share d but each only does a partial sum.
        for (int32_t d = tid; d < head_dim; d += tcount) {
            float acc = 0.0f;
            for (int32_t n = 0; n < state_dim; ++n) {
                acc += C_now[n] * state[d * state_dim + n];
            }
            y[y_bh_off + (int64_t)t * x_step + d] = store_from_f32<T>(acc);
        }
        __syncthreads();
    }
}

template <typename T>
int32_t launch_ssd_fwd(
    const T* x, const T* dt, const T* A, const T* B, const T* C,
    T* y,
    int32_t batch, int32_t seqlen, int32_t heads,
    int32_t head_dim, int32_t state_dim,
    int32_t /*chunk_size*/,  // reserved for the chunk-aware perf kernel
    cudaStream_t stream)
{
    if (batch == 0 || seqlen == 0 || heads == 0 || head_dim == 0 || state_dim == 0) return 0;
    if (head_dim > 256 || state_dim > 256) return 3;  // SMEM cap guard
    if (heads > 65535 || batch > 65535) return 3;     // grid dim cap

    const int32_t threads_per_block = 256;
    dim3 grid((unsigned)batch, (unsigned)heads);
    dim3 block(threads_per_block);

    // SMEM: state[D*N] + B_now[N] + C_now[N], all f32.
    const size_t smem_bytes = (size_t)(head_dim * state_dim + 2 * state_dim) * sizeof(float);
    if (smem_bytes > 48 * 1024) return 3;   // sm_80 default SMEM cap

    ssd_fwd_kernel<T><<<grid, block, smem_bytes, stream>>>(
        x, dt, A, B, C, y,
        batch, seqlen, heads, head_dim, state_dim);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::ssd

#define BARACUDA_SSD_CHUNK_SCAN_FWD_INSTANTIATE(NAME, T)                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                         \
        int32_t batch, int32_t seqlen, int32_t heads,                                         \
        int32_t head_dim, int32_t state_dim, int32_t chunk_size,                              \
        const void* x, const void* dt, const void* A,                                         \
        const void* B, const void* C,                                                         \
        void* y,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                      \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || seqlen < 0 || heads < 0 || head_dim < 0 || state_dim < 0) return 2;  \
        if (chunk_size <= 0) return 2;                                                        \
        if (head_dim > 256 || state_dim > 256) return 3;                                      \
        if (batch == 0 || seqlen == 0 || heads == 0) return 0;                                \
        if (x == nullptr || dt == nullptr || A == nullptr) return 2;                          \
        if (B == nullptr || C == nullptr || y == nullptr) return 2;                           \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::ssd::launch_ssd_fwd<T>(                                              \
            static_cast<const T*>(x), static_cast<const T*>(dt),                              \
            static_cast<const T*>(A), static_cast<const T*>(B),                               \
            static_cast<const T*>(C), static_cast<T*>(y),                                     \
            batch, seqlen, heads, head_dim, state_dim, chunk_size, stream);                   \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                               \
        int32_t batch, int32_t seqlen, int32_t heads,                                         \
        int32_t head_dim, int32_t state_dim, int32_t chunk_size)                              \
    {                                                                                          \
        (void)batch; (void)seqlen; (void)heads;                                               \
        if (chunk_size <= 0) return 2;                                                        \
        if (head_dim > 256 || state_dim > 256) return 3;                                      \
        return 0;                                                                              \
    }

BARACUDA_SSD_CHUNK_SCAN_FWD_INSTANTIATE(ssd_chunk_scan_f32,  float)
BARACUDA_SSD_CHUNK_SCAN_FWD_INSTANTIATE(ssd_chunk_scan_f16,  __half)
BARACUDA_SSD_CHUNK_SCAN_FWD_INSTANTIATE(ssd_chunk_scan_bf16, __nv_bfloat16)
