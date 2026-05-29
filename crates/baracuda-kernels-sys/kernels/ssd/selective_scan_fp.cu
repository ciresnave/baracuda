// baracuda-kernels Phase 50b — Mamba-1 selective_scan FW (FP types).
//
// Hand-port of state-spaces/mamba selective_scan algorithm (Apache-2.0).
// See `vendor/mamba/VENDOR.md` for upstream attribution.
//
// Mamba-1's selective state-space scan: per-channel recurrence with a
// per-(d, n) eigenvalue matrix `A`, time-varying `B[t, n]` / `C[t, n]`
// modulation, scalar timestep `delta[t]`, optional skip `D[d] * u[t, d]`,
// optional sigmoid-gated tail `silu(z[t, d])`, optional `delta_bias[d]`
// + optional `softplus(delta)` mapping.
//
//     dA      = exp(delta[b, t, d] * A[d, n])                  // per (d, n)
//     dBu     = delta[b, t, d] * B[b, t, n] * u[b, t, d]
//     h[d, n] = dA * h[d, n] + dBu                             // state update
//     y[t, d] = sum_n h[d, n] * C[b, t, n]
//     if D given: y[t, d] += D[d] * u[b, t, d]
//     if z given: y[t, d] *= silu(z[b, t, d])
//
// Shape convention (matches upstream `selective_scan_fn`):
//   u           : [B, L, D]      input (also "u" in upstream)
//   delta       : [B, L, D]      time-step
//   A           : [D, N]         per-channel state matrix
//   B           : [B, L, N]      time-varying input-side projection
//   C           : [B, L, N]      time-varying output-side projection
//   D           : [D] or null    skip-connection (optional)
//   z           : [B, L, D] or null  gating (optional, SiLU-gated)
//   delta_bias  : [D] or null    per-channel delta bias (optional)
//   y           : [B, L, D]      output
//   last_state  : [B, D, N] or null  optional final h saved out
//
// Where:
//   B = batch
//   L = sequence length
//   D = channel dim ("dim" in upstream)
//   N = state dim ("dstate" in upstream, typically 16)
//
// Work decomposition: one block per (b, d) pair. Each block keeps the
// length-N state vector `h[n]` in SMEM and walks t sequentially through
// the recurrence. Each thread owns a strided subset of N cells.
//
// Threads per block: 64 (N typically 16; we use 64 to amortize the
// per-time-step reductions and to cover the N ≤ 256 cap).

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace selective_scan {

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

// Block-wide sum reduction over `state_dim` lanes. The threads with
// `tid < state_dim` contribute `val`; result lives in `smem[0]`.
// `smem` must have at least `state_dim` floats.
__device__ __forceinline__ float block_sum(
    float val, float* smem, int32_t tid, int32_t state_dim, int32_t tcount)
{
    // Stage 1: write per-thread partials.
    if (tid < state_dim) smem[tid] = val;
    else if (tid < tcount) smem[tid] = 0.0f;
    __syncthreads();

    // Tree reduction over the first `state_dim` lanes (rounded up to
    // a power of two by zero-padding the upper lanes above).
    int32_t span = 1;
    while (span < tcount) span <<= 1;
    for (int32_t offset = span >> 1; offset > 0; offset >>= 1) {
        if (tid < offset && (tid + offset) < tcount) {
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }
    return smem[0];
}

// =========================================================================
// FW kernel — one block per (b, d), state `h[n]` in SMEM.
// =========================================================================

template <typename T>
__global__ void selective_scan_fwd_kernel(
    const T* __restrict__ u,           // [B, L, D]
    const T* __restrict__ delta,       // [B, L, D]
    const T* __restrict__ A,           // [D, N]
    const T* __restrict__ B,           // [B, L, N]
    const T* __restrict__ C,           // [B, L, N]
    const T* __restrict__ D_skip,      // [D] or null
    const T* __restrict__ z,           // [B, L, D] or null
    const T* __restrict__ delta_bias,  // [D] or null
    T* __restrict__ y,                 // [B, L, D]
    T* __restrict__ last_state,        // [B, D, N] or null
    int32_t batch,
    int32_t seqlen,
    int32_t dim,           // D
    int32_t dstate,        // N
    int32_t delta_softplus_flag)
{
    const int32_t b = blockIdx.x;
    const int32_t d = blockIdx.y;
    if (b >= batch || d >= dim) return;

    extern __shared__ float smem[];
    // Layout: state[N] + reduce_scratch[blockDim.x]
    float* state = smem;                    // N floats
    float* reduce_scratch = smem + dstate;  // blockDim.x floats

    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    // Zero state.
    for (int32_t n = tid; n < dstate; n += tcount) state[n] = 0.0f;
    __syncthreads();

    // Offsets.
    //   u, delta, y, z: per-(b, d) base = b*L*D + d, stride along t = D.
    //   B, C: per-b base = b*L*N, stride along t = N.
    const int64_t ud_bd_off = (int64_t)b * seqlen * dim + d;
    const int64_t bc_b_off  = (int64_t)b * seqlen * dstate;
    const int64_t ud_step   = (int64_t)dim;
    const int64_t bc_step   = (int64_t)dstate;
    const int64_t A_d_off   = (int64_t)d * dstate;

    const float db_d = (delta_bias != nullptr) ? load_as_f32<T>(delta_bias[d]) : 0.0f;
    const float D_d  = (D_skip != nullptr)     ? load_as_f32<T>(D_skip[d])     : 0.0f;

    for (int32_t t = 0; t < seqlen; ++t) {
        // Load scalar inputs.
        float delta_t = load_as_f32<T>(delta[ud_bd_off + (int64_t)t * ud_step]) + db_d;
        if (delta_softplus_flag != 0) {
            // softplus(x) = log1p(exp(x)); upstream's overflow guard.
            delta_t = (delta_t <= 20.0f) ? __logf(1.0f + __expf(delta_t)) : delta_t;
        }
        const float u_t = load_as_f32<T>(u[ud_bd_off + (int64_t)t * ud_step]);

        // State update + dot-with-C accumulator.
        const int64_t bc_t_off = bc_b_off + (int64_t)t * bc_step;
        float y_partial = 0.0f;
        for (int32_t n = tid; n < dstate; n += tcount) {
            const float A_dn = load_as_f32<T>(A[A_d_off + n]);
            const float B_n  = load_as_f32<T>(B[bc_t_off + n]);
            const float C_n  = load_as_f32<T>(C[bc_t_off + n]);
            const float dA   = __expf(delta_t * A_dn);
            const float dBu  = delta_t * B_n * u_t;
            const float h_new = dA * state[n] + dBu;
            state[n] = h_new;
            y_partial += h_new * C_n;
        }
        __syncthreads();

        // Reduce partials over threads (all threads participate; threads
        // outside `dstate` contributed nothing above so we need to gate
        // their contribution here too).
        const float y_state = block_sum(
            (tid < dstate) ? y_partial : 0.0f,
            reduce_scratch, tid, dstate, tcount);

        if (tid == 0) {
            float y_val = y_state;
            if (D_skip != nullptr) y_val += D_d * u_t;
            if (z != nullptr) {
                const float z_t = load_as_f32<T>(z[ud_bd_off + (int64_t)t * ud_step]);
                // silu(z) = z * sigmoid(z) = z / (1 + exp(-z)).
                const float sig = 1.0f / (1.0f + __expf(-z_t));
                y_val *= z_t * sig;
            }
            y[ud_bd_off + (int64_t)t * ud_step] = store_from_f32<T>(y_val);
        }
        __syncthreads();
    }

    // Optional last-state save: [b, d, :] = state[:].
    if (last_state != nullptr) {
        const int64_t ls_off = ((int64_t)b * dim + d) * dstate;
        for (int32_t n = tid; n < dstate; n += tcount) {
            last_state[ls_off + n] = store_from_f32<T>(state[n]);
        }
    }
}

template <typename T>
int32_t launch_selective_scan_fwd(
    const T* u, const T* delta, const T* A, const T* B, const T* C,
    const T* D_skip, const T* z, const T* delta_bias,
    T* y, T* last_state,
    int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate,
    int32_t delta_softplus_flag,
    cudaStream_t stream)
{
    if (batch == 0 || seqlen == 0 || dim == 0 || dstate == 0) return 0;
    if (dstate > 256) return 3;
    if (batch > 65535 || dim > 65535) return 3;

    // 64 threads/block — N is typically 16 in Mamba-1; we use 64 to
    // amortize the reduction over multiple warps and cover N ≤ 256.
    const int32_t threads_per_block = 64;
    dim3 grid((unsigned)batch, (unsigned)dim);
    dim3 block(threads_per_block);

    // SMEM: state[N] + reduce_scratch[threads_per_block], both f32.
    const size_t smem_bytes = (size_t)(dstate + threads_per_block) * sizeof(float);
    if (smem_bytes > 48 * 1024) return 3;

    selective_scan_fwd_kernel<T><<<grid, block, smem_bytes, stream>>>(
        u, delta, A, B, C, D_skip, z, delta_bias, y, last_state,
        batch, seqlen, dim, dstate, delta_softplus_flag);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::selective_scan

#define BARACUDA_SELECTIVE_SCAN_FWD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate,                            \
        int32_t delta_softplus,                                                                \
        const void* u, const void* delta, const void* A,                                       \
        const void* B, const void* C,                                                          \
        const void* D_skip, const void* z, const void* delta_bias,                             \
        void* y, void* last_state,                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (batch < 0 || seqlen < 0 || dim < 0 || dstate < 0) return 2;                        \
        if (dstate > 256) return 3;                                                            \
        if (batch == 0 || seqlen == 0 || dim == 0) return 0;                                   \
        if (u == nullptr || delta == nullptr || A == nullptr) return 2;                        \
        if (B == nullptr || C == nullptr || y == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::selective_scan::launch_selective_scan_fwd<T>(                         \
            static_cast<const T*>(u), static_cast<const T*>(delta),                            \
            static_cast<const T*>(A), static_cast<const T*>(B),                                \
            static_cast<const T*>(C),                                                          \
            static_cast<const T*>(D_skip), static_cast<const T*>(z),                           \
            static_cast<const T*>(delta_bias),                                                 \
            static_cast<T*>(y), static_cast<T*>(last_state),                                   \
            batch, seqlen, dim, dstate, delta_softplus, stream);                               \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate)                            \
    {                                                                                          \
        (void)batch; (void)seqlen; (void)dim;                                                  \
        if (dstate > 256) return 3;                                                            \
        return 0;                                                                              \
    }

BARACUDA_SELECTIVE_SCAN_FWD_INSTANTIATE(selective_scan_f32,  float)
BARACUDA_SELECTIVE_SCAN_FWD_INSTANTIATE(selective_scan_f16,  __half)
BARACUDA_SELECTIVE_SCAN_FWD_INSTANTIATE(selective_scan_bf16, __nv_bfloat16)
