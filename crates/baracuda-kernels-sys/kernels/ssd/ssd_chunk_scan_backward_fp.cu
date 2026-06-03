// baracuda-kernels Phase 50 — Mamba-2 SSD chunk-scan BW (FP types).
//
// Hand-port of state-spaces/mamba SSD backward (Apache-2.0).
// See `vendor/mamba/VENDOR.md` for upstream attribution.
//
// Mathematical derivation of the SSM BW for Mamba-2's scalar-A
// formulation:
//
//   FW:  decay[t] = exp(dt[t] * A)
//        h[t, d, n] = decay[t] * h[t-1, d, n] + dt[t] * B[t, n] * x[t, d]
//        y[t, d]    = sum_n C[t, n] * h[t, d, n]
//
//   BW (reverse-time recurrence on dh):
//        dh[t, d, n] = C[t, n] * dy[t, d] + decay[t+1] * dh[t+1, d, n]
//        with boundary dh[L, d, n] = 0
//        dC[t, n]   = sum_d h[t, d, n] * dy[t, d]
//        dB[t, n]   = sum_d dt[t] * x[t, d] * dh[t, d, n]
//        dx[t, d]   = sum_n dt[t] * B[t, n] * dh[t, d, n]
//        ddt[t]     = sum_{d,n} [A * decay[t] * h[t-1, d, n] * dh[t, d, n]
//                                + B[t, n] * x[t, d] * dh[t, d, n]]
//        dA[h]      = sum_{b, t, d, n} dt[t] * decay[t] * h[t-1, d, n]
//                                       * dh[t, d, n]
//
// Implementation: one block per (b, h). Pass 1 reruns FW recording
// every h[t] state to scratch ([B, L, H, D, N] of T) — caller provides
// the scratch buffer of size `workspace_bytes`. Pass 2 walks t = L-1
// down to 0 computing dh and emitting dx/dB/dC/ddt cells. dA is
// accumulated globally via atomicAdd.
//
// State residency: SMEM for state[D*N] + B_now[N] + C_now[N] + dh[D*N].
//   D=64, N=64 → 64*64*2*4 + 64*2*4 = 32768 + 512 = ~33 KiB (fits 48 KiB cap)
//   D=128, N=128 → 128*128*2*4 + 256*4 = 131072+1024 = ~128 KiB (REJECTED)
//
// We reject D*N > 4096 in `can_implement` to stay within the sm_80 SMEM
// budget. The brief documents this as the trailblazer scope; later
// fanout would push state to gmem with cp.async double-buffering.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace ssd_bw {

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

// AtomicAdd for f32/f64; for f16/bf16 use the native sm_80 atomicAdd.
template <typename T>
__device__ __forceinline__ void atomic_add_T(T* addr, float v) {
    atomicAdd(addr, (T)v);
}
template <>
__device__ __forceinline__ void atomic_add_T<__half>(__half* addr, float v) {
    atomicAdd(addr, __float2half(v));
}
template <>
__device__ __forceinline__ void atomic_add_T<__nv_bfloat16>(__nv_bfloat16* addr, float v) {
    atomicAdd(addr, __float2bfloat16(v));
}

// =========================================================================
// FW-record kernel — rerun FW but save every h[t] to scratch.
// =========================================================================

template <typename T>
__global__ void ssd_bw_pass1_record_states(
    const T* __restrict__ x,        // [B, L, H, D]
    const T* __restrict__ dt,       // [B, L, H]
    const T* __restrict__ A,        // [H]
    const T* __restrict__ B,        // [B, L, H, N]
    float* __restrict__ states,     // [B, H, L, D, N]  f32 scratch
    int32_t batch, int32_t seqlen, int32_t heads,
    int32_t head_dim, int32_t state_dim)
{
    const int32_t b = blockIdx.x;
    const int32_t h = blockIdx.y;
    if (b >= batch || h >= heads) return;

    extern __shared__ float smem[];
    float* state = smem;                                 // D * N
    float* B_now = smem + (int64_t)head_dim * state_dim; // N
    const int32_t dn = head_dim * state_dim;
    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    for (int32_t i = tid; i < dn; i += tcount) state[i] = 0.0f;
    __syncthreads();

    const float A_h = load_as_f32<T>(A[h]);
    const int64_t x_bh_off  = ((int64_t)b * seqlen) * (int64_t)heads * head_dim
                              + (int64_t)h * head_dim;
    const int64_t bn_bh_off = ((int64_t)b * seqlen) * (int64_t)heads * state_dim
                              + (int64_t)h * state_dim;
    const int64_t dt_bh_off = ((int64_t)b * seqlen) * (int64_t)heads + (int64_t)h;
    const int64_t x_step  = (int64_t)heads * head_dim;
    const int64_t bn_step = (int64_t)heads * state_dim;
    const int64_t dt_step = (int64_t)heads;

    // states layout: [B, H, L, D, N] — state[b, h, t, :, :] = bh_off + t*DN
    const int64_t states_bh_off = ((int64_t)b * heads + h) * (int64_t)seqlen
                                  * (int64_t)head_dim * state_dim;

    for (int32_t t = 0; t < seqlen; ++t) {
        const float dt_t = load_as_f32<T>(dt[dt_bh_off + (int64_t)t * dt_step]);
        const float decay = __expf(dt_t * A_h);

        const int64_t bn_t_off = bn_bh_off + (int64_t)t * bn_step;
        for (int32_t n = tid; n < state_dim; n += tcount) {
            B_now[n] = load_as_f32<T>(B[bn_t_off + n]);
        }
        __syncthreads();

        const int64_t x_t_off = x_bh_off + (int64_t)t * x_step;
        for (int32_t i = tid; i < dn; i += tcount) {
            const int32_t d = i / state_dim;
            const int32_t n = i % state_dim;
            const float xd = load_as_f32<T>(x[x_t_off + d]);
            const float s = decay * state[i] + dt_t * B_now[n] * xd;
            state[i] = s;
            // Save h[t, d, n].
            states[states_bh_off + (int64_t)t * dn + i] = s;
        }
        __syncthreads();
    }
}

// =========================================================================
// BW reverse-time pass — uses recorded states to emit dx/dB/dC/ddt/dA.
// =========================================================================

template <typename T>
__global__ void ssd_bw_pass2_kernel(
    const T* __restrict__ x,        // [B, L, H, D]
    const T* __restrict__ dt,       // [B, L, H]
    const T* __restrict__ A,        // [H]
    const T* __restrict__ B,        // [B, L, H, N]
    const T* __restrict__ C,        // [B, L, H, N]
    const T* __restrict__ dy,       // [B, L, H, D]
    const float* __restrict__ states, // [B, H, L, D, N] f32
    T* __restrict__ dx,             // [B, L, H, D]
    T* __restrict__ dB,             // [B, L, H, N]
    T* __restrict__ dC,             // [B, L, H, N]
    T* __restrict__ ddt,            // [B, L, H]
    T* __restrict__ dA,             // [H]  (atomicAdd)
    int32_t batch, int32_t seqlen, int32_t heads,
    int32_t head_dim, int32_t state_dim)
{
    const int32_t b = blockIdx.x;
    const int32_t h = blockIdx.y;
    if (b >= batch || h >= heads) return;

    extern __shared__ float smem[];
    float* dh    = smem;                                  // D * N
    float* B_now = smem + (int64_t)head_dim * state_dim;  // N
    float* C_now = B_now + state_dim;                     // N
    const int32_t dn = head_dim * state_dim;
    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    for (int32_t i = tid; i < dn; i += tcount) dh[i] = 0.0f;
    __syncthreads();

    const float A_h = load_as_f32<T>(A[h]);
    const int64_t x_bh_off  = ((int64_t)b * seqlen) * (int64_t)heads * head_dim
                              + (int64_t)h * head_dim;
    const int64_t bn_bh_off = ((int64_t)b * seqlen) * (int64_t)heads * state_dim
                              + (int64_t)h * state_dim;
    const int64_t dt_bh_off = ((int64_t)b * seqlen) * (int64_t)heads + (int64_t)h;
    const int64_t x_step  = (int64_t)heads * head_dim;
    const int64_t bn_step = (int64_t)heads * state_dim;
    const int64_t dt_step = (int64_t)heads;
    const int64_t states_bh_off = ((int64_t)b * heads + h) * (int64_t)seqlen
                                  * (int64_t)head_dim * state_dim;

    // Reverse time: dh[t] = C[t] * dy[t] + decay[t+1] * dh[t+1]
    // For dA accumulation we need decay[t] * h[t-1] * dh[t] * dt[t].
    // We walk backward t = L-1 .. 0.

    float dA_acc = 0.0f;  // per-block accumulator, atomicAdd at end

    for (int32_t t = seqlen - 1; t >= 0; --t) {
        const float dt_t = load_as_f32<T>(dt[dt_bh_off + (int64_t)t * dt_step]);
        const float decay_t = __expf(dt_t * A_h);

        // Load B_now, C_now.
        const int64_t bn_t_off = bn_bh_off + (int64_t)t * bn_step;
        for (int32_t n = tid; n < state_dim; n += tcount) {
            B_now[n] = load_as_f32<T>(B[bn_t_off + n]);
            C_now[n] = load_as_f32<T>(C[bn_t_off + n]);
        }
        __syncthreads();

        // dh[t] += C[t, n] * dy[t, d]  (matrix outer-product into dh[d, n])
        const int64_t y_t_off = x_bh_off + (int64_t)t * x_step;
        for (int32_t i = tid; i < dn; i += tcount) {
            const int32_t d = i / state_dim;
            const int32_t n = i % state_dim;
            const float dy_d = load_as_f32<T>(dy[y_t_off + d]);
            dh[i] += C_now[n] * dy_d;
        }
        __syncthreads();

        // dC[t, n] = sum_d h[t, d, n] * dy[t, d]
        // One thread per n; sum over d.
        const int64_t hts_off = states_bh_off + (int64_t)t * dn;
        for (int32_t n = tid; n < state_dim; n += tcount) {
            float dc_acc = 0.0f;
            for (int32_t d = 0; d < head_dim; ++d) {
                const float h_dn = states[hts_off + (int64_t)d * state_dim + n];
                const float dy_d = load_as_f32<T>(dy[y_t_off + d]);
                dc_acc += h_dn * dy_d;
            }
            dC[bn_t_off + n] = store_from_f32<T>(dc_acc);
        }

        // dB[t, n] = sum_d dt_t * x[t, d] * dh[t, d, n]
        for (int32_t n = tid; n < state_dim; n += tcount) {
            float db_acc = 0.0f;
            const int64_t x_t_off = x_bh_off + (int64_t)t * x_step;
            for (int32_t d = 0; d < head_dim; ++d) {
                const float xd = load_as_f32<T>(x[x_t_off + d]);
                db_acc += xd * dh[(int64_t)d * state_dim + n];
            }
            dB[bn_t_off + n] = store_from_f32<T>(dt_t * db_acc);
        }

        // dx[t, d] = sum_n dt_t * B[t, n] * dh[t, d, n]
        for (int32_t d = tid; d < head_dim; d += tcount) {
            float dx_acc = 0.0f;
            for (int32_t n = 0; n < state_dim; ++n) {
                dx_acc += B_now[n] * dh[(int64_t)d * state_dim + n];
            }
            dx[x_bh_off + (int64_t)t * x_step + d] = store_from_f32<T>(dt_t * dx_acc);
        }

        // ddt[t] = sum_{d, n} A * decay_t * h[t-1, d, n] * dh[t, d, n]
        //                    + B[n] * x[d] * dh[t, d, n]
        //        = A * decay_t * <h[t-1], dh[t]>  +  sum_{d, n} B[n]*x[d]*dh[d,n]
        // Compute via block-wide reduction.
        float ddt_local = 0.0f;
        for (int32_t i = tid; i < dn; i += tcount) {
            const int32_t d = i / state_dim;
            const int32_t n = i % state_dim;
            float h_prev = 0.0f;
            if (t > 0) {
                h_prev = states[states_bh_off + (int64_t)(t - 1) * dn + i];
            }
            const float xd = load_as_f32<T>(x[x_bh_off + (int64_t)t * x_step + d]);
            const float ddh = dh[i];
            ddt_local += A_h * decay_t * h_prev * ddh
                         + B_now[n] * xd * ddh;
        }
        // Block-reduction via SMEM (reuse C_now array — small reduction).
        // We do a simple warp shuffle + single-warp final reduction.
        __syncthreads();
        // Reuse the SMEM "C_now" region (length N) as scratch for the
        // reduction tree. N typically >= 32 for Mamba shapes.
        float* red_smem = C_now;
        // Per-warp sum.
        for (int32_t offset = 16; offset > 0; offset >>= 1) {
            ddt_local += __shfl_down_sync(0xFFFFFFFF, ddt_local, offset);
        }
        const int32_t warp_id = tid >> 5;
        const int32_t lane_id = tid & 31;
        const int32_t num_warps = (tcount + 31) >> 5;
        if (lane_id == 0 && warp_id < state_dim) red_smem[warp_id] = ddt_local;
        __syncthreads();
        float ddt_total = 0.0f;
        if (tid == 0) {
            for (int32_t w = 0; w < num_warps; ++w) ddt_total += red_smem[w];
            ddt[dt_bh_off + (int64_t)t * dt_step] = store_from_f32<T>(ddt_total);
        }

        // dA contribution: sum over d, n of dt_t * decay_t * h[t-1] * dh[t]
        float dA_local = 0.0f;
        for (int32_t i = tid; i < dn; i += tcount) {
            float h_prev = 0.0f;
            if (t > 0) h_prev = states[states_bh_off + (int64_t)(t - 1) * dn + i];
            dA_local += dt_t * decay_t * h_prev * dh[i];
        }
        for (int32_t offset = 16; offset > 0; offset >>= 1) {
            dA_local += __shfl_down_sync(0xFFFFFFFF, dA_local, offset);
        }
        if (lane_id == 0 && warp_id < state_dim) red_smem[warp_id] = dA_local;
        __syncthreads();
        if (tid == 0) {
            float dA_total = 0.0f;
            for (int32_t w = 0; w < num_warps; ++w) dA_total += red_smem[w];
            dA_acc += dA_total;
        }
        __syncthreads();

        // Propagate dh backward: dh[t-1] = decay_t * dh[t]
        // (the next iteration's "decay[t+1] * dh[t+1]" term — we apply
        // it now using the current decay, which becomes "decay_{t}" for
        // the prior iteration after the time step changes).
        // This is correct because h[t] = decay_t * h[t-1] + ... so
        // ∂h[t]/∂h[t-1] = decay_t, so dh[t-1] += decay_t * dh[t].
        // (The C[t-1] * dy[t-1] term is added at the next iteration's
        // dh accumulation step.)
        for (int32_t i = tid; i < dn; i += tcount) {
            dh[i] *= decay_t;
        }
        __syncthreads();
    }

    // dA[h] is accumulated across all (b) blocks for this h via atomic.
    if (tid == 0) {
        atomic_add_T<T>(&dA[h], dA_acc);
    }
}

template <typename T>
int32_t launch_ssd_bwd(
    const T* x, const T* dt, const T* A, const T* B, const T* C, const T* dy,
    T* dx, T* dB, T* dC, T* ddt, T* dA,
    int32_t batch, int32_t seqlen, int32_t heads,
    int32_t head_dim, int32_t state_dim,
    int32_t /*chunk_size*/,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
    if (batch == 0 || seqlen == 0 || heads == 0 || head_dim == 0 || state_dim == 0) return 0;
    if (head_dim > 64 || state_dim > 64) return 3;  // BW SMEM cap tighter

    // Workspace requirement: [B, H, L, D, N] f32 states.
    const int64_t states_count = (int64_t)batch * heads * seqlen
                                 * head_dim * state_dim;
    const size_t states_bytes = (size_t)states_count * sizeof(float);
    if (workspace_bytes < states_bytes || workspace == nullptr) return 4;
    float* states = (float*)workspace;

    const int32_t threads_per_block = 256;
    dim3 grid((unsigned)batch, (unsigned)heads);
    dim3 block(threads_per_block);

    // Pass 1: SMEM = state[D*N] + B_now[N], all f32.
    const size_t smem_pass1 = (size_t)(head_dim * state_dim + state_dim) * sizeof(float);
    if (smem_pass1 > 48 * 1024) return 3;

    ssd_bw_pass1_record_states<T><<<grid, block, smem_pass1, stream>>>(
        x, dt, A, B, states,
        batch, seqlen, heads, head_dim, state_dim);

    // Zero dA before atomic accumulation.
    cudaMemsetAsync(dA, 0, (size_t)heads * sizeof(T), stream);

    // Pass 2: SMEM = dh[D*N] + B_now[N] + C_now[N], all f32.
    const size_t smem_pass2 = (size_t)(head_dim * state_dim + 2 * state_dim) * sizeof(float);
    if (smem_pass2 > 48 * 1024) return 3;

    ssd_bw_pass2_kernel<T><<<grid, block, smem_pass2, stream>>>(
        x, dt, A, B, C, dy, states,
        dx, dB, dC, ddt, dA,
        batch, seqlen, heads, head_dim, state_dim);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::ssd_bw

// Returns the required workspace size in bytes for the BW launch.
extern "C" size_t baracuda_kernels_ssd_chunk_scan_workspace_bytes(
    int32_t batch, int32_t seqlen, int32_t heads,
    int32_t head_dim, int32_t state_dim,
    int32_t /*chunk_size*/, int32_t /*dtype_id*/)
{
    if (batch <= 0 || seqlen <= 0 || heads <= 0 || head_dim <= 0 || state_dim <= 0) return 0;
    const int64_t cells = (int64_t)batch * heads * seqlen * head_dim * state_dim;
    return (size_t)cells * sizeof(float);
}

#define BARACUDA_SSD_CHUNK_SCAN_BWD_INSTANTIATE(NAME, T)                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_backward_run(                                \
        int32_t batch, int32_t seqlen, int32_t heads,                                         \
        int32_t head_dim, int32_t state_dim, int32_t chunk_size,                              \
        const void* x, const void* dt, const void* A,                                         \
        const void* B, const void* C, const void* dy,                                         \
        void* dx, void* dB, void* dC, void* ddt, void* dA,                                    \
        void* workspace, size_t workspace_bytes,                                              \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || seqlen < 0 || heads < 0 || head_dim < 0 || state_dim < 0) return 2;  \
        if (chunk_size <= 0) return 2;                                                        \
        if (head_dim > 64 || state_dim > 64) return 3;                                        \
        if (batch == 0 || seqlen == 0 || heads == 0) return 0;                                \
        if (x == nullptr || dt == nullptr || A == nullptr) return 2;                          \
        if (B == nullptr || C == nullptr || dy == nullptr) return 2;                          \
        if (dx == nullptr || dB == nullptr || dC == nullptr ||                                \
            ddt == nullptr || dA == nullptr) return 2;                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::ssd_bw::launch_ssd_bwd<T>(                                           \
            static_cast<const T*>(x), static_cast<const T*>(dt),                              \
            static_cast<const T*>(A), static_cast<const T*>(B),                               \
            static_cast<const T*>(C), static_cast<const T*>(dy),                              \
            static_cast<T*>(dx), static_cast<T*>(dB), static_cast<T*>(dC),                    \
            static_cast<T*>(ddt), static_cast<T*>(dA),                                        \
            batch, seqlen, heads, head_dim, state_dim, chunk_size,                            \
            workspace, workspace_bytes, stream);                                              \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_backward_can_implement(                      \
        int32_t batch, int32_t seqlen, int32_t heads,                                         \
        int32_t head_dim, int32_t state_dim, int32_t chunk_size)                              \
    {                                                                                          \
        if (batch < 0 || seqlen < 0 || heads < 0 || head_dim < 0 || state_dim < 0) return 2;  \
        if (chunk_size <= 0) return 2;                                                        \
        /* BW SMEM cap is tighter than FW: dh[D*N] + B[N] + C[N] all f32. */                  \
        if (head_dim > 64 || state_dim > 64) return 3;                                        \
        return 0;                                                                              \
    }

BARACUDA_SSD_CHUNK_SCAN_BWD_INSTANTIATE(ssd_chunk_scan_f32,  float)
BARACUDA_SSD_CHUNK_SCAN_BWD_INSTANTIATE(ssd_chunk_scan_f16,  __half)
BARACUDA_SSD_CHUNK_SCAN_BWD_INSTANTIATE(ssd_chunk_scan_bf16, __nv_bfloat16)
