// baracuda-kernels Phase 50b — Mamba-1 selective_scan BW (FP types).
//
// Hand-port of state-spaces/mamba selective_scan backward (Apache-2.0).
// See `vendor/mamba/VENDOR.md` for upstream attribution.
//
// Mathematical derivation of the BW for the per-channel Mamba-1
// selective_scan recurrence:
//
//   FW:  delta_eff = softplus?(delta + delta_bias[d])
//        dA[d, n]  = exp(delta_eff * A[d, n])
//        dBu[n]    = delta_eff * B[t, n] * u[t, d]
//        h[t, n]   = dA[d, n] * h[t-1, n] + dBu[n]
//        y_state   = sum_n h[t, n] * C[t, n]
//        y[t, d]   = y_state + D?[d] * u[t, d]
//        y[t, d]  *= silu(z[t, d]) if z given
//
//   BW (reverse-time recurrence on dh):
//        let s     = silu(z[t, d]) if z given, else 1
//        let dy_s  = dy[t, d] * s
//        dz[t, d]  = dy[t, d] * y_pre_gate * silu'(z)         if z given
//        dD[d]    += dy[t, d] * u[t, d]                       if D given (atomicAdd)
//        du_skip   = dy[t, d] * D[d]                          if D given
//
//        dh[n]    += dy_s * C[t, n]
//        dC[t, n]  = dy_s * h[t, n]                           (deterministic, one writer)
//
//        dA[d, n] += delta_eff * h[t-1, n] * exp(delta_eff*A[d, n]) * dh[n]    (atomicAdd)
//                  = delta_eff * h[t-1, n] * dA_dn * dh[n]
//        dB[t, n] += delta_eff * u[t, d] * dh[n]              (atomicAdd over d)
//        ddelta_from_state =
//            Σ_n [ B[t, n] * u[t, d] * dh[n]
//                  + A[d, n] * dA_dn * h[t-1, n] * dh[n] ]
//        du[t, d]  = du_skip + delta_eff * Σ_n B[t, n] * dh[n]
//        Propagate: dh[n] = dh[n] * dA_dn
//
//   Apply softplus chain rule:
//        if delta_softplus and (delta+db) <= 20:
//           ddelta_pre = ddelta_from_state * sigmoid(delta + db)
//        else:
//           ddelta_pre = ddelta_from_state
//        ddelta[t, d]   = ddelta_pre
//        ddelta_bias[d] += ddelta_pre                          if delta_bias given (atomicAdd)
//
// Implementation: one block per (b, d). Pass 1 reruns FW recording every
// `h[t, :]` state vector to scratch ([B, D, L, N] of T) — caller
// provides workspace of size `B*D*L*N*sizeof(T)`. Pass 2 walks t = L-1
// down to 0 computing dh and emitting du/dB/dC/ddelta cells per-time;
// dA / dD / ddelta_bias are atomic-accumulated. Each (b, d) block keeps
// its own dh[n] vector and atomically accumulates to globally-shared
// gradient buffers (atomicAdd over b is the cost; dh is block-local).

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace selective_scan_bw {

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

__device__ __forceinline__ float block_sum(
    float val, float* smem, int32_t tid, int32_t dstate, int32_t tcount)
{
    if (tid < dstate) smem[tid] = val;
    else if (tid < tcount) smem[tid] = 0.0f;
    __syncthreads();
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
// Pass 1 — rerun FW, record h[t, :] states to scratch.
// =========================================================================

template <typename T>
__global__ void selective_scan_bw_pass1_record(
    const T* __restrict__ u,           // [B, L, D]
    const T* __restrict__ delta,       // [B, L, D]
    const T* __restrict__ A,           // [D, N]
    const T* __restrict__ B,           // [B, L, N]
    const T* __restrict__ delta_bias,  // [D] or null
    T* __restrict__ states,            // [B, D, L, N] (T-typed)
    int32_t batch, int32_t seqlen,
    int32_t dim, int32_t dstate,
    int32_t delta_softplus_flag)
{
    const int32_t b = blockIdx.x;
    const int32_t d = blockIdx.y;
    if (b >= batch || d >= dim) return;

    extern __shared__ float smem[];
    float* state = smem;                          // N floats
    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    for (int32_t n = tid; n < dstate; n += tcount) state[n] = 0.0f;
    __syncthreads();

    const int64_t ud_bd_off = (int64_t)b * seqlen * dim + d;
    const int64_t bc_b_off  = (int64_t)b * seqlen * dstate;
    const int64_t ud_step   = (int64_t)dim;
    const int64_t bc_step   = (int64_t)dstate;
    const int64_t A_d_off   = (int64_t)d * dstate;
    const int64_t st_bd_off = ((int64_t)b * dim + d) * (int64_t)seqlen * dstate;

    const float db_d = (delta_bias != nullptr) ? load_as_f32<T>(delta_bias[d]) : 0.0f;

    for (int32_t t = 0; t < seqlen; ++t) {
        float delta_t = load_as_f32<T>(delta[ud_bd_off + (int64_t)t * ud_step]) + db_d;
        if (delta_softplus_flag != 0) {
            delta_t = (delta_t <= 20.0f) ? __logf(1.0f + __expf(delta_t)) : delta_t;
        }
        const float u_t = load_as_f32<T>(u[ud_bd_off + (int64_t)t * ud_step]);
        const int64_t bc_t_off = bc_b_off + (int64_t)t * bc_step;

        for (int32_t n = tid; n < dstate; n += tcount) {
            const float A_dn = load_as_f32<T>(A[A_d_off + n]);
            const float B_n  = load_as_f32<T>(B[bc_t_off + n]);
            const float dA   = __expf(delta_t * A_dn);
            const float dBu  = delta_t * B_n * u_t;
            const float h_new = dA * state[n] + dBu;
            state[n] = h_new;
            states[st_bd_off + (int64_t)t * dstate + n] = store_from_f32<T>(h_new);
        }
        __syncthreads();
    }
}

// =========================================================================
// Pass 2 — reverse-time BW using saved states.
// =========================================================================

template <typename T>
__global__ void selective_scan_bw_pass2(
    const T* __restrict__ u,           // [B, L, D]
    const T* __restrict__ delta,       // [B, L, D]
    const T* __restrict__ A,           // [D, N]
    const T* __restrict__ B,           // [B, L, N]
    const T* __restrict__ C,           // [B, L, N]
    const T* __restrict__ D_skip,      // [D] or null
    const T* __restrict__ z,           // [B, L, D] or null
    const T* __restrict__ delta_bias,  // [D] or null
    const T* __restrict__ dy,          // [B, L, D]
    const T* __restrict__ states,      // [B, D, L, N] (T)
    T* __restrict__ du,                // [B, L, D]
    T* __restrict__ dB,                // [B, L, N]    atomicAdd over d
    T* __restrict__ dC,                // [B, L, N]    atomicAdd over d
    T* __restrict__ ddelta,            // [B, L, D]
    T* __restrict__ dA_out,            // [D, N]       atomicAdd over (b, t)
    T* __restrict__ dD_out,            // [D] or null  atomicAdd over (b, t)
    T* __restrict__ dz,                // [B, L, D] or null
    T* __restrict__ ddelta_bias_out,   // [D] or null  atomicAdd over (b, t)
    int32_t batch, int32_t seqlen,
    int32_t dim, int32_t dstate,
    int32_t delta_softplus_flag)
{
    const int32_t b = blockIdx.x;
    const int32_t d = blockIdx.y;
    if (b >= batch || d >= dim) return;

    extern __shared__ float smem[];
    float* dh = smem;                                // N floats
    float* reduce_scratch = smem + dstate;           // blockDim.x floats

    const int32_t tid = threadIdx.x;
    const int32_t tcount = blockDim.x;

    for (int32_t n = tid; n < dstate; n += tcount) dh[n] = 0.0f;
    __syncthreads();

    const int64_t ud_bd_off = (int64_t)b * seqlen * dim + d;
    const int64_t bc_b_off  = (int64_t)b * seqlen * dstate;
    const int64_t ud_step   = (int64_t)dim;
    const int64_t bc_step   = (int64_t)dstate;
    const int64_t A_d_off   = (int64_t)d * dstate;
    const int64_t st_bd_off = ((int64_t)b * dim + d) * (int64_t)seqlen * dstate;

    const float db_d = (delta_bias != nullptr) ? load_as_f32<T>(delta_bias[d]) : 0.0f;
    const float D_d  = (D_skip != nullptr)     ? load_as_f32<T>(D_skip[d])     : 0.0f;

    for (int32_t t = seqlen - 1; t >= 0; --t) {
        // Effective delta (with optional bias + softplus).
        float delta_raw = load_as_f32<T>(delta[ud_bd_off + (int64_t)t * ud_step]) + db_d;
        float delta_t;
        float dsp_factor = 1.0f;   // d(delta_eff) / d(delta_raw)
        if (delta_softplus_flag != 0) {
            if (delta_raw <= 20.0f) {
                delta_t = __logf(1.0f + __expf(delta_raw));
                // sigmoid(delta_raw) — softplus derivative.
                dsp_factor = 1.0f / (1.0f + __expf(-delta_raw));
            } else {
                delta_t = delta_raw;
                dsp_factor = 1.0f;
            }
        } else {
            delta_t = delta_raw;
        }

        const float u_t = load_as_f32<T>(u[ud_bd_off + (int64_t)t * ud_step]);
        const float dy_t = load_as_f32<T>(dy[ud_bd_off + (int64_t)t * ud_step]);
        const int64_t bc_t_off = bc_b_off + (int64_t)t * bc_step;

        // Pull h[t-1, :] from the recorded states (or zero at t=0).
        // h[t, :] is also at states[t]; we need h[t-1] for the dA / ddelta math.
        // (We *could* recompute via reverse-difference, but the saved states
        // make this a single read.)

        // Step 1: handle the y -> y_state chain through z gating and D skip.
        float dy_state;
        float dz_partial = 0.0f;
        if (z != nullptr) {
            // y[t, d] = y_pre_gate * silu(z)
            // y_pre_gate = y_state + (D_d * u_t if D_skip)
            // Recompute y_pre_gate (cheap: a length-N dot).
            float y_pre_partial = 0.0f;
            for (int32_t n = tid; n < dstate; n += tcount) {
                const float h_tn = load_as_f32<T>(states[st_bd_off + (int64_t)t * dstate + n]);
                const float C_n  = load_as_f32<T>(C[bc_t_off + n]);
                y_pre_partial += h_tn * C_n;
            }
            __syncthreads();
            float y_pre_gate = block_sum(
                (tid < dstate) ? y_pre_partial : 0.0f,
                reduce_scratch, tid, dstate, tcount);
            if (D_skip != nullptr) y_pre_gate += D_d * u_t;
            // silu and its derivative.
            const float z_t = load_as_f32<T>(z[ud_bd_off + (int64_t)t * ud_step]);
            const float sig = 1.0f / (1.0f + __expf(-z_t));
            const float silu_z = z_t * sig;
            // d(silu)/dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
            const float silu_p = sig * (1.0f + z_t * (1.0f - sig));
            dy_state = dy_t * silu_z;
            dz_partial = dy_t * y_pre_gate * silu_p;
        } else {
            dy_state = dy_t;
        }

        // Write dz[t, d] (one writer per (b, t, d), no atomic needed).
        if (tid == 0 && dz != nullptr) {
            dz[ud_bd_off + (int64_t)t * ud_step] = store_from_f32<T>(dz_partial);
        }

        // Step 2: dh[n] += dy_state * C[t, n]; dC[t, n] = dy_state * h[t, n]
        //         (dC is atomicAdd over d).
        for (int32_t n = tid; n < dstate; n += tcount) {
            const float C_n  = load_as_f32<T>(C[bc_t_off + n]);
            const float h_tn = load_as_f32<T>(states[st_bd_off + (int64_t)t * dstate + n]);
            dh[n] += dy_state * C_n;
            atomic_add_T<T>(dC + (bc_t_off + n), dy_state * h_tn);
        }
        __syncthreads();

        // Step 3: emit dA, dB, du, ddelta contributions for time t.
        //   h_prev[n] = states[t-1, n] (or 0 at t=0)
        //   dA_dn = exp(delta_t * A[d, n])
        //   dA[d, n] += delta_t * h_prev[n] * dA_dn * dh[n]    (atomicAdd)
        //   dB[t, n] += delta_t * u_t * dh[n]                  (atomicAdd over d)
        //   du_contrib = delta_t * Σ_n B[t, n] * dh[n]
        //   ddelta_contrib = Σ_n [ B[t, n] * u_t * dh[n]
        //                          + A[d, n] * dA_dn * h_prev[n] * dh[n] ]
        //   Then propagate: dh[n] = dh[n] * dA_dn

        float du_partial = 0.0f;
        float ddelta_partial = 0.0f;

        for (int32_t n = tid; n < dstate; n += tcount) {
            const float A_dn = load_as_f32<T>(A[A_d_off + n]);
            const float B_n  = load_as_f32<T>(B[bc_t_off + n]);
            const float h_prev = (t > 0)
                ? load_as_f32<T>(states[st_bd_off + (int64_t)(t - 1) * dstate + n])
                : 0.0f;
            const float dA_dn = __expf(delta_t * A_dn);
            const float dh_n = dh[n];

            // dA[d, n] += delta_t * h_prev * dA_dn * dh_n
            atomic_add_T<T>(dA_out + (A_d_off + n),
                            delta_t * h_prev * dA_dn * dh_n);

            // dB[t, n] += delta_t * u_t * dh_n
            atomic_add_T<T>(dB + (bc_t_off + n), delta_t * u_t * dh_n);

            // du partial: delta_t * B_n * dh_n
            du_partial += B_n * dh_n;
            // ddelta partial.
            ddelta_partial += B_n * u_t * dh_n + A_dn * dA_dn * h_prev * dh_n;

            // Propagate dh for previous time step.
            dh[n] = dh_n * dA_dn;
        }
        __syncthreads();

        const float du_state = block_sum(
            (tid < dstate) ? du_partial : 0.0f,
            reduce_scratch, tid, dstate, tcount);
        const float ddelta_state = block_sum(
            (tid < dstate) ? ddelta_partial : 0.0f,
            reduce_scratch, tid, dstate, tcount);

        if (tid == 0) {
            float du_val = delta_t * du_state;
            if (D_skip != nullptr) {
                du_val += D_d * dy_t;
                atomic_add_T<T>(dD_out + d, u_t * dy_t);
            }
            du[ud_bd_off + (int64_t)t * ud_step] = store_from_f32<T>(du_val);

            // Apply softplus chain rule then write ddelta + accumulate
            // ddelta_bias (if given).
            const float ddelta_pre = ddelta_state * dsp_factor;
            ddelta[ud_bd_off + (int64_t)t * ud_step] = store_from_f32<T>(ddelta_pre);
            if (ddelta_bias_out != nullptr) {
                atomic_add_T<T>(ddelta_bias_out + d, ddelta_pre);
            }
        }
        __syncthreads();
    }
}

template <typename T>
int32_t launch_selective_scan_bw(
    const T* u, const T* delta, const T* A, const T* B, const T* C,
    const T* D_skip, const T* z, const T* delta_bias,
    const T* dy,
    T* du, T* dB, T* dC, T* ddelta,
    T* dA_out, T* dD_out, T* dz, T* ddelta_bias_out,
    void* workspace, size_t workspace_bytes,
    int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate,
    int32_t delta_softplus_flag,
    cudaStream_t stream)
{
    if (batch == 0 || seqlen == 0 || dim == 0 || dstate == 0) return 0;
    if (dstate > 256) return 3;
    if (batch > 65535 || dim > 65535) return 3;

    const size_t need = (size_t)batch * dim * seqlen * dstate * sizeof(T);
    if (workspace_bytes < need || workspace == nullptr) return 4;
    T* states = static_cast<T*>(workspace);

    const int32_t threads_per_block = 64;
    dim3 grid((unsigned)batch, (unsigned)dim);
    dim3 block(threads_per_block);

    const size_t smem_pass1 = (size_t)dstate * sizeof(float);
    const size_t smem_pass2 = (size_t)(dstate + threads_per_block) * sizeof(float);
    if (smem_pass2 > 48 * 1024) return 3;

    selective_scan_bw_pass1_record<T><<<grid, block, smem_pass1, stream>>>(
        u, delta, A, B, delta_bias, states,
        batch, seqlen, dim, dstate, delta_softplus_flag);

    selective_scan_bw_pass2<T><<<grid, block, smem_pass2, stream>>>(
        u, delta, A, B, C, D_skip, z, delta_bias, dy, states,
        du, dB, dC, ddelta, dA_out, dD_out, dz, ddelta_bias_out,
        batch, seqlen, dim, dstate, delta_softplus_flag);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::selective_scan_bw

// Workspace size helper. `dtype_id`: 0 = f32, 1 = f16, 2 = bf16.
extern "C" size_t baracuda_kernels_selective_scan_workspace_bytes(
    int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate,
    int32_t dtype_id)
{
    if (batch <= 0 || seqlen <= 0 || dim <= 0 || dstate <= 0) return 0;
    size_t elem_size;
    switch (dtype_id) {
        case 0: elem_size = sizeof(float); break;
        case 1: elem_size = 2; break;
        case 2: elem_size = 2; break;
        default: return 0;
    }
    return (size_t)batch * dim * seqlen * dstate * elem_size;
}

#define BARACUDA_SELECTIVE_SCAN_BWD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_backward_run(                                 \
        int32_t batch, int32_t seqlen, int32_t dim, int32_t dstate,                            \
        int32_t delta_softplus,                                                                \
        const void* u, const void* delta, const void* A,                                       \
        const void* B, const void* C,                                                          \
        const void* D_skip, const void* z, const void* delta_bias,                             \
        const void* dy,                                                                        \
        void* du, void* dB, void* dC, void* ddelta,                                            \
        void* dA, void* dD, void* dz, void* ddelta_bias,                                       \
        void* workspace, size_t workspace_bytes,                                               \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (batch < 0 || seqlen < 0 || dim < 0 || dstate < 0) return 2;                        \
        if (dstate > 256) return 3;                                                            \
        if (batch == 0 || seqlen == 0 || dim == 0) return 0;                                   \
        if (u == nullptr || delta == nullptr || A == nullptr) return 2;                        \
        if (B == nullptr || C == nullptr || dy == nullptr) return 2;                           \
        if (du == nullptr || dB == nullptr || dC == nullptr) return 2;                         \
        if (ddelta == nullptr || dA == nullptr) return 2;                                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::selective_scan_bw::launch_selective_scan_bw<T>(                       \
            static_cast<const T*>(u), static_cast<const T*>(delta),                            \
            static_cast<const T*>(A), static_cast<const T*>(B),                                \
            static_cast<const T*>(C),                                                          \
            static_cast<const T*>(D_skip), static_cast<const T*>(z),                           \
            static_cast<const T*>(delta_bias),                                                 \
            static_cast<const T*>(dy),                                                         \
            static_cast<T*>(du), static_cast<T*>(dB), static_cast<T*>(dC),                     \
            static_cast<T*>(ddelta),                                                           \
            static_cast<T*>(dA), static_cast<T*>(dD),                                          \
            static_cast<T*>(dz), static_cast<T*>(ddelta_bias),                                 \
            workspace, workspace_bytes,                                                        \
            batch, seqlen, dim, dstate, delta_softplus, stream);                               \
    }

BARACUDA_SELECTIVE_SCAN_BWD_INSTANTIATE(selective_scan_f32,  float)
BARACUDA_SELECTIVE_SCAN_BWD_INSTANTIATE(selective_scan_f16,  __half)
BARACUDA_SELECTIVE_SCAN_BWD_INSTANTIATE(selective_scan_bf16, __nv_bfloat16)
