// baracuda_ctc.cuh
//
// CTCLoss kernels (Milestone 5.5 — Phase 5 final deferral).
//
// FW: per-sample forward dynamic programming over the CTC lattice using
// the log-sum-exp running-max trick. Each block handles one batch sample;
// each thread handles one position k in the extended target sequence
// (length L = 2·S + 1). Per time step the threads cooperate via shared
// memory to apply the recurrence
//
//   α[t, k] = log_probs[t, n, ext_target[k]]
//           + lse( α[t-1, k],
//                  α[t-1, k-1] if k >= 1,
//                  α[t-1, k-2] if k >= 2 AND ext_target[k] != blank
//                              AND ext_target[k] != ext_target[k-2] )
//
// Final loss for sample n is
//   loss[n] = -lse( α[T_n - 1, L_n - 1], α[T_n - 1, L_n - 2] )
//
// We persist α to a workspace tensor (shape [T, N, L_max]) for BW reuse.
// For Mean reduction we additionally save per-sample losses to a scratch
// `[N]` buffer and reduce to scalar with a single-block tree reduction
// (denominator = Σ target_lengths[n], i.e. the standard CTC mean
// convention).
//
// BW: re-running β over the lattice in reverse, then per-class gradient
//
//   dlog_probs[t, n, c] = (exp(log_probs[t, n, c])
//                          - (1/exp(loss[n])) · Σ_{k: ext_t[k]==c} exp(α + β))
//                         · scale
//
// where `scale = dloss / N` for Mean, `dloss` for Sum, `dloss[n]` for None.
//
// For numerical stability all DP math is f32 (or native f64 when T=double);
// f16 / bf16 load/store cast at the buffer boundary.
//
// Caps (trailblazer):
//   - max_target_len S ≤ 256  (so L = 2S+1 ≤ 513 fits in 512-thread block
//     when L_max ≤ 512, OR a 1024-thread block when L_max ≤ 1024)
//   - num_classes C ≤ 32
//
// Status codes (consistent with the rest of the loss family):
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_CTC_CUH
#define BARACUDA_CTC_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_loss.cuh" // for kBlockReduce and loss_reduce_finalize_kernel

namespace baracuda { namespace ctc {

// -----------------------------------------------------------------------------
// Numeric helpers
// -----------------------------------------------------------------------------

static __device__ __forceinline__ float lse2_f(float a, float b) {
    // numerically stable log-sum-exp of two values; treats -INF as identity.
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    float m = (a > b) ? a : b;
    return m + logf(expf(a - m) + expf(b - m));
}

static __device__ __forceinline__ float lse3_f(float a, float b, float c) {
    float m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (m == -INFINITY) return -INFINITY;
    float sum = 0.0f;
    if (a != -INFINITY) sum += expf(a - m);
    if (b != -INFINITY) sum += expf(b - m);
    if (c != -INFINITY) sum += expf(c - m);
    return m + logf(sum);
}

static __device__ __forceinline__ double lse2_d(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    double m = (a > b) ? a : b;
    return m + log(exp(a - m) + exp(b - m));
}

static __device__ __forceinline__ double lse3_d(double a, double b, double c) {
    double m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (m == -INFINITY) return -INFINITY;
    double sum = 0.0;
    if (a != -INFINITY) sum += exp(a - m);
    if (b != -INFINITY) sum += exp(b - m);
    if (c != -INFINITY) sum += exp(c - m);
    return m + log(sum);
}

// -----------------------------------------------------------------------------
// FW kernel (f32 accumulator path — covers f32 + f16 + bf16).
// One block per batch sample; blockDim.x ≥ L_max = 2·max_target_len + 1.
// -----------------------------------------------------------------------------
//
// Layout assumptions:
//   - log_probs: contiguous [T, N, C], row-major
//   - targets:   contiguous [N, S], row-major (S = max_target_len)
//   - alpha:     contiguous [T, N, L_max] workspace
//   - per_sample_loss: [N] buffer (also written to `out` directly for None mode
//     via the launcher; this kernel only writes to per_sample_loss)
//
// `zero_infinity`: when set, samples with `T_n < 2·L_n - 1` (which makes
// the lattice unreachable; CTC requires at least one blank between
// repeated labels) get loss = 0 and ignore α for BW.
//
// `loss_out`: also receives the per-sample loss for None reduction
// (and the launcher then ignores per_sample_loss).

template <typename T>
__global__ void ctc_forward_kernel_f32(
    const T* __restrict__ log_probs,
    const int64_t* __restrict__ targets,
    const int64_t* __restrict__ input_lengths,
    const int64_t* __restrict__ target_lengths,
    float* __restrict__ alpha,        // workspace [T, N, L_max], f32
    float* __restrict__ per_sample_loss, // [N] f32
    int32_t max_time,
    int32_t batch_size,
    int32_t num_classes,
    int32_t max_target_len,
    int32_t blank,
    int32_t zero_infinity)
{
    int n = blockIdx.x;
    if (n >= batch_size) return;

    int Tn = (int)input_lengths[n];
    int Sn = (int)target_lengths[n];
    int Ln = 2 * Sn + 1;
    int Lmax = 2 * max_target_len + 1;
    int tid = threadIdx.x;

    // Out-of-range threads still need to participate in __syncthreads.
    bool active = (tid < Ln);

    // Quick exits: empty sample (T_n == 0). Loss is 0 if S_n == 0, else
    // -INF (impossible). With zero_infinity we clamp -INF to 0.
    if (Tn <= 0) {
        if (tid == 0) {
            float v = (Sn == 0) ? 0.0f : (zero_infinity ? 0.0f : INFINITY);
            per_sample_loss[n] = v;
        }
        return;
    }

    // CTC requires T_n ≥ S_n + (#repeated_label_pairs). The classic sufficient
    // bound is T_n ≥ Ln - (number of non-repeat skips available). A simple
    // safe lower bound: T_n ≥ Sn (you need at least one timestep per label).
    // Tighter unreachability is handled implicitly: if path is unreachable,
    // α[Tn-1, Ln-1] = α[Tn-1, Ln-2] = -INF → loss = +INF, which we clamp to 0
    // when zero_infinity is set.

    // Index into the extended target sequence: position k in [0, Ln).
    // ext_t[k] = blank when k even, targets[n, k/2] when k odd.
    auto ext_t = [&](int k) -> int {
        if ((k & 1) == 0) return blank;
        int64_t lbl = targets[(int64_t)n * (int64_t)max_target_len + (int64_t)(k >> 1)];
        return (int)lbl;
    };

    int my_class = active ? ext_t(tid) : 0;

    // Alpha base pointer for this sample: alpha[t, n, *] strided as t*N*Lmax + n*Lmax.
    // Strides are computed in the launcher; here we just walk t.
    float* A = alpha + (int64_t)n * (int64_t)Lmax; // alpha[0, n, *]
    int64_t stride_t = (int64_t)batch_size * (int64_t)Lmax;

    // Initialize α[0, k]. CTC initialization: only k=0 (blank) and k=1
    // (first label) are reachable.
    float a_prev;
    if (active) {
        if (tid == 0) {
            // α[0, 0] = log P(blank at t=0)
            int c = my_class;
            float v = (float)log_probs[(int64_t)0 * (int64_t)batch_size * (int64_t)num_classes
                                       + (int64_t)n * (int64_t)num_classes
                                       + (int64_t)c];
            a_prev = v;
        } else if (tid == 1 && Sn >= 1) {
            int c = my_class;
            float v = (float)log_probs[(int64_t)0 * (int64_t)batch_size * (int64_t)num_classes
                                       + (int64_t)n * (int64_t)num_classes
                                       + (int64_t)c];
            a_prev = v;
        } else {
            a_prev = -INFINITY;
        }
        A[tid] = a_prev;
    } else {
        a_prev = -INFINITY;
    }
    __syncthreads();

    // Recurrence for t = 1..Tn-1.
    for (int t = 1; t < Tn; ++t) {
        float* A_prev_row = A + (int64_t)(t - 1) * stride_t;
        float* A_curr_row = A + (int64_t)t       * stride_t;

        float new_val = -INFINITY;
        if (active) {
            float a_k   = A_prev_row[tid];
            float a_km1 = (tid >= 1) ? A_prev_row[tid - 1] : -INFINITY;
            float a_km2 = -INFINITY;
            if (tid >= 2) {
                int curr_c = my_class;
                int km2_c  = ext_t(tid - 2);
                if (curr_c != blank && curr_c != km2_c) {
                    a_km2 = A_prev_row[tid - 2];
                }
            }
            float lse_term = lse3_f(a_k, a_km1, a_km2);
            int c = my_class;
            float emit = (float)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                          + (int64_t)n * (int64_t)num_classes
                                          + (int64_t)c];
            new_val = lse_term + emit;
            A_curr_row[tid] = new_val;
        }
        __syncthreads();
    }

    // Final loss: lse(α[Tn-1, Ln-1], α[Tn-1, Ln-2]).
    // Tn >= 1 here.
    if (tid == 0) {
        float* A_last = A + (int64_t)(Tn - 1) * stride_t;
        float v;
        if (Sn == 0) {
            // Only the blank lattice cell (k=0) terminates a zero-length target.
            v = A_last[0];
        } else if (Ln >= 2) {
            float a_end1 = A_last[Ln - 1];
            float a_end2 = A_last[Ln - 2];
            v = lse2_f(a_end1, a_end2);
        } else {
            v = -INFINITY;
        }
        float loss_val = -v; // CTC loss is the negative log-likelihood.
        if (zero_infinity && (isinf(loss_val) || isnan(loss_val))) {
            loss_val = 0.0f;
        }
        per_sample_loss[n] = loss_val;
    }
}

// f64 specialization
template <typename T>
__global__ void ctc_forward_kernel_f64(
    const T* __restrict__ log_probs,
    const int64_t* __restrict__ targets,
    const int64_t* __restrict__ input_lengths,
    const int64_t* __restrict__ target_lengths,
    double* __restrict__ alpha,
    double* __restrict__ per_sample_loss,
    int32_t max_time,
    int32_t batch_size,
    int32_t num_classes,
    int32_t max_target_len,
    int32_t blank,
    int32_t zero_infinity)
{
    int n = blockIdx.x;
    if (n >= batch_size) return;

    int Tn = (int)input_lengths[n];
    int Sn = (int)target_lengths[n];
    int Ln = 2 * Sn + 1;
    int Lmax = 2 * max_target_len + 1;
    int tid = threadIdx.x;

    bool active = (tid < Ln);

    if (Tn <= 0) {
        if (tid == 0) {
            double v = (Sn == 0) ? 0.0 : (zero_infinity ? 0.0 : INFINITY);
            per_sample_loss[n] = v;
        }
        return;
    }

    auto ext_t = [&](int k) -> int {
        if ((k & 1) == 0) return blank;
        int64_t lbl = targets[(int64_t)n * (int64_t)max_target_len + (int64_t)(k >> 1)];
        return (int)lbl;
    };

    int my_class = active ? ext_t(tid) : 0;
    double* A = alpha + (int64_t)n * (int64_t)Lmax;
    int64_t stride_t = (int64_t)batch_size * (int64_t)Lmax;

    if (active) {
        double a_prev;
        if (tid == 0) {
            int c = my_class;
            a_prev = (double)log_probs[(int64_t)0 * (int64_t)batch_size * (int64_t)num_classes
                                       + (int64_t)n * (int64_t)num_classes
                                       + (int64_t)c];
        } else if (tid == 1 && Sn >= 1) {
            int c = my_class;
            a_prev = (double)log_probs[(int64_t)0 * (int64_t)batch_size * (int64_t)num_classes
                                       + (int64_t)n * (int64_t)num_classes
                                       + (int64_t)c];
        } else {
            a_prev = -INFINITY;
        }
        A[tid] = a_prev;
    }
    __syncthreads();

    for (int t = 1; t < Tn; ++t) {
        double* A_prev_row = A + (int64_t)(t - 1) * stride_t;
        double* A_curr_row = A + (int64_t)t       * stride_t;

        if (active) {
            double a_k   = A_prev_row[tid];
            double a_km1 = (tid >= 1) ? A_prev_row[tid - 1] : -INFINITY;
            double a_km2 = -INFINITY;
            if (tid >= 2) {
                int curr_c = my_class;
                int km2_c  = ext_t(tid - 2);
                if (curr_c != blank && curr_c != km2_c) {
                    a_km2 = A_prev_row[tid - 2];
                }
            }
            double lse_term = lse3_d(a_k, a_km1, a_km2);
            int c = my_class;
            double emit = (double)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                            + (int64_t)n * (int64_t)num_classes
                                            + (int64_t)c];
            A_curr_row[tid] = lse_term + emit;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double* A_last = A + (int64_t)(Tn - 1) * stride_t;
        double v;
        if (Sn == 0) {
            v = A_last[0];
        } else if (Ln >= 2) {
            double a_end1 = A_last[Ln - 1];
            double a_end2 = A_last[Ln - 2];
            v = lse2_d(a_end1, a_end2);
        } else {
            v = -INFINITY;
        }
        double loss_val = -v;
        if (zero_infinity && (isinf(loss_val) || isnan(loss_val))) {
            loss_val = 0.0;
        }
        per_sample_loss[n] = loss_val;
    }
}

// -----------------------------------------------------------------------------
// Cast per-sample f32 (or f64) loss buffer into the T-typed output buffer.
// Used for None reduction.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void ctc_cast_loss_kernel(
    const float* __restrict__ src_f32,
    T* __restrict__ dst,
    int32_t batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    dst[i] = baracuda::loss::store_from_acc<T>(src_f32[i]);
}

template <typename T>
__global__ void ctc_cast_loss_kernel_d(
    const double* __restrict__ src_f64,
    T* __restrict__ dst,
    int32_t batch_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;
    dst[i] = (T)src_f64[i];
}

// -----------------------------------------------------------------------------
// Reduce per-sample loss buffer to scalar.
//   reduction_mode 1 (Mean): scalar = Σ loss[n] / max(1, Σ target_lengths[n])
//   reduction_mode 2 (Sum):  scalar = Σ loss[n]
// Single-block tree reduction (kBlockReduce threads).
// -----------------------------------------------------------------------------
template <typename T>
__global__ void ctc_reduce_loss_kernel(
    const float* __restrict__ per_sample_loss, // f32
    const int64_t* __restrict__ target_lengths,
    T* __restrict__ out,
    int32_t batch_size,
    int32_t reduction_mode)
{
    constexpr int B = baracuda::loss::kBlockReduce;
    __shared__ float smem[B];
    __shared__ int64_t tl_smem[B];
    int tid = threadIdx.x;
    float acc = 0.0f;
    int64_t tl_acc = 0;
    for (int i = tid; i < batch_size; i += B) {
        acc += per_sample_loss[i];
        tl_acc += target_lengths[i];
    }
    smem[tid] = acc;
    tl_smem[tid] = tl_acc;
    __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            tl_smem[tid] += tl_smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float total = smem[0];
        float final_val;
        if (reduction_mode == 1) {
            int64_t denom = tl_smem[0];
            if (denom <= 0) denom = 1;
            final_val = total / (float)denom;
        } else {
            final_val = total;
        }
        out[0] = baracuda::loss::store_from_acc<T>(final_val);
    }
}

template <typename T>
__global__ void ctc_reduce_loss_kernel_d(
    const double* __restrict__ per_sample_loss,
    const int64_t* __restrict__ target_lengths,
    T* __restrict__ out,
    int32_t batch_size,
    int32_t reduction_mode)
{
    constexpr int B = baracuda::loss::kBlockReduce;
    __shared__ double smem[B];
    __shared__ int64_t tl_smem[B];
    int tid = threadIdx.x;
    double acc = 0.0;
    int64_t tl_acc = 0;
    for (int i = tid; i < batch_size; i += B) {
        acc += per_sample_loss[i];
        tl_acc += target_lengths[i];
    }
    smem[tid] = acc;
    tl_smem[tid] = tl_acc;
    __syncthreads();
    for (int s = B / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            tl_smem[tid] += tl_smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        double total = smem[0];
        double final_val;
        if (reduction_mode == 1) {
            int64_t denom = tl_smem[0];
            if (denom <= 0) denom = 1;
            final_val = total / (double)denom;
        } else {
            final_val = total;
        }
        out[0] = (T)final_val;
    }
}

// =============================================================================
// BW kernel.
//
// Strategy: run β (backward DP) inside the kernel, combine with the saved
// α (from FW) to compute the CTC gradient.
//
//   β[Tn-1, Ln-1] = log P(emit ext_t[Ln-1] at t=Tn-1) = 0? no — we use the
//   convention β[Tn-1, Ln-1] = log_probs[Tn-1, n, ext_t[Ln-1]] and
//   β[Tn-1, Ln-2] = log_probs[Tn-1, n, ext_t[Ln-2]] (these are the terminal
//   states). All other β[Tn-1, *] = -INF.
//
// Actually we'll use the standard PyTorch convention where
//   β[Tn-1, k] = log_probs[Tn-1, n, ext_t[k]]  if k ∈ {Ln-1, Ln-2}
//              = -INF                          otherwise
//   β[t, k] = log_probs[t, n, ext_t[k]] + lse( β[t+1, k],
//                                              β[t+1, k+1] if k+1 < Ln,
//                                              β[t+1, k+2] if k+2 < Ln AND
//                                                ext_t[k] != blank AND
//                                                ext_t[k] != ext_t[k+2] )
//
// Then the gradient is
//   dlog_probs[t, n, c] =  exp(log_probs[t, n, c]) · scale_n
//                       -  scale_n · Σ_{k: ext_t[k]==c} exp(α[t,k]+β[t,k]−log_probs[t,n,c]−nll[n])
//
// Wait — the standard CTC gradient is
//
//   ∂L / ∂logits[t, n, c]   (where logits → log_probs via log_softmax) is
//   exp(log_probs[t,n,c]) − γ[t, n, c]
//
// where γ[t,n,c] = (1 / P) · Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k])
// and  P = exp(-nll[n]) = exp(α[Tn-1, Ln-1] + α[Tn-1, Ln-2] summed).
//
// But PyTorch's CTCLoss FW takes `log_probs` directly (NOT logits) and
// computes the gradient *of the loss w.r.t. log_probs*:
//   ∂L / ∂log_probs[t,n,c] = exp(log_probs[t,n,c]) − γ[t,n,c]
//
// The factor scale_n = dloss · (mean_factor) is applied. For None reduction
// scale_n = dloss[n]; for Mean scale_n = dloss[0] / Σ target_lengths; for
// Sum scale_n = dloss[0].
//
// However in PyTorch's convention there's an extra subtle factor — the
// derivative w.r.t. log_probs is just exp(log_probs) − γ (NOT divided by N
// in the per-sample sense). The Mean/Sum reduction's chain rule applies the
// scalar dloss factor.
//
// For samples where zero_infinity is set and the FW loss was clamped to 0,
// we must also zero out the gradient.
// =============================================================================

template <typename T>
__global__ void ctc_backward_kernel_f32(
    const T* __restrict__ log_probs,
    const int64_t* __restrict__ targets,
    const int64_t* __restrict__ input_lengths,
    const int64_t* __restrict__ target_lengths,
    const float* __restrict__ alpha,         // [T, N, L_max]
    const float* __restrict__ per_sample_loss, // [N] f32 (positive nll values)
    const T* __restrict__ dloss,             // [1] or [N]
    T* __restrict__ dlog_probs,              // [T, N, C]
    float* __restrict__ beta_ws,             // workspace [N, L_max] (running β; persistent across t)
    float* __restrict__ neg_gamma_ws,        // workspace [N, C] (per-sample, per-class accum)
    int32_t max_time,
    int32_t batch_size,
    int32_t num_classes,
    int32_t max_target_len,
    int32_t blank,
    int32_t reduction_mode,                  // 0/1/2
    float inv_denom,                         // 1.0 for None/Sum; 1/Σtargets for Mean
    int32_t zero_infinity)
{
    int n = blockIdx.x;
    if (n >= batch_size) return;

    int Tn = (int)input_lengths[n];
    int Sn = (int)target_lengths[n];
    int Ln = 2 * Sn + 1;
    int Lmax = 2 * max_target_len + 1;
    int tid = threadIdx.x;
    bool active = (tid < Ln);

    // Compute scale_n for this sample.
    float dloss_val;
    if (reduction_mode == 0) {
        dloss_val = (float)dloss[n];
    } else {
        dloss_val = (float)dloss[0];
    }
    float scale_n = dloss_val * inv_denom; // inv_denom: 1/N for Mean, 1 for Sum, 1 for None

    // Check zero_infinity gating.
    float fw_loss = per_sample_loss[n];
    bool zero_grad = (zero_infinity != 0) && (fw_loss == 0.0f) && (Tn > 0)
                     && (Sn > 0); // FW clamps unreachable losses to 0; here we treat them as zero-grad
    // Also, if Tn <= 0 or Sn > some unreachable threshold, gradient is undefined → zero it.
    if (Tn <= 0) {
        // Zero dlog_probs for this sample across all (t, c).
        for (int idx = tid; idx < num_classes; idx += blockDim.x) {
            // No timesteps for this sample — nothing to write.
            (void)idx;
        }
        return;
    }

    // ext_target lookup
    auto ext_t = [&](int k) -> int {
        if ((k & 1) == 0) return blank;
        int64_t lbl = targets[(int64_t)n * (int64_t)max_target_len + (int64_t)(k >> 1)];
        return (int)lbl;
    };
    int my_class = active ? ext_t(tid) : 0;

    // Alpha base pointer
    const float* A = alpha + (int64_t)n * (int64_t)Lmax;
    int64_t stride_t = (int64_t)batch_size * (int64_t)Lmax;

    // Beta workspace base pointer (one β row per sample, reused across t)
    float* B = beta_ws + (int64_t)n * (int64_t)Lmax;

    // The NLL value for this sample (positive). P = exp(-nll). When we compute
    // (α + β) - logP_total = α + β + nll. So the formula becomes:
    //   γ[t, n, c] = Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k] + nll[n] - log_probs[t,n,c])
    // Wait, but the standard derivation gives:
    //   γ[t,c] = (1/P) · Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k])
    // where β[t,k] is defined such that α[t,k] + β[t,k] = log P(path through k at t) + log_probs[t, ext_t[k]].
    //
    // Common PyTorch convention: β[t,k] = lse(β[t+1,...]) + log_probs[t, ext_t[k]]
    // and  α[t,k] + β[t,k] = log P(any path going through state k at time t) + log_probs[t, ext_t[k]]
    // and  γ[t,c] = (1/(P · exp(log_probs[t,c]))) · Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k])
    //             = exp(- log_probs[t,c]) / P · Σ_{k...} exp(α[t,k] + β[t,k])
    //
    // Then ∂L/∂log_probs[t,n,c] = exp(log_probs[t,n,c]) − γ[t,c]    (the standard formula).
    //
    // BUT a simpler/more common form is achieved by defining β so that α[t,k] + β[t,k] = log P(through k at t) (no log_probs factor), but then we need to divide by exp(log_probs[t, ext_t[k]]) when summing γ. We'll use that.
    //
    // To keep matching PyTorch behavior, we'll use the convention from Graves 2006:
    //   β[Tn-1, Ln-1] = log_probs[Tn-1, n, ext_t[Ln-1]]
    //   β[Tn-1, Ln-2] = log_probs[Tn-1, n, ext_t[Ln-2]]
    //   β[t, k] = log_probs[t, n, ext_t[k]] + lse(β[t+1,k], β[t+1,k+1], β[t+1,k+2]')
    // where the k+2 transition is forbidden if ext_t[k] == blank OR ext_t[k] == ext_t[k+2].
    //
    // Then  γ[t, c] = (1/P) · Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k] - log_probs[t,n,c])
    //               = exp(-nll[n] - log_probs[t,n,c]) · Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k])
    // And ∂L/∂log_probs[t,n,c] = exp(log_probs[t,n,c]) − γ[t,c].

    // Initialize β for the last timestep (Tn-1).
    if (active) {
        float v = -INFINITY;
        if (Sn == 0) {
            // Only k=0 (blank) is terminal for an empty target.
            if (tid == 0) {
                int c = my_class;
                v = (float)log_probs[(int64_t)(Tn - 1) * (int64_t)batch_size * (int64_t)num_classes
                                     + (int64_t)n * (int64_t)num_classes
                                     + (int64_t)c];
            }
        } else if (tid == Ln - 1 || tid == Ln - 2) {
            int c = my_class;
            v = (float)log_probs[(int64_t)(Tn - 1) * (int64_t)batch_size * (int64_t)num_classes
                                 + (int64_t)n * (int64_t)num_classes
                                 + (int64_t)c];
        }
        B[tid] = v;
    }
    __syncthreads();

    // For each (t, c), we need to accumulate γ contributions. We'll process
    // time steps from Tn-1 down to 0. At each t, after β is computed:
    //   neg_gamma_ws[n, c] = Σ_{k: ext_t[k]==c} exp(α[t,k] + β[t,k] - log_probs[t,n,c]) · exp(-nll[n])
    // To do this with parallelism but without atomic adds, we observe num_classes
    // is small (≤ 32). Each thread reads its own (k, my_class) and contributes
    // to one bucket in shared mem; we use atomicAdd in shared mem (small race
    // domain — fast on Ampere/Ada). Actually let's just use shared-mem
    // serial accumulation: thread 0 walks through Ln contributions per (t, c).
    // num_classes ≤ 32, Ln ≤ 513 → 32 * 513 ≈ 16K ops sequentially per t.
    // That's expensive but fine for trailblazer (one block per sample).
    //
    // Simpler: each thread tid (active) contributes exp(α + β - logp) to a
    // shared-mem bucket my_class. Use shared-mem atomicAdd (fast on modern arches)
    // OR a serial reduction over thread-by-class.
    //
    // We'll use shared-mem atomicAdd here. CTC's small num_classes makes
    // shared-mem contention manageable.

    __shared__ float gamma_sm[32]; // up to num_classes ≤ 32

    for (int t = Tn - 1; t >= 0; --t) {
        // Compute γ[t, c] in shared memory using β (current values in B).
        // gamma_sm[c] = Σ_{k: ext_t[k]==c, k active} exp(α[t,k] + β[t,k] - log_probs[t,n,c]) · exp(-fw_loss)
        // Note: fw_loss = +nll. exp(-fw_loss) = 1/P.
        // But (α+β−log_probs) can be huge negative; exp() of that is safe.

        // Zero out gamma_sm.
        if (tid < num_classes) gamma_sm[tid] = 0.0f;
        __syncthreads();

        const float* A_t = A + (int64_t)t * stride_t;
        // Each active thread reads its (α, β, logp) and contributes.
        if (active) {
            float a_tk = A_t[tid];
            float b_tk = B[tid];
            int c = my_class;
            float logp_tnc = (float)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                              + (int64_t)n * (int64_t)num_classes
                                              + (int64_t)c];
            // contribution = exp(α + β - logp_tnc + fw_loss)
            //
            // fw_loss = -log P (positive: stored nll). 1/P = exp(nll) = exp(fw_loss).
            // γ_contribution per k = (1/P) · exp(α + β − logp)
            //                      = exp(fw_loss) · exp(α + β − logp)
            //                      = exp(α + β − logp + fw_loss)
            //
            // Earlier we had `- fw_loss` here which gave `(P · exp(α+β−logp))`,
            // i.e. multiplied by P instead of dividing — the γ scatter
            // collapsed to ~0 because per-cell P is tiny, so dlog_probs
            // returned exp(log_probs) (the un-corrected term). Fixed
            // 2026-05-16; finite-difference cross-check now matches.
            float contribution = -INFINITY;
            float arg = a_tk + b_tk - logp_tnc + fw_loss;
            if (a_tk != -INFINITY && b_tk != -INFINITY) {
                contribution = expf(arg);
            }
            if (contribution > 0.0f && contribution == contribution /*not NaN*/) {
                atomicAdd(&gamma_sm[c], contribution);
            }
        }
        __syncthreads();

        // Now write dlog_probs[t, n, c] = (exp(log_probs[t,n,c]) - gamma_sm[c]) * scale_n
        // for all c in parallel. We use threads 0..num_classes-1.
        if (tid < num_classes) {
            int c = tid;
            float logp_tnc = (float)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                              + (int64_t)n * (int64_t)num_classes
                                              + (int64_t)c];
            float p = expf(logp_tnc);
            float grad_raw = p - gamma_sm[c];
            float grad = zero_grad ? 0.0f : (grad_raw * scale_n);
            int64_t off = (int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                          + (int64_t)n * (int64_t)num_classes
                          + (int64_t)c;
            dlog_probs[off] = baracuda::loss::store_from_acc<T>(grad);
        }
        __syncthreads();

        // Update β for the previous timestep (t-1). Need to use logp at t = t (not t-1!),
        // since the recurrence is β[t-1, k] = log_probs[t-1, n, ext_t[k]] + lse(β[t,k]...).
        // But we're processing t descending; at the *end* of iteration t we
        // want to compute B = β[t-1] from current B = β[t].
        if (t > 0) {
            // Compute β[t-1, k] using current B (which is β[t, *]).
            float b_new = -INFINITY;
            if (active) {
                float b_k   = B[tid];
                float b_kp1 = (tid + 1 < Ln) ? B[tid + 1] : -INFINITY;
                float b_kp2 = -INFINITY;
                if (tid + 2 < Ln) {
                    int curr_c = my_class;
                    int kp2_c  = ext_t(tid + 2);
                    if (curr_c != blank && curr_c != kp2_c) {
                        b_kp2 = B[tid + 2];
                    }
                }
                float lse_term = lse3_f(b_k, b_kp1, b_kp2);
                int c = my_class;
                float emit_prev = (float)log_probs[(int64_t)(t - 1) * (int64_t)batch_size * (int64_t)num_classes
                                                   + (int64_t)n * (int64_t)num_classes
                                                   + (int64_t)c];
                b_new = lse_term + emit_prev;
            }
            __syncthreads();
            if (active) {
                B[tid] = b_new;
            }
            __syncthreads();
        }
    }
    (void)neg_gamma_ws; // unused — gamma_sm is shared
    (void)max_time;
}

// f64 BW
template <typename T>
__global__ void ctc_backward_kernel_f64(
    const T* __restrict__ log_probs,
    const int64_t* __restrict__ targets,
    const int64_t* __restrict__ input_lengths,
    const int64_t* __restrict__ target_lengths,
    const double* __restrict__ alpha,
    const double* __restrict__ per_sample_loss,
    const T* __restrict__ dloss,
    T* __restrict__ dlog_probs,
    double* __restrict__ beta_ws,
    double* __restrict__ /*neg_gamma_ws*/,
    int32_t max_time,
    int32_t batch_size,
    int32_t num_classes,
    int32_t max_target_len,
    int32_t blank,
    int32_t reduction_mode,
    double inv_denom,
    int32_t zero_infinity)
{
    int n = blockIdx.x;
    if (n >= batch_size) return;

    int Tn = (int)input_lengths[n];
    int Sn = (int)target_lengths[n];
    int Ln = 2 * Sn + 1;
    int Lmax = 2 * max_target_len + 1;
    int tid = threadIdx.x;
    bool active = (tid < Ln);

    double dloss_val = (reduction_mode == 0) ? (double)dloss[n] : (double)dloss[0];
    double scale_n = dloss_val * inv_denom;

    double fw_loss = per_sample_loss[n];
    bool zero_grad = (zero_infinity != 0) && (fw_loss == 0.0) && (Tn > 0) && (Sn > 0);
    if (Tn <= 0) return;

    auto ext_t = [&](int k) -> int {
        if ((k & 1) == 0) return blank;
        int64_t lbl = targets[(int64_t)n * (int64_t)max_target_len + (int64_t)(k >> 1)];
        return (int)lbl;
    };
    int my_class = active ? ext_t(tid) : 0;

    const double* A = alpha + (int64_t)n * (int64_t)Lmax;
    int64_t stride_t = (int64_t)batch_size * (int64_t)Lmax;
    double* B = beta_ws + (int64_t)n * (int64_t)Lmax;

    if (active) {
        double v = -INFINITY;
        if (Sn == 0) {
            if (tid == 0) {
                int c = my_class;
                v = (double)log_probs[(int64_t)(Tn - 1) * (int64_t)batch_size * (int64_t)num_classes
                                      + (int64_t)n * (int64_t)num_classes + (int64_t)c];
            }
        } else if (tid == Ln - 1 || tid == Ln - 2) {
            int c = my_class;
            v = (double)log_probs[(int64_t)(Tn - 1) * (int64_t)batch_size * (int64_t)num_classes
                                  + (int64_t)n * (int64_t)num_classes + (int64_t)c];
        }
        B[tid] = v;
    }
    __syncthreads();

    __shared__ double gamma_sm[32];

    for (int t = Tn - 1; t >= 0; --t) {
        if (tid < num_classes) gamma_sm[tid] = 0.0;
        __syncthreads();
        const double* A_t = A + (int64_t)t * stride_t;
        if (active) {
            double a_tk = A_t[tid];
            double b_tk = B[tid];
            int c = my_class;
            double logp_tnc = (double)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                                + (int64_t)n * (int64_t)num_classes + (int64_t)c];
            double contribution = 0.0;
            // Fixed 2026-05-16: was `- fw_loss` (P-scaling — wrong direction);
            // correct factor is `1/P = exp(nll)`, so we ADD fw_loss.
            // See f32 kernel comment block above for derivation.
            double arg = a_tk + b_tk - logp_tnc + fw_loss;
            if (a_tk != -INFINITY && b_tk != -INFINITY) {
                contribution = exp(arg);
            }
            if (contribution > 0.0 && contribution == contribution) {
                // double atomicAdd is sm_60+. Sm_80+ definitely has it.
                atomicAdd(&gamma_sm[c], contribution);
            }
        }
        __syncthreads();

        if (tid < num_classes) {
            int c = tid;
            double logp_tnc = (double)log_probs[(int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                                                + (int64_t)n * (int64_t)num_classes + (int64_t)c];
            double p = exp(logp_tnc);
            double grad_raw = p - gamma_sm[c];
            double grad = zero_grad ? 0.0 : (grad_raw * scale_n);
            int64_t off = (int64_t)t * (int64_t)batch_size * (int64_t)num_classes
                          + (int64_t)n * (int64_t)num_classes + (int64_t)c;
            dlog_probs[off] = (T)grad;
        }
        __syncthreads();

        if (t > 0) {
            double b_new = -INFINITY;
            if (active) {
                double b_k = B[tid];
                double b_kp1 = (tid + 1 < Ln) ? B[tid + 1] : -INFINITY;
                double b_kp2 = -INFINITY;
                if (tid + 2 < Ln) {
                    int curr_c = my_class;
                    int kp2_c = ext_t(tid + 2);
                    if (curr_c != blank && curr_c != kp2_c) {
                        b_kp2 = B[tid + 2];
                    }
                }
                double lse_term = lse3_d(b_k, b_kp1, b_kp2);
                int c = my_class;
                double emit_prev = (double)log_probs[(int64_t)(t - 1) * (int64_t)batch_size * (int64_t)num_classes
                                                     + (int64_t)n * (int64_t)num_classes + (int64_t)c];
                b_new = lse_term + emit_prev;
            }
            __syncthreads();
            if (active) {
                B[tid] = b_new;
            }
            __syncthreads();
        }
    }
    (void)max_time;
}

} } // namespace baracuda::ctc

// =============================================================================
// INSTANTIATE macros
// =============================================================================
//
// FW launcher ABI:
//   (max_time, batch_size, num_classes, max_target_len, blank, reduction_mode,
//    zero_infinity,
//    log_probs, targets, input_lengths, target_lengths, alpha_ws, out,
//    workspace, workspace_bytes, stream_ptr)
//
// alpha_ws and a per-sample loss buffer come from the launcher: the launcher
// carves the workspace into (alpha_acc[T*N*Lmax], per_sample_loss[N]).
//
// For f32 / f16 / bf16, acc type is f32. For f64, acc type is f64.

#define BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F32_ACC(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t max_time,                                                                           \
        int32_t batch_size,                                                                         \
        int32_t num_classes,                                                                        \
        int32_t max_target_len,                                                                     \
        int32_t blank,                                                                              \
        int32_t reduction_mode,                                                                     \
        int32_t zero_infinity,                                                                      \
        const void* log_probs,                                                                      \
        const void* targets,                                                                        \
        const void* input_lengths,                                                                  \
        const void* target_lengths,                                                                 \
        void* alpha_ws,                                                                             \
        void* out,                                                                                  \
        void* workspace,                                                                            \
        size_t workspace_bytes,                                                                     \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (max_time < 0 || batch_size < 0 || num_classes < 0 || max_target_len < 0) return 2;     \
        if (blank < 0 || blank >= num_classes) return 2;                                            \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        if (num_classes > 32) return 3;                                                             \
        if (max_target_len > 256) return 3;                                                         \
        if (batch_size == 0 || max_time == 0) return 0;                                             \
        if (log_probs == nullptr || targets == nullptr || input_lengths == nullptr                  \
            || target_lengths == nullptr || alpha_ws == nullptr || out == nullptr) return 2;        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        int Lmax = 2 * max_target_len + 1;                                                          \
        if (Lmax <= 0) Lmax = 1;                                                                    \
        int block_dim = Lmax;                                                                       \
        if (block_dim < 32) block_dim = 32;                                                         \
        if (block_dim > 1024) return 3;                                                             \
        size_t per_sample_bytes = (size_t)batch_size * sizeof(float);                               \
        if (workspace == nullptr || workspace_bytes < per_sample_bytes) return 4;                   \
        float* per_sample_loss = static_cast<float*>(workspace);                                    \
        cudaMemsetAsync(per_sample_loss, 0, per_sample_bytes, stream);                              \
        baracuda::ctc::ctc_forward_kernel_f32<T><<<batch_size, block_dim, 0, stream>>>(             \
            static_cast<const T*>(log_probs),                                                       \
            static_cast<const int64_t*>(targets),                                                   \
            static_cast<const int64_t*>(input_lengths),                                             \
            static_cast<const int64_t*>(target_lengths),                                            \
            static_cast<float*>(alpha_ws),                                                          \
            per_sample_loss,                                                                        \
            max_time, batch_size, num_classes, max_target_len, blank, zero_infinity);               \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode == 0) {                                                                  \
            int blocks = (batch_size + 255) / 256;                                                  \
            if (blocks < 1) blocks = 1;                                                             \
            baracuda::ctc::ctc_cast_loss_kernel<T><<<blocks, 256, 0, stream>>>(                     \
                per_sample_loss, static_cast<T*>(out), batch_size);                                 \
        } else {                                                                                    \
            baracuda::ctc::ctc_reduce_loss_kernel<T>                                                \
                <<<1, baracuda::loss::kBlockReduce, 0, stream>>>(                                   \
                    per_sample_loss,                                                                \
                    static_cast<const int64_t*>(target_lengths),                                    \
                    static_cast<T*>(out), batch_size, reduction_mode);                              \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

#define BARACUDA_KERNELS_LOSS_CTC_FW_INSTANTIATE_F64_ACC(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t max_time,                                                                           \
        int32_t batch_size,                                                                         \
        int32_t num_classes,                                                                        \
        int32_t max_target_len,                                                                     \
        int32_t blank,                                                                              \
        int32_t reduction_mode,                                                                     \
        int32_t zero_infinity,                                                                      \
        const void* log_probs,                                                                      \
        const void* targets,                                                                        \
        const void* input_lengths,                                                                  \
        const void* target_lengths,                                                                 \
        void* alpha_ws,                                                                             \
        void* out,                                                                                  \
        void* workspace,                                                                            \
        size_t workspace_bytes,                                                                     \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (max_time < 0 || batch_size < 0 || num_classes < 0 || max_target_len < 0) return 2;     \
        if (blank < 0 || blank >= num_classes) return 2;                                            \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        if (num_classes > 32) return 3;                                                             \
        if (max_target_len > 256) return 3;                                                         \
        if (batch_size == 0 || max_time == 0) return 0;                                             \
        if (log_probs == nullptr || targets == nullptr || input_lengths == nullptr                  \
            || target_lengths == nullptr || alpha_ws == nullptr || out == nullptr) return 2;        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        int Lmax = 2 * max_target_len + 1;                                                          \
        if (Lmax <= 0) Lmax = 1;                                                                    \
        int block_dim = Lmax;                                                                       \
        if (block_dim < 32) block_dim = 32;                                                         \
        if (block_dim > 1024) return 3;                                                             \
        size_t per_sample_bytes = (size_t)batch_size * sizeof(double);                              \
        if (workspace == nullptr || workspace_bytes < per_sample_bytes) return 4;                   \
        double* per_sample_loss = static_cast<double*>(workspace);                                  \
        cudaMemsetAsync(per_sample_loss, 0, per_sample_bytes, stream);                              \
        baracuda::ctc::ctc_forward_kernel_f64<T><<<batch_size, block_dim, 0, stream>>>(             \
            static_cast<const T*>(log_probs),                                                       \
            static_cast<const int64_t*>(targets),                                                   \
            static_cast<const int64_t*>(input_lengths),                                             \
            static_cast<const int64_t*>(target_lengths),                                            \
            static_cast<double*>(alpha_ws),                                                         \
            per_sample_loss,                                                                        \
            max_time, batch_size, num_classes, max_target_len, blank, zero_infinity);               \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode == 0) {                                                                  \
            int blocks = (batch_size + 255) / 256;                                                  \
            if (blocks < 1) blocks = 1;                                                             \
            baracuda::ctc::ctc_cast_loss_kernel_d<T><<<blocks, 256, 0, stream>>>(                   \
                per_sample_loss, static_cast<T*>(out), batch_size);                                 \
        } else {                                                                                    \
            baracuda::ctc::ctc_reduce_loss_kernel_d<T>                                              \
                <<<1, baracuda::loss::kBlockReduce, 0, stream>>>(                                   \
                    per_sample_loss,                                                                \
                    static_cast<const int64_t*>(target_lengths),                                    \
                    static_cast<T*>(out), batch_size, reduction_mode);                              \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// BW launcher ABI:
//   (max_time, batch_size, num_classes, max_target_len, blank, reduction_mode,
//    zero_infinity, inv_denom,
//    log_probs, targets, input_lengths, target_lengths,
//    alpha_ws, per_sample_loss, dloss, dlog_probs,
//    workspace, workspace_bytes, stream_ptr)
//
// workspace: per_sample_loss buffer ([N] floats/doubles) was passed in from
// FW. The BW launcher additionally needs β workspace [N * Lmax].
// We require workspace = [per_sample_loss(N) | beta_ws(N*Lmax)] all-in.

#define BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F32_ACC(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t max_time,                                                                           \
        int32_t batch_size,                                                                         \
        int32_t num_classes,                                                                        \
        int32_t max_target_len,                                                                     \
        int32_t blank,                                                                              \
        int32_t reduction_mode,                                                                     \
        int32_t zero_infinity,                                                                      \
        float inv_denom,                                                                            \
        const void* log_probs,                                                                      \
        const void* targets,                                                                        \
        const void* input_lengths,                                                                  \
        const void* target_lengths,                                                                 \
        const void* alpha_ws,                                                                       \
        const void* per_sample_loss,                                                                \
        const void* dloss,                                                                          \
        void* dlog_probs,                                                                           \
        void* workspace,                                                                            \
        size_t workspace_bytes,                                                                     \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (max_time < 0 || batch_size < 0 || num_classes < 0 || max_target_len < 0) return 2;     \
        if (blank < 0 || blank >= num_classes) return 2;                                            \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        if (num_classes > 32) return 3;                                                             \
        if (max_target_len > 256) return 3;                                                         \
        if (batch_size == 0 || max_time == 0) return 0;                                             \
        if (log_probs == nullptr || targets == nullptr || input_lengths == nullptr                  \
            || target_lengths == nullptr || alpha_ws == nullptr || per_sample_loss == nullptr       \
            || dloss == nullptr || dlog_probs == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        int Lmax = 2 * max_target_len + 1;                                                          \
        if (Lmax <= 0) Lmax = 1;                                                                    \
        int block_dim = Lmax;                                                                       \
        if (block_dim < 32) block_dim = 32;                                                         \
        if (block_dim > 1024) return 3;                                                             \
        size_t beta_bytes = (size_t)batch_size * (size_t)Lmax * sizeof(float);                      \
        if (workspace == nullptr || workspace_bytes < beta_bytes) return 4;                         \
        float* beta_ws = static_cast<float*>(workspace);                                            \
        cudaMemsetAsync(dlog_probs, 0,                                                              \
            (size_t)max_time * (size_t)batch_size * (size_t)num_classes * sizeof(T), stream);       \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        baracuda::ctc::ctc_backward_kernel_f32<T><<<batch_size, block_dim, 0, stream>>>(            \
            static_cast<const T*>(log_probs),                                                       \
            static_cast<const int64_t*>(targets),                                                   \
            static_cast<const int64_t*>(input_lengths),                                             \
            static_cast<const int64_t*>(target_lengths),                                            \
            static_cast<const float*>(alpha_ws),                                                    \
            static_cast<const float*>(per_sample_loss),                                             \
            static_cast<const T*>(dloss),                                                           \
            static_cast<T*>(dlog_probs),                                                            \
            beta_ws,                                                                                \
            nullptr,                                                                                \
            max_time, batch_size, num_classes, max_target_len, blank,                               \
            reduction_mode, inv_denom, zero_infinity);                                              \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

#define BARACUDA_KERNELS_LOSS_CTC_BW_INSTANTIATE_F64_ACC(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t max_time,                                                                           \
        int32_t batch_size,                                                                         \
        int32_t num_classes,                                                                        \
        int32_t max_target_len,                                                                     \
        int32_t blank,                                                                              \
        int32_t reduction_mode,                                                                     \
        int32_t zero_infinity,                                                                      \
        float inv_denom,                                                                            \
        const void* log_probs,                                                                      \
        const void* targets,                                                                        \
        const void* input_lengths,                                                                  \
        const void* target_lengths,                                                                 \
        const void* alpha_ws,                                                                       \
        const void* per_sample_loss,                                                                \
        const void* dloss,                                                                          \
        void* dlog_probs,                                                                           \
        void* workspace,                                                                            \
        size_t workspace_bytes,                                                                     \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (max_time < 0 || batch_size < 0 || num_classes < 0 || max_target_len < 0) return 2;     \
        if (blank < 0 || blank >= num_classes) return 2;                                            \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        if (num_classes > 32) return 3;                                                             \
        if (max_target_len > 256) return 3;                                                         \
        if (batch_size == 0 || max_time == 0) return 0;                                             \
        if (log_probs == nullptr || targets == nullptr || input_lengths == nullptr                  \
            || target_lengths == nullptr || alpha_ws == nullptr || per_sample_loss == nullptr       \
            || dloss == nullptr || dlog_probs == nullptr) return 2;                                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        int Lmax = 2 * max_target_len + 1;                                                          \
        if (Lmax <= 0) Lmax = 1;                                                                    \
        int block_dim = Lmax;                                                                       \
        if (block_dim < 32) block_dim = 32;                                                         \
        if (block_dim > 1024) return 3;                                                             \
        size_t beta_bytes = (size_t)batch_size * (size_t)Lmax * sizeof(double);                     \
        if (workspace == nullptr || workspace_bytes < beta_bytes) return 4;                         \
        double* beta_ws = static_cast<double*>(workspace);                                          \
        cudaMemsetAsync(dlog_probs, 0,                                                              \
            (size_t)max_time * (size_t)batch_size * (size_t)num_classes * sizeof(T), stream);       \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        baracuda::ctc::ctc_backward_kernel_f64<T><<<batch_size, block_dim, 0, stream>>>(            \
            static_cast<const T*>(log_probs),                                                       \
            static_cast<const int64_t*>(targets),                                                   \
            static_cast<const int64_t*>(input_lengths),                                             \
            static_cast<const int64_t*>(target_lengths),                                            \
            static_cast<const double*>(alpha_ws),                                                   \
            static_cast<const double*>(per_sample_loss),                                            \
            static_cast<const T*>(dloss),                                                           \
            static_cast<T*>(dlog_probs),                                                            \
            beta_ws,                                                                                \
            nullptr,                                                                                \
            max_time, batch_size, num_classes, max_target_len, blank,                               \
            reduction_mode, (double)inv_denom, zero_infinity);                                      \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

#endif // BARACUDA_CTC_CUH
