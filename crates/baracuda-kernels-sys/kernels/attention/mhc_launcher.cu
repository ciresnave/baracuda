// baracuda-kernels Phase 43 — AndreSlavescu/mHC.cu launcher.
//
// Bridges baracuda's `extern "C"` FFI surface to the vendored mHC
// `MHCLayer` API (manifold-constrained hyper-connections residual
// mixing) shipped from `vendor/mhc/src/mhc_layer.cuh`.
//
// Scope (Tier 1):
//   - Static-H forward path only (the simpler of the two upstream
//     code paths — H values are fixed parameters supplied by the
//     caller). Dynamic-H FW (where H is recomputed per-batch via
//     learned projections) deferred to Tier 2.
//   - Backward deferred to Tier 2.
//   - bf16 weights / f32 activations only (matches upstream's
//     `floatX = nv_bfloat16` typedef; f16 / f32 paths require
//     additional convert kernels and are deferred).
//
// Memory contract:
//   * MHCLayer is a stateful object — it owns ~5 MiB of device-side
//     scratch (Sinkhorn workspace, RMS save, intermediate streams,
//     post-mix buffer). We expose three C-ABI symbols:
//
//        baracuda_kernels_mhc_layer_static_bf16_create(...)  -> handle
//        baracuda_kernels_mhc_layer_static_bf16_destroy(handle)
//        baracuda_kernels_mhc_layer_static_bf16_run(handle, …)
//
//     so callers can pay the alloc cost once and reuse the handle
//     across many forward calls. The Rust `HyperConnectionPlan`
//     pairs `select()` with `create` and `Drop` with `destroy`.
//
//   * The MHCLayer internal stream is configurable — we let the
//     baracuda caller pass a `cudaStream_t` at run time so the layer
//     joins the caller's launch graph rather than fighting an
//     internal stream.
//
// Layout / shape contract:
//   * x_expanded: [B, n, C] — one "stream" per row of the residual
//     fanout. Row-major contiguous.
//   * rmsnorm_weight: [C] — bf16 per-channel scale (Llama-style
//     RMSNorm gamma).
//   * H_pre, H_post, H_res: caller-supplied static H values, all
//     float. H_pre and H_post are length n; H_res is [n, n] (the
//     pre-Sinkhorn mixing matrix).
//   * output: [B, n, C] — same layout as x_expanded. Each output row
//     is `M @ x_expanded[b] + H_post[i] * RMSNorm(aggregate)`.

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <new>     // std::nothrow
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "../../vendor/mhc/src/mhc_layer.cuh"

// Exact adjoint of the Sinkhorn-Knopp iteration (row normalise, then column
// normalise, `iters` times) with respect to its INPUT `K = exp(H_res)`.
//
// baracuda-owned replacement for upstream `sinkhorn_knopp_backward`. For
// y = x / sum(x) the adjoint is dx = (dy - sum(dy * y)) / sum(x); the vendored
// kernel subtracts `y * sum(dy * y)` (an extra factor of y) and drops the
// 1/sum, so its result is wrong, by a factor of about `n` when `K = exp(H)` is
// far from doubly stochastic. Measured against central differences of an f64
// reference (tests/hyper_connection_smoke.rs) and checked in pure Python at
// 1, 2 and 20 iterations (max error 2e-10). The vendored tree is left
// byte-identical (VENDOR.md).
//
// One block; N <= 32, iters <= 20. Dynamic shared memory, in floats:
//   iters*N*N (post-row-normalise checkpoints) + 2*iters*N (inverse sums, 0 = the
//   normalisation was skipped because the sum was <= eps) + N*N (running tile)
//   + N*N (running gradient) + N (scratch).
__global__ void mhc_sinkhorn_backward_exact_kernel(float* __restrict__ d_inp,
                                                   const float* __restrict__ grad,
                                                   const float* __restrict__ K, int N, int iters,
                                                   float eps) {
    extern __shared__ float smem[];
    const int NN = N * N;
    float* chk = smem;                    // [iters][N*N]
    float* rinv = chk + iters * NN;       // [iters][N]
    float* cinv = rinv + iters * N;       // [iters][N]
    float* tile = cinv + iters * N;       // [N*N]
    float* d = tile + NN;                 // [N*N]
    float* buf = d + NN;                  // [N]

    for (int i = threadIdx.x; i < NN; i += blockDim.x) {
        tile[i] = K[i];
        d[i] = grad[i];
    }
    __syncthreads();

    // Forward replay, mirroring the forward kernel (including the `> eps` guard).
    for (int it = 0; it < iters; it++) {
        for (int r = threadIdx.x; r < N; r += blockDim.x) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) sum += tile[r * N + c];
            rinv[it * N + r] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NN; i += blockDim.x) {
            const float f = rinv[it * N + i / N];
            if (f != 0.0f) tile[i] *= f;
            chk[it * NN + i] = tile[i];
        }
        __syncthreads();
        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            float sum = 0.0f;
            for (int r = 0; r < N; r++) sum += tile[r * N + c];
            cinv[it * N + c] = (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NN; i += blockDim.x) {
            const float f = cinv[it * N + i % N];
            if (f != 0.0f) tile[i] *= f;
        }
        __syncthreads();
    }

    for (int it = iters - 1; it >= 0; it--) {
        // Column step. B = A * c;  dA_ij = c_j * (dB_ij - sum_k dB_kj B_kj).
        for (int c = threadIdx.x; c < N; c += blockDim.x) {
            const float f = cinv[it * N + c];
            float dot = 0.0f;
            if (f != 0.0f) {
                for (int r = 0; r < N; r++) dot += d[r * N + c] * (chk[it * NN + r * N + c] * f);
            }
            buf[c] = dot;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NN; i += blockDim.x) {
            const float f = cinv[it * N + i % N];
            if (f != 0.0f) d[i] = f * (d[i] - buf[i % N]);
        }
        __syncthreads();
        // Row step. A = K * r;  dK_ij = r_i * (dA_ij - sum_l dA_il A_il).
        for (int r = threadIdx.x; r < N; r += blockDim.x) {
            const float f = rinv[it * N + r];
            float dot = 0.0f;
            if (f != 0.0f) {
                for (int c = 0; c < N; c++) dot += d[r * N + c] * chk[it * NN + r * N + c];
            }
            buf[r] = dot;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < NN; i += blockDim.x) {
            const float f = rinv[it * N + i / N];
            if (f != 0.0f) d[i] = f * (d[i] - buf[i / N]);
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < NN; i += blockDim.x) d_inp[i] = d[i];
}

// Same sequence as upstream `MHCLayer::backward` (static-H), with the exact
// Sinkhorn adjoint above. Returns false when the shape is outside the kernel's
// range (n > 32 or more than 20 iterations).
static bool mhc_backward_exact(mhc::MHCLayer* layer, const float* d_output) {
    const int B = layer->config.batch_size;
    const int C = layer->config.hidden_dim;
    const int n = layer->config.expansion_rate;
    const int iters = layer->config.sinkhorn_iters;
    if (n > 32 || iters > 20) return false;
    cudaStream_t stream = layer->stream;
    auto& g = layer->grads;
    auto& b = layer->buffers;

    g.zero_weight_grads(C, n, stream);
    mhc::stream_distribute_mix_backward_fused(
        g.d_x_mixed, g.d_layer_out, g.d_M, g.d_H_post_activated, d_output, b.x_expanded,
        b.layer_out_f32, b.sinkhorn_M, b.H_post_activated, B, n, C, g.workspace_dM,
        g.workspace_dH, g.workspace_num_blocks, stream);

    const size_t smem = ((size_t)iters * n * n + 2 * (size_t)iters * n + 2 * (size_t)n * n + n) *
                        sizeof(float);
    cudaFuncSetAttribute(mhc_sinkhorn_backward_exact_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    mhc_sinkhorn_backward_exact_kernel<<<1, 256, smem, stream>>>(
        g.d_H_res_exp, g.d_M, b.H_res_tilde, n, iters, layer->config.eps);
    mhc::exp_backward(g.d_H_res, g.d_H_res_exp, b.H_res_tilde, n * n, stream);

    mhc::sigmoid_scale_backward(g.d_H_post, g.d_H_post_activated, b.H_post_activated, 2.0f, n,
                                stream);
    mhc::rmsnorm_backward(g.d_x_aggregated, g.d_rmsnorm_weight, g.d_layer_out, b.x_aggregated_f32,
                          layer->weights.rmsnorm_weight, b.rms_values, B, C, stream);
    mhc::stream_aggregate_backward(g.d_x_expanded, g.d_H_pre_activated, g.d_x_aggregated,
                                   b.x_expanded, b.H_pre_activated, B, n, C, g.workspace_dH,
                                   g.workspace_num_blocks, stream);
    mhc::sigmoid_backward(g.d_H_pre, g.d_H_pre_activated, b.H_pre_activated, n, stream);
    return true;
}

__global__ void mhc_add2_kernel(float* __restrict__ out, const float* __restrict__ a,
                                const float* __restrict__ b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

extern "C" {

// Status codes match baracuda-kernels-sys convention:
//   0 = ok, 2 = invalid_problem, 3 = unsupported, 4 = workspace_too_small.
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;
constexpr int STATUS_UNSUPPORTED = 3;

// Create an MHCLayer handle configured for static-H, bf16, given
// (B, C, n). Returns a heap-allocated `mhc::MHCLayer*` cast to
// `void*`. Returns nullptr on failure (only when allocation fails or
// the (B, C, n) tuple is rejected upstream — there is no upstream
// `bool` return for init failure so we do our own range checks first).
//
// Upstream allocates ~B*n*C*sizeof(float) bytes of scratch on the
// device per handle. For (B=32, n=4, C=2048) that's ~1 MiB.
static void* mhc_create_impl(int B, int C, int n, int sinkhorn_iters, float eps,
                             bool enable_backward) {
    if (B <= 0 || C <= 0 || n <= 0) {
        return nullptr;
    }
    if (sinkhorn_iters <= 0 || sinkhorn_iters > 1000) {
        return nullptr;
    }
    if (!(eps > 0.0f) || !(eps < 1.0f)) {
        return nullptr;
    }
    // mHC's stream_aggregate_bf16_fused_sigmoid kernel assumes n <= 32
    // (stream count fits in shared memory). Stream-mix-TC threshold is
    // also 32 (`n >= 32` activates the cuBLAS-Lt path which we have not
    // yet validated in the C-ABI shim) — we cap at n < 32 to stay on
    // the bespoke-kernel path.
    if (n >= 32) {
        return nullptr;
    }

    auto* layer = new (std::nothrow) mhc::MHCLayer();
    if (!layer) {
        return nullptr;
    }

    mhc::MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = n;
    cfg.sinkhorn_iters = sinkhorn_iters;
    cfg.eps = eps;
    cfg.alpha_init = 0.01f;
    // PDL (programmatic stream serialization) is a Hopper/sm_90+ feature.
    // Disable it unconditionally — we don't enable -DMHC_ENABLE_PDL in
    // the build script either.
    cfg.use_pdl = false;
    // Tier 1 = static H only.
    cfg.use_dynamic_h = false;

    // Pass nullptr stream so the layer creates its own; we override
    // per-call in `_run` below by NOT using `forward_device`'s internal
    // stream member and instead launching with the caller's stream
    // directly through the lower-level kernel inline functions.
    //
    // NOTE: the upstream `forward_device` reads layer.stream. We work
    // around this by patching layer.stream to the caller's stream
    // before each call and back to the owned one after. See _run.
    //
    // Backward off for Tier 1; pipelining off (only beneficial for n>=16
    // anyway).
    layer->init(cfg, /*s=*/nullptr,
                /*enable_backward=*/enable_backward,
                /*enable_pipelining=*/false);

    return reinterpret_cast<void*>(layer);
}

void* baracuda_kernels_mhc_layer_static_bf16_create(int B, int C, int n,
                                                    int sinkhorn_iters, float eps) {
    return mhc_create_impl(B, C, n, sinkhorn_iters, eps, /*enable_backward=*/false);
}

// Same as `create`, but also allocates the gradient buffers so
// `baracuda_kernels_mhc_layer_static_bf16_backward` can follow a `_run`.
void* baracuda_kernels_mhc_layer_static_bf16_create_training(int B, int C, int n,
                                                             int sinkhorn_iters, float eps) {
    // The exact Sinkhorn adjoint keeps its checkpoints in shared memory: <= 20 iterations.
    if (sinkhorn_iters > 20) return nullptr;
    return mhc_create_impl(B, C, n, sinkhorn_iters, eps, /*enable_backward=*/true);
}

// Destroy an MHCLayer handle. Safe to pass nullptr.
void baracuda_kernels_mhc_layer_static_bf16_destroy(void* handle) {
    if (!handle) return;
    auto* layer = reinterpret_cast<mhc::MHCLayer*>(handle);
    layer->destroy();
    delete layer;
}

// Forward kernel — static-H path, bf16 weights / f32 activations.
//
//   handle           : opaque MHCLayer handle from `create`.
//   x_expanded       : device-pointer, [B, n, C] f32, row-major.
//                      In Llama transformer terms: each of the n
//                      residual streams.
//   rmsnorm_weight   : device-pointer, [C] bf16, the gamma scale for
//                      the internal RMSNorm.
//   H_pre, H_post    : device-pointer, [n] f32. Pre/post mixing
//                      coefficients (pass through sigmoid internally).
//   H_res            : device-pointer, [n, n] f32. The pre-Sinkhorn
//                      mixing matrix (passes through Sinkhorn-Knopp
//                      iteration internally).
//   out              : device-pointer, [B, n, C] f32, row-major.
//   stream           : cudaStream_t (cast to void*). Caller-supplied;
//                      the layer launches kernels on this stream.
//
// Returns 0 on success, 2 if an argument fails validation, 3 if the
// (B, C, n) of the handle disagrees with the call (a contract violation).
int baracuda_kernels_mhc_layer_static_bf16_run(
    void* handle,
    const void* x_expanded,
    const void* rmsnorm_weight,
    const void* H_pre,
    const void* H_post,
    const void* H_res,
    void* out,
    int B, int C, int n,
    void* /*workspace*/,            // unused — internal scratch is in the handle
    uint64_t /*workspace_bytes*/,   // unused
    void* stream)
{
    if (!handle) return STATUS_INVALID_ARG;
    if (!x_expanded || !rmsnorm_weight || !H_pre || !H_post || !H_res || !out) {
        return STATUS_INVALID_ARG;
    }

    auto* layer = reinterpret_cast<mhc::MHCLayer*>(handle);

    if (layer->config.batch_size != B ||
        layer->config.hidden_dim != C ||
        layer->config.expansion_rate != n) {
        return STATUS_UNSUPPORTED;
    }
    if (layer->config.use_dynamic_h) {
        return STATUS_UNSUPPORTED;   // Tier-1 launcher only handles static H
    }

    // 1. Patch in the caller's stream so all kernels launched by
    //    `forward_device` route there. Stash + restore the
    //    layer-owned stream so destroy() still closes it cleanly.
    cudaStream_t saved_stream = layer->stream;
    cudaStream_t caller_stream = reinterpret_cast<cudaStream_t>(stream);
    layer->stream = caller_stream;

    // 2. Upload the static weights into the layer's internal buffers
    //    (upstream stages weights via host pointers — but we have
    //    device-side weights already so use cudaMemcpyAsync D2D).
    cudaError_t err;
    err = cudaMemcpyAsync(layer->weights.rmsnorm_weight, rmsnorm_weight,
                          C * sizeof(mhc::floatX),
                          cudaMemcpyDeviceToDevice, caller_stream);
    if (err != cudaSuccess) { layer->stream = saved_stream; return STATUS_UNSUPPORTED; }

    // mHC's static-H path treats `b_pre` / `b_post` / `b_res` as the
    // pre-sigmoid logits (the upstream `set_weights_static` overload
    // takes them directly — see stream_aggregate_bf16_fused_sigmoid).
    err = cudaMemcpyAsync(layer->weights.b_pre, H_pre,
                          n * sizeof(float),
                          cudaMemcpyDeviceToDevice, caller_stream);
    if (err != cudaSuccess) { layer->stream = saved_stream; return STATUS_UNSUPPORTED; }

    err = cudaMemcpyAsync(layer->weights.b_post, H_post,
                          n * sizeof(float),
                          cudaMemcpyDeviceToDevice, caller_stream);
    if (err != cudaSuccess) { layer->stream = saved_stream; return STATUS_UNSUPPORTED; }

    err = cudaMemcpyAsync(layer->weights.b_res, H_res,
                          n * n * sizeof(float),
                          cudaMemcpyDeviceToDevice, caller_stream);
    if (err != cudaSuccess) { layer->stream = saved_stream; return STATUS_UNSUPPORTED; }

    // 3. Launch the static-H forward. `forward_device` takes a
    //    device pointer for `x_expanded` and writes into the layer's
    //    internal `output` buffer.
    layer->forward_device(reinterpret_cast<const float*>(x_expanded));

    // 4. Copy the output out. Upstream owns it in a per-handle buffer
    //    `buffers.output` of size [B, n, C] f32.
    err = cudaMemcpyAsync(out, layer->buffers.output,
                          B * n * C * sizeof(float),
                          cudaMemcpyDeviceToDevice, caller_stream);

    // 5. Restore the layer's owned stream.
    layer->stream = saved_stream;

    if (err != cudaSuccess) {
        return STATUS_UNSUPPORTED;
    }
    return STATUS_OK;
}

// Backward (static-H). MUST follow a `_run` on the SAME handle: it reads the
// activations that forward left in the handle (x_expanded, the normalised
// aggregate, the Sinkhorn matrix, the activated H vectors, the RMS values).
//
//   d_out          : [B, n, C] f32, upstream gradient of `out`.
//   d_x_expanded   : [B, n, C] f32 out, gradient w.r.t. x_expanded.
//   d_rmsnorm_w    : [C] f32 out, gradient w.r.t. the RMSNorm gamma (f32 even
//                    though gamma itself is bf16).
//   d_h_pre, d_h_post : [n] f32 out, gradients w.r.t. the pre-activation logits.
//   d_h_res        : [n, n] f32 out, gradient w.r.t. the pre-exp H_res.
//
// Returns 0 ok, 2 invalid arg, 3 if the handle was not created for training or
// its (B, C, n) disagrees with the call.
int baracuda_kernels_mhc_layer_static_bf16_backward(
    void* handle,
    const void* d_out,
    void* d_x_expanded,
    void* d_rmsnorm_w,
    void* d_h_pre,
    void* d_h_post,
    void* d_h_res,
    int B, int C, int n,
    void* stream)
{
    if (!handle) return STATUS_INVALID_ARG;
    if (!d_out || !d_x_expanded || !d_rmsnorm_w || !d_h_pre || !d_h_post || !d_h_res) {
        return STATUS_INVALID_ARG;
    }
    auto* layer = reinterpret_cast<mhc::MHCLayer*>(handle);
    if (!layer->backward_enabled) return STATUS_UNSUPPORTED;
    if (layer->config.batch_size != B ||
        layer->config.hidden_dim != C ||
        layer->config.expansion_rate != n) {
        return STATUS_UNSUPPORTED;
    }

    cudaStream_t saved_stream = layer->stream;
    cudaStream_t caller_stream = reinterpret_cast<cudaStream_t>(stream);
    layer->stream = caller_stream;

    if (!mhc_backward_exact(layer, reinterpret_cast<const float*>(d_out))) {
        layer->stream = saved_stream;
        return STATUS_UNSUPPORTED;
    }

    // Upstream leaves the gradient of x_expanded in two buffers: `d_x_expanded`
    // holds the aggregate (RMSNorm) path and `d_x_mixed` the distribute / mix
    // path. The true gradient is their sum.
    {
        const size_t total = (size_t)B * n * C;
        const int block = 256;
        const int grid = (int)((total + block - 1) / block);
        mhc_add2_kernel<<<grid, block, 0, caller_stream>>>(
            reinterpret_cast<float*>(d_x_expanded), layer->grads.d_x_expanded,
            layer->grads.d_x_mixed, total);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(d_rmsnorm_w, layer->grads.d_rmsnorm_weight,
                              (size_t)C * sizeof(float), cudaMemcpyDeviceToDevice, caller_stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(d_h_pre, layer->grads.d_H_pre, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToDevice, caller_stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(d_h_post, layer->grads.d_H_post, (size_t)n * sizeof(float),
                              cudaMemcpyDeviceToDevice, caller_stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(d_h_res, layer->grads.d_H_res, (size_t)n * n * sizeof(float),
                              cudaMemcpyDeviceToDevice, caller_stream);

    layer->stream = saved_stream;
    return err == cudaSuccess ? STATUS_OK : STATUS_UNSUPPORTED;
}

// can_implement — pure-host validation, no GPU work. Caller wires
// this into the Rust plan's `can_implement` body. The shape sanity
// checks duplicate the ones in `create` so callers can probe support
// before allocating the handle.
int baracuda_kernels_mhc_layer_static_bf16_can_implement(int B, int C, int n) {
    if (B <= 0 || C <= 0 || n <= 0) return STATUS_INVALID_ARG;
    if (n >= 32) return STATUS_UNSUPPORTED;
    // Upstream's vec4 path assumes C % 4 == 0 in several kernels; the
    // fallback covers non-multiples-of-4 so we don't reject them, but
    // we DO require C >= n (the aggregate kernel needs at least one
    // column per stream to make progress).
    if (C < n) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

}  // extern "C"
