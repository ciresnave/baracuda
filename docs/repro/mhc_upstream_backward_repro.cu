// Standalone repro for two defects in AndreSlavescu/mHC.cu `MHCLayer::backward` (static-H).
// Uses ONLY the upstream headers (src/mhc_layer.cuh) and the CUDA runtime: no baracuda code.
//
// Build (from a checkout of mHC.cu; adjust -arch; on Windows add -Xcompiler /Zc:preprocessor):
//   nvcc -std=c++17 -arch=sm_89 -I include -I src mhc_upstream_backward_repro.cu -o repro -lcublasLt
//
// Setup: B=2, C=16, n=4, 20 Sinkhorn iterations, eps=1e-5, static H. The loss is
// L = sum(d_out * out). For a few entries it prints the analytic gradient that
// `backward()` leaves in `grads` next to a central finite difference of the same
// GPU forward (float32, step 1e-2, so the finite difference is good to ~1e-3).
//
// Expected with a correct backward: both columns agree.
// Observed (RTX 4070, CUDA 13.3, upstream commit a426939c):
//   defect 1: `grads.d_x_expanded` (aggregate path only) differs from the finite
//             difference by about the size of `grads.d_x_mixed` (mix path); their SUM agrees
//             (x[0]: -2.8914 + 0.3249 = -2.5665 vs finite difference -2.5646).
//   defect 2: `grads.d_H_res` is about n (4) times larger than the finite difference
//             (h_res[0]: -0.6343 vs -0.1673, ratio 3.79).
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "mhc_layer.cuh"

using namespace mhc;

static const int B = 2, C = 16, N = 4;

static float run_loss(MHCLayer& L, const std::vector<float>& x, const std::vector<float>& dout,
                      const std::vector<float>& bpre, const std::vector<float>& bpost,
                      const std::vector<float>& bres, const std::vector<nv_bfloat16>& gamma) {
    L.set_weights(gamma.data(), bpre.data(), bpost.data(), bres.data());
    L.forward(x.data());
    L.sync();
    std::vector<float> out(B * N * C);
    cudaMemcpy(out.data(), L.get_output(), out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0;
    for (size_t i = 0; i < out.size(); i++) s += (double)out[i] * dout[i];
    return (float)s;
}

int main() {
    std::vector<float> x(B * N * C), dout(B * N * C), bpre(N), bpost(N), bres(N * N, 0.0f);
    std::vector<nv_bfloat16> gamma(C);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = sinf((float)i * 0.013f - 0.5f) * 0.4f;
        dout[i] = cosf((float)i * 0.021f + 0.3f) * 0.5f;
    }
    for (int i = 0; i < N; i++) { bpre[i] = 0.1f * i - 0.15f; bpost[i] = 0.2f * i - 0.3f; bres[i * N + i] = 0.05f; }
    for (int c = 0; c < C; c++) gamma[c] = __float2bfloat16(0.9f + 0.2f * sinf(c * 0.31f));

    MHCLayerConfig cfg;
    cfg.batch_size = B; cfg.hidden_dim = C; cfg.expansion_rate = N;
    cfg.sinkhorn_iters = 20; cfg.eps = 1e-5f; cfg.use_dynamic_h = false; cfg.use_pdl = false;
    MHCLayer L;
    L.init(cfg, nullptr, /*enable_backward=*/true, /*enable_pipelining=*/false);

    run_loss(L, x, dout, bpre, bpost, bres, gamma);
    float* d_dout; cudaMalloc(&d_dout, dout.size() * sizeof(float));
    cudaMemcpy(d_dout, dout.data(), dout.size() * sizeof(float), cudaMemcpyHostToDevice);
    L.backward(d_dout);
    L.sync();
    std::vector<float> dx(B * N * C), dxm(B * N * C), dres(N * N);
    cudaMemcpy(dx.data(), L.grads.d_x_expanded, dx.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dxm.data(), L.grads.d_x_mixed, dxm.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dres.data(), L.grads.d_H_res, dres.size() * 4, cudaMemcpyDeviceToHost);

    const float h = 1e-2f;
    printf("defect 1: d_x_expanded (aggregate path) + d_x_mixed (mix path) vs finite difference\n");
    for (int i : {0, 5, 37}) {
        auto xp = x, xm = x; xp[i] += h; xm[i] -= h;
        float fd = (run_loss(L, xp, dout, bpre, bpost, bres, gamma) -
                    run_loss(L, xm, dout, bpre, bpost, bres, gamma)) / (2 * h);
        printf("  x[%2d]: d_x_expanded=%9.5f  d_x_mixed=%9.5f  sum=%9.5f  finite-diff=%9.5f\n",
               i, dx[i], dxm[i], dx[i] + dxm[i], fd);
    }
    printf("defect 2: d_H_res vs finite difference (H_res enters through exp then Sinkhorn)\n");
    for (int i : {0, 1, 6, 12}) {
        auto rp = bres, rm = bres; rp[i] += h; rm[i] -= h;
        float fd = (run_loss(L, x, dout, bpre, bpost, rp, gamma) -
                    run_loss(L, x, dout, bpre, bpost, rm, gamma)) / (2 * h);
        printf("  h_res[%2d]: d_H_res=%9.5f  finite-diff=%9.5f  ratio=%.2f\n", i, dres[i], fd, dres[i] / fd);
    }
    return 0;
}
