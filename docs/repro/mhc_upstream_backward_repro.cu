// Standalone repro / regression test for two defects in AndreSlavescu/mHC.cu backward.
// Uses ONLY the upstream headers (src/csrc/kernels/*.cuh, src/csrc/include/*) and the CUDA
// runtime: no code from any other project.
//
// Build, from a checkout of mHC.cu (adjust -arch; on Windows add -Xcompiler /Zc:preprocessor):
//   nvcc -std=c++17 -arch=sm_89 -I src/csrc/include -I src/csrc/kernels repro.cu -o repro -lcublasLt -lcublas
//
// Every check compares an analytic gradient against a central finite difference of the SAME GPU
// forward (float32, step 1e-2). The loss is L = sum(G * output) for a fixed G. A check passes when
// the gradient is within 5% of the largest finite difference in its group (the finite difference
// itself is good to ~1e-3 relative). The program exits 1 if any check fails.
//
//   check A  MHCLayer::backward, dL/dx left in grads.d_x_expanded   (B=2, C=16, n=4, 20 iters)
//   check B  MHCLayer::backward, grads.d_H_res                       (same layer)
//   check C  sinkhorn_knopp_backward, N=4,  20 iters (checkpointed kernel)
//   check D  sinkhorn_knopp_backward, N=40,  5 iters (generic kernel)
//
// Observed on unmodified upstream a426939c (RTX 4070 Laptop, CUDA 13.3): A, B, C, D all FAIL.
// With the patch next to this file: all PASS.
#include <cmath>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "mhc_layer.cuh"

using namespace mhc;

static const int B = 2, C = 16, N = 4;
static int g_fail = 0;

static void verdict(const char* name, double worst_err, double scale) {
    bool ok = worst_err <= 0.05 * scale;
    printf("  -> %s: worst |analytic - finite-diff| = %.5f, scale %.5f : %s\n", name, worst_err,
           scale, ok ? "PASS" : "FAIL");
    if (!ok) g_fail++;
}

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

static void layer_checks() {
    std::vector<float> x(B * N * C), dout(B * N * C), bpre(N), bpost(N), bres(N * N, 0.0f);
    std::vector<nv_bfloat16> gamma(C);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = sinf((float)i * 0.013f - 0.5f) * 0.4f;
        dout[i] = cosf((float)i * 0.021f + 0.3f) * 0.5f;
    }
    for (int i = 0; i < N; i++) {
        bpre[i] = 0.1f * i - 0.15f;
        bpost[i] = 0.2f * i - 0.3f;
        bres[i * N + i] = 0.05f;
    }
    for (int c = 0; c < C; c++) gamma[c] = __float2bfloat16(0.9f + 0.2f * sinf(c * 0.31f));

    MHCLayerConfig cfg;
    cfg.batch_size = B;
    cfg.hidden_dim = C;
    cfg.expansion_rate = N;
    cfg.sinkhorn_iters = 20;
    cfg.eps = 1e-5f;
    cfg.use_dynamic_h = false;
    cfg.use_pdl = false;
    MHCLayer L;
    L.init(cfg, nullptr, /*enable_backward=*/true, /*enable_pipelining=*/false);

    run_loss(L, x, dout, bpre, bpost, bres, gamma);
    float* d_dout;
    cudaMalloc(&d_dout, dout.size() * sizeof(float));
    cudaMemcpy(d_dout, dout.data(), dout.size() * sizeof(float), cudaMemcpyHostToDevice);
    L.backward(d_dout);
    L.sync();
    std::vector<float> dx(B * N * C), dxm(B * N * C), dres(N * N);
    cudaMemcpy(dx.data(), L.grads.d_x_expanded, dx.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dxm.data(), L.grads.d_x_mixed, dxm.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(dres.data(), L.grads.d_H_res, dres.size() * 4, cudaMemcpyDeviceToHost);

    const float h = 1e-2f;
    printf("check A: dL/dx in grads.d_x_expanded (d_x_mixed and the sum are shown for reference)\n");
    double worst = 0, scale = 1e-6;
    for (int i : {0, 5, 37}) {
        auto xp = x, xm = x;
        xp[i] += h;
        xm[i] -= h;
        float fd = (run_loss(L, xp, dout, bpre, bpost, bres, gamma) -
                    run_loss(L, xm, dout, bpre, bpost, bres, gamma)) / (2 * h);
        printf("  x[%2d]: d_x_expanded=%9.5f  d_x_mixed=%9.5f  sum=%9.5f  finite-diff=%9.5f\n", i,
               dx[i], dxm[i], dx[i] + dxm[i], fd);
        worst = fmax(worst, fabs(dx[i] - fd));
        scale = fmax(scale, fabs(fd));
    }
    verdict("A", worst, scale);

    printf("check B: grads.d_H_res (H_res enters through exp, then Sinkhorn-Knopp)\n");
    worst = 0;
    scale = 1e-6;
    for (int i : {0, 1, 6, 12}) {
        auto rp = bres, rm = bres;
        rp[i] += h;
        rm[i] -= h;
        float fd = (run_loss(L, x, dout, bpre, bpost, rp, gamma) -
                    run_loss(L, x, dout, bpre, bpost, rm, gamma)) / (2 * h);
        printf("  h_res[%2d]: d_H_res=%9.5f  finite-diff=%9.5f  ratio=%.2f\n", i, dres[i], fd,
               dres[i] / fd);
        worst = fmax(worst, fabs(dres[i] - fd));
        scale = fmax(scale, fabs(fd));
    }
    verdict("B", worst, scale);
    cudaFree(d_dout);
}

// sinkhorn_knopp_backward alone: K in [0.2, 1.2], loss = sum(G * sinkhorn(K)).
static float sk_loss(const std::vector<float>& K, const std::vector<float>& G, int n, int iters,
                     float* d_in, float* d_out) {
    cudaMemcpy(d_in, K.data(), K.size() * 4, cudaMemcpyHostToDevice);
    sinkhorn_knopp_forward(d_out, d_in, n, n, iters, 1e-8f);
    cudaDeviceSynchronize();
    std::vector<float> y(n * n);
    cudaMemcpy(y.data(), d_out, y.size() * 4, cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < n * n; i++) s += (double)G[i] * y[i];
    return (float)s;
}

static void sinkhorn_check(const char* name, int n, int iters) {
    std::vector<float> K(n * n), G(n * n), dk(n * n);
    for (int i = 0; i < n * n; i++) {
        K[i] = 0.7f + 0.5f * sinf((float)i * 0.37f + 1.0f);
        G[i] = cosf((float)i * 0.11f + 0.2f);
    }
    float *d_in, *d_out, *d_g, *d_dk;
    cudaMalloc(&d_in, n * n * 4);
    cudaMalloc(&d_out, n * n * 4);
    cudaMalloc(&d_g, n * n * 4);
    cudaMalloc(&d_dk, n * n * 4);
    cudaMemcpy(d_g, G.data(), n * n * 4, cudaMemcpyHostToDevice);
    sk_loss(K, G, n, iters, d_in, d_out);  // leaves K in d_in, sinkhorn(K) in d_out
    sinkhorn_knopp_backward(d_dk, d_g, d_out, d_in, n, iters, 1e-8f);
    cudaDeviceSynchronize();
    cudaMemcpy(dk.data(), d_dk, n * n * 4, cudaMemcpyDeviceToHost);

    printf("check %s: sinkhorn_knopp_backward, N=%d, %d iterations\n", name, n, iters);
    const float h = 1e-2f;
    double worst = 0, scale = 1e-9;
    for (int i : {0, 1, n + 2, n * n - 1}) {
        auto kp = K, km = K;
        kp[i] += h;
        km[i] -= h;
        float fd = (sk_loss(kp, G, n, iters, d_in, d_out) - sk_loss(km, G, n, iters, d_in, d_out)) /
                   (2 * h);
        printf("  dK[%4d]: backward=%11.6f  finite-diff=%11.6f  ratio=%.2f\n", i, dk[i], fd,
               dk[i] / fd);
        worst = fmax(worst, fabs(dk[i] - fd));
        scale = fmax(scale, fabs(fd));
    }
    verdict(name, worst, scale);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_g);
    cudaFree(d_dk);
}

int main() {
    layer_checks();
    sinkhorn_check("C", 4, 20);
    sinkhorn_check("D", 40, 5);
    printf("%s (%d failing)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
