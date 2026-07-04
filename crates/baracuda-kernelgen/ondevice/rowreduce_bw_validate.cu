// rowreduce_bw_validate.cu — increment-2 compound-backward RowReduce audit.
//
// Launches the GENERATED fused backward kernels (one block per row, block-parallel
// tree reduce) and diffs them against (a) an f64 CPU oracle and (b) the BESPOKE
// baracuda-kernels-sys siblings called through their own launchers (the path
// dispatch actually takes). The generated kernels are the increment-2 proof
// vehicles: a second row-streamed input (softmax bw) and per-row saved-stat
// scalars hoisted once per row (layer_norm bw dx).
//
//   softmax bw:      dx[j] = y[j]·(dy[j] - Σ_l y[l]·dy[l])          (y, dy streamed)
//   layer_norm bw dx: x_hat=(x-mean)·rstd;
//                     dx = rstd·(dy - mean(dy) - x_hat·mean(dy·x_hat))
//                     (x, dy streamed; mean, rstd per-row scalars)
//
// Build (from a VS dev shell so nvcc finds cl.exe):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <kernels-sys>/kernels/include rowreduce_bw_validate.cu -o rowreduce_bw_validate
//   compute-sanitizer --tool memcheck  ./rowreduce_bw_validate
//   compute-sanitizer --tool racecheck ./rowreduce_bw_validate
//
// The generated .cu names track the bin/kernelgen.rs catalog cells; regenerate with
//   cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
// then copy this harness beside them (same convention as the other ondevice files).

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

// Bespoke siblings (templated launchers in baracuda::softmax / baracuda::norm).
#include "baracuda_softmax.cuh"
#include "baracuda_norm.cuh"

// Generated fused backward kernels (extern "C" __global__, headerless).
#include "baracuda_gen_softmax_bw_f32_rowreduce.cu"
#include "baracuda_gen_layer_norm_bw_f32_rowreduce.cu"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1);} } while(0)

static int g_pass = 0, g_fail = 0;

// Scale-relative L-inf error: worst |got-ref| normalized by the tensor's PEAK
// magnitude. Per-element relative error is meaningless for backward grads that
// legitimately cancel to near-zero (softmax/layer-norm dx sum ≈ 0 per row); this
// measures how far the worst element is off relative to the signal scale.
static double max_relerr(const std::vector<float>& got, const std::vector<double>& ref) {
    double maxabs = 0.0, maxref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        maxabs = std::max(maxabs, std::fabs((double)got[i] - ref[i]));
        maxref = std::max(maxref, std::fabs(ref[i]));
    }
    return maxabs / (maxref + 1e-30);
}
static double max_absdiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}
static void report(const char* name, double oracle, double bespoke, double tol) {
    bool ok = oracle <= tol;
    printf("  %-46s oracle relerr %.3e | vs bespoke absdiff %.3e  %s\n",
           name, oracle, bespoke, ok ? "PASS" : "FAIL");
    if (ok) ++g_pass; else ++g_fail;
}

// ---------------------------------------------------------------------------
// softmax backward
// ---------------------------------------------------------------------------
static void test_softmax_bw(int64_t n, int64_t k) {
    int64_t numel = n * k;
    std::mt19937 rng(1234 + (unsigned)(n * 131 + k));
    std::uniform_real_distribution<float> ud(-2.f, 2.f);

    // y = softmax(random logits) per row (a valid saved forward output);
    // dy = random upstream grad.
    std::vector<float> y(numel), dy(numel);
    for (int64_t i = 0; i < n; ++i) {
        double mx = -1e30, s = 0.0;
        std::vector<double> row(k);
        for (int64_t j = 0; j < k; ++j) { row[j] = ud(rng); mx = std::max(mx, row[j]); }
        for (int64_t j = 0; j < k; ++j) { row[j] = std::exp(row[j] - mx); s += row[j]; }
        for (int64_t j = 0; j < k; ++j) { y[i*k+j] = (float)(row[j] / s); dy[i*k+j] = ud(rng); }
    }
    // f64 oracle: dx = y·(dy - Σ y·dy).
    std::vector<double> ref(numel);
    for (int64_t i = 0; i < n; ++i) {
        double dot = 0.0;
        for (int64_t j = 0; j < k; ++j) dot += (double)y[i*k+j] * (double)dy[i*k+j];
        for (int64_t j = 0; j < k; ++j)
            ref[i*k+j] = (double)y[i*k+j] * ((double)dy[i*k+j] - dot);
    }

    float *dY, *dDy, *dDx_g, *dDx_b;
    CK(cudaMalloc(&dY, numel*4)); CK(cudaMalloc(&dDy, numel*4));
    CK(cudaMalloc(&dDx_g, numel*4)); CK(cudaMalloc(&dDx_b, numel*4));
    CK(cudaMemcpy(dY, y.data(), numel*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDy, dy.data(), numel*4, cudaMemcpyHostToDevice));

    // Generated: one block per row, block=256, grid-stride over rows.
    int block = 256, grid = (int)std::min<int64_t>(n, 65535);
    baracuda_gen_softmax_bw_f32_rowreduce<<<grid, block>>>(dY, dDy, dDx_g, n, k);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // Bespoke: launch_softmax_backward_fp(dy, y, dx, ...), softmax_axis = last.
    int32_t shape[2] = {(int32_t)n, (int32_t)k};
    int64_t st[2]    = {k, 1};
    int rc = baracuda::softmax::launch_softmax_backward_fp<float>(
        dDy, dY, dDx_b, numel, 2, shape, st, st, st,
        /*axis*/1, /*extent*/(int32_t)k, /*stride_dy*/1, /*stride_y*/1, 0);
    if (rc != 0) { printf("  bespoke softmax_bw launch rc=%d\n", rc); }
    CK(cudaDeviceSynchronize());

    std::vector<float> g(numel), b(numel);
    CK(cudaMemcpy(g.data(), dDx_g, numel*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), dDx_b, numel*4, cudaMemcpyDeviceToHost));
    char nm[64]; snprintf(nm, sizeof nm, "softmax_bw [%lldx%lld]", (long long)n, (long long)k);
    report(nm, max_relerr(g, ref), max_absdiff(g, b), 2e-4);

    CK(cudaFree(dY)); CK(cudaFree(dDy)); CK(cudaFree(dDx_g)); CK(cudaFree(dDx_b));
}

// ---------------------------------------------------------------------------
// layer_norm backward dx (gamma == 1; dx-only path)
// ---------------------------------------------------------------------------
static void test_layer_norm_bw(int64_t n, int64_t k) {
    int64_t numel = n * k;
    const float eps = 1e-5f;
    std::mt19937 rng(9871 + (unsigned)(n * 131 + k));
    std::uniform_real_distribution<float> ud(-3.f, 3.f);

    std::vector<float> x(numel), dy(numel), mean(n), rstd(n);
    for (int64_t i = 0; i < n; ++i) {
        double m = 0.0;
        for (int64_t j = 0; j < k; ++j) { x[i*k+j] = ud(rng); dy[i*k+j] = ud(rng); m += x[i*k+j]; }
        m /= (double)k;
        double v = 0.0;
        for (int64_t j = 0; j < k; ++j) { double d = (double)x[i*k+j] - m; v += d*d; }
        v /= (double)k;
        mean[i] = (float)m;
        rstd[i] = (float)(1.0 / std::sqrt(v + eps));   // rstd == inv_std
    }
    // f64 oracle: dx = rstd·(dy - mean(dy) - x_hat·mean(dy·x_hat)).
    std::vector<double> ref(numel);
    for (int64_t i = 0; i < n; ++i) {
        double mu = mean[i], rs = rstd[i];
        double sdxh = 0.0, sdxhxh = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            double dyj = dy[i*k+j];
            double xh  = ((double)x[i*k+j] - mu) * rs;
            sdxh += dyj; sdxhxh += dyj * xh;
        }
        double mdxh = sdxh / (double)k, mdxhxh = sdxhxh / (double)k;
        for (int64_t j = 0; j < k; ++j) {
            double xh = ((double)x[i*k+j] - mu) * rs;
            ref[i*k+j] = rs * ((double)dy[i*k+j] - mdxh - xh * mdxhxh);
        }
    }

    float *dX, *dDy, *dMean, *dRstd, *dDx_g, *dDx_b;
    CK(cudaMalloc(&dX, numel*4)); CK(cudaMalloc(&dDy, numel*4));
    CK(cudaMalloc(&dMean, n*4)); CK(cudaMalloc(&dRstd, n*4));
    CK(cudaMalloc(&dDx_g, numel*4)); CK(cudaMalloc(&dDx_b, numel*4));
    CK(cudaMemcpy(dX, x.data(), numel*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dDy, dy.data(), numel*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dMean, mean.data(), n*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dRstd, rstd.data(), n*4, cudaMemcpyHostToDevice));

    int block = 256, grid = (int)std::min<int64_t>(n, 65535);
    baracuda_gen_layer_norm_bw_f32_rowreduce<<<grid, block>>>(dX, dDy, dMean, dRstd, dDx_g, n, k);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    // Bespoke: launch(dy, x, gamma=null, mean, inv_std, dx, dgamma=null, dbeta=null, ...).
    int32_t shape[2] = {(int32_t)n, (int32_t)k};
    int64_t st[2]    = {k, 1};        // dy / x / dx strides
    int64_t ssv[2]   = {1, 0};        // saved-stat stride: [row]=1, [feature]=0
    int rc = baracuda::norm::launch_layer_norm_backward_fp<float>(
        dDy, dX, /*gamma*/nullptr, dMean, dRstd, dDx_b, /*dgamma*/nullptr, /*dbeta*/nullptr,
        numel, 2, shape, st, st, ssv, st,
        /*norm_axes_mask*/0b10, /*norm_total_extent*/(int32_t)k, 0);
    if (rc != 0) { printf("  bespoke layer_norm_bw launch rc=%d\n", rc); }
    CK(cudaDeviceSynchronize());

    std::vector<float> g(numel), b(numel);
    CK(cudaMemcpy(g.data(), dDx_g, numel*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), dDx_b, numel*4, cudaMemcpyDeviceToHost));
    char nm[64]; snprintf(nm, sizeof nm, "layer_norm_bw [%lldx%lld]", (long long)n, (long long)k);
    report(nm, max_relerr(g, ref), max_absdiff(g, b), 2e-4);

    CK(cudaFree(dX)); CK(cudaFree(dDy)); CK(cudaFree(dMean)); CK(cudaFree(dRstd));
    CK(cudaFree(dDx_g)); CK(cudaFree(dDx_b));
}

// ---------------------------------------------------------------------------
// Bench: generated vs bespoke GB/s on a large square-ish cell.
// ---------------------------------------------------------------------------
static float time_ms(std::function<void()> f, int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f(); CK(cudaDeviceSynchronize());               // warm up
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / iters;
}

// gi/bi = generated / bespoke iteration counts. The bespoke backwards are
// one-thread-per-cell with an inner O(extent) recompute (O(numel·extent) total),
// so at wide rows even a single iter is seconds — bi is kept tiny there.
static void bench(int64_t n, int64_t k, int gi, int bi) {
    int64_t numel = n * k;
    float *dY, *dDy, *dX, *dMean, *dRstd, *dDx;
    CK(cudaMalloc(&dY, numel*4)); CK(cudaMalloc(&dDy, numel*4)); CK(cudaMalloc(&dX, numel*4));
    CK(cudaMalloc(&dMean, n*4)); CK(cudaMalloc(&dRstd, n*4)); CK(cudaMalloc(&dDx, numel*4));
    CK(cudaMemset(dY, 1, numel*4)); CK(cudaMemset(dDy, 1, numel*4)); CK(cudaMemset(dX, 1, numel*4));
    CK(cudaMemset(dMean, 0, n*4)); CK(cudaMemset(dRstd, 1, n*4));
    int block = 256, grid = (int)std::min<int64_t>(n, 65535);
    int32_t shape[2] = {(int32_t)n, (int32_t)k};
    int64_t st[2] = {k, 1}, ssv[2] = {1, 0};
    double rw = 3.0 * (double)numel * 4.0; // ~read 2 + write 1

    float g_sm = time_ms([&]{ baracuda_gen_softmax_bw_f32_rowreduce<<<grid,block>>>(dY,dDy,dDx,n,k); }, gi);
    float b_sm = time_ms([&]{ baracuda::softmax::launch_softmax_backward_fp<float>(
        dDy,dY,dDx,numel,2,shape,st,st,st,1,(int32_t)k,1,1,0); }, bi);
    float g_ln = time_ms([&]{ baracuda_gen_layer_norm_bw_f32_rowreduce<<<grid,block>>>(dX,dDy,dMean,dRstd,dDx,n,k); }, gi);
    float b_ln = time_ms([&]{ baracuda::norm::launch_layer_norm_backward_fp<float>(
        dDy,dX,nullptr,dMean,dRstd,dDx,nullptr,nullptr,numel,2,shape,st,st,ssv,st,0b10,(int32_t)k,0); }, bi);

    printf("  bench [%lldx%lld]  softmax_bw:    gen %6.1f GB/s | bespoke %8.3f ms/%.0f GB/s | %.1fx\n",
           (long long)n,(long long)k, rw/g_sm/1e6, b_sm, rw/b_sm/1e6, b_sm/g_sm);
    printf("  bench [%lldx%lld]  layer_norm_bw: gen %6.1f GB/s | bespoke %8.3f ms/%.0f GB/s | %.1fx\n",
           (long long)n,(long long)k, rw/g_ln/1e6, b_ln, rw/b_ln/1e6, b_ln/g_ln);

    CK(cudaFree(dY)); CK(cudaFree(dDy)); CK(cudaFree(dX));
    CK(cudaFree(dMean)); CK(cudaFree(dRstd)); CK(cudaFree(dDx));
}

int main(int argc, char** argv) {
    // `san` mode: GENERATED kernels only, small shapes, no bench — keeps
    // compute-sanitizer (10–100× slowdown) fast and points memcheck/racecheck at the
    // kernels actually under test (block-reduce smem, the per-row-scalar hoist, the
    // dual row-streamed loads). The bespoke siblings are already-shipped, not tested.
    bool san = (argc > 1 && std::string(argv[1]) == "san");

    if (san) {
        printf("== sanitizer mode: generated kernels, small + non-square + partial-warp ==\n");
        const int64_t ss[][2] = {{7, 33}, {256, 128}, {1000, 777}, {3, 1290}};
        for (auto& s : ss) test_softmax_bw(s[0], s[1]);
        for (auto& s : ss) test_layer_norm_bw(s[0], s[1]);
        printf("\n%d passed, %d failed\n", g_pass, g_fail);
        return g_fail ? 1 : 0;
    }

    // Shapes: square-ish small, non-square, a >47KB row (16384*4=64KB — the
    // smemrow-variant regime), the catalog cell, and many-rows / tiny-k.
    const int64_t shapes[][2] = {
        {256, 128}, {1000, 777}, {64, 16384}, {4096, 1024}, {131072, 32},
    };
    printf("== correctness (generated vs f64 oracle; vs bespoke sibling) ==\n");
    for (auto& s : shapes) test_softmax_bw(s[0], s[1]);
    for (auto& s : shapes) test_layer_norm_bw(s[0], s[1]);

    printf("== bench (extract-the-delta; ~3·numel·4 bytes) ==\n");
    bench(8192, 2048, 100, 5);
    bench(2048, 16384, 50, 1);   // wide-row regime bespoke's per-thread recompute can't amortize

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
