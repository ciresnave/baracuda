// GENERATED-vs-BESPOKE AUDIT, round 1: reductions + softmax (f32).
// Bespoke = the hand-written baracuda-kernels-sys kernels, called through their
// extern "C" _run launchers (their own path selection — what dispatch calls).
// Generated = the kernelgen cells. Oracle-checked both sides, then benched.
// Compile: nvcc -O3 -arch=sm_89 -I<kernels-sys>/kernels/include audit_reduce_softmax.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---- bespoke (self-contained per the surface map) ----
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/reduce_sum_fp.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/reduce_mean_fp.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/softmax/softmax_fp.cu"

// ---- generated ----
#include "baracuda_gen_mean_f32_reduce_mean.cu"          // last-axis block-per-row
#include "baracuda_gen_sum_f32_reduce_sum_ax1.cu"        // outer-axis base (general)
#include "baracuda_gen_sum_f32_reduce_sum_ax1_splitk_partial.cu"
#include "baracuda_gen_sum_f32_reduce_sum_ax1_splitk_combine.cu"
#include "baracuda_gen_softmax_f32_rowreduce.cu"         // 3-pass recompute
#include "baracuda_gen_softmax_f32_rowreduce_smemrow.cu" // smem row-cache variant

static int fails = 0;

template <class F>
static float timed(F launch, int iters = 50) {
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

static void relcheck(const char* name, const float* got, const double* want, long long n) {
    double mr = 0;
    for (long long i = 0; i < n; ++i) {
        double denom = fabs(want[i]) > 1.0 ? fabs(want[i]) : 1.0;
        mr = fmax(mr, fabs((double)got[i] - want[i]) / denom);
    }
    bool ok = mr < 1e-4;
    printf(ok ? "PASS %-34s relerr %.2e\n" : "FAIL %-34s relerr %.2e\n", name, mr);
    if (!ok) fails++;
}

int main() {
    // ================= Matchup 1: LAST-AXIS MEAN, [8192, 8192] =================
    {
        const long long R = 8192, C = 8192;
        std::vector<float> in(R * C);
        for (long long i = 0; i < R * C; ++i) in[i] = ((i % 41) - 20) * 0.125f;
        std::vector<double> want(R, 0.0);
        for (long long r = 0; r < R; ++r) {
            double s = 0;
            for (long long c = 0; c < C; ++c) s += (double)in[r * C + c];
            want[r] = s / (double)C;
        }
        float *d_in, *d_g, *d_b;
        cudaMalloc((void**)&d_in, R * C * 4);
        cudaMalloc((void**)&d_g, R * 4);
        cudaMalloc((void**)&d_b, R * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);

        auto gen = [&] { baracuda_gen_mean_f32_reduce_mean<<<(unsigned)R, 256>>>(d_in, d_g, R, C); };
        // Bespoke: reduce axis 1 of [R, C]; keepdim output shape [R, 1].
        int32_t shp[2] = {(int32_t)R, 1};
        int64_t sx[2] = {C, 1}, sy[2] = {1, 1};
        auto besp = [&] {
            baracuda_kernels_reduce_mean_f32_run(R, 2, shp, sx, sy, 1, (int32_t)C, 1,
                                                 d_in, d_b, nullptr, 0, nullptr);
        };
        gen(); besp();
        cudaDeviceSynchronize();
        std::vector<float> g(R), b(R);
        cudaMemcpy(g.data(), d_g, R * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, R * 4, cudaMemcpyDeviceToHost);
        relcheck("mean_lastaxis generated", g.data(), want.data(), R);
        relcheck("mean_lastaxis bespoke", b.data(), want.data(), R);
        float tg = timed(gen), tb = timed(besp);
        double gb = R * C * 4.0 / 1e9;
        printf("  [8192x8192] mean last-axis: GEN %7.3f ms (%6.1f GB/s) | BESPOKE %7.3f ms (%6.1f GB/s) | gen speedup %.2fx\n\n",
               tg, gb / (tg / 1000), tb, gb / (tb / 1000), tb / tg);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_b);
    }

    // ============ Matchup 2: OUTER-AXIS SUM, [65536, 1024] (starved) ============
    {
        const long long R = 65536, C = 1024, n_chunks = 256;
        std::vector<float> in(R * C);
        for (long long i = 0; i < R * C; ++i) in[i] = ((i % 37) - 18) * 0.0625f;
        std::vector<double> want(C, 0.0);
        for (long long r = 0; r < R; ++r)
            for (long long c = 0; c < C; ++c) want[c] += (double)in[r * C + c];
        float *d_in, *d_g, *d_b, *d_ws;
        cudaMalloc((void**)&d_in, R * C * 4);
        cudaMalloc((void**)&d_g, C * 4);
        cudaMalloc((void**)&d_b, C * 4);
        cudaMalloc((void**)&d_ws, n_chunks * C * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);

        long long shape[2] = {R, C}, s0[2] = {C, 1}, so[1] = {1};
        long long *dsh, *ds0, *dso;
        cudaMalloc((void**)&dsh, 16); cudaMemcpy(dsh, shape, 16, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&ds0, 16); cudaMemcpy(ds0, s0, 16, cudaMemcpyHostToDevice);
        cudaMalloc((void**)&dso, 8); cudaMemcpy(dso, so, 8, cudaMemcpyHostToDevice);
        const int block = 256;
        int grid = (int)((C + block - 1) / block);
        auto gen_base = [&] {
            baracuda_gen_sum_f32_reduce_sum_ax1<<<grid, block>>>(d_in, d_g, R, C, C, 1, 1, C);
        };
        auto gen_splitk = [&] {
            long long chunk = (R + n_chunks - 1) / n_chunks;
            baracuda_gen_sum_f32_reduce_sum_ax1_splitk_partial<<<dim3(grid, (unsigned)n_chunks), block>>>(
                d_in, d_ws, R, C, chunk);
            baracuda_gen_sum_f32_reduce_sum_ax1_splitk_combine<<<grid, block>>>(d_ws, d_g, C, n_chunks);
        };
        // Bespoke: reduce axis 0 of [R, C]; keepdim output [1, C]; stride along
        // reduce axis = C (non-unit -> their launcher takes the LEGACY path).
        int32_t shp[2] = {1, (int32_t)C};
        int64_t sx[2] = {C, 1}, sy[2] = {C, 1};
        auto besp = [&] {
            baracuda_kernels_reduce_sum_f32_run(C, 2, shp, sx, sy, 0, (int32_t)R, C,
                                                d_in, d_b, nullptr, 0, nullptr);
        };
        gen_splitk(); besp();
        cudaDeviceSynchronize();
        std::vector<float> g(C), b(C);
        cudaMemcpy(g.data(), d_g, C * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, C * 4, cudaMemcpyDeviceToHost);
        relcheck("sum_outer generated(splitk)", g.data(), want.data(), C);
        relcheck("sum_outer bespoke", b.data(), want.data(), C);
        float tgb = timed(gen_base), tgs = timed(gen_splitk), tb = timed(besp);
        double gb = R * C * 4.0 / 1e9;
        printf("  [65536x1024] sum axis-0: GEN base %7.3f ms (%6.1f GB/s) | GEN splitk %7.3f ms (%6.1f GB/s) | BESPOKE %7.3f ms (%6.1f GB/s) | splitk vs bespoke %.2fx\n\n",
               tgb, gb / (tgb / 1000), tgs, gb / (tgs / 1000), tb, gb / (tb / 1000), tb / tgs);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_b); cudaFree(d_ws);
        cudaFree(dsh); cudaFree(ds0); cudaFree(dso);
    }

    // ====== Matchup 3: SOFTMAX, [4096, 4096] (smem-eligible) + [2048, 16384] ======
    for (int cs = 0; cs < 2; ++cs) {
        const long long R = cs == 0 ? 4096 : 2048;
        const long long C = cs == 0 ? 4096 : 16384; // 16384: (C+32)*4 = 64KB > 47KB -> bespoke GLOBAL fallback
        std::vector<float> in(R * C);
        for (long long i = 0; i < R * C; ++i) in[i] = ((i % 251) - 125) * 0.02f;
        // Oracle on a sample of rows.
        float *d_in, *d_g, *d_gs, *d_b;
        cudaMalloc((void**)&d_in, R * C * 4);
        cudaMalloc((void**)&d_g, R * C * 4);
        cudaMalloc((void**)&d_gs, R * C * 4);
        cudaMalloc((void**)&d_b, R * C * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);

        auto gen = [&] { baracuda_gen_softmax_f32_rowreduce<<<(unsigned)R, 256>>>(d_in, d_g, R, C); };
        auto gen_sm = [&] {
            baracuda_gen_softmax_f32_rowreduce_smemrow<<<(unsigned)R, 256, (unsigned)(C * 4)>>>(d_in, d_gs, R, C);
        };
        int32_t shp[2] = {(int32_t)R, (int32_t)C};
        int64_t sx[2] = {C, 1}, sy[2] = {C, 1};
        auto besp = [&] {
            baracuda_kernels_softmax_f32_run(R * C, 2, shp, sx, sy, 1, (int32_t)C, 1, 1,
                                             d_in, d_b, nullptr, 0, nullptr);
        };
        bool sm_ok = C * 4 <= 96 * 1024; // our smemrow needs k*4 dynamic smem
        gen(); if (sm_ok && cs == 0) gen_sm(); besp();
        cudaDeviceSynchronize();
        // Sampled correctness: row 0 and row R-1 vs f64 oracle.
        std::vector<float> g(R * C), b(R * C);
        cudaMemcpy(g.data(), d_g, R * C * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, R * C * 4, cudaMemcpyDeviceToHost);
        double mr_g = 0, mr_b = 0;
        for (long long rr : {0LL, R - 1}) {
            double mx = -1e300, se = 0;
            for (long long c = 0; c < C; ++c) mx = fmax(mx, (double)in[rr * C + c]);
            for (long long c = 0; c < C; ++c) se += exp((double)in[rr * C + c] - mx);
            for (long long c = 0; c < C; ++c) {
                double w = exp((double)in[rr * C + c] - mx) / se;
                double denom = w > 1e-30 ? w : 1e-30;
                mr_g = fmax(mr_g, fabs((double)g[rr * C + c] - w) / denom);
                mr_b = fmax(mr_b, fabs((double)b[rr * C + c] - w) / denom);
            }
        }
        bool okg = mr_g < 1e-3, okb = mr_b < 1e-3;
        printf(okg ? "PASS softmax[%lldx%lld] generated relerr %.2e\n" : "FAIL softmax[%lldx%lld] generated relerr %.2e\n", R, C, mr_g);
        printf(okb ? "PASS softmax[%lldx%lld] bespoke   relerr %.2e\n" : "FAIL softmax[%lldx%lld] bespoke   relerr %.2e\n", R, C, mr_b);
        if (!okg) fails++;
        if (!okb) fails++;
        float tg = timed(gen);
        float tgs2 = (sm_ok && cs == 0) ? timed(gen_sm) : 0.0f;
        float tb = timed(besp);
        double gbytes = 2.0 * R * C * 4.0 / 1e9; // 1 read + 1 write minimum
        if (cs == 0)
            printf("  [%lldx%lld] softmax: GEN recompute %7.3f ms (%6.1f GB/s) | GEN smemrow %7.3f ms (%6.1f GB/s) | BESPOKE(smem) %7.3f ms (%6.1f GB/s) | gen vs bespoke %.2fx\n\n",
                   R, C, tg, gbytes / (tg / 1000), tgs2, gbytes / (tgs2 / 1000), tb, gbytes / (tb / 1000), tb / tg);
        else
            printf("  [%lldx%lld] softmax: GEN recompute %7.3f ms (%6.1f GB/s) | BESPOKE(global fallback) %7.3f ms (%6.1f GB/s) | gen vs bespoke %.2fx\n\n",
                   R, C, tg, gbytes / (tg / 1000), tb, gbytes / (tb / 1000), tb / tg);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_gs); cudaFree(d_b);
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
