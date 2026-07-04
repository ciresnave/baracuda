// Increment 0e — reduction upgrades on device: (1) ReduceOp::Prod, (2) fused
// reduction post-expression, (3) hetero output dtype. Generated kernelgen cells
// vs a CPU reference AND (where a sibling exists) vs the hand-written
// baracuda-kernels-sys reduce kernels, called through their extern "C" _run
// launchers. Prod-int and Prod-fp / Norm2 have bespoke siblings; the hetero-out
// any/count reductions have no bespoke OpKind (CPU is the sole oracle).
//
// Compile (from a VS dev shell; the bespoke reduce headers want c++17):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <kernels-sys>/kernels/include reduction_upgrades_validate.cu -o ru
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

// ---- bespoke siblings ----
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/reduce_prod_fp.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/reduce_prod_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/reduce_norm2_fp.cu"

// ---- generated (regenerate with the README snippet) ----
#include "baracuda_gen_prod_f32_reduce_prod.cu"    // Prod fp, block_prod tree
#include "baracuda_gen_prod_i32_reduce_prod.cu"    // Prod int, long long accum
#include "baracuda_gen_norm2_f32_reduce_sum.cu"    // Sqrt(Sum(Sqr(x))) post-expr
#include "baracuda_gen_anyv_f32_reduce_sum.cu"     // any -> u8 (Cmp* post)
#include "baracuda_gen_countv_f32_reduce_sum.cu"   // count -> i64 (identity post)

static int fails = 0;

static void relcheck(const char* name, const float* got, const double* want, long long n) {
    double mr = 0;
    for (long long i = 0; i < n; ++i) {
        double denom = fabs(want[i]) > 1.0 ? fabs(want[i]) : 1.0;
        mr = fmax(mr, fabs((double)got[i] - want[i]) / denom);
    }
    bool ok = mr < 1e-4;
    printf(ok ? "PASS %-40s relerr %.2e\n" : "FAIL %-40s relerr %.2e\n", name, mr);
    if (!ok) fails++;
}

static void exact_i32(const char* name, const int32_t* got, const int32_t* want, long long n) {
    long long bad = 0;
    for (long long i = 0; i < n; ++i) if (got[i] != want[i]) bad++;
    printf(bad == 0 ? "PASS %-40s bit-exact\n" : "FAIL %-40s %lld mismatches\n", name, bad);
    if (bad) fails++;
}

static void exact_i64(const char* name, const int64_t* got, const int64_t* want, long long n) {
    long long bad = 0;
    for (long long i = 0; i < n; ++i) if (got[i] != want[i]) bad++;
    printf(bad == 0 ? "PASS %-40s bit-exact\n" : "FAIL %-40s %lld mismatches\n", name, bad);
    if (bad) fails++;
}

static void exact_u8(const char* name, const uint8_t* got, const uint8_t* want, long long n) {
    long long bad = 0;
    for (long long i = 0; i < n; ++i) if (got[i] != want[i]) bad++;
    printf(bad == 0 ? "PASS %-40s bit-exact\n" : "FAIL %-40s %lld mismatches\n", name, bad);
    if (bad) fails++;
}

int main() {
    const long long R = 1024;

    // ===================== (1a) Prod f32, last axis =====================
    {
        const long long C = 64; // values near 1 so the product stays O(1)
        std::vector<float> in(R * C);
        for (long long i = 0; i < R * C; ++i)
            in[i] = 0.9f + (float)((i * 7 + 3) % 21) * 0.01f; // [0.90, 1.10]
        std::vector<double> want(R);
        for (long long r = 0; r < R; ++r) {
            double p = 1.0;
            for (long long c = 0; c < C; ++c) p *= (double)in[r * C + c];
            want[r] = p;
        }
        float *d_in, *d_g, *d_b;
        cudaMalloc(&d_in, R * C * 4); cudaMalloc(&d_g, R * 4); cudaMalloc(&d_b, R * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);
        baracuda_gen_prod_f32_reduce_prod<<<(unsigned)R, 256>>>(d_in, d_g, R, C);
        int32_t shp[2] = {(int32_t)R, 1}; int64_t sx[2] = {C, 1}, sy[2] = {1, 1};
        baracuda_kernels_reduce_prod_f32_run(R, 2, shp, sx, sy, 1, (int32_t)C, 1,
                                             d_in, d_b, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> g(R), b(R);
        cudaMemcpy(g.data(), d_g, R * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, R * 4, cudaMemcpyDeviceToHost);
        relcheck("prod_f32 generated vs CPU(f64)", g.data(), want.data(), R);
        relcheck("prod_f32 bespoke   vs CPU(f64)", b.data(), want.data(), R);
        // generated vs bespoke: both fp, different fold order -> tolerance.
        std::vector<double> bwant(R);
        for (long long r = 0; r < R; ++r) bwant[r] = (double)b[r];
        relcheck("prod_f32 generated vs BESPOKE", g.data(), bwant.data(), R);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_b);
    }

    // ===================== (1b) Prod i32, last axis =====================
    {
        const long long C = 20; // 3^20 ~ 3.5e9: fits i64, WRAPS i32 (exercises the truncate)
        std::vector<int32_t> in(R * C);
        for (long long i = 0; i < R * C; ++i)
            in[i] = (int32_t)(((i * 13 + 5) % 7) - 3); // {-3..3}, includes 0
        std::vector<int32_t> want(R);
        for (long long r = 0; r < R; ++r) {
            int64_t p = 1; // widened accumulator, exactly as both kernels
            for (long long c = 0; c < C; ++c) p *= (int64_t)in[r * C + c];
            want[r] = (int32_t)p; // wrap-on-store
        }
        int32_t *d_in, *d_g, *d_b;
        cudaMalloc(&d_in, R * C * 4); cudaMalloc(&d_g, R * 4); cudaMalloc(&d_b, R * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);
        baracuda_gen_prod_i32_reduce_prod<<<(unsigned)R, 256>>>(d_in, d_g, R, C);
        int32_t shp[2] = {(int32_t)R, 1}; int64_t sx[2] = {C, 1}, sy[2] = {1, 1};
        baracuda_kernels_reduce_prod_i32_run(R, 2, shp, sx, sy, 1, (int32_t)C, 1,
                                             d_in, d_b, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<int32_t> g(R), b(R);
        cudaMemcpy(g.data(), d_g, R * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, R * 4, cudaMemcpyDeviceToHost);
        exact_i32("prod_i32 generated vs CPU(i64->i32)", g.data(), want.data(), R);
        exact_i32("prod_i32 generated vs BESPOKE", g.data(), b.data(), R);
        exact_i32("prod_i32 bespoke   vs CPU(i64->i32)", b.data(), want.data(), R);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_b);
    }

    // ================ (2) norm2 = Sqrt(Sum(Sqr(x))) post-expr ================
    {
        const long long C = 256;
        std::vector<float> in(R * C);
        for (long long i = 0; i < R * C; ++i)
            in[i] = (float)(((i * 11 + 1) % 41) - 20) * 0.125f;
        std::vector<double> want(R);
        for (long long r = 0; r < R; ++r) {
            double s = 0;
            for (long long c = 0; c < C; ++c) { double x = in[r * C + c]; s += x * x; }
            want[r] = sqrt(s);
        }
        float *d_in, *d_g, *d_b;
        cudaMalloc(&d_in, R * C * 4); cudaMalloc(&d_g, R * 4); cudaMalloc(&d_b, R * 4);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);
        baracuda_gen_norm2_f32_reduce_sum<<<(unsigned)R, 256>>>(d_in, d_g, R, C);
        int32_t shp[2] = {(int32_t)R, 1}; int64_t sx[2] = {C, 1}, sy[2] = {1, 1};
        baracuda_kernels_reduce_norm2_f32_run(R, 2, shp, sx, sy, 1, (int32_t)C, 1,
                                              d_in, d_b, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> g(R), b(R);
        cudaMemcpy(g.data(), d_g, R * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_b, R * 4, cudaMemcpyDeviceToHost);
        relcheck("norm2_f32 generated vs CPU(f64)", g.data(), want.data(), R);
        relcheck("norm2_f32 bespoke   vs CPU(f64)", b.data(), want.data(), R);
        std::vector<double> bwant(R);
        for (long long r = 0; r < R; ++r) bwant[r] = (double)b[r];
        relcheck("norm2_f32 generated vs BESPOKE", g.data(), bwant.data(), R);
        cudaFree(d_in); cudaFree(d_g); cudaFree(d_b);
    }

    // ============ (3a) hetero-out ANY -> u8 (Cmp* post, no bespoke) ============
    {
        const long long C = 128;
        std::vector<float> in(R * C, 0.0f);
        std::vector<uint8_t> want(R, 0);
        for (long long r = 0; r < R; ++r) {
            // Every 3rd row has exactly one nonzero; the rest are all-zero.
            if (r % 3 == 0) { in[r * C + (r % C)] = 2.5f; want[r] = 1; }
        }
        float *d_in; uint8_t *d_g;
        cudaMalloc(&d_in, R * C * 4); cudaMalloc(&d_g, R * 1);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);
        baracuda_gen_anyv_f32_reduce_sum<<<(unsigned)R, 256>>>(d_in, d_g, R, C);
        cudaDeviceSynchronize();
        std::vector<uint8_t> g(R);
        cudaMemcpy(g.data(), d_g, R * 1, cudaMemcpyDeviceToHost);
        exact_u8("any_u8 generated vs CPU", g.data(), want.data(), R);
        cudaFree(d_in); cudaFree(d_g);
    }

    // ============ (3b) hetero-out COUNT -> i64 (identity post, no bespoke) ============
    {
        const long long C = 128;
        std::vector<float> in(R * C);
        std::vector<int64_t> want(R, 0);
        for (long long r = 0; r < R; ++r)
            for (long long c = 0; c < C; ++c) {
                float v = ((r + c) % 4 == 0) ? 0.0f : 1.0f; // ~3/4 nonzero
                in[r * C + c] = v;
                if (v != 0.0f) want[r]++;
            }
        float *d_in; int64_t *d_g;
        cudaMalloc(&d_in, R * C * 4); cudaMalloc(&d_g, R * 8);
        cudaMemcpy(d_in, in.data(), R * C * 4, cudaMemcpyHostToDevice);
        baracuda_gen_countv_f32_reduce_sum<<<(unsigned)R, 256>>>(d_in, d_g, R, C);
        cudaDeviceSynchronize();
        std::vector<int64_t> g(R);
        cudaMemcpy(g.data(), d_g, R * 8, cudaMemcpyDeviceToHost);
        exact_i64("count_i64 generated vs CPU", g.data(), want.data(), R);
        cudaFree(d_in); cudaFree(d_g);
    }

    printf(fails == 0 ? "\nALL PASSED\n" : "\n%d FAILED\n", fails);
    return fails ? 1 : 0;
}
