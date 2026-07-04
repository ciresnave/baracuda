// On-device validation of the increment-6 SCAN kernels (prefix scan):
//   * the serial-fold BASE (VariantFidelity::BitIdentical) — bit-exact vs a CPU
//     float-serial oracle scanned in the MATCHING direction (memcmp / NaN-aware);
//   * the block-scan VARIANT (ReassociatedDeterministic, FP Sum/Prod) — run-to-run
//     stable, within-ULP of an f64 oracle, degenerate single-chunk within-ULP.
//
// The generated .cu kernels are #included by name (they must sit in this dir — the
// `dump_scan_sources` test regenerates them; see ondevice/README.md). Kernel
// signature is uniform: `(const T* in0, T* out, long long n_out, long long k)`.
//
// Build (pure, self-contained):
//   nvcc -O3 -arch=sm_89 scan_validate.cu -o scan_validate && ./scan_validate
// Sanitizers (small shapes via the `san` argv):
//   compute-sanitizer --tool memcheck  ./scan_validate san
//   compute-sanitizer --tool racecheck ./scan_validate san
//   compute-sanitizer --tool synccheck ./scan_validate san
//   compute-sanitizer --tool initcheck ./scan_validate san
// Extract-the-delta audit vs the bespoke naive scan (adds the -run launcher):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -DWITH_BESPOKE -I <kernels-sys>/kernels/include scan_validate.cu -o scan_validate
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---- generated serial base (f32: 4 combines x incl/excl x fwd/rev) ----
#include "baracuda_gen_cumsum_f32_scan_sum.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_excl.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_rev.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_rev_excl.cu"
#include "baracuda_gen_cumprod_f32_scan_prod.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_excl.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_rev.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_rev_excl.cu"
#include "baracuda_gen_cummax_f32_scan_max.cu"
#include "baracuda_gen_cummax_f32_scan_max_excl.cu"
#include "baracuda_gen_cummax_f32_scan_max_rev.cu"
#include "baracuda_gen_cummax_f32_scan_max_rev_excl.cu"
#include "baracuda_gen_cummin_f32_scan_min.cu"
#include "baracuda_gen_cummin_f32_scan_min_excl.cu"
#include "baracuda_gen_cummin_f32_scan_min_rev.cu"
#include "baracuda_gen_cummin_f32_scan_min_rev_excl.cu"
// f64 base (double-precision bit-exact case)
#include "baracuda_gen_cumsum_f64_scan_sum.cu"
// block-scan variant (FP Sum/Prod, incl/excl)
#include "baracuda_gen_cumsum_f32_scan_sum_blockscan.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_excl_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_excl_blockscan.cu"

#ifdef WITH_BESPOKE
#include "scan_cumsum_fp.cu" // bespoke naive one-thread-per-cell (audit; -I .../elementwise)
#endif

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    fails++; } } while (0)

static int fails = 0;

enum Combine { SUM, PROD, MAX, MIN };
static const char* cname(Combine c) {
    return c == SUM ? "sum" : c == PROD ? "prod" : c == MAX ? "max" : "min";
}

// ---- CPU float oracle: EXACTLY mirrors the generated serial base (float acc,
// same op order, same direction) so a correct base is memcmp-bit-exact to it. ----
static void cpu_scan_f32(const float* x, float* y, long long rows, long long k,
                         Combine c, bool reverse, bool exclusive) {
    for (long long r = 0; r < rows; ++r) {
        const float* xr = x + r * k;
        float* yr = y + r * k;
        float acc = (c == SUM) ? 0.0f : (c == PROD) ? 1.0f : 0.0f; // max/min use `have`
        bool have = false;
        for (long long t = 0; t < k; ++t) {
            long long j = reverse ? (k - 1 - t) : t;
            float v = xr[j];
            if (exclusive) {
                float prefix;
                if (c == SUM || c == PROD) prefix = acc;
                else prefix = have ? acc : (c == MAX ? -INFINITY : INFINITY);
                yr[j] = prefix;
            }
            // combine
            if (c == SUM) acc = acc + v;
            else if (c == PROD) acc = acc * v;
            else { // MAX / MIN (NaN-propagating: v != v takes the NaN)
                bool take = !have || v != v || (c == MAX ? v > acc : v < acc);
                if (take) { acc = v; have = true; }
            }
            if (!exclusive) yr[j] = acc;
        }
    }
}

// ---- CPU f64 oracle for the block-scan within-ULP check (Sum/Prod only). ----
static void cpu_scan_f64(const float* x, double* y, long long rows, long long k,
                         Combine c, bool exclusive) {
    for (long long r = 0; r < rows; ++r) {
        const float* xr = x + r * k;
        double* yr = y + r * k;
        double acc = (c == SUM) ? 0.0 : 1.0;
        for (long long j = 0; j < k; ++j) {
            double v = (double)xr[j];
            if (exclusive) yr[j] = acc;
            acc = (c == SUM) ? acc + v : acc * v;
            if (!exclusive) yr[j] = acc;
        }
    }
}

static float* dev_f32(const std::vector<float>& h) {
    float* d = nullptr; cudaMalloc((void**)&d, h.size() * 4);
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    return d;
}

// Launch the serial BASE for (combine, reverse, exclusive). Thread 0 owns the row.
static void launch_base_f32(Combine c, bool rev, bool exc,
                            const float* d_in, float* d_out, long long n_out, long long k) {
    int g = (int)(n_out < 65535 ? n_out : 65535); if (g < 1) g = 1;
    const int b = 1; // only thread 0 works in the base
#define B(K) K<<<g, b>>>(d_in, d_out, n_out, k)
    if (c == SUM)  { if (!rev && !exc) B(baracuda_gen_cumsum_f32_scan_sum);
                     else if (!rev && exc) B(baracuda_gen_cumsum_f32_scan_sum_excl);
                     else if (rev && !exc) B(baracuda_gen_cumsum_f32_scan_sum_rev);
                     else B(baracuda_gen_cumsum_f32_scan_sum_rev_excl); }
    else if (c == PROD) { if (!rev && !exc) B(baracuda_gen_cumprod_f32_scan_prod);
                     else if (!rev && exc) B(baracuda_gen_cumprod_f32_scan_prod_excl);
                     else if (rev && !exc) B(baracuda_gen_cumprod_f32_scan_prod_rev);
                     else B(baracuda_gen_cumprod_f32_scan_prod_rev_excl); }
    else if (c == MAX) { if (!rev && !exc) B(baracuda_gen_cummax_f32_scan_max);
                     else if (!rev && exc) B(baracuda_gen_cummax_f32_scan_max_excl);
                     else if (rev && !exc) B(baracuda_gen_cummax_f32_scan_max_rev);
                     else B(baracuda_gen_cummax_f32_scan_max_rev_excl); }
    else { if (!rev && !exc) B(baracuda_gen_cummin_f32_scan_min);
                     else if (!rev && exc) B(baracuda_gen_cummin_f32_scan_min_excl);
                     else if (rev && !exc) B(baracuda_gen_cummin_f32_scan_min_rev);
                     else B(baracuda_gen_cummin_f32_scan_min_rev_excl); }
#undef B
}

// Launch the block-scan VARIANT (Sum/Prod, incl/excl). blockDim multiple of 32.
static void launch_block_f32(Combine c, bool exc,
                             const float* d_in, float* d_out, long long n_out, long long k, int block) {
    int g = (int)(n_out < 65535 ? n_out : 65535); if (g < 1) g = 1;
#define B(K) K<<<g, block>>>(d_in, d_out, n_out, k)
    if (c == SUM) { if (!exc) B(baracuda_gen_cumsum_f32_scan_sum_blockscan);
                    else B(baracuda_gen_cumsum_f32_scan_sum_excl_blockscan); }
    else { if (!exc) B(baracuda_gen_cumprod_f32_scan_prod_blockscan);
           else B(baracuda_gen_cumprod_f32_scan_prod_excl_blockscan); }
#undef B
}

// NaN-aware EXACT compare (bit-exact where finite; class-match where NaN).
static bool cmp_exact_nan(const char* nm, const std::vector<float>& got,
                          const std::vector<float>& oracle) {
    bool ok = true; long long bad = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = got[i], o = oracle[i];
        bool eq = std::isnan(o) ? std::isnan(g)
                                : (memcmp(&g, &o, 4) == 0); // bit-exact (incl signed zero)
        if (!eq) { ok = false; if (bad < 0) bad = (long long)i; }
    }
    printf(ok ? "PASS %-30s (bit-exact vs float oracle)\n"
              : "FAIL %-30s (first diff @%lld got %g want %g)\n",
           nm, ok ? 0 : bad, ok ? 0.0 : (double)got[bad], ok ? 0.0 : (double)oracle[bad]);
    if (!ok) fails++;
    return ok;
}

// ---- BASE: 4 combines x incl/excl x fwd/rev, bit-exact vs the float oracle. ----
static void base_matrix(long long rows, long long k, const char* tag) {
    std::vector<float> in((size_t)rows * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(((int)(i % 41) - 20)) * 0.125f;
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
    for (int ci = 0; ci < 4; ++ci) {
        Combine c = (Combine)ci;
        for (int rev = 0; rev < 2; ++rev)
            for (int exc = 0; exc < 2; ++exc) {
                cudaMemset(d_out, 0xAA, in.size() * 4);
                launch_base_f32(c, rev, exc, d_in, d_out, rows, k);
                cudaDeviceSynchronize();
                std::vector<float> got(in.size());
                cudaMemcpy(got.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
                std::vector<float> oracle(in.size());
                cpu_scan_f32(in.data(), oracle.data(), rows, k, c, rev, exc);
                char nm[96];
                snprintf(nm, sizeof nm, "base %s%s%s %s", cname(c),
                         rev ? "/rev" : "", exc ? "/excl" : "", tag);
                cmp_exact_nan(nm, got, oracle);
            }
    }
    cudaFree(d_in); cudaFree(d_out);
}

// ---- signed-zero: an all-(-0.0) row summed with a 0.0f seed maps to +0. ----
static void signed_zero_case() {
    const long long rows = 1, k = 8;
    std::vector<float> in((size_t)rows * k, -0.0f);
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
    launch_base_f32(SUM, false, false, d_in, d_out, rows, k);
    cudaDeviceSynchronize();
    std::vector<float> got(in.size());
    cudaMemcpy(got.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    unsigned bits; memcpy(&bits, &got[k - 1], 4);
    // 0.0f + (-0.0f) = +0.0f, so the running inclusive sum is +0 at every position.
    bool ok = (bits == 0x00000000u);
    printf(ok ? "PASS %-30s (all -0.0 sum -> +0, bits 0x%08x)\n"
              : "FAIL %-30s (bits 0x%08x, expected 0x00000000)\n",
           "signed_zero_sum", bits);
    if (!ok) fails++;
    cudaFree(d_in); cudaFree(d_out);
}

// ---- NaN propagation: plant a NaN; downstream (in scan direction) is NaN. ----
static void nan_case() {
    const long long rows = 1, k = 8; const int p = 3; // NaN position
    std::vector<float> in((size_t)rows * k);
    for (long long i = 0; i < k; ++i) in[i] = (float)(i + 1);
    in[p] = nanf("");
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
    for (int ci = 0; ci < 4; ++ci) {
        Combine c = (Combine)ci;
        launch_base_f32(c, false, false, d_in, d_out, rows, k); // forward inclusive
        cudaDeviceSynchronize();
        std::vector<float> got(in.size());
        cudaMemcpy(got.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (long long j = 0; j < k; ++j) {
            if (j < p) { if (std::isnan(got[j])) ok = false; }      // upstream unaffected
            else { if (!std::isnan(got[j])) ok = false; }           // NaN propagates forward
        }
        char nm[64]; snprintf(nm, sizeof nm, "nan_prop %s", cname(c));
        printf(ok ? "PASS %-30s (NaN@%d propagates forward)\n"
                  : "FAIL %-30s (NaN@%d not propagated)\n", nm, p);
        if (!ok) fails++;
    }
    cudaFree(d_in); cudaFree(d_out);
}

// ---- edge shapes: empty (k=0 untouched), single element (k=1). ----
static void edge_shapes() {
    // k = 0: the kernel returns immediately; the output must be UNTOUCHED.
    {
        const long long rows = 4, k = 0;
        float* d_out = nullptr; cudaMalloc((void**)&d_out, rows * 1 * 4 + 4);
        cudaMemset(d_out, 0xAB, 4);
        float* d_in = nullptr; cudaMalloc((void**)&d_in, 4);
        launch_base_f32(SUM, false, false, d_in, d_out, rows, k);
        cudaDeviceSynchronize();
        unsigned bits; cudaMemcpy(&bits, d_out, 4, cudaMemcpyDeviceToHost);
        bool ok = (bits == 0xABABABABu);
        printf(ok ? "PASS %-30s (k=0 leaves output untouched)\n"
                  : "FAIL %-30s (k=0 wrote 0x%08x)\n", "empty_row", bits);
        if (!ok) fails++;
        cudaFree(d_in); cudaFree(d_out);
    }
    // k = 1: inclusive out=in; exclusive out=identity (0 for sum).
    {
        const long long rows = 3, k = 1;
        std::vector<float> in{2.5f, -7.0f, 4.0f};
        float* d_in = dev_f32(in);
        float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
        launch_base_f32(SUM, false, false, d_in, d_out, rows, k);
        cudaDeviceSynchronize();
        std::vector<float> inc(in.size());
        cudaMemcpy(inc.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
        launch_base_f32(SUM, false, true, d_in, d_out, rows, k);
        cudaDeviceSynchronize();
        std::vector<float> exc(in.size());
        cudaMemcpy(exc.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
        bool ok = true;
        for (size_t i = 0; i < in.size(); ++i) { if (inc[i] != in[i] || exc[i] != 0.0f) ok = false; }
        printf(ok ? "PASS %-30s (incl=in, excl=identity)\n"
                  : "FAIL %-30s\n", "single_element_row");
        if (!ok) fails++;
        cudaFree(d_in); cudaFree(d_out);
    }
}

// ---- BLOCK-SCAN variant: run-to-run stable, within-ULP of f64, degenerate
//      within-ULP of base. (Sum/Prod, incl/excl.) ----
static void block_variant(Combine c, bool exc, long long rows, long long k, int block, const char* tag) {
    std::vector<float> in((size_t)rows * k);
    for (size_t i = 0; i < in.size(); ++i) {
        // small magnitudes so cumprod stays finite over long rows.
        in[i] = (c == PROD) ? (0.999f + (float)((i % 7)) * 0.0003f)
                            : (float)(((int)(i % 61) - 30)) * 0.05f;
    }
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
    float* d_base = nullptr; cudaMalloc((void**)&d_base, in.size() * 4);

    // within-ULP of an f64 oracle
    launch_block_f32(c, exc, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> got(in.size());
    cudaMemcpy(got.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<double> oracle(in.size());
    cpu_scan_f64(in.data(), oracle.data(), rows, k, c, exc);
    double maxrel = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        double den = fabs(oracle[i]) > 1.0 ? fabs(oracle[i]) : 1.0;
        maxrel = fmax(maxrel, fabs((double)got[i] - oracle[i]) / den);
    }
    // Depth-aware bound: an f32 scan of k elements accumulates ~sqrt(k)*eps rounding
    // (the reassociated tree is deterministic — run-to-run PASSES — this only sizes
    // the value-correctness tolerance for deep rows; a fixed 1e-5 is right only for
    // shallow rows). A well-conditioned k=16384 f32 sum genuinely carries ~1.4e-5.
    double tol = fmax(1e-5, 5e-7 * sqrt((double)k));
    bool okrel = maxrel < tol;
    char nm[96]; snprintf(nm, sizeof nm, "block %s%s %s ulp", cname(c), exc ? "/excl" : "", tag);
    printf(okrel ? "PASS %-30s (relerr %.2e < %.1e, k=%lld)\n" : "FAIL %-30s (relerr %.2e >= %.1e)\n",
           nm, maxrel, tol, okrel ? k : 0);
    if (!okrel) fails++;

    // run-to-run determinism (memcmp of two identical launches)
    cudaMemset(d_out, 0x00, in.size() * 4);
    launch_block_f32(c, exc, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> a(in.size()); cudaMemcpy(a.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemset(d_out, 0xAA, in.size() * 4);
    launch_block_f32(c, exc, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> b(in.size()); cudaMemcpy(b.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    bool okdet = memcmp(a.data(), b.data(), in.size() * 4) == 0;
    snprintf(nm, sizeof nm, "block %s%s %s determ", cname(c), exc ? "/excl" : "", tag);
    printf(okdet ? "PASS %-30s (run-to-run memcmp)\n" : "FAIL %-30s\n", nm);
    if (!okdet) fails++;

    // degenerate single-chunk (k<=block): within-ULP of the serial base (the tree
    // reassociates, so NOT memcmp-identical for FP Sum/Prod — the honest verdict).
    if (k <= block) {
        launch_base_f32(c, false, exc, d_in, d_base, rows, k);
        cudaDeviceSynchronize();
        std::vector<float> bs(in.size());
        cudaMemcpy(bs.data(), d_base, in.size() * 4, cudaMemcpyDeviceToHost);
        double mr = 0;
        for (size_t i = 0; i < in.size(); ++i) {
            double den = fabs((double)bs[i]) > 1.0 ? fabs((double)bs[i]) : 1.0;
            mr = fmax(mr, fabs((double)got[i] - (double)bs[i]) / den);
        }
        bool okd = mr < 1e-5;
        snprintf(nm, sizeof nm, "block %s%s %s degen", cname(c), exc ? "/excl" : "", tag);
        printf(okd ? "PASS %-30s (1-chunk within-ULP of base, relerr %.2e)\n"
                   : "FAIL %-30s (relerr %.2e)\n", nm, mr);
        if (!okd) fails++;
    }
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_base);
}

// ---- f64 base bit-exact vs a double oracle. ----
static void f64_base_case() {
    const long long rows = 4, k = 17;
    std::vector<double> in((size_t)rows * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = ((double)(i % 23) - 11) * 0.1;
    double* d_in = nullptr; cudaMalloc((void**)&d_in, in.size() * 8);
    cudaMemcpy(d_in, in.data(), in.size() * 8, cudaMemcpyHostToDevice);
    double* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 8);
    baracuda_gen_cumsum_f64_scan_sum<<<(unsigned)rows, 1>>>(d_in, d_out, rows, k);
    cudaDeviceSynchronize();
    std::vector<double> got(in.size());
    cudaMemcpy(got.data(), d_out, in.size() * 8, cudaMemcpyDeviceToHost);
    std::vector<double> oracle(in.size());
    for (long long r = 0; r < rows; ++r) { double a = 0; for (long long j = 0; j < k; ++j) { a += in[r*k+j]; oracle[r*k+j] = a; } }
    bool ok = memcmp(got.data(), oracle.data(), in.size() * 8) == 0;
    printf(ok ? "PASS %-30s (f64 memcmp-exact)\n" : "FAIL %-30s\n", "base sum f64");
    if (!ok) fails++;
    cudaFree(d_in); cudaFree(d_out);
}

#ifdef WITH_BESPOKE
// ---- Extract-the-delta: generated serial base vs the bespoke naive one-thread-
//      per-cell scan (same math order) — memcmp-exact for f32 cumsum. ----
static void bespoke_audit() {
    const long long rows = 64, k = 777;
    std::vector<float> in((size_t)rows * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(((int)(i % 51) - 25)) * 0.1f;
    float* d_in = dev_f32(in);
    float* d_gen = nullptr;  cudaMalloc((void**)&d_gen, in.size() * 4);
    float* d_bes = nullptr;  cudaMalloc((void**)&d_bes, in.size() * 4);
    int shape[2] = {(int)rows, (int)k};
    long long sx[2] = {k, 1}, sy[2] = {k, 1};
    for (int rev = 0; rev < 2; ++rev) {
        launch_base_f32(SUM, rev != 0, false, d_in, d_gen, rows, k);
        baracuda_kernels_scan_cumsum_f32_run(
            rows * k, 2, shape, sx, sy, /*scan_axis*/1, /*scan_extent*/(int)k,
            /*scan_stride_x*/1, rev, d_in, d_bes, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> g(in.size()), b(in.size());
        cudaMemcpy(g.data(), d_gen, in.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d_bes, in.size() * 4, cudaMemcpyDeviceToHost);
        bool ok = memcmp(g.data(), b.data(), in.size() * 4) == 0;
        printf(ok ? "PASS %-30s (gen base == bespoke, memcmp)\n" : "FAIL %-30s\n",
               rev ? "audit cumsum rev" : "audit cumsum fwd");
        if (!ok) fails++;
    }
    // bandwidth line (memory-bound; GB/s read+write vs copy peak ~195 on this card).
    // gen base = one-thread-PER-ROW (O(numel)); gen blockscan = cooperative; bespoke
    // = one-thread-PER-CELL (O(numel*extent) — quadratic in the scanned extent).
    {
        const long long R = 4096, K = 4096; const long long tot = R * K;
        std::vector<float> big(tot, 1.0f);
        float* dx = dev_f32(big); float* dy = nullptr; cudaMalloc((void**)&dy, tot * 4);
        int shp[2] = {(int)R, (int)K}; long long sx2[2] = {K, 1}, sy2[2] = {K, 1};
        auto timeit = [&](auto fn) {
            cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
            for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
            for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
            float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / 20;
        };
        double gb = tot * 4.0 * 2 / 1e9; // read + write
        float t_base = timeit([&] { launch_base_f32(SUM, false, false, dx, dy, R, K); });
        float t_blk  = timeit([&] { launch_block_f32(SUM, false, dx, dy, R, K, 256); });
        float t_bes  = timeit([&] {
            baracuda_kernels_scan_cumsum_f32_run(tot, 2, shp, sx2, sy2, 1, (int)K, 1, 0,
                                                 dx, dy, nullptr, 0, nullptr);
        });
        printf("[bench] cumsum %lldx%lld (read+write %.2f GB): base %.3f ms %.1f GB/s | "
               "blockscan %.3f ms %.1f GB/s | bespoke(naive) %.3f ms %.1f GB/s | base %.0fx bespoke\n",
               R, K, gb, t_base, gb / (t_base / 1000), t_blk, gb / (t_blk / 1000),
               t_bes, gb / (t_bes / 1000), t_bes / t_base);
        cudaFree(dx); cudaFree(dy);
    }
    cudaFree(d_in); cudaFree(d_gen); cudaFree(d_bes);
}
#endif

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== scan_validate (increment 6) ==\n");
    if (san) {
        // Small shapes for compute-sanitizer (memcheck/racecheck/synccheck/initcheck).
        base_matrix(4, 9, "small");
        signed_zero_case();
        nan_case();
        edge_shapes();
        block_variant(SUM, false, 4, 40, 32, "small");   // multi-chunk (k>block)
        block_variant(SUM, true, 4, 40, 32, "small");
        block_variant(PROD, false, 4, 20, 32, "small");
        f64_base_case();
    } else {
        printf("- serial BASE (bit-exact vs float oracle) -\n");
        base_matrix(8, 13, "8x13");
        base_matrix(64, 200, "64x200");
        f64_base_case();
        printf("- FP boundary semantics -\n");
        signed_zero_case();
        nan_case();
        edge_shapes();
        printf("- block-scan VARIANT (reassociated Sum/Prod) -\n");
        block_variant(SUM, false, 64, 100, 256, "degen100");     // single chunk
        block_variant(SUM, false, 64, 16384, 256, "multiwarp");  // 64 KB row, cross-warp+chunk
        block_variant(SUM, true, 64, 16384, 256, "multiwarp");
        block_variant(PROD, false, 64, 4096, 256, "multiwarp");
        block_variant(PROD, true, 64, 100, 256, "degen100");
#ifdef WITH_BESPOKE
        printf("- extract-the-delta audit vs bespoke naive scan -\n");
        bespoke_audit();
#endif
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); fails++; }
    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
