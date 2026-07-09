// On-device validation of the SCAN kernels (prefix scan) — increment 6 base +
// the BLOCKSCAN-VARIANTS increment (reverse, Max/Min, integer block-scans):
//   * the serial-fold BASE (VariantFidelity::BitIdentical) — bit-exact vs a CPU
//     float-serial oracle scanned in the MATCHING direction (memcmp / NaN-aware);
//   * the block-scan VARIANT — one cooperative kernel serving Sum/Prod/Max/Min over
//     FP {f16,bf16,f32,f32-strict,f64} + integer {i32,i64}, forward + reverse. FP
//     Sum/Prod reassociate (ReassociatedDeterministic — within-ULP of an f64 oracle,
//     run-to-run stable); FP Max/Min + ALL integer scans are BitIdentical to the
//     serial base — proven here by whole-buffer raw-byte memcmp==0 vs the base, plus
//     the hand-derived FP Max/Min combiner probe rows (raw-bit expected).
//
// IEEE-STRICT REQUIREMENT: the FP Max/Min bit-identity relies on `v != v` NOT being
// folded to false and signed zeros NOT being flushed — build WITHOUT -ffast-math /
// --use_fast_math (nvcc/nvrtc default IEEE, matched FTZ). Same requirement the
// existing Sum/Prod block-scan already relies on.
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
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
// block-scan variant (FP Sum/Prod f32, incl/excl, forward — reassociated)
#include "baracuda_gen_cumsum_f32_scan_sum_blockscan.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_excl_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_excl_blockscan.cu"

// ---- BLOCKSCAN-VARIANTS: new blockscan + base pairs for the BitIdentical matrix
//      (FP Max/Min f32/f64/f16/bf16/f32-strict; integer Sum/Prod/Max/Min i32/i64;
//      forward + reverse; incl + excl) and the reverse FP Sum/Prod reassoc block. ----
#include "baracuda_gen_cumsum_f32_scan_sum_rev_blockscan.cu"
#include "baracuda_gen_cumsum_f32_scan_sum_rev_excl_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_rev_blockscan.cu"
#include "baracuda_gen_cumprod_f32_scan_prod_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_f32_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_f32_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_f32_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_f32_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_f32_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_f32_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_f32_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_f32_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_f64_scan_max.cu"
#include "baracuda_gen_cummax_f64_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_f64_scan_max_excl.cu"
#include "baracuda_gen_cummax_f64_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_f64_scan_max_rev.cu"
#include "baracuda_gen_cummax_f64_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_f64_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_f64_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_f64_scan_min.cu"
#include "baracuda_gen_cummin_f64_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_f64_scan_min_excl.cu"
#include "baracuda_gen_cummin_f64_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_f64_scan_min_rev.cu"
#include "baracuda_gen_cummin_f64_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_f64_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_f64_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_f16_scan_max.cu"
#include "baracuda_gen_cummax_f16_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_f16_scan_max_excl.cu"
#include "baracuda_gen_cummax_f16_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_f16_scan_max_rev.cu"
#include "baracuda_gen_cummax_f16_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_f16_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_f16_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_f16_scan_min.cu"
#include "baracuda_gen_cummin_f16_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_f16_scan_min_excl.cu"
#include "baracuda_gen_cummin_f16_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_f16_scan_min_rev.cu"
#include "baracuda_gen_cummin_f16_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_f16_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_f16_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_bf16_scan_max.cu"
#include "baracuda_gen_cummax_bf16_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_bf16_scan_max_excl.cu"
#include "baracuda_gen_cummax_bf16_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_bf16_scan_max_rev.cu"
#include "baracuda_gen_cummax_bf16_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_bf16_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_bf16_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_bf16_scan_min.cu"
#include "baracuda_gen_cummin_bf16_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_bf16_scan_min_excl.cu"
#include "baracuda_gen_cummin_bf16_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_bf16_scan_min_rev.cu"
#include "baracuda_gen_cummin_bf16_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_bf16_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_bf16_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_f32s_scan_max.cu"
#include "baracuda_gen_cummax_f32s_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_f32s_scan_max_excl.cu"
#include "baracuda_gen_cummax_f32s_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_f32s_scan_max_rev.cu"
#include "baracuda_gen_cummax_f32s_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_f32s_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_f32s_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_f32s_scan_min.cu"
#include "baracuda_gen_cummin_f32s_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_f32s_scan_min_excl.cu"
#include "baracuda_gen_cummin_f32s_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_f32s_scan_min_rev.cu"
#include "baracuda_gen_cummin_f32s_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_f32s_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_f32s_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cumsum_i32_scan_sum.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_blockscan.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_excl.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_excl_blockscan.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_rev.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_rev_blockscan.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_rev_excl.cu"
#include "baracuda_gen_cumsum_i32_scan_sum_rev_excl_blockscan.cu"
#include "baracuda_gen_cumprod_i32_scan_prod.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_blockscan.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_excl.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_excl_blockscan.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_rev.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_rev_blockscan.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_rev_excl.cu"
#include "baracuda_gen_cumprod_i32_scan_prod_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_i32_scan_max.cu"
#include "baracuda_gen_cummax_i32_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_i32_scan_max_excl.cu"
#include "baracuda_gen_cummax_i32_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_i32_scan_max_rev.cu"
#include "baracuda_gen_cummax_i32_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_i32_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_i32_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_i32_scan_min.cu"
#include "baracuda_gen_cummin_i32_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_i32_scan_min_excl.cu"
#include "baracuda_gen_cummin_i32_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_i32_scan_min_rev.cu"
#include "baracuda_gen_cummin_i32_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_i32_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_i32_scan_min_rev_excl_blockscan.cu"
#include "baracuda_gen_cumsum_i64_scan_sum.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_blockscan.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_excl.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_excl_blockscan.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_rev.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_rev_blockscan.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_rev_excl.cu"
#include "baracuda_gen_cumsum_i64_scan_sum_rev_excl_blockscan.cu"
#include "baracuda_gen_cumprod_i64_scan_prod.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_blockscan.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_excl.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_excl_blockscan.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_rev.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_rev_blockscan.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_rev_excl.cu"
#include "baracuda_gen_cumprod_i64_scan_prod_rev_excl_blockscan.cu"
#include "baracuda_gen_cummax_i64_scan_max.cu"
#include "baracuda_gen_cummax_i64_scan_max_blockscan.cu"
#include "baracuda_gen_cummax_i64_scan_max_excl.cu"
#include "baracuda_gen_cummax_i64_scan_max_excl_blockscan.cu"
#include "baracuda_gen_cummax_i64_scan_max_rev.cu"
#include "baracuda_gen_cummax_i64_scan_max_rev_blockscan.cu"
#include "baracuda_gen_cummax_i64_scan_max_rev_excl.cu"
#include "baracuda_gen_cummax_i64_scan_max_rev_excl_blockscan.cu"
#include "baracuda_gen_cummin_i64_scan_min.cu"
#include "baracuda_gen_cummin_i64_scan_min_blockscan.cu"
#include "baracuda_gen_cummin_i64_scan_min_excl.cu"
#include "baracuda_gen_cummin_i64_scan_min_excl_blockscan.cu"
#include "baracuda_gen_cummin_i64_scan_min_rev.cu"
#include "baracuda_gen_cummin_i64_scan_min_rev_blockscan.cu"
#include "baracuda_gen_cummin_i64_scan_min_rev_excl.cu"
#include "baracuda_gen_cummin_i64_scan_min_rev_excl_blockscan.cu"

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

// ---- CPU f64 oracle for the block-scan within-ULP check (Sum/Prod, fwd/rev). ----
static void cpu_scan_f64(const float* x, double* y, long long rows, long long k,
                         Combine c, bool exclusive, bool reverse) {
    for (long long r = 0; r < rows; ++r) {
        const float* xr = x + r * k;
        double* yr = y + r * k;
        double acc = (c == SUM) ? 0.0 : 1.0;
        for (long long t = 0; t < k; ++t) {
            long long j = reverse ? (k - 1 - t) : t;
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

// Launch the f32 block-scan VARIANT (Sum/Prod, incl/excl, fwd/rev). B mult of 32.
static void launch_block_f32(Combine c, bool exc, bool rev,
                             const float* d_in, float* d_out, long long n_out, long long k, int block) {
    int g = (int)(n_out < 65535 ? n_out : 65535); if (g < 1) g = 1;
#define B(K) K<<<g, block>>>(d_in, d_out, n_out, k)
    if (c == SUM) { if (!rev && !exc) B(baracuda_gen_cumsum_f32_scan_sum_blockscan);
                    else if (!rev && exc) B(baracuda_gen_cumsum_f32_scan_sum_excl_blockscan);
                    else if (rev && !exc) B(baracuda_gen_cumsum_f32_scan_sum_rev_blockscan);
                    else B(baracuda_gen_cumsum_f32_scan_sum_rev_excl_blockscan); }
    else { if (!rev && !exc) B(baracuda_gen_cumprod_f32_scan_prod_blockscan);
           else if (!rev && exc) B(baracuda_gen_cumprod_f32_scan_prod_excl_blockscan);
           else if (rev && !exc) B(baracuda_gen_cumprod_f32_scan_prod_rev_blockscan);
           else B(baracuda_gen_cumprod_f32_scan_prod_rev_excl_blockscan); }
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

// ============================================================================
// BLOCKSCAN-VARIANTS: BitIdentical base-vs-blockscan raw-byte memcmp matrix.
// The serial base is the bit-reference; a BitIdentical variant MUST be memcmp==0
// to it (house rule #4). dtype-agnostic (raw bytes) — one templated launcher.
// ============================================================================
static int mtx_pass = 0, mtx_total = 0;

template <typename T>
static void bitcmp(const char* nm,
                   void (*fbase)(const T*, T*, long long, long long),
                   void (*fblock)(const T*, T*, long long, long long),
                   const std::vector<T>& in, long long rows, long long k, int block) {
    mtx_total++;
    size_t n = in.size(), bytes = n * sizeof(T);
    T *din = nullptr, *db = nullptr, *dv = nullptr;
    cudaMalloc((void**)&din, bytes); cudaMalloc((void**)&db, bytes); cudaMalloc((void**)&dv, bytes);
    cudaMemcpy(din, in.data(), bytes, cudaMemcpyHostToDevice);
    int g = (int)(rows < 65535 ? rows : 65535); if (g < 1) g = 1;
    cudaMemset(db, 0xA5, bytes); fbase<<<g, 1>>>(din, db, rows, k);
    cudaMemset(dv, 0x5A, bytes); fblock<<<g, block>>>(din, dv, rows, k);
    cudaDeviceSynchronize();
    std::vector<T> hb(n), hv(n);
    cudaMemcpy(hb.data(), db, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hv.data(), dv, bytes, cudaMemcpyDeviceToHost);
    if (memcmp(hb.data(), hv.data(), bytes) == 0) { mtx_pass++; }
    else {
        const unsigned char* pb = (const unsigned char*)hb.data();
        const unsigned char* pv = (const unsigned char*)hv.data();
        size_t bi = 0; while (bi < bytes && pb[bi] == pv[bi]) ++bi;
        printf("FAIL %-30s (base!=block @byte %zu base 0x%02x block 0x%02x, rows=%lld k=%lld blk=%d)\n",
               nm, bi, pb[bi], pv[bi], rows, k, block);
        fails++;
    }
    cudaFree(din); cudaFree(db); cudaFree(dv);
}

// base + block symbols for a (tag,dtype,op) across fwd/rev x incl/excl.
#define CMP4(TAG, CT, DT, OP, IN) \
    bitcmp<CT>(#TAG "/" #DT " fwd/inc", baracuda_gen_##TAG##_##DT##_scan_##OP,           baracuda_gen_##TAG##_##DT##_scan_##OP##_blockscan,           IN, rows, k, block); \
    bitcmp<CT>(#TAG "/" #DT " fwd/exc", baracuda_gen_##TAG##_##DT##_scan_##OP##_excl,     baracuda_gen_##TAG##_##DT##_scan_##OP##_excl_blockscan,     IN, rows, k, block); \
    bitcmp<CT>(#TAG "/" #DT " rev/inc", baracuda_gen_##TAG##_##DT##_scan_##OP##_rev,      baracuda_gen_##TAG##_##DT##_scan_##OP##_rev_blockscan,      IN, rows, k, block); \
    bitcmp<CT>(#TAG "/" #DT " rev/exc", baracuda_gen_##TAG##_##DT##_scan_##OP##_rev_excl, baracuda_gen_##TAG##_##DT##_scan_##OP##_rev_excl_blockscan, IN, rows, k, block);

// Adversarial FP seed: a signed-zero / NaN-payload / ±inf / subnormal palette at
// each row head (rotated per row) + a finite ramp with planted NaN/inf/-0 mid-row
// so the whole-buffer memcmp exercises the tie/NaN combiner paths at scale.
static const unsigned FP_PAL[] = {
    0x80000000u /*-0*/, 0x00000000u /*+0*/, 0x7fc00001u /*N1*/, 0x7fc00002u /*N2*/,
    0x7f800000u /*+inf*/, 0xff800000u /*-inf*/, 0x00800000u /*min-normal*/,
    0x00000001u /*min-subnormal*/, 0x40400000u /*3*/, 0x40a00000u /*5*/,
    0xc0a00000u /*-5*/, 0x461c4000u /*10000*/,
};
static std::vector<float> fp_seed(long long rows, long long k) {
    std::vector<float> v((size_t)rows * k);
    const int np = (int)(sizeof(FP_PAL) / sizeof(FP_PAL[0]));
    for (long long r = 0; r < rows; ++r)
        for (long long j = 0; j < k; ++j) {
            unsigned bits;
            if (j < np) bits = FP_PAL[(int)((j + r) % np)];
            else {
                long long m = (long long)(((unsigned long long)j * 2654435761ull + (unsigned long long)r * 40503ull) % 97ull);
                float f = (float)((int)m - 48) * 0.125f;
                memcpy(&bits, &f, 4);
                if (m == 0) bits = 0x7fc00003u;      // planted NaN mid-row
                else if (m == 7) bits = 0x7f800000u; // planted +inf
                else if (m == 11) bits = 0x80000000u;// planted -0
                else if (m == 13) bits = 0xff800000u;// planted -inf
            }
            float f; memcpy(&f, &bits, 4); v[(size_t)r * k + j] = f;
        }
    return v;
}
// Adversarial integer seed: extremes (INT_MIN/MAX), 0, -1, ties, and a wide ramp
// so Sum/Prod exercise modular wrap and Max/Min select the extremes.
template <typename T>
static std::vector<T> int_seed(long long rows, long long k) {
    std::vector<T> v((size_t)rows * k);
    const bool is32 = (sizeof(T) == 4);
    const T IMIN = is32 ? (T)(-2147483647 - 1) : (T)(-9223372036854775807LL - 1);
    const T IMAX = is32 ? (T)2147483647 : (T)9223372036854775807LL;
    for (long long r = 0; r < rows; ++r)
        for (long long j = 0; j < k; ++j) {
            T val;
            if (j == 0) val = (T)0;
            else if (j == 1) val = IMIN;
            else if (j == 2) val = IMAX;
            else if (j == 3) val = (T)-1;
            else if (j == 4) val = (T)7;   // tie set (Max/Min keep-left)
            else if (j == 5) val = (T)7;
            else {
                long long m = (long long)(((unsigned long long)j * 2654435761ull + (unsigned long long)r * 2246822519ull) % 200003ull) - 100000;
                val = (T)m;
            }
            v[(size_t)r * k + j] = val;
        }
    return v;
}

// FP Max/Min BitIdentical matrix over ALL FP dtypes (f32/f64/f16/bf16/f32-strict).
static void fp_bitcmp_matrix(long long rows, long long k, int block, const char* tag) {
    mtx_pass = 0; mtx_total = 0;
    std::vector<float> sf = fp_seed(rows, k);
    std::vector<float> in_f32 = sf;
    std::vector<double> in_f64(sf.size());
    std::vector<__half> in_f16(sf.size());
    std::vector<__nv_bfloat16> in_bf16(sf.size());
    for (size_t i = 0; i < sf.size(); ++i) {
        in_f64[i] = (double)sf[i];
        in_f16[i] = __float2half(sf[i]);
        in_bf16[i] = __float2bfloat16(sf[i]);
    }
    CMP4(cummax, float, f32, max, in_f32)
    CMP4(cummin, float, f32, min, in_f32)
    CMP4(cummax, double, f64, max, in_f64)
    CMP4(cummin, double, f64, min, in_f64)
    CMP4(cummax, __half, f16, max, in_f16)
    CMP4(cummin, __half, f16, min, in_f16)
    CMP4(cummax, __nv_bfloat16, bf16, max, in_bf16)
    CMP4(cummin, __nv_bfloat16, bf16, min, in_bf16)
    CMP4(cummax, float, f32s, max, in_f32)
    CMP4(cummin, float, f32s, min, in_f32)
    printf("%s %-24s (%d/%d FP Max/Min cells memcmp==0 vs base)\n",
           mtx_pass == mtx_total ? "PASS" : "FAIL", tag, mtx_pass, mtx_total);
}

// Integer Sum/Prod/Max/Min BitIdentical matrix over i32/i64.
static void int_bitcmp_matrix(long long rows, long long k, int block, const char* tag) {
    mtx_pass = 0; mtx_total = 0;
    std::vector<int> in_i32 = int_seed<int>(rows, k);
    std::vector<long long> in_i64 = int_seed<long long>(rows, k);
    CMP4(cumsum, int, i32, sum, in_i32)
    CMP4(cumprod, int, i32, prod, in_i32)
    CMP4(cummax, int, i32, max, in_i32)
    CMP4(cummin, int, i32, min, in_i32)
    CMP4(cumsum, long long, i64, sum, in_i64)
    CMP4(cumprod, long long, i64, prod, in_i64)
    CMP4(cummax, long long, i64, max, in_i64)
    CMP4(cummin, long long, i64, min, in_i64)
    printf("%s %-24s (%d/%d integer cells memcmp==0 vs base)\n",
           mtx_pass == mtx_total ? "PASS" : "FAIL", tag, mtx_pass, mtx_total);
}

// ---- MANDATORY FP Max/Min combiner probes: raw-bit memcmp of the f32 blockscan
//      output vs the hand-derived expected (§7). These distinguish the CORRECT
//      combiner from the three likely-wrong ones (fmaxf NaN-scrub / right-wins-tie /
//      earlier-NaN-wins). `==` on values would hide the signed-zero P1 defect, so
//      we compare RAW BITS. ----
static void probe_fwd(const char* nm, void (*fv)(const float*, float*, long long, long long),
                      const unsigned* inb, const unsigned* expb, int n) {
    std::vector<float> in(n); memcpy(in.data(), inb, (size_t)n * 4);
    float* din = dev_f32(in); float* dv = nullptr; cudaMalloc((void**)&dv, (size_t)n * 4);
    cudaMemset(dv, 0xCC, (size_t)n * 4);
    fv<<<1, 256>>>(din, dv, 1, n);
    cudaDeviceSynchronize();
    std::vector<unsigned> got(n); cudaMemcpy(got.data(), dv, (size_t)n * 4, cudaMemcpyDeviceToHost);
    bool ok = (memcmp(got.data(), expb, (size_t)n * 4) == 0);
    if (ok) printf("PASS %-30s (raw-bit == hand-derived)\n", nm);
    else { int i = 0; while (i < n && got[i] == expb[i]) ++i;
        printf("FAIL %-30s (@%d got 0x%08x want 0x%08x)\n", nm, i, got[i], expb[i]); fails++; }
    cudaFree(din); cudaFree(dv);
}
// Reverse probe: block-reverse vs base-reverse (the position-remapped bit-oracle).
static void probe_rev(const char* nm,
                      void (*fbase)(const float*, float*, long long, long long),
                      void (*fblock)(const float*, float*, long long, long long),
                      const unsigned* inb, int n) {
    std::vector<float> in(n); memcpy(in.data(), inb, (size_t)n * 4);
    float* din = dev_f32(in);
    float* db = nullptr; cudaMalloc((void**)&db, (size_t)n * 4);
    float* dv = nullptr; cudaMalloc((void**)&dv, (size_t)n * 4);
    cudaMemset(db, 0xA5, (size_t)n * 4); fbase<<<1, 1>>>(din, db, 1, n);
    cudaMemset(dv, 0x5A, (size_t)n * 4); fblock<<<1, 256>>>(din, dv, 1, n);
    cudaDeviceSynchronize();
    std::vector<unsigned> gb(n), gv(n);
    cudaMemcpy(gb.data(), db, (size_t)n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gv.data(), dv, (size_t)n * 4, cudaMemcpyDeviceToHost);
    bool ok = (memcmp(gb.data(), gv.data(), (size_t)n * 4) == 0);
    if (ok) printf("PASS %-30s (rev block==base raw-bit)\n", nm);
    else { int i = 0; while (i < n && gb[i] == gv[i]) ++i;
        printf("FAIL %-30s (@%d base 0x%08x block 0x%08x)\n", nm, i, gb[i], gv[i]); fails++; }
    cudaFree(din); cudaFree(db); cudaFree(dv);
}

static void fp_probes() {
    // ---- forward MAX (inclusive unless noted) ----
    { const unsigned in[] = {0x80000000u, 0x00000000u};
      const unsigned ex[] = {0x80000000u, 0x80000000u};
      probe_fwd("P1 max signed-zero tie", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    { const unsigned in[] = {0x40a00000u, 0x7fc00001u, 0x40400000u};
      const unsigned ex[] = {0x40a00000u, 0x7fc00001u, 0x7fc00001u};
      probe_fwd("P2 max NaN absorb", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 3); }
    { const unsigned in[] = {0x7fc00001u, 0x7fc00002u};
      const unsigned ex[] = {0x7fc00001u, 0x7fc00002u};
      probe_fwd("P3 max later-NaN wins", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    { const unsigned in[] = {0xff800000u, 0x00000000u, 0x80000000u, 0x7f800000u,
                             0x00800000u, 0x7fc00001u, 0x40400000u, 0x7fc00002u};
      const unsigned ex[] = {0xff800000u, 0x00000000u, 0x00000000u, 0x7f800000u,
                             0x7f800000u, 0x7fc00001u, 0x7fc00001u, 0x7fc00002u};
      probe_fwd("P4 max comprehensive", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 8); }
    { const unsigned in[] = {0x80000000u, 0x00000000u, 0x7fc00001u, 0x40a00000u,
                             0x7fc00002u, 0xff800000u, 0x7f800000u};
      const unsigned ex[] = {0x80000000u, 0x80000000u, 0x7fc00001u, 0x7fc00001u,
                             0x7fc00002u, 0x7fc00002u, 0x7fc00002u};
      probe_fwd("P5 max 7-wide all-three", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 7); }
    // exclusive Max: out[0] = ident (-inf) even though in[0] is NaN.
    { const unsigned in[] = {0x7fc00001u, 0x80000000u, 0x00000000u};
      const unsigned ex[] = {0xff800000u, 0x7fc00001u, 0x7fc00001u};
      probe_fwd("exclusive max out0=ident", baracuda_gen_cummax_f32_scan_max_excl_blockscan, in, ex, 3); }
    // identity probes (2-elt rows; out[0] = comb_max(-inf, x0) = x0 exactly).
    { const unsigned in[] = {0xff800000u, 0x00000000u}; const unsigned ex[] = {0xff800000u, 0x00000000u};
      probe_fwd("identity comb_max(-inf,+0)", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    { const unsigned in[] = {0xff800000u, 0x80000000u}; const unsigned ex[] = {0xff800000u, 0x80000000u};
      probe_fwd("identity comb_max(-inf,-0)", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    { const unsigned in[] = {0xff800000u, 0x7fc00001u}; const unsigned ex[] = {0xff800000u, 0x7fc00001u};
      probe_fwd("identity comb_max(-inf,N1)", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    { const unsigned in[] = {0xff800000u, 0xff800000u}; const unsigned ex[] = {0xff800000u, 0xff800000u};
      probe_fwd("identity comb_max(-inf,-inf)", baracuda_gen_cummax_f32_scan_max_blockscan, in, ex, 2); }
    // ---- forward MIN ----
    { const unsigned in[] = {0x00000000u, 0x80000000u};
      const unsigned ex[] = {0x00000000u, 0x00000000u};
      probe_fwd("P1 min signed-zero tie", baracuda_gen_cummin_f32_scan_min_blockscan, in, ex, 2); }
    { const unsigned in[] = {0x00000000u, 0x80000000u, 0x7fc00001u, 0xc0a00000u,
                             0x7fc00002u, 0x7f800000u, 0xff800000u};
      const unsigned ex[] = {0x00000000u, 0x00000000u, 0x7fc00001u, 0x7fc00001u,
                             0x7fc00002u, 0x7fc00002u, 0x7fc00002u};
      probe_fwd("P5-mirror min 7-wide", baracuda_gen_cummin_f32_scan_min_blockscan, in, ex, 7); }
    // exclusive Min: out[0] = ident (+inf).
    { const unsigned in[] = {0x7fc00001u, 0x00000000u, 0x80000000u};
      const unsigned ex[] = {0x7f800000u, 0x7fc00001u, 0x7fc00001u};
      probe_fwd("exclusive min out0=ident", baracuda_gen_cummin_f32_scan_min_excl_blockscan, in, ex, 3); }
    // ---- reverse (position-remapped) MAX/MIN vs the base oracle ----
    { const unsigned in[] = {0xff800000u, 0x00000000u, 0x80000000u, 0x7f800000u,
                             0x00800000u, 0x7fc00001u, 0x40400000u, 0x7fc00002u};
      probe_rev("P4 max reverse", baracuda_gen_cummax_f32_scan_max_rev,
                baracuda_gen_cummax_f32_scan_max_rev_blockscan, in, 8); }
    { const unsigned in[] = {0x80000000u, 0x00000000u, 0x7fc00001u, 0x40a00000u,
                             0x7fc00002u, 0xff800000u, 0x7f800000u};
      probe_rev("P5 max reverse", baracuda_gen_cummax_f32_scan_max_rev,
                baracuda_gen_cummax_f32_scan_max_rev_blockscan, in, 7); }
    { const unsigned in[] = {0x00000000u, 0x80000000u, 0x7fc00001u, 0xc0a00000u,
                             0x7fc00002u, 0x7f800000u, 0xff800000u};
      probe_rev("P5-mirror min reverse", baracuda_gen_cummin_f32_scan_min_rev,
                baracuda_gen_cummin_f32_scan_min_rev_blockscan, in, 7); }
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

// ---- BLOCK-SCAN reassociated (FP Sum/Prod): run-to-run stable, within-ULP of f64,
//      degenerate single-chunk within-ULP of base. (fwd + rev, incl/excl.) ----
static void block_variant(Combine c, bool exc, bool rev, long long rows, long long k, int block, const char* tag) {
    std::vector<float> in((size_t)rows * k);
    for (size_t i = 0; i < in.size(); ++i) {
        // small magnitudes so cumprod stays finite over long rows.
        in[i] = (c == PROD) ? (0.999f + (float)((i % 7)) * 0.0003f)
                            : (float)(((int)(i % 61) - 30)) * 0.05f;
    }
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, in.size() * 4);
    float* d_base = nullptr; cudaMalloc((void**)&d_base, in.size() * 4);

    // within-ULP of an f64 oracle (matching scan direction)
    launch_block_f32(c, exc, rev, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> got(in.size());
    cudaMemcpy(got.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<double> oracle(in.size());
    cpu_scan_f64(in.data(), oracle.data(), rows, k, c, exc, rev);
    double maxrel = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        double den = fabs(oracle[i]) > 1.0 ? fabs(oracle[i]) : 1.0;
        maxrel = fmax(maxrel, fabs((double)got[i] - oracle[i]) / den);
    }
    // Depth-aware bound: an f32 scan of k elements accumulates ~sqrt(k)*eps rounding
    // (the reassociated tree is deterministic — run-to-run PASSES — this only sizes
    // the value-correctness tolerance for deep rows).
    double tol = fmax(1e-5, 5e-7 * sqrt((double)k));
    bool okrel = maxrel < tol;
    char nm[96]; snprintf(nm, sizeof nm, "block %s%s%s %s ulp", cname(c), rev ? "/rev" : "", exc ? "/excl" : "", tag);
    printf(okrel ? "PASS %-30s (relerr %.2e < %.1e, k=%lld)\n" : "FAIL %-30s (relerr %.2e >= %.1e)\n",
           nm, maxrel, tol, okrel ? k : 0);
    if (!okrel) fails++;

    // run-to-run determinism (memcmp of two identical launches)
    cudaMemset(d_out, 0x00, in.size() * 4);
    launch_block_f32(c, exc, rev, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> a(in.size()); cudaMemcpy(a.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemset(d_out, 0xAA, in.size() * 4);
    launch_block_f32(c, exc, rev, d_in, d_out, rows, k, block);
    cudaDeviceSynchronize();
    std::vector<float> b(in.size()); cudaMemcpy(b.data(), d_out, in.size() * 4, cudaMemcpyDeviceToHost);
    bool okdet = memcmp(a.data(), b.data(), in.size() * 4) == 0;
    snprintf(nm, sizeof nm, "block %s%s%s %s determ", cname(c), rev ? "/rev" : "", exc ? "/excl" : "", tag);
    printf(okdet ? "PASS %-30s (run-to-run memcmp)\n" : "FAIL %-30s\n", nm);
    if (!okdet) fails++;

    // degenerate single-chunk (k<=block): within-ULP of the serial base (the tree
    // reassociates, so NOT memcmp-identical for FP Sum/Prod — the honest verdict).
    if (k <= block) {
        launch_base_f32(c, rev, exc, d_in, d_base, rows, k);
        cudaDeviceSynchronize();
        std::vector<float> bs(in.size());
        cudaMemcpy(bs.data(), d_base, in.size() * 4, cudaMemcpyDeviceToHost);
        double mr = 0;
        for (size_t i = 0; i < in.size(); ++i) {
            double den = fabs((double)bs[i]) > 1.0 ? fabs((double)bs[i]) : 1.0;
            mr = fmax(mr, fabs((double)got[i] - (double)bs[i]) / den);
        }
        bool okd = mr < 1e-5;
        snprintf(nm, sizeof nm, "block %s%s%s %s degen", cname(c), rev ? "/rev" : "", exc ? "/excl" : "", tag);
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
        float t_blk  = timeit([&] { launch_block_f32(SUM, false, false, dx, dy, R, K, 256); });
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

// ---- Bench: blockscan (256 thr/row cooperative) vs the serial base (1 thr/row)
//      for a cummax f32 and an i32 cumsum (memory-bound; GB/s read+write vs the
//      ~195 GB/s copy peak on this card). ----
template <typename T>
static float time_launch(void (*fn)(const T*, T*, long long, long long), int threads,
                         const T* di, T* dou, long long rows, long long k) {
    int g = (int)(rows < 65535 ? rows : 65535); if (g < 1) g = 1;
    cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
    for (int i = 0; i < 3; ++i) fn<<<g, threads>>>(di, dou, rows, k);
    cudaDeviceSynchronize(); cudaEventRecord(a);
    for (int i = 0; i < 20; ++i) fn<<<g, threads>>>(di, dou, rows, k);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, a, e);
    cudaEventDestroy(a); cudaEventDestroy(e);
    return ms / 20;
}
template <typename T>
static void bench_pair(const char* nm, void (*fb)(const T*, T*, long long, long long),
                       void (*fv)(const T*, T*, long long, long long), long long rows, long long k) {
    long long tot = rows * k; std::vector<T> h(tot);
    for (long long i = 0; i < tot; ++i) h[i] = (T)((i % 97) - 40);
    T* di = nullptr; cudaMalloc((void**)&di, tot * sizeof(T));
    cudaMemcpy(di, h.data(), tot * sizeof(T), cudaMemcpyHostToDevice);
    T* dou = nullptr; cudaMalloc((void**)&dou, tot * sizeof(T));
    float tb = time_launch<T>(fb, 1, di, dou, rows, k);
    float tv = time_launch<T>(fv, 256, di, dou, rows, k);
    double gb = (double)tot * sizeof(T) * 2 / 1e9;
    printf("[bench] %-12s %lldx%lld: base(1thr/row) %.3f ms %.1f GB/s | blockscan %.3f ms %.1f GB/s | blockscan %.1fx base\n",
           nm, rows, k, tb, gb / (tb / 1000), tv, gb / (tv / 1000), tb / tv);
    cudaFree(di); cudaFree(dou);
}
static void variant_bench() {
    bench_pair<float>("cummax f32", baracuda_gen_cummax_f32_scan_max,
                      baracuda_gen_cummax_f32_scan_max_blockscan, 4096, 4096);
    bench_pair<int>("cumsum i32", baracuda_gen_cumsum_i32_scan_sum,
                    baracuda_gen_cumsum_i32_scan_sum_blockscan, 4096, 4096);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== scan_validate (increment 6 + blockscan-variants) ==\n");
    if (san) {
        // Small shapes for compute-sanitizer (memcheck/racecheck/synccheck/initcheck).
        base_matrix(4, 9, "small");
        signed_zero_case();
        nan_case();
        edge_shapes();
        fp_probes();
        block_variant(SUM, false, false, 4, 40, 32, "small");   // multi-chunk (k>block)
        block_variant(SUM, true, true, 4, 40, 32, "small");     // reverse exclusive
        block_variant(PROD, false, false, 4, 20, 32, "small");
        fp_bitcmp_matrix(4, 40, 32, "san fp maxmin");            // multi-chunk smem carry
        int_bitcmp_matrix(4, 40, 32, "san integer");
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
        printf("- FP Max/Min combiner probes (raw-bit, kill fmaxf/right-tie/earlier-NaN) -\n");
        fp_probes();
        printf("- block-scan BitIdentical matrix (memcmp==0 vs serial base) -\n");
        fp_bitcmp_matrix(8, 200, 64, "fp maxmin 200/64");     // k not mult of 32, >block(64)
        fp_bitcmp_matrix(8, 1000, 256, "fp maxmin 1000/256"); // multi-chunk, k>block
        fp_bitcmp_matrix(70016, 8, 32, "fp maxmin manyrows"); // rows>gridDim grid-stride
        int_bitcmp_matrix(8, 200, 64, "integer 200/64");
        int_bitcmp_matrix(8, 1000, 256, "integer 1000/256");
        int_bitcmp_matrix(70016, 8, 32, "integer manyrows");
        printf("- block-scan reassociated (FP Sum/Prod, fwd + reverse) -\n");
        block_variant(SUM, false, false, 64, 100, 256, "degen100");     // single chunk
        block_variant(SUM, false, false, 64, 16384, 256, "multiwarp");  // 64 KB row
        block_variant(SUM, true, false, 64, 16384, 256, "multiwarp");
        block_variant(SUM, false, true, 64, 16384, 256, "multiwarp");   // reverse
        block_variant(SUM, true, true, 64, 16384, 256, "multiwarp");    // reverse excl
        block_variant(PROD, false, false, 64, 4096, 256, "multiwarp");
        block_variant(PROD, false, true, 64, 4096, 256, "multiwarp");   // reverse
        block_variant(PROD, true, false, 64, 100, 256, "degen100");
        printf("- bench (blockscan vs serial base) -\n");
        variant_bench();
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
