// On-device validation of the increment-7 WINDOW kernels (sliding-window pooling:
// max_pool / min_pool / sum_pool / avg_pool). The generated serial base is
// VariantFidelity::BitIdentical — each output is an independent fixed-order fold,
// so it is bit-exact vs a CPU float/double oracle that mirrors the tap order
// (memcmp where finite; NaN-class match; within-ULP for avg_pool's FP divide).
//
// The generated .cu kernels are #included by name (they must sit in this dir — the
// `dump_window_sources` test regenerates them; see ondevice/README.md). Kernel
// signature is uniform:
//   `(const T* in0, T* out, long long n_out, long long k_in, long long k_out)`.
// The window geometry (size/stride/dilation/pad) is baked into each kernel as
// compile-time literals — the op_name encodes it so distinct geometries emit
// distinct symbols (see the case table below, which mirrors dump_window_sources).
//
// Build (pure, self-contained):
//   nvcc -O3 -arch=sm_89 window_validate.cu -o window_validate && ./window_validate
// Sanitizers (small shapes via the `san` argv):
//   compute-sanitizer --tool memcheck  ./window_validate san
//   compute-sanitizer --tool racecheck ./window_validate san
//   compute-sanitizer --tool synccheck ./window_validate san
//   compute-sanitizer --tool initcheck ./window_validate san
//
// NOTE on the "extract-the-delta" audit: the bespoke FIXED-window MaxPool1d /
// AvgPool1d ride cuDNN's Nd-pooling descriptor (crates/baracuda-kernels/src/pool/
// {max,avg}_pool1d.rs) — an OPAQUE library path with no exposed `_run` launcher
// and no fixed-window `.cu` whose math order we could match bit-for-bit. (The only
// bespoke pool `.cu` with a `_run` launcher is `adaptive_pool.cu`, a DIFFERENT
// non-uniform windowing scheme.) So the delta here is measured against the
// memory-bandwidth ceiling (pooling is memory-bound) rather than a math-matched
// sibling; the perf section reports GB/s vs a cudaMemcpy copy.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---- generated serial base kernels (mirrors dump_window_sources) ----
#include "baracuda_gen_mx_a_f32_window_max.cu"       // max  size2 s2 d1 p0
#include "baracuda_gen_mx_b_f32_window_max.cu"       // max  size3 s1 d1 p1 (NaN)
#include "baracuda_gen_mx_c_f32_window_max.cu"       // max  size3 s2 d2 p2 (edge overhang)
#include "baracuda_gen_mn_b_f32_window_min.cu"       // min  size3 s1 d1 p1
#include "baracuda_gen_sm_d_f32_window_sum.cu"       // sum  size3 s2 d1 p0
#include "baracuda_gen_av_b_f32_window_mean.cu"      // avg  size3 s1 d1 p1 (exclude pad)
#include "baracuda_gen_ai_b_f32_window_mean_cip.cu"  // avg  size3 s1 d1 p1 (include pad)
#include "baracuda_gen_av_c_f32_window_mean.cu"      // avg  size3 s2 d2 p2 (exclude, dilated)
#include "baracuda_gen_av_c64_f64_window_mean.cu"    // avg  size3 s2 d2 p2 f64
#include "baracuda_gen_mx_c64_f64_window_max.cu"     // max  size3 s2 d2 p2 f64

static int fails = 0;

enum Combine { MAX, MIN, SUM, MEAN };

// out_len = floor((in_len + pad_lo + pad_hi - dilation*(size-1) - 1)/stride) + 1
static long long out_len(long long k_in, int size, int stride, int dil, int plo, int phi) {
    long long span = (long long)dil * (size - 1) + 1;
    return (k_in + plo + phi - span) / stride + 1;
}

// ---- CPU float oracle: EXACTLY mirrors the generated pooling fold (float acc,
// ascending tap order, matching padding + divisor policy). ----
static void cpu_pool_f32(const float* x, float* y, long long rows, long long k_in, long long k_out,
                         Combine c, int size, int stride, int dil, int plo, int cip) {
    for (long long r = 0; r < rows; ++r) {
        const float* xr = x + r * k_in;
        float* yr = y + r * k_out;
        for (long long o = 0; o < k_out; ++o) {
            if (c == MAX || c == MIN) {
                float best = 0.0f; bool have = false;
                for (int kk = 0; kk < size; ++kk) {
                    long long p = o * stride - plo + (long long)kk * dil;
                    if (p >= 0 && p < k_in) {
                        float v = xr[p];
                        bool take = !have || v != v || (c == MAX ? v > best : v < best);
                        if (take) { best = v; have = true; }
                    }
                }
                yr[o] = have ? best : (c == MAX ? -INFINITY : INFINITY);
            } else {
                float acc = 0.0f; long long cnt = 0;
                for (int kk = 0; kk < size; ++kk) {
                    long long p = o * stride - plo + (long long)kk * dil;
                    if (p >= 0 && p < k_in) { acc = acc + xr[p]; cnt += 1; }
                }
                if (c == MEAN) yr[o] = cip ? acc / (float)size : (cnt > 0 ? acc / (float)cnt : 0.0f);
                else yr[o] = acc;
            }
        }
    }
}

static void cpu_pool_f64(const double* x, double* y, long long rows, long long k_in, long long k_out,
                         Combine c, int size, int stride, int dil, int plo, int cip) {
    for (long long r = 0; r < rows; ++r) {
        const double* xr = x + r * k_in;
        double* yr = y + r * k_out;
        for (long long o = 0; o < k_out; ++o) {
            if (c == MAX || c == MIN) {
                double best = 0.0; bool have = false;
                for (int kk = 0; kk < size; ++kk) {
                    long long p = o * stride - plo + (long long)kk * dil;
                    if (p >= 0 && p < k_in) {
                        double v = xr[p];
                        bool take = !have || v != v || (c == MAX ? v > best : v < best);
                        if (take) { best = v; have = true; }
                    }
                }
                yr[o] = have ? best : (c == MAX ? -INFINITY : INFINITY);
            } else {
                double acc = 0.0; long long cnt = 0;
                for (int kk = 0; kk < size; ++kk) {
                    long long p = o * stride - plo + (long long)kk * dil;
                    if (p >= 0 && p < k_in) { acc = acc + xr[p]; cnt += 1; }
                }
                if (c == MEAN) yr[o] = cip ? acc / (double)size : (cnt > 0 ? acc / (double)cnt : 0.0);
                else yr[o] = acc;
            }
        }
    }
}

static float* dev_f32(const std::vector<float>& h) {
    float* d = nullptr; cudaMalloc((void**)&d, h.size() * 4);
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    return d;
}
static double* dev_f64(const std::vector<double>& h) {
    double* d = nullptr; cudaMalloc((void**)&d, h.size() * 8);
    cudaMemcpy(d, h.data(), h.size() * 8, cudaMemcpyHostToDevice);
    return d;
}

// NaN-aware EXACT compare (bit-exact where finite; class-match where NaN).
static bool cmp_exact_f32(const char* nm, const std::vector<float>& got,
                          const std::vector<float>& oracle) {
    bool ok = got.size() == oracle.size(); long long bad = -1;
    for (size_t i = 0; ok && i < got.size(); ++i) {
        float g = got[i], o = oracle[i];
        bool eq = std::isnan(o) ? std::isnan(g) : (memcmp(&g, &o, 4) == 0);
        if (!eq) { ok = false; bad = (long long)i; }
    }
    printf(ok ? "PASS %-34s (bit-exact vs float oracle)\n"
              : "FAIL %-34s (first diff @%lld got %g want %g)\n",
           nm, ok ? 0 : bad, ok ? 0.0 : (double)got[bad], ok ? 0.0 : (double)oracle[bad]);
    if (!ok) fails++;
    return ok;
}

// within-1-ULP compare (avg_pool's FP divide); reports the max ULP distance seen.
static bool cmp_ulp_f32(const char* nm, const std::vector<float>& got,
                        const std::vector<float>& oracle, int max_ulp) {
    bool ok = got.size() == oracle.size(); long long bad = -1; long long worst = 0;
    for (size_t i = 0; ok && i < got.size(); ++i) {
        float g = got[i], o = oracle[i];
        if (std::isnan(o)) { if (!std::isnan(g)) { ok = false; bad = (long long)i; } continue; }
        int gi, oi; memcpy(&gi, &g, 4); memcpy(&oi, &o, 4);
        if (gi < 0) gi = 0x80000000 - gi; if (oi < 0) oi = 0x80000000 - oi;
        long long d = gi > oi ? gi - oi : oi - gi;
        if (d > worst) worst = d;
        if (d > max_ulp) { ok = false; bad = (long long)i; }
    }
    printf(ok ? "PASS %-34s (<= %d ULP; worst %lld)\n"
              : "FAIL %-34s (first diff @%lld got %g want %g)\n",
           nm, ok ? max_ulp : (int)worst, ok ? worst : bad,
           ok ? 0.0 : (double)got[bad], ok ? 0.0 : (double)oracle[bad]);
    if (!ok) fails++;
    return ok;
}

static bool cmp_exact_f64(const char* nm, const std::vector<double>& got,
                          const std::vector<double>& oracle, int max_ulp) {
    bool ok = got.size() == oracle.size(); long long bad = -1; long long worst = 0;
    for (size_t i = 0; ok && i < got.size(); ++i) {
        double g = got[i], o = oracle[i];
        if (std::isnan(o)) { if (!std::isnan(g)) { ok = false; bad = (long long)i; } continue; }
        long long gi, oi; memcpy(&gi, &g, 8); memcpy(&oi, &o, 8);
        if (gi < 0) gi = 0x8000000000000000LL - gi; if (oi < 0) oi = 0x8000000000000000LL - oi;
        long long d = gi > oi ? gi - oi : oi - gi;
        if (d > worst) worst = d;
        if (d > max_ulp) { ok = false; bad = (long long)i; }
    }
    printf(ok ? "PASS %-34s (f64 <= %d ULP; worst %lld)\n"
              : "FAIL %-34s (first diff @%lld got %g want %g)\n",
           nm, ok ? max_ulp : (int)worst, ok ? worst : bad,
           ok ? 0.0 : got[bad], ok ? 0.0 : oracle[bad]);
    if (!ok) fails++;
    return ok;
}

static int grid_for(long long total) {
    long long b = (total + 255) / 256;
    return (int)(b < 65535 ? (b < 1 ? 1 : b) : 65535);
}

// ---- an f32 pooling case: launch the named kernel, diff vs the CPU oracle. ----
template <typename K>
static void case_f32(const char* nm, K kernel, Combine c, long long rows, long long k_in,
                     int size, int stride, int dil, int plo, int phi, int cip, bool ulp) {
    long long k_out = out_len(k_in, size, stride, dil, plo, phi);
    std::vector<float> in((size_t)rows * k_in);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)(((int)(i % 37) - 18)) * 0.25f;
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, (size_t)rows * k_out * 4);
    cudaMemset(d_out, 0xAA, (size_t)rows * k_out * 4);
    kernel<<<grid_for(rows * k_out), 256>>>(d_in, d_out, rows, k_in, k_out);
    cudaDeviceSynchronize();
    std::vector<float> got((size_t)rows * k_out);
    cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> oracle((size_t)rows * k_out);
    cpu_pool_f32(in.data(), oracle.data(), rows, k_in, k_out, c, size, stride, dil, plo, cip);
    if (ulp) cmp_ulp_f32(nm, got, oracle, 1); else cmp_exact_f32(nm, got, oracle);
    cudaFree(d_in); cudaFree(d_out);
}

template <typename K>
static void case_f64(const char* nm, K kernel, Combine c, long long rows, long long k_in,
                     int size, int stride, int dil, int plo, int phi, int cip) {
    long long k_out = out_len(k_in, size, stride, dil, plo, phi);
    std::vector<double> in((size_t)rows * k_in);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (double)(((int)(i % 37) - 18)) * 0.1;
    double* d_in = dev_f64(in);
    double* d_out = nullptr; cudaMalloc((void**)&d_out, (size_t)rows * k_out * 8);
    cudaMemset(d_out, 0xAA, (size_t)rows * k_out * 8);
    kernel<<<grid_for(rows * k_out), 256>>>(d_in, d_out, rows, k_in, k_out);
    cudaDeviceSynchronize();
    std::vector<double> got((size_t)rows * k_out);
    cudaMemcpy(got.data(), d_out, got.size() * 8, cudaMemcpyDeviceToHost);
    std::vector<double> oracle((size_t)rows * k_out);
    cpu_pool_f64(in.data(), oracle.data(), rows, k_in, k_out, c, size, stride, dil, plo, cip);
    cmp_exact_f64(nm, got, oracle, 0);
    cudaFree(d_in); cudaFree(d_out);
}

// ---- NaN propagation: plant a NaN inside a max window; the covering outputs are
// NaN (v != v probe), the others unaffected. ----
static void nan_case() {
    const long long rows = 1, k_in = 16;
    const int size = 3, stride = 1, dil = 1, plo = 1, phi = 1;
    long long k_out = out_len(k_in, size, stride, dil, plo, phi);
    std::vector<float> in((size_t)k_in);
    for (long long i = 0; i < k_in; ++i) in[i] = (float)i;
    in[7] = std::nanf("");
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, (size_t)k_out * 4);
    baracuda_gen_mx_b_f32_window_max<<<grid_for(k_out), 256>>>(d_in, d_out, rows, k_in, k_out);
    cudaDeviceSynchronize();
    std::vector<float> got((size_t)k_out);
    cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost);
    std::vector<float> oracle((size_t)k_out);
    cpu_pool_f32(in.data(), oracle.data(), rows, k_in, k_out, MAX, size, stride, dil, plo, 0);
    // Any output window covering index 7 must be NaN; count them to prove propagation.
    int nan_out = 0; for (auto v : got) if (std::isnan(v)) nan_out++;
    bool ok = (nan_out > 0);
    printf(ok ? "PASS %-34s (%d NaN outputs, propagated via v!=v)\n"
              : "FAIL %-34s (no NaN propagated)\n", "max_pool NaN propagation", nan_out);
    if (!ok) fails++;
    cmp_exact_f32("max_pool NaN-window bit-exact", got, oracle);
    cudaFree(d_in); cudaFree(d_out);
}

// ---- perf: max_pool 2x downsample GB/s vs a cudaMemcpy copy (bandwidth ceiling;
// the bespoke fixed-window pool is cuDNN, opaque — no math-matched sibling). ----
static void perf_case() {
    const long long rows = 8192, k_in = 8192;          // 256 MiB f32 input
    const int size = 2, stride = 2, dil = 1, plo = 0, phi = 0;
    long long k_out = out_len(k_in, size, stride, dil, plo, phi);
    std::vector<float> in((size_t)rows * k_in, 1.0f);
    float* d_in = dev_f32(in);
    float* d_out = nullptr; cudaMalloc((void**)&d_out, (size_t)rows * k_out * 4);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    // warm up + time the pool.
    baracuda_gen_mx_a_f32_window_max<<<grid_for(rows * k_out), 256>>>(d_in, d_out, rows, k_in, k_out);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int it = 0; it < 20; ++it)
        baracuda_gen_mx_a_f32_window_max<<<grid_for(rows * k_out), 256>>>(d_in, d_out, rows, k_in, k_out);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e); ms /= 20.0f;
    double bytes = (double)(rows * k_in + rows * k_out) * 4.0; // read in + write out
    double gbps = bytes / (ms * 1e-3) / 1e9;
    // copy reference (read+write the SAME input size).
    float* d_cpy = nullptr; cudaMalloc((void**)&d_cpy, (size_t)rows * k_in * 4);
    cudaEventRecord(s);
    for (int it = 0; it < 20; ++it) cudaMemcpy(d_cpy, d_in, (size_t)rows * k_in * 4, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float cms = 0; cudaEventElapsedTime(&cms, s, e); cms /= 20.0f;
    double cgbps = (double)(rows * k_in) * 4.0 * 2.0 / (cms * 1e-3) / 1e9;
    printf("PERF max_pool 2x [%lldx%lld]: %.1f GB/s (%.2f ms); copy ceiling %.1f GB/s -> %.0f%% of copy\n",
           rows, k_in, gbps, ms, cgbps, 100.0 * gbps / cgbps);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_cpy);
    cudaEventDestroy(s); cudaEventDestroy(e);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    long long rows = san ? 3 : 512;
    long long k_in = 64; // matches the dumped cells' k_in; oracle is geometry-driven

    // f32 exact cells: max/min/sum + the geometry spread (stride, dilation, pad).
    case_f32("max_pool s2 d1 p0",  baracuda_gen_mx_a_f32_window_max, MAX, rows, k_in, 2, 2, 1, 0, 0, 0, false);
    case_f32("max_pool s1 d1 p1",  baracuda_gen_mx_b_f32_window_max, MAX, rows, k_in, 3, 1, 1, 1, 1, 0, false);
    case_f32("max_pool s2 d2 p2",  baracuda_gen_mx_c_f32_window_max, MAX, rows, k_in, 3, 2, 2, 2, 2, 0, false);
    case_f32("min_pool s1 d1 p1",  baracuda_gen_mn_b_f32_window_min, MIN, rows, k_in, 3, 1, 1, 1, 1, 0, false);
    case_f32("sum_pool s2 d1 p0",  baracuda_gen_sm_d_f32_window_sum, SUM, rows, k_in, 3, 2, 1, 0, 0, 0, false);
    // avg_pool: exclude-pad (divide by valid count) and include-pad (divide by size).
    case_f32("avg_pool excl s1 p1", baracuda_gen_av_b_f32_window_mean,     MEAN, rows, k_in, 3, 1, 1, 1, 1, 0, true);
    case_f32("avg_pool incl s1 p1", baracuda_gen_ai_b_f32_window_mean_cip, MEAN, rows, k_in, 3, 1, 1, 1, 1, 1, true);
    case_f32("avg_pool excl s2 d2 p2", baracuda_gen_av_c_f32_window_mean,  MEAN, rows, k_in, 3, 2, 2, 2, 2, 0, true);

    // f64 oracle-exact avg + max (dilated + padded, edge windows overhang).
    case_f64("f64 avg_pool s2 d2 p2", baracuda_gen_av_c64_f64_window_mean, MEAN, rows, k_in, 3, 2, 2, 2, 2, 0);
    case_f64("f64 max_pool s2 d2 p2", baracuda_gen_mx_c64_f64_window_max,  MAX,  rows, k_in, 3, 2, 2, 2, 2, 0);

    nan_case();
    if (!san) perf_case();

    printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
