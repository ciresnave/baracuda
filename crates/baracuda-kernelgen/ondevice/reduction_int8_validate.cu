// int8-reductions Task 5 — on-device validation of the S8/U8 reduction cells
// (sum/max/min/prod + the any/count hetero-out predicate folds) vs a CPU
// wrapping-reduce oracle. No bespoke int8 reduce sibling exists (mirrors
// int_validate.cu:14) — the CPU oracle is the sole reference.
//
// Covers:
//   1. S8/U8 sum/max/min/prod: bit-exact vs a CPU `long long`-accumulate,
//      wrap-at-store oracle (`wrap8s`/`wrap8u`), including an overflow-WRAP
//      input for sum/prod (max/min never overflow — a monoid PICK of
//      already-in-range operands never leaves range).
//   2. Empty-axis identity (k=0): sum -> 0, prod -> 1, max -> dtype-min
//      (S8 -128 / U8 0), min -> dtype-max (S8 127 / U8 255) — KISS-OPS-6.11-0002.
//   3. any -> U8 keep-mask, count -> I64, S8 + U8 input dtype.
//   4. THE >2^24 count case (the Task 3b regression probe, 6fdfe478): a
//      ~20,000,000-element all-nonzero axis must count EXACTLY 20,000,000,
//      not stall at 2^24 (16,777,216) the way a float accumulator would.
//
// Build (from a VS dev shell, or with -ccbin pointed at cl.exe):
//   cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
//   cp crates/baracuda-kernelgen/ondevice/reduction_int8_validate.cu <outdir>/
//   nvcc -O3 -arch=sm_89 <outdir>/reduction_int8_validate.cu \
//        -o <outdir>/reduction_int8_validate && <outdir>/reduction_int8_validate

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// ---- generated kernels (place this harness beside them; the S8/U8 catalog
// section of bin/kernelgen.rs emits these) -----------------------------------
#include "baracuda_gen_sum_i8_reduce_sum.cu"
#include "baracuda_gen_amax_i8_reduce_max.cu"
#include "baracuda_gen_amin_i8_reduce_min.cu"
#include "baracuda_gen_prod_i8_reduce_prod.cu"
#include "baracuda_gen_any_i8_reduce_sum.cu"
#include "baracuda_gen_count_i8_reduce_sum.cu"
#include "baracuda_gen_sum_u8_reduce_sum.cu"
#include "baracuda_gen_amax_u8_reduce_max.cu"
#include "baracuda_gen_amin_u8_reduce_min.cu"
#include "baracuda_gen_prod_u8_reduce_prod.cu"
#include "baracuda_gen_any_u8_reduce_sum.cu"
#include "baracuda_gen_count_u8_reduce_sum.cu"

static int g_failures = 0;

#define CUDA_OK(call)                                                        \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                   __FILE__, __LINE__);                                      \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ---- CPU wrapping-reduce oracle ---------------------------------------------
// Truncate the low 8 bits of a `long long`-accumulated value and reinterpret
// per signed/unsigned — the exact effect of the emitted kernel's implicit
// narrowing `signed char*`/`unsigned char*` store (two's-complement wrap,
// KISS-OPS-6.2-0002). Accumulation itself is unwrapped end-to-end in the
// oracle, matching the kernel's own widened `long long acc`.
static inline int8_t wrap8s(int64_t x) { return (int8_t)(uint8_t)(x & 0xFF); }
static inline uint8_t wrap8u(int64_t x) { return (uint8_t)(x & 0xFF); }

enum class RedOp { Sum, Max, Min, Prod };

// Per-dtype monoid identity (KISS-OPS-6.11-0002): sum 0, prod 1, max =
// dtype-min, min = dtype-max. An EMPTY reduced extent (k=0) must fold to
// exactly this value, untouched.
static int64_t identity(RedOp op, bool is_signed) {
    switch (op) {
        case RedOp::Sum: return 0;
        case RedOp::Prod: return 1;
        case RedOp::Max: return is_signed ? -128 : 0;
        case RedOp::Min: return is_signed ? 127 : 255;
    }
    return 0;
}

// CPU reference: fold `in` (R rows of C cols, row-major) with `op`, unwrapped
// `int64_t` accumulate, wrap-to-width only at the very end — mirrors the
// kernel's `long long acc` + narrow-pointer store exactly.
template <typename T>
static std::vector<T> cpu_reduce_wrap(RedOp op, bool is_signed, const std::vector<T>& in,
                                      long long R, long long C) {
    std::vector<T> want((size_t)R);
    for (long long r = 0; r < R; ++r) {
        int64_t acc = identity(op, is_signed);
        for (long long c = 0; c < C; ++c) {
            int64_t x = (int64_t)in[(size_t)(r * C + c)];
            switch (op) {
                case RedOp::Sum: acc += x; break;
                case RedOp::Prod: acc *= x; break;
                case RedOp::Max: if (x > acc) acc = x; break;
                case RedOp::Min: if (x < acc) acc = x; break;
            }
        }
        want[(size_t)r] = is_signed ? (T)wrap8s(acc) : (T)wrap8u(acc);
    }
    return want;
}

// ---- comparators: bit-exact, no tolerance (int sum/prod/max/min are class
// exact-byte) — print "div=<n>", the mismatch count, as the pass signal. ----
static void exact_i8(const char* name, const int8_t* got, const int8_t* want, long long n) {
    long long bad = 0, firsti = -1;
    for (long long i = 0; i < n; ++i)
        if (got[i] != want[i]) { if (firsti < 0) firsti = i; ++bad; }
    if (bad == 0) {
        printf("[PASS] %-30s div=0  (n=%lld)\n", name, n);
    } else {
        printf("[FAIL] %-30s div=%lld/%lld  first[%lld] got=%d want=%d\n", name, bad, n,
               firsti, (int)got[firsti], (int)want[firsti]);
        ++g_failures;
    }
}

static void exact_u8(const char* name, const uint8_t* got, const uint8_t* want, long long n) {
    long long bad = 0, firsti = -1;
    for (long long i = 0; i < n; ++i)
        if (got[i] != want[i]) { if (firsti < 0) firsti = i; ++bad; }
    if (bad == 0) {
        printf("[PASS] %-30s div=0  (n=%lld)\n", name, n);
    } else {
        printf("[FAIL] %-30s div=%lld/%lld  first[%lld] got=%u want=%u\n", name, bad, n,
               firsti, (unsigned)got[firsti], (unsigned)want[firsti]);
        ++g_failures;
    }
}

static void exact_i64(const char* name, const int64_t* got, const int64_t* want, long long n) {
    long long bad = 0, firsti = -1;
    for (long long i = 0; i < n; ++i)
        if (got[i] != want[i]) { if (firsti < 0) firsti = i; ++bad; }
    if (bad == 0) {
        printf("[PASS] %-30s div=0  (n=%lld)\n", name, n);
    } else {
        printf("[FAIL] %-30s div=%lld/%lld  first[%lld] got=%lld want=%lld\n", name, bad, n,
               firsti, (long long)got[firsti], (long long)want[firsti]);
        ++g_failures;
    }
}

// ---- generic S8/U8 sum/max/min/prod runner ----------------------------------
typedef void (*I8Kern)(const signed char*, signed char*, long long, long long);
typedef void (*U8Kern)(const unsigned char*, unsigned char*, long long, long long);

static void run_i8(const char* name, I8Kern kern, RedOp op, long long R, long long C,
                   const std::vector<int8_t>& in) {
    size_t nin = in.size();
    int8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_in, nin > 0 ? nin : 1));
    CUDA_OK(cudaMalloc(&d_out, (size_t)R));
    if (nin > 0) CUDA_OK(cudaMemcpy(d_in, in.data(), nin, cudaMemcpyHostToDevice));
    long long grid = R > 0 ? R : 1;
    kern<<<(unsigned)grid, 256>>>((const signed char*)d_in, (signed char*)d_out, R, C);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int8_t> got((size_t)R);
    CUDA_OK(cudaMemcpy(got.data(), d_out, (size_t)R, cudaMemcpyDeviceToHost));
    std::vector<int8_t> want = cpu_reduce_wrap<int8_t>(op, /*is_signed=*/true, in, R, C);
    exact_i8(name, got.data(), want.data(), R);
    cudaFree(d_in); cudaFree(d_out);
}

static void run_u8(const char* name, U8Kern kern, RedOp op, long long R, long long C,
                   const std::vector<uint8_t>& in) {
    size_t nin = in.size();
    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_OK(cudaMalloc(&d_in, nin > 0 ? nin : 1));
    CUDA_OK(cudaMalloc(&d_out, (size_t)R));
    if (nin > 0) CUDA_OK(cudaMemcpy(d_in, in.data(), nin, cudaMemcpyHostToDevice));
    long long grid = R > 0 ? R : 1;
    kern<<<(unsigned)grid, 256>>>((const unsigned char*)d_in, (unsigned char*)d_out, R, C);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> got((size_t)R);
    CUDA_OK(cudaMemcpy(got.data(), d_out, (size_t)R, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> want = cpu_reduce_wrap<uint8_t>(op, /*is_signed=*/false, in, R, C);
    exact_u8(name, got.data(), want.data(), R);
    cudaFree(d_in); cudaFree(d_out);
}

int main() {
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);

    // ===================== 1. S8/U8 sum/max/min/prod =====================
    // In-range (no wrap) + a deliberate overflow-WRAP case for sum/prod;
    // max/min get a boundary-value case (dtype extremes present, no wrap
    // possible for a monoid pick).
    {
        // -- S8 sum: in-range --
        const long long R = 4, C = 4;
        std::vector<int8_t> in((size_t)(R * C));
        for (long long i = 0; i < R * C; ++i) in[(size_t)i] = (int8_t)(((i * 3 + 1) % 9) - 4);
        run_i8("sum_i8 in-range", baracuda_gen_sum_i8_reduce_sum, RedOp::Sum, R, C, in);
    }
    {
        // -- S8 sum: overflow-WRAP (constant 5 x 100 = 500 -> wraps) --
        const long long R = 3, C = 100;
        std::vector<int8_t> in((size_t)(R * C), (int8_t)5);
        run_i8("sum_i8 OVERFLOW-WRAP", baracuda_gen_sum_i8_reduce_sum, RedOp::Sum, R, C, in);
    }
    {
        // -- U8 sum: in-range --
        const long long R = 4, C = 4;
        std::vector<uint8_t> in((size_t)(R * C));
        for (long long i = 0; i < R * C; ++i) in[(size_t)i] = (uint8_t)((i * 3 + 1) % 10);
        run_u8("sum_u8 in-range", baracuda_gen_sum_u8_reduce_sum, RedOp::Sum, R, C, in);
    }
    {
        // -- U8 sum: overflow-WRAP (constant 5 x 100 = 500 -> wraps) --
        const long long R = 3, C = 100;
        std::vector<uint8_t> in((size_t)(R * C), (uint8_t)5);
        run_u8("sum_u8 OVERFLOW-WRAP", baracuda_gen_sum_u8_reduce_sum, RedOp::Sum, R, C, in);
    }
    {
        // -- S8 prod: in-range --
        const long long R = 4, C = 3;
        std::vector<int8_t> in = {1, 2, 3, -1, 2, -3, 2, 2, 2, -1, -1, -1};
        run_i8("prod_i8 in-range", baracuda_gen_prod_i8_reduce_prod, RedOp::Prod, R, C, in);
    }
    {
        // -- S8 prod: overflow-WRAP (3^20 ~= 3.49e9, wraps mod 256 many times) --
        const long long R = 2, C = 20;
        std::vector<int8_t> in((size_t)(R * C), (int8_t)3);
        run_i8("prod_i8 OVERFLOW-WRAP", baracuda_gen_prod_i8_reduce_prod, RedOp::Prod, R, C, in);
    }
    {
        // -- U8 prod: in-range --
        const long long R = 4, C = 3;
        std::vector<uint8_t> in = {1, 2, 3, 2, 2, 2, 1, 1, 5, 3, 1, 2};
        run_u8("prod_u8 in-range", baracuda_gen_prod_u8_reduce_prod, RedOp::Prod, R, C, in);
    }
    {
        // -- U8 prod: overflow-WRAP (3^20, wraps mod 256) --
        const long long R = 2, C = 20;
        std::vector<uint8_t> in((size_t)(R * C), (uint8_t)3);
        run_u8("prod_u8 OVERFLOW-WRAP", baracuda_gen_prod_u8_reduce_prod, RedOp::Prod, R, C, in);
    }
    {
        // -- S8 max/min: dtype-boundary values present (-128, 127) --
        const long long R = 2, C = 8;
        std::vector<int8_t> in = {-128, 3, -5, 127, 0, -1, 42, -100,
                                   10, -128, 127, 5, -7, 99, -3, 0};
        run_i8("amax_i8 boundary", baracuda_gen_amax_i8_reduce_max, RedOp::Max, R, C, in);
        run_i8("amin_i8 boundary", baracuda_gen_amin_i8_reduce_min, RedOp::Min, R, C, in);
    }
    {
        // -- U8 max/min: dtype-boundary values present (0, 255) --
        const long long R = 2, C = 8;
        std::vector<uint8_t> in = {0, 3, 5, 255, 128, 1, 42, 200,
                                    10, 0, 255, 5, 7, 99, 3, 1};
        run_u8("amax_u8 boundary", baracuda_gen_amax_u8_reduce_max, RedOp::Max, R, C, in);
        run_u8("amin_u8 boundary", baracuda_gen_amin_u8_reduce_min, RedOp::Min, R, C, in);
    }
    printf("\n");

    // ===================== 2. Empty-axis identity (k=0) =====================
    // n_out rows, k=0 reduced extent: the fold loop never runs, so `out[row]`
    // is the bare seed — must be the KISS-OPS-6.11-0002 monoid identity, not 0
    // for max/min.
    {
        const long long R = 8, C = 0;
        std::vector<int8_t> empty_i8;
        run_i8("sum_i8 EMPTY-AXIS", baracuda_gen_sum_i8_reduce_sum, RedOp::Sum, R, C, empty_i8);
        run_i8("prod_i8 EMPTY-AXIS", baracuda_gen_prod_i8_reduce_prod, RedOp::Prod, R, C, empty_i8);
        run_i8("amax_i8 EMPTY-AXIS(-128)", baracuda_gen_amax_i8_reduce_max, RedOp::Max, R, C,
               empty_i8);
        run_i8("amin_i8 EMPTY-AXIS(127)", baracuda_gen_amin_i8_reduce_min, RedOp::Min, R, C,
               empty_i8);
        std::vector<uint8_t> empty_u8;
        run_u8("sum_u8 EMPTY-AXIS", baracuda_gen_sum_u8_reduce_sum, RedOp::Sum, R, C, empty_u8);
        run_u8("prod_u8 EMPTY-AXIS", baracuda_gen_prod_u8_reduce_prod, RedOp::Prod, R, C, empty_u8);
        run_u8("amax_u8 EMPTY-AXIS(0)", baracuda_gen_amax_u8_reduce_max, RedOp::Max, R, C,
               empty_u8);
        run_u8("amin_u8 EMPTY-AXIS(255)", baracuda_gen_amin_u8_reduce_min, RedOp::Min, R, C,
               empty_u8);
    }
    printf("\n");

    // ===================== 3. any -> U8, count -> I64 =====================
    {
        const long long R = 1024, C = 128;
        std::vector<int8_t> in((size_t)(R * C), 0);
        std::vector<uint8_t> want_any((size_t)R, 0);
        std::vector<int64_t> want_count((size_t)R, 0);
        for (long long r = 0; r < R; ++r) {
            for (long long c = 0; c < C; ++c) {
                // ~3/4 nonzero, deterministic pattern (mirrors reduction_upgrades_validate.cu).
                int8_t v = ((r + c) % 4 == 0) ? (int8_t)0 : (int8_t)(1 + (int8_t)((r + c) % 5));
                in[(size_t)(r * C + c)] = v;
                if (v != 0) { want_any[(size_t)r] = 1; want_count[(size_t)r]++; }
            }
        }
        int8_t *d_in = nullptr;
        uint8_t *d_any = nullptr;
        int64_t *d_count = nullptr;
        CUDA_OK(cudaMalloc(&d_in, in.size()));
        CUDA_OK(cudaMalloc(&d_any, (size_t)R));
        CUDA_OK(cudaMalloc(&d_count, (size_t)R * 8));
        CUDA_OK(cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice));
        baracuda_gen_any_i8_reduce_sum<<<(unsigned)R, 256>>>((const signed char*)d_in,
                                                             (unsigned char*)d_any, R, C);
        CUDA_OK(cudaGetLastError());
        baracuda_gen_count_i8_reduce_sum<<<(unsigned)R, 256>>>((const signed char*)d_in,
                                                                (long long*)d_count, R, C);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint8_t> got_any((size_t)R);
        std::vector<int64_t> got_count((size_t)R);
        CUDA_OK(cudaMemcpy(got_any.data(), d_any, (size_t)R, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(got_count.data(), d_count, (size_t)R * 8, cudaMemcpyDeviceToHost));
        exact_u8("any_i8", got_any.data(), want_any.data(), R);
        exact_i64("count_i8", got_count.data(), want_count.data(), R);
        cudaFree(d_in); cudaFree(d_any); cudaFree(d_count);
    }
    {
        const long long R = 1024, C = 128;
        std::vector<uint8_t> in((size_t)(R * C), 0);
        std::vector<uint8_t> want_any((size_t)R, 0);
        std::vector<int64_t> want_count((size_t)R, 0);
        for (long long r = 0; r < R; ++r) {
            for (long long c = 0; c < C; ++c) {
                uint8_t v = ((r + c) % 4 == 0) ? (uint8_t)0 : (uint8_t)(1 + ((r + c) % 5));
                in[(size_t)(r * C + c)] = v;
                if (v != 0) { want_any[(size_t)r] = 1; want_count[(size_t)r]++; }
            }
        }
        uint8_t *d_in = nullptr, *d_any = nullptr;
        int64_t *d_count = nullptr;
        CUDA_OK(cudaMalloc(&d_in, in.size()));
        CUDA_OK(cudaMalloc(&d_any, (size_t)R));
        CUDA_OK(cudaMalloc(&d_count, (size_t)R * 8));
        CUDA_OK(cudaMemcpy(d_in, in.data(), in.size(), cudaMemcpyHostToDevice));
        baracuda_gen_any_u8_reduce_sum<<<(unsigned)R, 256>>>((const unsigned char*)d_in,
                                                             (unsigned char*)d_any, R, C);
        CUDA_OK(cudaGetLastError());
        baracuda_gen_count_u8_reduce_sum<<<(unsigned)R, 256>>>((const unsigned char*)d_in,
                                                                (long long*)d_count, R, C);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint8_t> got_any((size_t)R);
        std::vector<int64_t> got_count((size_t)R);
        CUDA_OK(cudaMemcpy(got_any.data(), d_any, (size_t)R, cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(got_count.data(), d_count, (size_t)R * 8, cudaMemcpyDeviceToHost));
        exact_u8("any_u8", got_any.data(), want_any.data(), R);
        exact_i64("count_u8", got_count.data(), want_count.data(), R);
        cudaFree(d_in); cudaFree(d_any); cudaFree(d_count);
    }
    printf("\n");

    // ============ 4. THE >2^24 count case (Task 3b regression probe) ============
    // A single row of ~20,000,000 ALL-NONZERO elements. 2^24 = 16,777,216 — a
    // float accumulator (the pre-6fdfe478 bug: the predicate lowered as a FLOAT
    // 0.0f/1.0f, silently promoting `acc` to `float` in C) stalls exactly there.
    // The fix lowers the predicate as a genuine integer comparison
    // (`in0[idx] != 0 ? 1 : 0`), so `acc` stays `long long` end-to-end and the
    // count must be EXACT at 20,000,000 — not 16,777,216.
    {
        const long long R = 1, C = 20000003; // > 2^24 by a wide margin
        std::vector<int8_t> in((size_t)C, (int8_t)1); // all nonzero
        int8_t *d_in = nullptr;
        int64_t *d_count = nullptr;
        CUDA_OK(cudaMalloc(&d_in, (size_t)C));
        CUDA_OK(cudaMalloc(&d_count, sizeof(int64_t)));
        CUDA_OK(cudaMemcpy(d_in, in.data(), (size_t)C, cudaMemcpyHostToDevice));
        baracuda_gen_count_i8_reduce_sum<<<(unsigned)R, 256>>>((const signed char*)d_in,
                                                                (long long*)d_count, R, C);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        int64_t got = 0;
        CUDA_OK(cudaMemcpy(&got, d_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
        printf(">2^24 count_i8: C=%lld (2^24=16777216), got=%lld, want=%lld  %s\n", C, got, C,
               got == C ? "[PASS] div=0" : "[FAIL]");
        if (got != C) { ++g_failures; printf("       *** REGRESSION: count stalled/diverged — the Task 3b integer-predicate fix did not hold on device ***\n"); }
        cudaFree(d_in); cudaFree(d_count);
    }
    {
        const long long R = 1, C = 20000003;
        std::vector<uint8_t> in((size_t)C, (uint8_t)1); // all nonzero
        uint8_t *d_in = nullptr;
        int64_t *d_count = nullptr;
        CUDA_OK(cudaMalloc(&d_in, (size_t)C));
        CUDA_OK(cudaMalloc(&d_count, sizeof(int64_t)));
        CUDA_OK(cudaMemcpy(d_in, in.data(), (size_t)C, cudaMemcpyHostToDevice));
        baracuda_gen_count_u8_reduce_sum<<<(unsigned)R, 256>>>((const unsigned char*)d_in,
                                                                (long long*)d_count, R, C);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaDeviceSynchronize());
        int64_t got = 0;
        CUDA_OK(cudaMemcpy(&got, d_count, sizeof(int64_t), cudaMemcpyDeviceToHost));
        printf(">2^24 count_u8: C=%lld (2^24=16777216), got=%lld, want=%lld  %s\n", C, got, C,
               got == C ? "[PASS] div=0" : "[FAIL]");
        if (got != C) { ++g_failures; printf("       *** REGRESSION: count stalled/diverged — the Task 3b integer-predicate fix did not hold on device ***\n"); }
        cudaFree(d_in); cudaFree(d_count);
    }

    printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
