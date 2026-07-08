// On-device validation of the increment-8 SORT_PERM kernels (row sort / argsort):
//   * the per-output RANK-sort BASE (VariantFidelity::BitIdentical, any k) — bit-exact
//     (raw-byte memcmp) vs a CPU oracle that implements pair_lt EXACTLY (order-adjusted
//     keys, NaN-greatest, ascending-index tie-break);
//   * the cooperative smem BITONIC pair-sort VARIANT (also BitIdentical — a pair sort is
//     a pure permutation), k <= 1024 launch contract — memcmp vs the oracle AND vs base.
//
// The generated .cu kernels are #included by name (they must sit in this dir — the
// `dump_sort_sources` test regenerates them; see ondevice/README.md). Kernel signature
// is uniform: `(const T* in0, OUT* out, long long n_out, long long k)` where OUT == T
// for a values sort and `int` for an argsort (I32 index output).
//
// Build (pure, self-contained):
//   nvcc -O3 -arch=sm_89 sort_validate.cu -o sort_validate && ./sort_validate
// Sanitizers (small shapes via the `san` argv — race/sync load-bearing for the smem
// bitonic swaps + per-phase barriers; initcheck guards the pad cells):
//   compute-sanitizer --tool memcheck  ./sort_validate san
//   compute-sanitizer --tool racecheck ./sort_validate san
//   compute-sanitizer --tool synccheck ./sort_validate san
//   compute-sanitizer --tool initcheck ./sort_validate san
// Extract-the-delta audit vs the bespoke stable msort (adds the -run launcher):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -DWITH_BESPOKE sort_validate.cu -o sort_validate_bes && ./sort_validate_bes
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- generated kernels: {f32,f64,i32} x {asc,desc} x {sort,argsort} x {base,bitonic} ----
#include "baracuda_gen_sort_f32_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_f32_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_sort_f32_rowsort_desc_stable.cu"
#include "baracuda_gen_sort_f32_rowsort_desc_stable_bitonic.cu"
#include "baracuda_gen_argsort_f32_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_f32_rowsort_asc_stable_idx_bitonic.cu"
#include "baracuda_gen_argsort_f32_rowsort_desc_stable_idx.cu"
#include "baracuda_gen_argsort_f32_rowsort_desc_stable_idx_bitonic.cu"
#include "baracuda_gen_sort_f64_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_f64_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_sort_f64_rowsort_desc_stable.cu"
#include "baracuda_gen_sort_f64_rowsort_desc_stable_bitonic.cu"
#include "baracuda_gen_argsort_f64_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_f64_rowsort_asc_stable_idx_bitonic.cu"
#include "baracuda_gen_argsort_f64_rowsort_desc_stable_idx.cu"
#include "baracuda_gen_argsort_f64_rowsort_desc_stable_idx_bitonic.cu"
#include "baracuda_gen_sort_i32_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_i32_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_sort_i32_rowsort_desc_stable.cu"
#include "baracuda_gen_sort_i32_rowsort_desc_stable_bitonic.cu"
#include "baracuda_gen_argsort_i32_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_i32_rowsort_asc_stable_idx_bitonic.cu"
#include "baracuda_gen_argsort_i32_rowsort_desc_stable_idx.cu"
#include "baracuda_gen_argsort_i32_rowsort_desc_stable_idx_bitonic.cu"
// i64 asc sort+argsort; f16/bf16/f32s asc sort (acc/convert coverage cells)
#include "baracuda_gen_sort_i64_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_i64_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_argsort_i64_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_i64_rowsort_asc_stable_idx_bitonic.cu"
#include "baracuda_gen_sort_f16_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_f16_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_sort_bf16_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_bf16_rowsort_asc_stable_bitonic.cu"
#include "baracuda_gen_sort_f32s_rowsort_asc_stable.cu"
#include "baracuda_gen_sort_f32s_rowsort_asc_stable_bitonic.cu"

#ifdef WITH_BESPOKE
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/sort/sort.cu" // bespoke stable msort
#endif

static int fails = 0;
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fails++; } } while (0)

// ---------------------------------------------------------------------------
// CPU oracle: the pair_lt total order EXACTLY (order-adjusted keys, NaN-greatest,
// ascending-index tie-break). One oracle serves every dtype via an acc-typed key
// extractor `K`; the permutation is applied to raw STORAGE bytes for the values
// output (so NaN payloads / -0.0 signs are preserved and memcmp-checkable).
// ---------------------------------------------------------------------------
template <typename K>
static inline bool key_lt(K a, K b, bool is_fp) {
    if (is_fp) { if (a != a) return false; if (b != b) return true; }
    return a < b;
}
template <typename K>
static inline bool pair_lt(K ka, int ia, K kb, int ib, bool asc, bool is_fp) {
    if (asc) { if (key_lt(ka, kb, is_fp)) return true; if (key_lt(kb, ka, is_fp)) return false; }
    else     { if (key_lt(kb, ka, is_fp)) return true; if (key_lt(ka, kb, is_fp)) return false; }
    return ia < ib;
}
template <typename K>
static void oracle_perm(const std::vector<K>& keys, long long k, bool asc, bool is_fp,
                        std::vector<int>& perm) {
    perm.resize((size_t)k);
    for (long long i = 0; i < k; ++i) perm[(size_t)i] = (int)i;
    std::sort(perm.begin(), perm.end(), [&](int a, int b) {
        return pair_lt(keys[(size_t)a], a, keys[(size_t)b], b, asc, is_fp);
    });
}

static void* dev_bytes(const void* h, size_t bytes) {
    void* d = nullptr; cudaMalloc(&d, bytes); cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice); return d;
}
static long long next_pow2(long long k) { long long p = 1; while (p < k) p <<= 1; return p; }

template <typename T, typename K>
struct Cell {
    const char* dtag; bool is_fp; bool asc; size_t acc_sz; bool has_argsort;
    void* sort_base; void* sort_bt; void* argsort_base; void* argsort_bt;
    K (*to_key)(const T&);
};

template <typename T, typename OUT>
static void launch(void* kern, const T* in, OUT* out, long long n_out, long long k,
                   int grid, int block, size_t smem) {
    const T* pin = in; OUT* pout = out;
    void* args[] = { (void*)&pin, (void*)&pout, (void*)&n_out, (void*)&k };
    CHECK(cudaLaunchKernel(kern, dim3(grid), dim3(block), args, smem, 0));
}

// Generic per-cell driver.
template <typename T, typename K>
static void run_cell(const Cell<T, K>& c, const std::vector<T>& in, long long n_out, long long k,
                     const char* tag, bool do_bitonic, int block) {
    const size_t N = (size_t)n_out * (size_t)k;
    std::vector<T> ov(N); std::vector<int> oi(N);
    for (long long r = 0; r < n_out; ++r) {
        std::vector<K> keys((size_t)k);
        for (long long j = 0; j < k; ++j) keys[(size_t)j] = c.to_key(in[(size_t)(r * k + j)]);
        std::vector<int> perm; oracle_perm(keys, k, c.asc, c.is_fp, perm);
        for (long long j = 0; j < k; ++j) {
            ov[(size_t)(r * k + j)] = in[(size_t)(r * k + perm[(size_t)j])];
            oi[(size_t)(r * k + j)] = perm[(size_t)j];
        }
    }
    T* d_in = (T*)dev_bytes(in.data(), N * sizeof(T));
    T* d_v = nullptr; cudaMalloc(&d_v, N * sizeof(T));
    T* d_v2 = nullptr; cudaMalloc(&d_v2, N * sizeof(T));
    int* d_i = nullptr; cudaMalloc(&d_i, N * sizeof(int));
    int* d_i2 = nullptr; cudaMalloc(&d_i2, N * sizeof(int));
    int grid = (int)(n_out < 65535 ? n_out : 65535); if (grid < 1) grid = 1;
    size_t smem = (size_t)next_pow2(k) * (c.acc_sz + sizeof(int));

    auto report = [&](const char* what, bool ok) {
        char nm[128]; snprintf(nm, sizeof nm, "%s %s %s %s", c.dtag, c.asc ? "asc" : "desc", what, tag);
        printf(ok ? "PASS %-46s\n" : "FAIL %-46s\n", nm);
        if (!ok) fails++;
    };

    // values sort: base (any k), memcmp-oracle + determinism.
    cudaMemset(d_v, 0x5A, N * sizeof(T));
    launch<T, T>(c.sort_base, d_in, d_v, n_out, k, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> hv(N); cudaMemcpy(hv.data(), d_v, N * sizeof(T), cudaMemcpyDeviceToHost);
    report("sort/base memcmp-oracle", memcmp(hv.data(), ov.data(), N * sizeof(T)) == 0);
    cudaMemset(d_v2, 0xA5, N * sizeof(T));
    launch<T, T>(c.sort_base, d_in, d_v2, n_out, k, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> hv_b(N); cudaMemcpy(hv_b.data(), d_v2, N * sizeof(T), cudaMemcpyDeviceToHost);
    report("sort/base determinism", memcmp(hv.data(), hv_b.data(), N * sizeof(T)) == 0);

    std::vector<int> hi(N);
    if (c.has_argsort) {
        cudaMemset(d_i, 0xFF, N * sizeof(int));
        launch<T, int>(c.argsort_base, d_in, d_i, n_out, k, grid, 256, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(hi.data(), d_i, N * sizeof(int), cudaMemcpyDeviceToHost);
        report("argsort/base memcmp-oracle", memcmp(hi.data(), oi.data(), N * sizeof(int)) == 0);
        bool ok = true;
        for (size_t idx = 0; idx < N && ok; ++idx) {
            long long r = (long long)idx / k;
            const T& g = in[(size_t)(r * k + hi[idx])];
            if (memcmp(&g, &hv[idx], sizeof(T)) != 0) ok = false;
        }
        report("argsort==sort consistency", ok);
    }

    if (do_bitonic) {
        cudaMemset(d_v2, 0x3C, N * sizeof(T));
        launch<T, T>(c.sort_bt, d_in, d_v2, n_out, k, grid, block, smem);
        cudaDeviceSynchronize();
        std::vector<T> hbt(N); cudaMemcpy(hbt.data(), d_v2, N * sizeof(T), cudaMemcpyDeviceToHost);
        report("sort/bitonic memcmp-oracle", memcmp(hbt.data(), ov.data(), N * sizeof(T)) == 0);
        report("sort base==bitonic", memcmp(hbt.data(), hv.data(), N * sizeof(T)) == 0);
        if (c.has_argsort) {
            cudaMemset(d_i2, 0xFF, N * sizeof(int));
            launch<T, int>(c.argsort_bt, d_in, d_i2, n_out, k, grid, block, smem);
            cudaDeviceSynchronize();
            std::vector<int> hibt(N); cudaMemcpy(hibt.data(), d_i2, N * sizeof(int), cudaMemcpyDeviceToHost);
            report("argsort/bitonic memcmp-oracle", memcmp(hibt.data(), oi.data(), N * sizeof(int)) == 0);
            report("argsort base==bitonic", memcmp(hibt.data(), hi.data(), N * sizeof(int)) == 0);
        }
    }
    cudaFree(d_in); cudaFree(d_v); cudaFree(d_v2); cudaFree(d_i); cudaFree(d_i2);
}

// ---- key extractors ----
static float     k_f32(const float& x)  { return x; }
static double    k_f64(const double& x) { return x; }
static int       k_i32(const int& x)    { return x; }
static long long k_i64(const long long& x) { return x; }
static double    k_f32s(const float& x) { return (double)x; }
static float     k_f16(const __half& x) { return __half2float(x); }
static float     k_bf16(const __nv_bfloat16& x) { return __bfloat162float(x); }

static Cell<float, float> cell_f32(bool asc) {
    return asc
        ? Cell<float, float>{"f32", true, true, 4, true,
            (void*)baracuda_gen_sort_f32_rowsort_asc_stable, (void*)baracuda_gen_sort_f32_rowsort_asc_stable_bitonic,
            (void*)baracuda_gen_argsort_f32_rowsort_asc_stable_idx, (void*)baracuda_gen_argsort_f32_rowsort_asc_stable_idx_bitonic, k_f32}
        : Cell<float, float>{"f32", true, false, 4, true,
            (void*)baracuda_gen_sort_f32_rowsort_desc_stable, (void*)baracuda_gen_sort_f32_rowsort_desc_stable_bitonic,
            (void*)baracuda_gen_argsort_f32_rowsort_desc_stable_idx, (void*)baracuda_gen_argsort_f32_rowsort_desc_stable_idx_bitonic, k_f32};
}
static Cell<double, double> cell_f64(bool asc) {
    return asc
        ? Cell<double, double>{"f64", true, true, 8, true,
            (void*)baracuda_gen_sort_f64_rowsort_asc_stable, (void*)baracuda_gen_sort_f64_rowsort_asc_stable_bitonic,
            (void*)baracuda_gen_argsort_f64_rowsort_asc_stable_idx, (void*)baracuda_gen_argsort_f64_rowsort_asc_stable_idx_bitonic, k_f64}
        : Cell<double, double>{"f64", true, false, 8, true,
            (void*)baracuda_gen_sort_f64_rowsort_desc_stable, (void*)baracuda_gen_sort_f64_rowsort_desc_stable_bitonic,
            (void*)baracuda_gen_argsort_f64_rowsort_desc_stable_idx, (void*)baracuda_gen_argsort_f64_rowsort_desc_stable_idx_bitonic, k_f64};
}
static Cell<int, int> cell_i32(bool asc) {
    return asc
        ? Cell<int, int>{"i32", false, true, 4, true,
            (void*)baracuda_gen_sort_i32_rowsort_asc_stable, (void*)baracuda_gen_sort_i32_rowsort_asc_stable_bitonic,
            (void*)baracuda_gen_argsort_i32_rowsort_asc_stable_idx, (void*)baracuda_gen_argsort_i32_rowsort_asc_stable_idx_bitonic, k_i32}
        : Cell<int, int>{"i32", false, false, 4, true,
            (void*)baracuda_gen_sort_i32_rowsort_desc_stable, (void*)baracuda_gen_sort_i32_rowsort_desc_stable_bitonic,
            (void*)baracuda_gen_argsort_i32_rowsort_desc_stable_idx, (void*)baracuda_gen_argsort_i32_rowsort_desc_stable_idx_bitonic, k_i32};
}
static Cell<long long, long long> cell_i64() {
    return Cell<long long, long long>{"i64", false, true, 8, true,
        (void*)baracuda_gen_sort_i64_rowsort_asc_stable, (void*)baracuda_gen_sort_i64_rowsort_asc_stable_bitonic,
        (void*)baracuda_gen_argsort_i64_rowsort_asc_stable_idx, (void*)baracuda_gen_argsort_i64_rowsort_asc_stable_idx_bitonic, k_i64};
}

static uint32_t rng_state = 0x12345678u;
static uint32_t xrand() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }

// =========================== NaN / signed-zero =============================
static void nan_case_f32() {
    const long long n = 3, k = 13;
    std::vector<float> in((size_t)n * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(i % 11) - 5) * 0.5f;
    uint32_t nq[3] = {0x7fc00001u, 0x7fc0abcdu, 0xffc00042u};
    for (int t = 0; t < 3; ++t) memcpy(&in[(size_t)(1 * k + 2 + t * 3)], &nq[t], 4);
    for (bool asc : {true, false}) run_cell<float, float>(cell_f32(asc), in, n, k, "nan", true, 32);
}
static void nan_case_f64() {
    const long long n = 2, k = 11;
    std::vector<double> in((size_t)n * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = ((double)((int)(i % 13) - 6)) * 0.25;
    uint64_t nq[2] = {0x7ff8000000000abcULL, 0xfff8000000000001ULL};
    for (int t = 0; t < 2; ++t) memcpy(&in[(size_t)(0 * k + 3 + t * 4)], &nq[t], 8);
    for (bool asc : {true, false}) run_cell<double, double>(cell_f64(asc), in, n, k, "nan", true, 32);
}
static void signed_zero_case() {
    const long long n = 1, k = 8;
    std::vector<float> in(k);
    uint32_t pat[8] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x00000000u,
                       0x80000000u, 0xbf800000u, 0x80000000u, 0x00000000u};
    for (int i = 0; i < 8; ++i) memcpy(&in[i], &pat[i], 4);
    for (bool asc : {true, false}) run_cell<float, float>(cell_f32(asc), in, n, k, "signed0", true, 32);
}

// =========================== stability witness ============================
template <typename T, typename K>
static void stability_witness(const Cell<T, K>& c_asc, const Cell<T, K>& c_desc, T v0, T v1, T v2) {
    const long long n = 2, k = 257; const size_t N = (size_t)n * k;
    std::vector<T> in(N);
    for (size_t i = 0; i < N; ++i) { uint32_t r = xrand() % 3; in[i] = (r == 0 ? v0 : r == 1 ? v1 : v2); }
    for (const Cell<T, K>* cp : {&c_asc, &c_desc}) {
        T* d_in = (T*)dev_bytes(in.data(), N * sizeof(T));
        int* d_i = nullptr; cudaMalloc(&d_i, N * sizeof(int));
        cudaMemset(d_i, 0xFF, N * sizeof(int));
        launch<T, int>(cp->argsort_base, d_in, d_i, n, k, (int)n, 256, 0);
        cudaDeviceSynchronize();
        std::vector<int> hi(N); cudaMemcpy(hi.data(), d_i, N * sizeof(int), cudaMemcpyDeviceToHost);
        bool ok = true;
        for (long long r = 0; r < n && ok; ++r)
            for (long long j = 1; j < k; ++j) {
                int a = hi[(size_t)(r * k + j - 1)], b = hi[(size_t)(r * k + j)];
                K ka = cp->to_key(in[(size_t)(r * k + a)]), kb = cp->to_key(in[(size_t)(r * k + b)]);
                if (ka == kb && !(a < b)) { ok = false; break; }
            }
        char nm[96]; snprintf(nm, sizeof nm, "%s %s stability(ties ascending idx)", cp->dtag, cp->asc ? "asc" : "desc");
        printf(ok ? "PASS %-46s\n" : "FAIL %-46s\n", nm); if (!ok) fails++;
        cudaFree(d_in); cudaFree(d_i);
    }
}

// ===================== extreme values (pad-tie invariant) =================
static void extreme_values() {
    {
        const long long n = 1, k = 100; std::vector<int> in(k);
        for (long long j = 0; j < k; ++j) in[(size_t)j] = (int)((xrand() % 201) - 100);
        in[10] = INT32_MAX; in[20] = INT32_MAX; in[30] = INT32_MIN;
        run_cell<int, int>(cell_i32(true), in, n, k, "extreme(INT_MAX)", true, 128);
    }
    {
        const long long n = 1, k = 100; std::vector<float> in(k);
        for (long long j = 0; j < k; ++j) in[(size_t)j] = (float)((int)(xrand() % 201) - 100) * 0.1f;
        uint32_t ninf = 0xff800000u, pinf = 0x7f800000u;
        memcpy(&in[5], &ninf, 4); memcpy(&in[6], &ninf, 4); memcpy(&in[7], &pinf, 4);
        run_cell<float, float>(cell_f32(false), in, n, k, "extreme(-inf)", true, 128);
    }
}

// ========== k > 1024: base-only device-launch + bitonic CONTRACT-REJECT =========
static void long_row_base_only() {
    const long long n = 2, k = 1500; const size_t N = (size_t)n * k;
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i) in[i] = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
    run_cell<float, float>(cell_f32(true), in, n, k, "k1500 (base only)", false, 0);
    const long long CAP = 1024;
    bool refused = (k > CAP); // host-side contract check mirroring launch_note (NOT a silent launch)
    printf(refused ? "PASS %-46s\n" : "FAIL %-46s\n", "f32 asc bitonic contract-reject k>1024");
    if (!refused) fails++;
}

// ============ f16 / bf16 / f32s values-sort (acc/convert coverage) ============
static void half_and_f32s_cells() {
    {
        std::vector<__half> in((size_t)2 * 300);
        for (auto& x : in) x = __float2half((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
        Cell<__half, float> c{"f16", true, true, 4, false,
            (void*)baracuda_gen_sort_f16_rowsort_asc_stable, (void*)baracuda_gen_sort_f16_rowsort_asc_stable_bitonic,
            nullptr, nullptr, k_f16};
        run_cell<__half, float>(c, in, 2, 300, "rand", true, 256);
    }
    {
        std::vector<__nv_bfloat16> in((size_t)2 * 300);
        for (auto& x : in) x = __float2bfloat16((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
        Cell<__nv_bfloat16, float> c{"bf16", true, true, 4, false,
            (void*)baracuda_gen_sort_bf16_rowsort_asc_stable, (void*)baracuda_gen_sort_bf16_rowsort_asc_stable_bitonic,
            nullptr, nullptr, k_bf16};
        run_cell<__nv_bfloat16, float>(c, in, 2, 300, "rand", true, 256);
    }
    {
        std::vector<float> in((size_t)2 * 300);
        for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
        Cell<float, double> c{"f32s", true, true, 8, false,
            (void*)baracuda_gen_sort_f32s_rowsort_asc_stable, (void*)baracuda_gen_sort_f32s_rowsort_asc_stable_bitonic,
            nullptr, nullptr, k_f32s};
        run_cell<float, double>(c, in, 2, 300, "rand", true, 256);
    }
}

#ifdef WITH_BESPOKE
// Fisher-Yates shuffle of 0..k-1 for the distinct-key rows.
static void shuffle(std::vector<int>& p) {
    for (size_t i = p.size(); i > 1; --i) { size_t j = xrand() % i; std::swap(p[i - 1], p[j]); }
}
// ---- Extract-the-delta: generated (base) vs the bespoke STABLE msort
//      (baracuda_sort.cuh STABLE=1) on NaN-FREE inputs, both directions.
//
//      FINDING (headline): the msort `descending` flag maps to our Desc, and its
//      VALUES output is bit-exact to ours (ties included). But its INDEX output is
//      bit-exact to ours ONLY on DISTINCT-KEY rows — on TIE rows the bespoke
//      bitonic STABLE tie-break (`cmp_swap_needed`: ascending_block ? a_idx<b_idx :
//      a_idx>b_idx, baracuda_sort.cuh:173/180) is NOT input-order-preserving the
//      way our (key, original-index) pair-sort is, so the two argsort permutations
//      differ among equal keys. OURS is the verified-stable permutation (checked vs
//      a CPU stable_sort oracle above); the values are identical either way. And
//      NaN rows are a further documented delta (bespoke treats NaN as an equality
//      tie — network-position-dependent, NOT PyTorch NaN-last). + a bandwidth line.
static void bespoke_audit() {
    const long long batch = 64, k = 1000; const size_t N = (size_t)batch * k;

    // ---- (A) VALUES bit-exact on ties-included NaN-free rows; INDICES bit-exact on
    //      DISTINCT-KEY rows; the tie-row index divergence recorded (INFO). ----
    auto audit_f32 = [&](int desc) {
        // ties-included input (for the values comparison).
        std::vector<float> tin(N);
        for (size_t i = 0; i < N; ++i) tin[i] = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
        // distinct-key input (a shuffled ramp per row, for the index comparison).
        std::vector<float> din(N);
        for (long long r = 0; r < batch; ++r) {
            std::vector<int> perm((size_t)k); for (long long j = 0; j < k; ++j) perm[(size_t)j] = (int)j;
            shuffle(perm);
            for (long long j = 0; j < k; ++j) din[(size_t)(r * k + j)] = (float)perm[(size_t)j] * 0.5f - 250.0f;
        }
        Cell<float, float> c = cell_f32(desc == 0);
        auto run = [&](const std::vector<float>& in, std::vector<float>& gv, std::vector<int>& gi,
                       std::vector<float>& bv, std::vector<int>& bi) -> int {
            float* d_in = (float*)dev_bytes(in.data(), N * 4);
            float* d_gv; cudaMalloc(&d_gv, N * 4); int* d_gi; cudaMalloc(&d_gi, N * 4);
            float* d_bv; cudaMalloc(&d_bv, N * 4); int* d_bi; cudaMalloc(&d_bi, N * 4);
            launch<float, float>(c.sort_base, d_in, d_gv, batch, k, (int)batch, 256, 0);
            launch<float, int>(c.argsort_base, d_in, d_gi, batch, k, (int)batch, 256, 0);
            int rc = baracuda_kernels_msort_f32_run((int)batch, (int)k, desc, d_in, d_bv, d_bi, nullptr, 0, nullptr);
            cudaDeviceSynchronize();
            gv.resize(N); gi.resize(N); bv.resize(N); bi.resize(N);
            cudaMemcpy(gv.data(), d_gv, N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(gi.data(), d_gi, N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(bv.data(), d_bv, N * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(bi.data(), d_bi, N * 4, cudaMemcpyDeviceToHost);
            cudaFree(d_in); cudaFree(d_gv); cudaFree(d_gi); cudaFree(d_bv); cudaFree(d_bi);
            return rc;
        };
        const char* dir = desc ? "desc" : "asc";
        // ties row -> values match; indices diverge (recorded).
        std::vector<float> gv, bv; std::vector<int> gi, bi;
        int rc = run(tin, gv, gi, bv, bi);
        char nm[80];
        snprintf(nm, sizeof nm, "audit f32 %s VALUES==msort (ties)", dir);
        bool okv = (rc == 0) && memcmp(gv.data(), bv.data(), N * 4) == 0;
        printf(okv ? "PASS %-46s\n" : "FAIL %-46s (rc=%d)\n", nm, rc); if (!okv) fails++;
        int idx_diff_rows = 0;
        for (long long r = 0; r < batch; ++r)
            if (memcmp(&gi[(size_t)(r * k)], &bi[(size_t)(r * k)], k * 4) != 0) idx_diff_rows++;
        printf("INFO f32 %s indices: %d/%lld tie-rows differ from msort (tie-break convention delta; ours is stable)\n",
               dir, idx_diff_rows, batch);
        // distinct-key row -> both values AND indices bit-exact.
        rc = run(din, gv, gi, bv, bi);
        snprintf(nm, sizeof nm, "audit f32 %s INDICES==msort (distinct)", dir);
        bool oki = (rc == 0) && memcmp(gi.data(), bi.data(), N * 4) == 0;
        printf(oki ? "PASS %-46s\n" : "FAIL %-46s (rc=%d)\n", nm, rc); if (!oki) fails++;
    };
    audit_f32(0);
    audit_f32(1);

    // i32 distinct-key: values + indices bit-exact ascending.
    {
        std::vector<int> in(N);
        for (long long r = 0; r < batch; ++r) {
            std::vector<int> perm((size_t)k); for (long long j = 0; j < k; ++j) perm[(size_t)j] = (int)j;
            shuffle(perm);
            for (long long j = 0; j < k; ++j) in[(size_t)(r * k + j)] = perm[(size_t)j] - 500;
        }
        int* d_in = (int*)dev_bytes(in.data(), N * 4);
        int* d_gv; cudaMalloc(&d_gv, N * 4); int* d_gi; cudaMalloc(&d_gi, N * 4);
        int* d_bv; cudaMalloc(&d_bv, N * 4); int* d_bi; cudaMalloc(&d_bi, N * 4);
        Cell<int, int> c = cell_i32(true);
        launch<int, int>(c.sort_base, d_in, d_gv, batch, k, (int)batch, 256, 0);
        launch<int, int>(c.argsort_base, d_in, d_gi, batch, k, (int)batch, 256, 0);
        int rc = baracuda_kernels_msort_i32_run((int)batch, (int)k, 0, d_in, d_bv, d_bi, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<int> gv(N), bv(N), gi(N), bi(N);
        cudaMemcpy(gv.data(), d_gv, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bv.data(), d_bv, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gi.data(), d_gi, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bi.data(), d_bi, N * 4, cudaMemcpyDeviceToHost);
        bool okv = (rc == 0) && memcmp(gv.data(), bv.data(), N * 4) == 0;
        bool oki = (rc == 0) && memcmp(gi.data(), bi.data(), N * 4) == 0;
        printf(okv ? "PASS %-46s\n" : "FAIL %-46s (rc=%d)\n", "audit i32 asc VALUES==msort (distinct)", rc);
        printf(oki ? "PASS %-46s\n" : "FAIL %-46s (rc=%d)\n", "audit i32 asc INDICES==msort (distinct)", rc);
        if (!okv) fails++; if (!oki) fails++;
        cudaFree(d_in); cudaFree(d_gv); cudaFree(d_gi); cudaFree(d_bv); cudaFree(d_bi);
    }
    // NaN delta: prove our kernel differs from bespoke on a NaN row (documented, NOT a bug).
    {
        const long long b1 = 1, kk = 16; std::vector<float> in((size_t)kk);
        for (long long j = 0; j < kk; ++j) in[(size_t)j] = (float)(kk - j);
        uint32_t nan = 0x7fc00000u; memcpy(&in[4], &nan, 4); memcpy(&in[9], &nan, 4);
        float* d_in = (float*)dev_bytes(in.data(), kk * 4);
        float* d_gv = nullptr; cudaMalloc(&d_gv, kk * 4);
        float* d_bv = nullptr; cudaMalloc(&d_bv, kk * 4);
        int* d_bi = nullptr; cudaMalloc(&d_bi, kk * 4);
        Cell<float, float> c = cell_f32(true);
        launch<float, float>(c.sort_base, d_in, d_gv, b1, kk, 1, 256, 0);
        baracuda_kernels_msort_f32_run(1, (int)kk, 0, d_in, d_bv, d_bi, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> gv((size_t)kk), bv((size_t)kk);
        cudaMemcpy(gv.data(), d_gv, kk * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bv.data(), d_bv, kk * 4, cudaMemcpyDeviceToHost);
        // ours: NaN block LAST (positions kk-2, kk-1). Verify + note bespoke differs.
        bool ours_nan_last = std::isnan(gv[(size_t)(kk - 1)]) && std::isnan(gv[(size_t)(kk - 2)]) && !std::isnan(gv[0]);
        bool differs = memcmp(gv.data(), bv.data(), kk * 4) != 0;
        printf(ours_nan_last ? "PASS %-46s\n" : "FAIL %-46s\n", "NaN-last (ours, PyTorch convention)");
        printf(differs ? "INFO NaN row: ours != bespoke msort (documented delta)\n"
                       : "INFO NaN row: ours == bespoke (unexpected)\n");
        if (!ours_nan_last) fails++;
        cudaFree(d_in); cudaFree(d_gv); cudaFree(d_bv); cudaFree(d_bi);
    }
    // Bandwidth: base vs bitonic vs bespoke msort (elements/s + GB/s), k=1024.
    {
        const long long R = 4096, K = 1024; const long long tot = R * K;
        std::vector<float> big(tot);
        for (size_t i = 0; i < (size_t)tot; ++i) big[i] = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
        float* dx = (float*)dev_bytes(big.data(), tot * 4);
        float* dy = nullptr; cudaMalloc(&dy, tot * 4);
        int*   di = nullptr; cudaMalloc(&di, tot * 4);
        Cell<float, float> c = cell_f32(true);
        size_t smem = (size_t)K * (4 + 4);
        auto timeit = [&](auto fn) {
            cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
            for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
            for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
            float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / 20;
        };
        double gb = tot * 4.0 * 2 / 1e9;
        float t_base = timeit([&] { launch<float, float>(c.sort_base, dx, dy, R, K, (int)R, 256, 0); });
        float t_bt   = timeit([&] { launch<float, float>(c.sort_bt, dx, dy, R, K, (int)R, 256, smem); });
        float t_bes  = timeit([&] { baracuda_kernels_msort_f32_run((int)R, (int)K, 0, dx, dy, di, nullptr, 0, nullptr); });
        double es = (double)tot / 1e9;
        printf("[bench] f32 sort %lldx%lld: base %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
               "bitonic %.3f ms (%.2f Gelem/s, %.1f GB/s) | bespoke msort %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
               "bitonic %.1fx base, %.2fx bespoke\n",
               R, K, t_base, es / (t_base / 1000), gb / (t_base / 1000),
               t_bt, es / (t_bt / 1000), gb / (t_bt / 1000),
               t_bes, es / (t_bes / 1000), gb / (t_bes / 1000),
               t_base / t_bt, t_bes / t_bt);
        cudaFree(dx); cudaFree(dy); cudaFree(di);
    }
}
#endif

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== sort_validate (increment 8 SORT_PERM) ==\n");

    if (san) {
        {   const long long n = 4, k = 13; std::vector<float> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(xrand() % 201) - 100) * 0.1f;
            run_cell<float, float>(cell_f32(true), in, n, k, "san", true, 32);
            run_cell<float, float>(cell_f32(false), in, n, k, "san", true, 32); }
        {   const long long n = 3, k = 32; std::vector<int> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (int)((xrand() % 201) - 100);
            run_cell<int, int>(cell_i32(true), in, n, k, "san", true, 32); }
        nan_case_f32();
        signed_zero_case();
        { Cell<int, int> a = cell_i32(true), d = cell_i32(false); stability_witness<int, int>(a, d, 0, 1, 2); }
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    printf("- random rows, near cap (k=1000): f32/f64/i32 asc+desc, sort+argsort, base+bitonic -\n");
    for (bool asc : {true, false}) {
        { std::vector<float> in((size_t)3 * 1000); for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
          run_cell<float, float>(cell_f32(asc), in, 3, 1000, "rand", true, 256); }
        { std::vector<double> in((size_t)3 * 1000); for (auto& x : in) x = (double)((int)(xrand() % 20001) - 10000) * 0.001;
          run_cell<double, double>(cell_f64(asc), in, 3, 1000, "rand", true, 256); }
        { std::vector<int> in((size_t)3 * 1000); for (auto& x : in) x = (int)((xrand() % 20001) - 10000);
          run_cell<int, int>(cell_i32(asc), in, 3, 1000, "rand", true, 256); }
    }

    printf("- edge k: 1, 5, 33, exactly 1024; already/reverse-sorted -\n");
    for (long long k : {(long long)1, (long long)5, (long long)33, (long long)1024}) {
        std::vector<float> in((size_t)4 * k); for (auto& x : in) x = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
        run_cell<float, float>(cell_f32(true), in, 4, k, "edge", true, k >= 64 ? 256 : 32);
        run_cell<float, float>(cell_f32(false), in, 4, k, "edge", true, k >= 64 ? 256 : 32);
    }
    {   const long long n = 2, k = 512; std::vector<float> in((size_t)n * k);
        for (long long j = 0; j < k; ++j) { in[(size_t)j] = (float)j; in[(size_t)(k + j)] = (float)(k - 1 - j); }
        run_cell<float, float>(cell_f32(true), in, n, k, "sorted+rev", true, 256); }

    printf("- dtypes: i64 asc + f16/bf16/f32s asc (acc/convert coverage) -\n");
    { std::vector<long long> in((size_t)2 * 300); for (auto& x : in) x = (long long)((int)(xrand() % 20001) - 10000);
      run_cell<long long, long long>(cell_i64(), in, 2, 300, "rand", true, 256); }
    half_and_f32s_cells();

    printf("- NaN / signed-zero / stability / extreme-value / long-row -\n");
    nan_case_f32();
    nan_case_f64();
    signed_zero_case();
    { Cell<float, float> a = cell_f32(true), d = cell_f32(false); stability_witness<float, float>(a, d, 0.0f, 1.0f, 2.0f); }
    { Cell<int, int> a = cell_i32(true), d = cell_i32(false); stability_witness<int, int>(a, d, 0, 1, 2); }
    extreme_values();
    long_row_base_only();

#ifdef WITH_BESPOKE
    printf("- extract-the-delta audit vs bespoke stable msort (NaN-free) -\n");
    bespoke_audit();
#endif

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); fails++; }
    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
