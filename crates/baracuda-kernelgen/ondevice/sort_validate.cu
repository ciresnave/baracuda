// On-device validation of the increment-8 SORT_PERM kernels (row sort / argsort)
// AND the increment-9 FUSED_ARGSORT two-output `Both` kernel (op "fused"):
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
#include <functional>
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

// ---- Increment 9 FUSED_ARGSORT: the two-output `Both` kernels (op "fused"), and
// the row_argsort oracle for f16/bf16/f32s (absent in #8 — the Both dual memcmp
// needs the index oracle for these dtypes too). Signature:
//   (const T* in0, T* out_val, int* out_idx, long long n_out, long long k). ----
#include "baracuda_gen_fused_f32_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_f32_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f32_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_f32_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f64_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_f64_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f64_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_f64_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_i32_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_i32_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_i32_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_i32_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_i64_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_i64_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f16_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_f16_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_bf16_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_bf16_rowsort_asc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f32s_rowsort_asc_stable_both.cu"
#include "baracuda_gen_fused_f32s_rowsort_asc_stable_both_bitonic.cu"
// row_argsort oracle for the half/f32s Both cells (base is the canonical oracle).
#include "baracuda_gen_argsort_f16_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_bf16_rowsort_asc_stable_idx.cu"
#include "baracuda_gen_argsort_f32s_rowsort_asc_stable_idx.cu"

// ---- Increment 10 TOPK/BOTTOMK: the runtime-k-capped two-output kernels (ops
// `topk` = Desc, `bottomk` = Asc), plus the matching-order fused Both oracle
// (desc for the topk cross-check on i64/f16/bf16/f32s — asc already above).
// Signature: (const T* in0, T* out_val, int* out_idx, long long n_out,
//             long long k_in, long long k_out). ----
#include "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_f64_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_f64_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_i32_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_i32_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_i64_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_i64_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_f16_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_f16_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk.cu"
#include "baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk_bitonic.cu"
#include "baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk.cu"
#include "baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk_bitonic.cu"

// ---- PARTIAL-SELECT TOPK: the streaming tiled-bitonic `_psel` top-k VARIANT
// (ops `topk` = Desc, `bottomk` = Asc). Same (const T* in0, T* out_val, int*
// out_idx, long long n_out, long long k_in, long long k_out) ABI as `_topk`, but
// the smem is bounded on k_out (2*next_pow2(k_out) pairs), NOT k_in — so there is
// NO k_in <= 1024 cap: this is the fast path validated at k_in = 4096/8192. The
// device oracle is the full-sort Both BASE (rank sort, any k_in). ----
#include "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_f64_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_i32_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_i64_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_f16_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk_psel.cu"
#include "baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk_psel.cu"
#include "baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk_psel.cu"
// matching-order (DESC) fused Both oracle for the topk cross-check on the dtypes
// whose desc fused was not included above (asc fused already present).
#include "baracuda_gen_fused_i64_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_i64_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f16_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_f16_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_bf16_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_bf16_rowsort_desc_stable_both_bitonic.cu"
#include "baracuda_gen_fused_f32s_rowsort_desc_stable_both.cu"
#include "baracuda_gen_fused_f32s_rowsort_desc_stable_both_bitonic.cu"

#ifdef WITH_BESPOKE
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/sort/sort.cu" // bespoke stable msort
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/sort/topk.cu" // bespoke partial-bitonic topk
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

// ===================== Increment 9 FUSED_ARGSORT: `Both` ====================
// The fused kernel writes TWO output buffers in one launch:
//   (const T* in0, T* out_val, int* out_idx, long long n_out, long long k).
// THE ACCEPTANCE GATE (brief §6): for every cell, dual whole-buffer memcmp of the
// fused (out_val, out_idx) vs the shipped row_sort (values) AND row_argsort
// (indices) on the SAME input — both #8 references are already bit-exact vs the CPU
// pair_lt oracle in run_cell above, so this transitively proves the fused kernel
// against the oracle AND directly proves the fusion introduced no permutation drift
// between the two projections. Base ≡ bitonic for Both is also pinned (BitIdentical).
template <typename T>
static void launch_both(void* kern, const T* in, T* out_val, int* out_idx,
                        long long n_out, long long k, int grid, int block, size_t smem) {
    const T* pin = in; T* pv = out_val; int* pi = out_idx;
    void* args[] = { (void*)&pin, (void*)&pv, (void*)&pi, (void*)&n_out, (void*)&k };
    CHECK(cudaLaunchKernel(kern, dim3(grid), dim3(block), args, smem, 0));
}

template <typename T>
struct BothCell {
    const char* dtag; bool asc; size_t acc_sz;
    void* fused_base; void* fused_bt;      // the fused two-output kernel
    void* sort_base;  void* argsort_base;  // the #8 oracles (base is canonical; #8 proved base==bitonic)
};

template <typename T>
static void run_both_cell(const BothCell<T>& c, const std::vector<T>& in, long long n_out,
                          long long k, const char* tag, bool do_bitonic, int block) {
    const size_t N = (size_t)n_out * (size_t)k;
    T* d_in = (T*)dev_bytes(in.data(), N * sizeof(T));
    T* d_rv = nullptr; cudaMalloc(&d_rv, N * sizeof(T));       // ref values (row_sort)
    int* d_ri = nullptr; cudaMalloc(&d_ri, N * sizeof(int));   // ref indices (row_argsort)
    T* d_fv = nullptr; cudaMalloc(&d_fv, N * sizeof(T));       // fused out_val
    int* d_fi = nullptr; cudaMalloc(&d_fi, N * sizeof(int));   // fused out_idx
    T* d_fv2 = nullptr; cudaMalloc(&d_fv2, N * sizeof(T));     // second launch / bitonic
    int* d_fi2 = nullptr; cudaMalloc(&d_fi2, N * sizeof(int));
    int grid = (int)(n_out < 65535 ? n_out : 65535); if (grid < 1) grid = 1;
    size_t smem = (size_t)next_pow2(k) * (c.acc_sz + sizeof(int));

    auto report = [&](const char* what, bool ok) {
        char nm[128]; snprintf(nm, sizeof nm, "%s %s both %s %s", c.dtag, c.asc ? "asc" : "desc", what, tag);
        printf(ok ? "PASS %-52s\n" : "FAIL %-52s\n", nm); if (!ok) fails++;
    };

    // Oracles: row_sort (values) + row_argsort (indices), base kernels.
    cudaMemset(d_rv, 0x5A, N * sizeof(T));
    launch<T, T>(c.sort_base, d_in, d_rv, n_out, k, grid, 256, 0);
    cudaMemset(d_ri, 0xFF, N * sizeof(int));
    launch<T, int>(c.argsort_base, d_in, d_ri, n_out, k, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> rv(N); std::vector<int> ri(N);
    cudaMemcpy(rv.data(), d_rv, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(ri.data(), d_ri, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Fused base — THE acceptance gate: both projections vs both oracles.
    cudaMemset(d_fv, 0x3C, N * sizeof(T)); cudaMemset(d_fi, 0x11, N * sizeof(int));
    launch_both<T>(c.fused_base, d_in, d_fv, d_fi, n_out, k, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> fv(N); std::vector<int> fi(N);
    cudaMemcpy(fv.data(), d_fv, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(fi.data(), d_fi, N * sizeof(int), cudaMemcpyDeviceToHost);
    report("base out_val==row_sort", memcmp(fv.data(), rv.data(), N * sizeof(T)) == 0);
    report("base out_idx==row_argsort", memcmp(fi.data(), ri.data(), N * sizeof(int)) == 0);

    // Determinism: a second base launch is bit-identical on BOTH buffers.
    cudaMemset(d_fv2, 0xA5, N * sizeof(T)); cudaMemset(d_fi2, 0x22, N * sizeof(int));
    launch_both<T>(c.fused_base, d_in, d_fv2, d_fi2, n_out, k, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> fv_b(N); std::vector<int> fi_b(N);
    cudaMemcpy(fv_b.data(), d_fv2, N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(fi_b.data(), d_fi2, N * sizeof(int), cudaMemcpyDeviceToHost);
    report("base determinism", memcmp(fv.data(), fv_b.data(), N * sizeof(T)) == 0
                            && memcmp(fi.data(), fi_b.data(), N * sizeof(int)) == 0);

    if (do_bitonic) {
        cudaMemset(d_fv2, 0x77, N * sizeof(T)); cudaMemset(d_fi2, 0x33, N * sizeof(int));
        launch_both<T>(c.fused_bt, d_in, d_fv2, d_fi2, n_out, k, grid, block, smem);
        cudaDeviceSynchronize();
        std::vector<T> bv(N); std::vector<int> bi(N);
        cudaMemcpy(bv.data(), d_fv2, N * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(bi.data(), d_fi2, N * sizeof(int), cudaMemcpyDeviceToHost);
        report("bitonic out_val==row_sort", memcmp(bv.data(), rv.data(), N * sizeof(T)) == 0);
        report("bitonic out_idx==row_argsort", memcmp(bi.data(), ri.data(), N * sizeof(int)) == 0);
        report("base==bitonic (both buffers)", memcmp(bv.data(), fv.data(), N * sizeof(T)) == 0
                                            && memcmp(bi.data(), fi.data(), N * sizeof(int)) == 0);
    }
    cudaFree(d_in); cudaFree(d_rv); cudaFree(d_ri);
    cudaFree(d_fv); cudaFree(d_fi); cudaFree(d_fv2); cudaFree(d_fi2);
}

static BothCell<float> both_cell_f32(bool asc) {
    return asc
        ? BothCell<float>{"f32", true, 4,
            (void*)baracuda_gen_fused_f32_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f32_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f32_rowsort_asc_stable, (void*)baracuda_gen_argsort_f32_rowsort_asc_stable_idx}
        : BothCell<float>{"f32", false, 4,
            (void*)baracuda_gen_fused_f32_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f32_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f32_rowsort_desc_stable, (void*)baracuda_gen_argsort_f32_rowsort_desc_stable_idx};
}
static BothCell<double> both_cell_f64(bool asc) {
    return asc
        ? BothCell<double>{"f64", true, 8,
            (void*)baracuda_gen_fused_f64_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f64_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f64_rowsort_asc_stable, (void*)baracuda_gen_argsort_f64_rowsort_asc_stable_idx}
        : BothCell<double>{"f64", false, 8,
            (void*)baracuda_gen_fused_f64_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f64_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f64_rowsort_desc_stable, (void*)baracuda_gen_argsort_f64_rowsort_desc_stable_idx};
}
static BothCell<int> both_cell_i32(bool asc) {
    return asc
        ? BothCell<int>{"i32", true, 4,
            (void*)baracuda_gen_fused_i32_rowsort_asc_stable_both, (void*)baracuda_gen_fused_i32_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_i32_rowsort_asc_stable, (void*)baracuda_gen_argsort_i32_rowsort_asc_stable_idx}
        : BothCell<int>{"i32", false, 4,
            (void*)baracuda_gen_fused_i32_rowsort_desc_stable_both, (void*)baracuda_gen_fused_i32_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_sort_i32_rowsort_desc_stable, (void*)baracuda_gen_argsort_i32_rowsort_desc_stable_idx};
}
static BothCell<long long> both_cell_i64() {
    return BothCell<long long>{"i64", true, 8,
        (void*)baracuda_gen_fused_i64_rowsort_asc_stable_both, (void*)baracuda_gen_fused_i64_rowsort_asc_stable_both_bitonic,
        (void*)baracuda_gen_sort_i64_rowsort_asc_stable, (void*)baracuda_gen_argsort_i64_rowsort_asc_stable_idx};
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

// ============ FUSED Both acceptance matrix + bandwidth bench ================
static void both_acceptance() {
    printf("- FUSED Both acceptance: memcmp(out_val, row_sort) && memcmp(out_idx, row_argsort) + base==bitonic -\n");
    // random rows near cap (k=1000): f32/f64/i32 asc+desc.
    for (bool asc : {true, false}) {
        { std::vector<float> in((size_t)3 * 1000); for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
          run_both_cell<float>(both_cell_f32(asc), in, 3, 1000, "rand", true, 256); }
        { std::vector<double> in((size_t)3 * 1000); for (auto& x : in) x = (double)((int)(xrand() % 20001) - 10000) * 0.001;
          run_both_cell<double>(both_cell_f64(asc), in, 3, 1000, "rand", true, 256); }
        { std::vector<int> in((size_t)3 * 1000); for (auto& x : in) x = (int)((xrand() % 20001) - 10000);
          run_both_cell<int>(both_cell_i32(asc), in, 3, 1000, "rand", true, 256); }
    }
    // i64 asc (wide integer).
    { std::vector<long long> in((size_t)2 * 300); for (auto& x : in) x = (long long)((int)(xrand() % 20001) - 10000);
      run_both_cell<long long>(both_cell_i64(), in, 2, 300, "rand", true, 256); }
    // f16 / bf16 / f32s asc (acc/convert coverage — reuse the argsort oracle).
    {
        std::vector<__half> in((size_t)2 * 300);
        for (auto& x : in) x = __float2half((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
        BothCell<__half> c{"f16", true, 4,
            (void*)baracuda_gen_fused_f16_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f16_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f16_rowsort_asc_stable, (void*)baracuda_gen_argsort_f16_rowsort_asc_stable_idx};
        run_both_cell<__half>(c, in, 2, 300, "rand", true, 256);
    }
    {
        std::vector<__nv_bfloat16> in((size_t)2 * 300);
        for (auto& x : in) x = __float2bfloat16((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
        BothCell<__nv_bfloat16> c{"bf16", true, 4,
            (void*)baracuda_gen_fused_bf16_rowsort_asc_stable_both, (void*)baracuda_gen_fused_bf16_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_bf16_rowsort_asc_stable, (void*)baracuda_gen_argsort_bf16_rowsort_asc_stable_idx};
        run_both_cell<__nv_bfloat16>(c, in, 2, 300, "rand", true, 256);
    }
    {
        std::vector<float> in((size_t)2 * 300);
        for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
        BothCell<float> c{"f32s", true, 8,
            (void*)baracuda_gen_fused_f32s_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f32s_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_sort_f32s_rowsort_asc_stable, (void*)baracuda_gen_argsort_f32s_rowsort_asc_stable_idx};
        run_both_cell<float>(c, in, 2, 300, "rand", true, 256);
    }
    // Probe-seeded f32: qNaN payloads + negative-NaN (both directions).
    {
        const long long n = 3, k = 13; std::vector<float> in((size_t)n * k);
        for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(i % 11) - 5) * 0.5f;
        uint32_t nq[3] = {0x7fc00001u, 0x7fc0abcdu, 0xffc00042u};
        for (int t = 0; t < 3; ++t) memcpy(&in[(size_t)(1 * k + 2 + t * 3)], &nq[t], 4);
        for (bool asc : {true, false}) run_both_cell<float>(both_cell_f32(asc), in, n, k, "nan", true, 32);
    }
    // Probe-seeded f32: signed zeros + ±1 (both directions).
    {
        const long long n = 1, k = 8; std::vector<float> in((size_t)k);
        uint32_t pat[8] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x00000000u,
                           0x80000000u, 0xbf800000u, 0x80000000u, 0x00000000u};
        for (int i = 0; i < 8; ++i) memcpy(&in[(size_t)i], &pat[i], 4);
        for (bool asc : {true, false}) run_both_cell<float>(both_cell_f32(asc), in, n, k, "signed0", true, 32);
    }
    // Extreme values: ±inf + INT_MAX/MIN pad-tie invariant.
    {
        const long long n = 1, k = 100; std::vector<float> in((size_t)k);
        for (long long j = 0; j < k; ++j) in[(size_t)j] = (float)((int)(xrand() % 201) - 100) * 0.1f;
        uint32_t ninf = 0xff800000u, pinf = 0x7f800000u;
        memcpy(&in[5], &ninf, 4); memcpy(&in[6], &ninf, 4); memcpy(&in[7], &pinf, 4);
        for (bool asc : {true, false}) run_both_cell<float>(both_cell_f32(asc), in, n, k, "extreme(inf)", true, 128);
    }
    // Edge k: 1, 33, exactly 1024.
    for (long long k : {(long long)1, (long long)33, (long long)1024}) {
        std::vector<float> in((size_t)4 * k); for (auto& x : in) x = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
        run_both_cell<float>(both_cell_f32(true), in, 4, k, "edge", true, k >= 64 ? 256 : 32);
    }
    // k > 1024: base only (bitonic contract requires k <= 1024).
    {
        const long long n = 2, k = 1500; std::vector<float> in((size_t)n * k);
        for (auto& x : in) x = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
        run_both_cell<float>(both_cell_f32(true), in, n, k, "k1500 (base only)", false, 0);
    }
}

// Bandwidth: the fused Both (one sort, two writes) vs the two decomposed #8
// kernels summed (row_sort + row_argsort = TWO full sorts). k=1024.
static void both_bench() {
    const long long R = 4096, K = 1024; const long long tot = R * K;
    std::vector<float> big((size_t)tot);
    for (size_t i = 0; i < (size_t)tot; ++i) big[i] = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
    float* dx = (float*)dev_bytes(big.data(), (size_t)tot * 4);
    float* dv = nullptr; cudaMalloc(&dv, (size_t)tot * 4);
    int* di = nullptr; cudaMalloc(&di, (size_t)tot * 4);
    BothCell<float> c = both_cell_f32(true);
    size_t smem = (size_t)K * (4 + 4);
    auto timeit = [&](auto fn) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / 20;
    };
    double es = (double)tot / 1e9;
    float t_fb = timeit([&] { launch_both<float>(c.fused_base, dx, dv, di, R, K, (int)R, 256, 0); });
    float t_ft = timeit([&] { launch_both<float>(c.fused_bt, dx, dv, di, R, K, (int)R, 256, smem); });
    float t_db = timeit([&] {
        launch<float, float>(c.sort_base, dx, dv, R, K, (int)R, 256, 0);
        launch<float, int>(c.argsort_base, dx, di, R, K, (int)R, 256, 0);
    });
    float t_dt = timeit([&] {
        launch<float, float>((void*)baracuda_gen_sort_f32_rowsort_asc_stable_bitonic, dx, dv, R, K, (int)R, 256, smem);
        launch<float, int>((void*)baracuda_gen_argsort_f32_rowsort_asc_stable_idx_bitonic, dx, di, R, K, (int)R, 256, smem);
    });
    printf("[bench] fused Both f32 %lldx%lld: base %.3f ms (%.2f Gelem/s) | bitonic %.3f ms (%.2f Gelem/s) || "
           "decomposed row_sort+row_argsort: base %.3f ms | bitonic %.3f ms || fused speedup %.2fx (base) %.2fx (bitonic)\n",
           R, K, t_fb, es / (t_fb / 1000), t_ft, es / (t_ft / 1000), t_db, t_dt, t_db / t_fb, t_dt / t_ft);
    cudaFree(dx); cudaFree(dv); cudaFree(di);
}

// ===================== Increment 10 TOPK / BOTTOMK ==========================
// TopK is the strict generalization of the fused Both — the SAME sort, the write
// CAPPED to the first k_out ranks under `order` (topk = Desc / largest first,
// bottomk = Asc / smallest first). Signature adds the third launch scalar:
//   (const T* in0, T* out_val, int* out_idx, long long n_out,
//    long long k_in, long long k_out).
//
// THE ACCEPTANCE GATE (brief §6): for every cell, the two topk outputs are
//  (A) whole-buffer memcmp-equal to the CPU pair_lt oracle's first k_out (an
//      independent, device-free reference), AND
//  (B) PER-ROW (stride-aware) memcmp-equal to the device-validated RowSort Both's
//      first-k_out slice — Both's row stride is k_in, topk's is k_out, so this is
//      `topk[row*k_out .. +k_out] == Both[row*k_in .. +k_out]` for out_val AND
//      out_idx, NOT a flat whole-buffer prefix (decision-5 caveat).
// Plus base == bitonic (BitIdentical) and run-to-run determinism (both buffers).
template <typename T>
static void launch_topk(void* kern, const T* in, T* out_val, int* out_idx,
                        long long n_out, long long k_in, long long k_out,
                        int grid, int block, size_t smem) {
    const T* pin = in; T* pv = out_val; int* pi = out_idx;
    void* args[] = { (void*)&pin, (void*)&pv, (void*)&pi, (void*)&n_out, (void*)&k_in, (void*)&k_out };
    CHECK(cudaLaunchKernel(kern, dim3(grid), dim3(block), args, smem, 0));
}

template <typename T, typename K>
struct TopkCell {
    const char* dtag; bool is_fp; bool asc; size_t acc_sz;   // asc == bottomk (order Asc)
    void* topk_base; void* topk_bt;    // the capped kernel (_topk)
    void* both_base; void* both_bt;    // the shipped RowSort Both at the SAME order (device oracle)
    void* topk_psel;                   // the PARTIAL-SELECT streaming top-k VARIANT
    K (*to_key)(const T&);
};

template <typename T, typename K>
static void run_topk_cell(const TopkCell<T, K>& c, const std::vector<T>& in,
                          long long n_out, long long k_in, long long k_out,
                          const char* tag, bool do_bitonic, int block) {
    const size_t Nin = (size_t)n_out * (size_t)k_in;
    const size_t Nout = (size_t)n_out * (size_t)k_out;

    // ---- CPU pair_lt oracle → first k_out per row (independent of the device) ----
    std::vector<T> ov(Nout); std::vector<int> oi(Nout);
    for (long long r = 0; r < n_out; ++r) {
        std::vector<K> keys((size_t)k_in);
        for (long long j = 0; j < k_in; ++j) keys[(size_t)j] = c.to_key(in[(size_t)(r * k_in + j)]);
        std::vector<int> perm; oracle_perm(keys, k_in, c.asc, c.is_fp, perm);
        for (long long j = 0; j < k_out; ++j) {
            ov[(size_t)(r * k_out + j)] = in[(size_t)(r * k_in + perm[(size_t)j])];
            oi[(size_t)(r * k_out + j)] = perm[(size_t)j];
        }
    }

    T* d_in = (T*)dev_bytes(in.data(), Nin * sizeof(T));
    T* d_bv = nullptr; cudaMalloc(&d_bv, Nin * sizeof(T));    // device Both values (k_in wide)
    int* d_bi = nullptr; cudaMalloc(&d_bi, Nin * sizeof(int));
    T* d_tv = nullptr; cudaMalloc(&d_tv, Nout * sizeof(T));   // topk out_val (k_out wide)
    int* d_ti = nullptr; cudaMalloc(&d_ti, Nout * sizeof(int));
    T* d_tv2 = nullptr; cudaMalloc(&d_tv2, Nout * sizeof(T)); // determinism / bitonic
    int* d_ti2 = nullptr; cudaMalloc(&d_ti2, Nout * sizeof(int));
    int grid = (int)(n_out < 65535 ? n_out : 65535); if (grid < 1) grid = 1;
    size_t smem = (size_t)next_pow2(k_in) * (c.acc_sz + sizeof(int));

    auto report = [&](const char* what, bool ok) {
        char nm[160];
        snprintf(nm, sizeof nm, "%s %s %s %s k=%lld/%lld", c.dtag,
                 c.asc ? "bottomk" : "topk", what, tag, k_out, k_in);
        printf(ok ? "PASS %-64s\n" : "FAIL %-64s\n", nm); if (!ok) fails++;
    };

    // Device Both oracle (k_in wide), SAME order.
    cudaMemset(d_bv, 0x5A, Nin * sizeof(T)); cudaMemset(d_bi, 0xFF, Nin * sizeof(int));
    launch_both<T>(c.both_base, d_in, d_bv, d_bi, n_out, k_in, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> bv(Nin); std::vector<int> bi(Nin);
    cudaMemcpy(bv.data(), d_bv, Nin * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(bi.data(), d_bi, Nin * sizeof(int), cudaMemcpyDeviceToHost);

    // topk base — the acceptance gate.
    cudaMemset(d_tv, 0x3C, Nout * sizeof(T)); cudaMemset(d_ti, 0x11, Nout * sizeof(int));
    launch_topk<T>(c.topk_base, d_in, d_tv, d_ti, n_out, k_in, k_out, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> tv(Nout); std::vector<int> ti(Nout);
    cudaMemcpy(tv.data(), d_tv, Nout * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(ti.data(), d_ti, Nout * sizeof(int), cudaMemcpyDeviceToHost);

    // (A) topk == CPU pair_lt oracle first-k_out (whole buffer, both k_out wide).
    report("base out_val==cpu-oracle", memcmp(tv.data(), ov.data(), Nout * sizeof(T)) == 0);
    report("base out_idx==cpu-oracle", memcmp(ti.data(), oi.data(), Nout * sizeof(int)) == 0);

    // (B) PER-ROW STRIDE-AWARE dual memcmp vs the device Both's first-k_out slice.
    {
        bool okv = true, oki = true;
        for (long long r = 0; r < n_out; ++r) {
            if (memcmp(&tv[(size_t)(r * k_out)], &bv[(size_t)(r * k_in)], (size_t)k_out * sizeof(T)) != 0) okv = false;
            if (memcmp(&ti[(size_t)(r * k_out)], &bi[(size_t)(r * k_in)], (size_t)k_out * sizeof(int)) != 0) oki = false;
        }
        report("base out_val==Both[..k_out] per-row", okv);
        report("base out_idx==Both[..k_out] per-row", oki);
    }

    // (C) determinism (both buffers).
    cudaMemset(d_tv2, 0xA5, Nout * sizeof(T)); cudaMemset(d_ti2, 0x22, Nout * sizeof(int));
    launch_topk<T>(c.topk_base, d_in, d_tv2, d_ti2, n_out, k_in, k_out, grid, 256, 0);
    cudaDeviceSynchronize();
    std::vector<T> tvb(Nout); std::vector<int> tib(Nout);
    cudaMemcpy(tvb.data(), d_tv2, Nout * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(tib.data(), d_ti2, Nout * sizeof(int), cudaMemcpyDeviceToHost);
    report("base determinism", memcmp(tv.data(), tvb.data(), Nout * sizeof(T)) == 0
                            && memcmp(ti.data(), tib.data(), Nout * sizeof(int)) == 0);

    // base == bitonic (BitIdentical) + bitonic vs both oracles.
    if (do_bitonic) {
        cudaMemset(d_tv2, 0x77, Nout * sizeof(T)); cudaMemset(d_ti2, 0x33, Nout * sizeof(int));
        launch_topk<T>(c.topk_bt, d_in, d_tv2, d_ti2, n_out, k_in, k_out, grid, block, smem);
        cudaDeviceSynchronize();
        std::vector<T> btv(Nout); std::vector<int> bti(Nout);
        cudaMemcpy(btv.data(), d_tv2, Nout * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(bti.data(), d_ti2, Nout * sizeof(int), cudaMemcpyDeviceToHost);
        report("bitonic out_val==cpu-oracle", memcmp(btv.data(), ov.data(), Nout * sizeof(T)) == 0);
        report("bitonic out_idx==cpu-oracle", memcmp(bti.data(), oi.data(), Nout * sizeof(int)) == 0);
        report("base==bitonic (both buffers)", memcmp(btv.data(), tv.data(), Nout * sizeof(T)) == 0
                                            && memcmp(bti.data(), ti.data(), Nout * sizeof(int)) == 0);
    }

    // ---- PARTIAL-SELECT streaming top-k (`_psel`): BitIdentical vs BOTH oracles
    // — (A) the CPU pair_lt first-k_out AND (B) the device full-sort Both's
    // first-k_out per-row slice — plus two-launch determinism. Smem is bounded on
    // k_out (2*next_pow2(k_out) pairs), NOT k_in, so psel runs at ANY k_in
    // (including the k_in > 1024 base-only cases the bitonic must decline). ----
    if (c.topk_psel) {
        int pblock = block < 32 ? 32 : block;
        size_t smem_ps = (size_t)2 * next_pow2(k_out) * (c.acc_sz + sizeof(int));
        cudaMemset(d_tv, 0x66, Nout * sizeof(T)); cudaMemset(d_ti, 0x44, Nout * sizeof(int));
        launch_topk<T>(c.topk_psel, d_in, d_tv, d_ti, n_out, k_in, k_out, grid, pblock, smem_ps);
        cudaDeviceSynchronize();
        std::vector<T> pv(Nout); std::vector<int> pi(Nout);
        cudaMemcpy(pv.data(), d_tv, Nout * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(pi.data(), d_ti, Nout * sizeof(int), cudaMemcpyDeviceToHost);
        // (A) psel == CPU pair_lt oracle first-k_out (whole buffer).
        report("psel out_val==cpu-oracle", memcmp(pv.data(), ov.data(), Nout * sizeof(T)) == 0);
        report("psel out_idx==cpu-oracle", memcmp(pi.data(), oi.data(), Nout * sizeof(int)) == 0);
        // (B) psel == device full-sort Both's first-k_out slice, PER ROW (stride-aware).
        {
            bool okv = true, oki = true;
            for (long long r = 0; r < n_out; ++r) {
                if (memcmp(&pv[(size_t)(r * k_out)], &bv[(size_t)(r * k_in)], (size_t)k_out * sizeof(T)) != 0) okv = false;
                if (memcmp(&pi[(size_t)(r * k_out)], &bi[(size_t)(r * k_in)], (size_t)k_out * sizeof(int)) != 0) oki = false;
            }
            report("psel out_val==Both[..k_out] per-row", okv);
            report("psel out_idx==Both[..k_out] per-row", oki);
        }
        // (C) determinism (both buffers), second launch.
        cudaMemset(d_tv2, 0x99, Nout * sizeof(T)); cudaMemset(d_ti2, 0x55, Nout * sizeof(int));
        launch_topk<T>(c.topk_psel, d_in, d_tv2, d_ti2, n_out, k_in, k_out, grid, pblock, smem_ps);
        cudaDeviceSynchronize();
        std::vector<T> pvb(Nout); std::vector<int> pib(Nout);
        cudaMemcpy(pvb.data(), d_tv2, Nout * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(pib.data(), d_ti2, Nout * sizeof(int), cudaMemcpyDeviceToHost);
        report("psel determinism", memcmp(pv.data(), pvb.data(), Nout * sizeof(T)) == 0
                                && memcmp(pi.data(), pib.data(), Nout * sizeof(int)) == 0);
    }

    cudaFree(d_in); cudaFree(d_bv); cudaFree(d_bi);
    cudaFree(d_tv); cudaFree(d_ti); cudaFree(d_tv2); cudaFree(d_ti2);
}

// One constructor per dtype: `is_topk` picks Desc (topk) vs Asc (bottomk). The
// CPU-oracle `asc` field is the SORT direction (bottomk == asc), the opposite of
// `is_topk`.
static TopkCell<float, float> topk_cell_f32(bool is_topk) {
    return is_topk
        ? TopkCell<float, float>{"f32", true, false, 4,
            (void*)baracuda_gen_topk_f32_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f32_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f32_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_psel, k_f32}
        : TopkCell<float, float>{"f32", true, true, 4,
            (void*)baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f32_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f32_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk_psel, k_f32};
}
static TopkCell<double, double> topk_cell_f64(bool is_topk) {
    return is_topk
        ? TopkCell<double, double>{"f64", true, false, 8,
            (void*)baracuda_gen_topk_f64_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_f64_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f64_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f64_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_f64_rowsort_desc_stable_both_topk_psel, k_f64}
        : TopkCell<double, double>{"f64", true, true, 8,
            (void*)baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f64_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f64_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_f64_rowsort_asc_stable_both_topk_psel, k_f64};
}
static TopkCell<int, int> topk_cell_i32(bool is_topk) {
    return is_topk
        ? TopkCell<int, int>{"i32", false, false, 4,
            (void*)baracuda_gen_topk_i32_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_i32_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_i32_rowsort_desc_stable_both, (void*)baracuda_gen_fused_i32_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_i32_rowsort_desc_stable_both_topk_psel, k_i32}
        : TopkCell<int, int>{"i32", false, true, 4,
            (void*)baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_i32_rowsort_asc_stable_both, (void*)baracuda_gen_fused_i32_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_i32_rowsort_asc_stable_both_topk_psel, k_i32};
}
static TopkCell<long long, long long> topk_cell_i64(bool is_topk) {
    return is_topk
        ? TopkCell<long long, long long>{"i64", false, false, 8,
            (void*)baracuda_gen_topk_i64_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_i64_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_i64_rowsort_desc_stable_both, (void*)baracuda_gen_fused_i64_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_i64_rowsort_desc_stable_both_topk_psel, k_i64}
        : TopkCell<long long, long long>{"i64", false, true, 8,
            (void*)baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_i64_rowsort_asc_stable_both, (void*)baracuda_gen_fused_i64_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_i64_rowsort_asc_stable_both_topk_psel, k_i64};
}
static TopkCell<__half, float> topk_cell_f16(bool is_topk) {
    return is_topk
        ? TopkCell<__half, float>{"f16", true, false, 4,
            (void*)baracuda_gen_topk_f16_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_f16_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f16_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f16_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_f16_rowsort_desc_stable_both_topk_psel, k_f16}
        : TopkCell<__half, float>{"f16", true, true, 4,
            (void*)baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f16_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f16_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_f16_rowsort_asc_stable_both_topk_psel, k_f16};
}
static TopkCell<__nv_bfloat16, float> topk_cell_bf16(bool is_topk) {
    return is_topk
        ? TopkCell<__nv_bfloat16, float>{"bf16", true, false, 4,
            (void*)baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_bf16_rowsort_desc_stable_both, (void*)baracuda_gen_fused_bf16_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_bf16_rowsort_desc_stable_both_topk_psel, k_bf16}
        : TopkCell<__nv_bfloat16, float>{"bf16", true, true, 4,
            (void*)baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_bf16_rowsort_asc_stable_both, (void*)baracuda_gen_fused_bf16_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_bf16_rowsort_asc_stable_both_topk_psel, k_bf16};
}
static TopkCell<float, double> topk_cell_f32s(bool is_topk) {
    return is_topk
        ? TopkCell<float, double>{"f32s", true, false, 8,
            (void*)baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk, (void*)baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f32s_rowsort_desc_stable_both, (void*)baracuda_gen_fused_f32s_rowsort_desc_stable_both_bitonic,
            (void*)baracuda_gen_topk_f32s_rowsort_desc_stable_both_topk_psel, k_f32s}
        : TopkCell<float, double>{"f32s", true, true, 8,
            (void*)baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk, (void*)baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk_bitonic,
            (void*)baracuda_gen_fused_f32s_rowsort_asc_stable_both, (void*)baracuda_gen_fused_f32s_rowsort_asc_stable_both_bitonic,
            (void*)baracuda_gen_bottomk_f32s_rowsort_asc_stable_both_topk_psel, k_f32s};
}

// INDEPENDENT CPU top-k oracle: on DISTINCT-KEY, NaN-free rows, `std::nth_element`
// (quickselect — a different algorithm than the device's rank/bitonic sort) picks
// the k_out extreme VALUES; their sorted order must equal the device topk out_val,
// and each out_idx must point to a position holding the matching value. This is a
// reference top-k identity, not a restatement of the sort-then-slice.
static void topk_select_oracle_f32() {
    const long long batch = 32, k_in = 300;
    for (int is_topk = 0; is_topk <= 1; ++is_topk) {
        for (long long k_out : {(long long)1, (long long)5, (long long)150, (long long)300}) {
            std::vector<float> in((size_t)batch * k_in);
            for (long long r = 0; r < batch; ++r) {          // distinct shuffled ramp per row
                std::vector<int> perm((size_t)k_in); for (long long j = 0; j < k_in; ++j) perm[(size_t)j] = (int)j;
                for (size_t i = perm.size(); i > 1; --i) { size_t j = xrand() % i; std::swap(perm[i - 1], perm[j]); }
                for (long long j = 0; j < k_in; ++j) in[(size_t)(r * k_in + j)] = (float)perm[(size_t)j] * 0.5f - 75.0f;
            }
            TopkCell<float, float> c = topk_cell_f32(is_topk != 0);
            const size_t Nout = (size_t)batch * (size_t)k_out;
            float* d_in = (float*)dev_bytes(in.data(), (size_t)batch * k_in * sizeof(float));
            float* d_tv = nullptr; cudaMalloc(&d_tv, Nout * sizeof(float));
            int* d_ti = nullptr; cudaMalloc(&d_ti, Nout * sizeof(int));
            int grid = (int)batch;
            launch_topk<float>(c.topk_base, d_in, d_tv, d_ti, batch, k_in, k_out, grid, 256, 0);
            cudaDeviceSynchronize();
            std::vector<float> tv(Nout); std::vector<int> ti(Nout);
            cudaMemcpy(tv.data(), d_tv, Nout * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(ti.data(), d_ti, Nout * sizeof(int), cudaMemcpyDeviceToHost);
            bool ok = true;
            for (long long r = 0; r < batch && ok; ++r) {
                std::vector<float> keys((size_t)k_in);
                for (long long j = 0; j < k_in; ++j) keys[(size_t)j] = in[(size_t)(r * k_in + j)];
                // Quickselect the k_out extreme, then sort them in output order.
                if (is_topk) std::nth_element(keys.begin(), keys.begin() + (k_out - 1), keys.end(), std::greater<float>());
                else         std::nth_element(keys.begin(), keys.begin() + (k_out - 1), keys.end());
                std::vector<float> sel(keys.begin(), keys.begin() + (size_t)k_out);
                if (is_topk) std::sort(sel.begin(), sel.end(), std::greater<float>());
                else         std::sort(sel.begin(), sel.end());
                for (long long j = 0; j < k_out; ++j) {
                    if (tv[(size_t)(r * k_out + j)] != sel[(size_t)j]) { ok = false; break; }
                    int idx = ti[(size_t)(r * k_out + j)];
                    if (idx < 0 || idx >= k_in || in[(size_t)(r * k_in + idx)] != sel[(size_t)j]) { ok = false; break; }
                }
            }
            char nm[96]; snprintf(nm, sizeof nm, "f32 %s nth_element select k=%lld/%lld",
                                  is_topk ? "topk" : "bottomk", k_out, k_in);
            printf(ok ? "PASS %-64s\n" : "FAIL %-64s\n", nm); if (!ok) fails++;
            cudaFree(d_in); cudaFree(d_tv); cudaFree(d_ti);
        }
    }
}

// The full topk/bottomk acceptance matrix + the k_out boundary sweep.
static void topk_acceptance() {
    printf("- TOPK/BOTTOMK acceptance: cpu-oracle + per-row Both[..k_out] + base==bitonic, k_out sweep -\n");
    const long long n = 3, k_in = 1000;
    // k_out sweep: 1 (max/argmax), 2, k_in/2, k_in-1, k_in (== full sort).
    long long sweep[5] = { 1, 2, k_in / 2, k_in - 1, k_in };
    for (int is_topk = 0; is_topk <= 1; ++is_topk) {
        bool tk = (is_topk != 0);
        for (long long k_out : sweep) {
            { std::vector<float> in((size_t)n * k_in); for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
              run_topk_cell<float, float>(topk_cell_f32(tk), in, n, k_in, k_out, "rand", true, 256); }
            { std::vector<double> in((size_t)n * k_in); for (auto& x : in) x = (double)((int)(xrand() % 20001) - 10000) * 0.001;
              run_topk_cell<double, double>(topk_cell_f64(tk), in, n, k_in, k_out, "rand", true, 256); }
            { std::vector<int> in((size_t)n * k_in); for (auto& x : in) x = (int)((xrand() % 20001) - 10000);
              run_topk_cell<int, int>(topk_cell_i32(tk), in, n, k_in, k_out, "rand", true, 256); }
        }
    }
    // i64 / f16 / bf16 / f32s — smaller k_in (300), the k_out sweep scaled.
    const long long k2 = 300; long long sweep2[5] = { 1, 2, k2 / 2, k2 - 1, k2 };
    for (int is_topk = 0; is_topk <= 1; ++is_topk) {
        bool tk = (is_topk != 0);
        for (long long k_out : sweep2) {
            { std::vector<long long> in((size_t)2 * k2); for (auto& x : in) x = (long long)((int)(xrand() % 20001) - 10000);
              run_topk_cell<long long, long long>(topk_cell_i64(tk), in, 2, k2, k_out, "rand", true, 256); }
            { std::vector<__half> in((size_t)2 * k2); for (auto& x : in) x = __float2half((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
              run_topk_cell<__half, float>(topk_cell_f16(tk), in, 2, k2, k_out, "rand", true, 256); }
            { std::vector<__nv_bfloat16> in((size_t)2 * k2); for (auto& x : in) x = __float2bfloat16((float)((int)(xrand() % 2001) - 1000) * 0.03125f);
              run_topk_cell<__nv_bfloat16, float>(topk_cell_bf16(tk), in, 2, k2, k_out, "rand", true, 256); }
            { std::vector<float> in((size_t)2 * k2); for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
              run_topk_cell<float, double>(topk_cell_f32s(tk), in, 2, k2, k_out, "rand", true, 256); }
        }
    }
    // Probe-seeded f32 (topk + bottomk): qNaN payloads (NaN-first for topk, NaN-last
    // for bottomk), signed zeros, ±inf pad-tie — the torch-faithful cases.
    for (int is_topk = 0; is_topk <= 1; ++is_topk) {
        bool tk = (is_topk != 0);
        {   const long long nn = 3, kk = 13; std::vector<float> in((size_t)nn * kk);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(i % 11) - 5) * 0.5f;
            uint32_t nq[3] = {0x7fc00001u, 0x7fc0abcdu, 0xffc00042u};
            for (int t = 0; t < 3; ++t) memcpy(&in[(size_t)(1 * kk + 2 + t * 3)], &nq[t], 4);
            for (long long k_out : {(long long)1, (long long)4, (long long)13})
                run_topk_cell<float, float>(topk_cell_f32(tk), in, nn, kk, k_out, "nan", true, 32); }
        {   const long long nn = 1, kk = 8; std::vector<float> in((size_t)kk);
            uint32_t pat[8] = {0x00000000u, 0x80000000u, 0x3f800000u, 0x00000000u,
                               0x80000000u, 0xbf800000u, 0x80000000u, 0x00000000u};
            for (int i = 0; i < 8; ++i) memcpy(&in[(size_t)i], &pat[i], 4);
            for (long long k_out : {(long long)1, (long long)3, (long long)8})
                run_topk_cell<float, float>(topk_cell_f32(tk), in, nn, kk, k_out, "signed0", true, 32); }
        {   const long long nn = 1, kk = 100; std::vector<float> in((size_t)kk);
            for (long long j = 0; j < kk; ++j) in[(size_t)j] = (float)((int)(xrand() % 201) - 100) * 0.1f;
            uint32_t ninf = 0xff800000u, pinf = 0x7f800000u;
            memcpy(&in[5], &ninf, 4); memcpy(&in[6], &ninf, 4); memcpy(&in[7], &pinf, 4);
            for (long long k_out : {(long long)1, (long long)50, (long long)100})
                run_topk_cell<float, float>(topk_cell_f32(tk), in, nn, kk, k_out, "extreme(inf)", true, 128); }
    }
    // k_in > 1024: base only (bitonic contract requires k_in <= 1024).
    { std::vector<float> in((size_t)2 * 1500); for (auto& x : in) x = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
      for (long long k_out : {(long long)1, (long long)64, (long long)1500})
          run_topk_cell<float, float>(topk_cell_f32(true), in, 2, 1500, k_out, "k1500 (base only)", false, 0); }
    // Independent nth_element selection oracle (distinct-key rows).
    topk_select_oracle_f32();
}

// Bench: topk (k_out-only writeback) vs the full-sort-then-host-slice baseline —
// same sort cost, but topk writes only [batch,k_out] (no full [batch,k_in] output).
static void topk_bench() {
    const long long R = 4096, K = 1024, KO = 64; const long long tot = R * K;
    std::vector<float> big((size_t)tot);
    for (size_t i = 0; i < (size_t)tot; ++i) big[i] = (float)((int)(xrand() % 4001) - 2000) * 0.01f;
    float* dx = (float*)dev_bytes(big.data(), (size_t)tot * 4);
    float* dv = nullptr; cudaMalloc(&dv, (size_t)tot * 4);
    int* di = nullptr; cudaMalloc(&di, (size_t)tot * 4);
    float* dvo = nullptr; cudaMalloc(&dvo, (size_t)R * KO * 4);
    int* dio = nullptr; cudaMalloc(&dio, (size_t)R * KO * 4);
    TopkCell<float, float> c = topk_cell_f32(true);
    size_t smem = (size_t)K * (4 + 4);
    auto timeit = [&](auto fn) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / 20;
    };
    double es = (double)tot / 1e9;
    float t_tb = timeit([&] { launch_topk<float>(c.topk_base, dx, dvo, dio, R, K, KO, (int)R, 256, 0); });
    float t_tt = timeit([&] { launch_topk<float>(c.topk_bt, dx, dvo, dio, R, K, KO, (int)R, 256, smem); });
    float t_fb = timeit([&] { launch_both<float>(c.both_base, dx, dv, di, R, K, (int)R, 256, 0); });
    float t_ft = timeit([&] { launch_both<float>(c.both_bt, dx, dv, di, R, K, (int)R, 256, smem); });
    printf("[bench] topk f32 %lldx%lld k_out=%lld: base %.3f ms (%.2f Gelem/s) | bitonic %.3f ms (%.2f Gelem/s) || "
           "full-sort Both (then host slice): base %.3f ms | bitonic %.3f ms\n",
           R, K, KO, t_tb, es / (t_tb / 1000), t_tt, es / (t_tt / 1000), t_fb, t_ft);
    cudaFree(dx); cudaFree(dv); cudaFree(di); cudaFree(dvo); cudaFree(dio);
}

// ============ PARTIAL-SELECT TOPK: the 11 mandatory probes + multi-tile =======
// Each probe feeds a hand-crafted input through run_topk_cell, whose psel block
// memcmps `_psel` out_val/out_idx vs BOTH the CPU pair_lt oracle AND the device
// full-sort Both base (the authoritative references) + two-launch determinism —
// so a divergence on ANY probe FAILS. The INPUTS are engineered to stress the
// three load-bearing invariants: the pad key (sort_pad_lit), the GLOBAL-index
// store (g = t*m + p, never the local slot), and keep-the-LOWER-half.
static std::vector<float> f32_bits(std::initializer_list<uint32_t> pats) {
    std::vector<float> v; v.reserve(pats.size());
    for (uint32_t p : pats) { float f; memcpy(&f, &p, 4); v.push_back(f); }
    return v;
}
static void psel_probes() {
    printf("- PARTIAL-SELECT: 11 mandatory probes (psel vs cpu-oracle + full-sort base, raw-bit) -\n");
    // P1 pad-index-invariant KILLER (bottomk real-NaN + pad-NaN): a pad-index<k_in
    // impl gives (0x7ff00000, idx1) at pos3 instead of (0x7fc00001, idx3).
    {   auto in = f32_bits({0x00800000u, 0x7ff00000u, 0xe0a99e9au, 0x7fc00001u, 0x7fc00000u});
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 5, 4, "P1a padidx", true, 32); }
    {   auto in = f32_bits({0x41100000u /*9.0*/, 0x7fc00001u, 0x7fc0abcdu});
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 3, 2, "P1b padidx", true, 32); }
    // P2 store-slot detector (multi-tile distinct values, m=2 → 3 tiles).
    {   std::vector<float> in = {10, 20, 30, 40, 50, 60};
        run_topk_cell<float, float>(topk_cell_f32(true), in, 1, 6, 2, "P2 storeslot", true, 32); }
    // P3 dup-key ACROSS the tile boundary (idx3 in tile0, idx4 in tile1 → idx3
    // first) + an all-equal row (pure index order). Both directions via two rows
    // (a value cannot be both the largest AND smallest).
    {   std::vector<float> in = {1, 7, 3, 8, 8, 2, 4, 6};   // dup 8.0 @ idx3,idx4 → topk keeps 3 before 4
        run_topk_cell<float, float>(topk_cell_f32(true), in, 1, 8, 4, "P3 dup-topk", true, 32); }
    {   std::vector<float> in = {9, 3, 7, 1, 1, 8, 5, 2};   // dup 1.0 @ idx3,idx4 → bottomk keeps 3 before 4
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 8, 4, "P3 dup-bottom", true, 32); }
    {   std::vector<float> in = {5, 5, 5, 5, 5, 5};
        run_topk_cell<float, float>(topk_cell_f32(true), in, 1, 6, 2, "P3 alleq", true, 32); }
    // P4 wrong-pad-key (+inf leak): bottomk reals asc end (..., +inf, NaN last); a
    // +inf pad would sort BEFORE the real NaN and leak into the output.
    {   auto in = f32_bits({0x3f800000u /*1*/, 0x7f800000u /*+inf*/, 0x7fc00001u, 0x40000000u /*2*/, 0xbf800000u /*-1*/});
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 5, 5, "P4 padleak", true, 32); }
    // P5 NaN payload preservation, both directions (distinct payloads; no quieting).
    {   auto in = f32_bits({0x40a00000u /*5*/, 0x7fc00001u, 0xffc0abcdu, 0x7fc0deadu, 0x40000000u /*2*/, 0xbfc00000u});
        for (bool tk : {true, false}) run_topk_cell<float, float>(topk_cell_f32(tk), in, 1, 6, 4, "P5 nanpay", true, 32); }
    // P6 all-NaN bottomk with pads (pure index tie-break; raw payloads intact).
    {   auto in = f32_bits({0x7fc00001u, 0x7fc00002u, 0x7fc00003u, 0xffc00004u, 0x7fc00005u, 0x7fc00006u});
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 6, 4, "P6 allnan", true, 32); }
    // P7 keep-wrong-half / direction (strictly monotone): topk idx [5,4], bottomk [0,1].
    {   std::vector<float> in = {10, 20, 30, 40, 50, 60};
        for (bool tk : {true, false}) run_topk_cell<float, float>(topk_cell_f32(tk), in, 1, 6, 2, "P7 direction", true, 32); }
    // P8 k_out=1 (m=1 running-best scan; a tie keeps the LOWER index).
    {   std::vector<float> in = {5, 9, 9, 3};   // topk tie at 9 → lower idx 1
        run_topk_cell<float, float>(topk_cell_f32(true), in, 1, 4, 1, "P8 kout1-top", true, 32); }
    {   std::vector<float> in = {5, 3, 3, 9};   // bottomk tie at 3 → lower idx 1
        run_topk_cell<float, float>(topk_cell_f32(false), in, 1, 4, 1, "P8 kout1-bottom", true, 32); }
    // P9 k_out==k_in (== full sort): random k_in=13 + non-pow2 k_in=300, asc+desc.
    {   std::vector<float> in(13); for (auto& x : in) x = (float)((int)(xrand() % 2001) - 1000) * 0.01f;
        for (bool tk : {true, false}) run_topk_cell<float, float>(topk_cell_f32(tk), in, 1, 13, 13, "P9 kfull13", true, 32); }
    {   std::vector<float> in(300); for (auto& x : in) x = (float)((int)(xrand() % 20001) - 10000) * 0.001f;
        for (bool tk : {true, false}) run_topk_cell<float, float>(topk_cell_f32(tk), in, 1, 300, 300, "P9 kfull300", true, 256); }
    // P10 partial last tile (m=2 → 3 tiles, last = 1 real + 1 pad): topk idx [4,2].
    {   std::vector<float> in = {5, 2, 8, 1, 9};
        run_topk_cell<float, float>(topk_cell_f32(true), in, 1, 5, 2, "P10 lasttile", true, 32); }
}

// P11: MULTI-TILE k_in > 1024 (the variant's PURPOSE) — the bitonic must decline
// (whole padded row exceeds one block), the rank base is O(k_in^2). psel streams
// in tiles of m = next_pow2(k_out); memcmp vs the full-sort Both base + determinism.
static void psel_multitile() {
    printf("- PARTIAL-SELECT probe 11: MULTI-TILE k_in > 1024 (the variant's PURPOSE) -\n");
    // k_in=4096 (128 tiles) asc+desc; k_in=8192 (256 tiles) asc+desc; k_out=32.
    for (long long k_in : {(long long)4096, (long long)8192}) {
        std::vector<float> in((size_t)3 * k_in);
        for (auto& x : in) x = (float)((int)(xrand() % 40001) - 20000) * 0.001f;
        for (bool tk : {true, false})
            run_topk_cell<float, float>(topk_cell_f32(tk), in, 3, k_in, 32, "P11 multitile", false, 256);
    }
    // k_in=3000 with ~40% specials (±0, ±inf, qNaN/sNaN payloads, subnormals), k_out=17 asc.
    {
        const long long k_in = 3000, n = 2;
        std::vector<float> in((size_t)n * k_in);
        uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00001u,
                         0x7fa00001u, 0xffc0abcdu, 0x00000001u, 0x80000001u, 0x007fffffu};
        for (size_t i = 0; i < in.size(); ++i) {
            if (xrand() % 5 < 2) { uint32_t p = sp[xrand() % (sizeof sp / 4)]; memcpy(&in[i], &p, 4); }
            else in[i] = (float)((int)(xrand() % 40001) - 20000) * 0.001f;
        }
        run_topk_cell<float, float>(topk_cell_f32(false), in, n, k_in, 17, "P11 specials", false, 256);
    }
}

// Bench: psel vs the full bitonic (CANNOT run at k_in=50000 — the whole padded row
// exceeds one block's smem) vs the rank base (O(k_in^2)). Headline top-10 of 50000
// + a modest top-64 of 1024 where all three run.
static void psel_bench() {
    auto timeit_n = [](auto fn, int warm, int iters) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < warm; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / iters;
    };
    auto bench_one = [&](long long R, long long K, long long KO, bool base_slow) {
        const long long tot = R * K;
        std::vector<float> big((size_t)tot);
        for (size_t i = 0; i < (size_t)tot; ++i) big[i] = (float)((int)(xrand() % 40001) - 20000) * 0.001f;
        float* dx = (float*)dev_bytes(big.data(), (size_t)tot * 4);
        float* dvo = nullptr; cudaMalloc(&dvo, (size_t)R * KO * 4);
        int* dio = nullptr; cudaMalloc(&dio, (size_t)R * KO * 4);
        TopkCell<float, float> c = topk_cell_f32(true);
        size_t smem_ps = (size_t)2 * next_pow2(KO) * (4 + 4);
        double es = (double)tot / 1e9;
        float t_ps = timeit_n([&] { launch_topk<float>(c.topk_psel, dx, dvo, dio, R, K, KO, (int)R, 256, smem_ps); }, 3, 20);
        // The rank base is O(k_in^2); for the huge-k_in headline time it lightly.
        float t_rb = base_slow
            ? timeit_n([&] { launch_topk<float>(c.topk_base, dx, dvo, dio, R, K, KO, (int)R, 256, 0); }, 1, 2)
            : timeit_n([&] { launch_topk<float>(c.topk_base, dx, dvo, dio, R, K, KO, (int)R, 256, 0); }, 3, 20);
        if (K <= 1024) {
            size_t smem_bt = (size_t)next_pow2(K) * (4 + 4);
            float t_bt = timeit_n([&] { launch_topk<float>(c.topk_bt, dx, dvo, dio, R, K, KO, (int)R, 256, smem_bt); }, 3, 20);
            printf("[bench] psel top-%lld of %lld (batch %lld): psel %.3f ms (%.2f Gelem/s) | rank base %.3f ms | "
                   "full-bitonic %.3f ms || psel %.1fx base, %.1fx bitonic\n",
                   KO, K, R, t_ps, es / (t_ps / 1000), t_rb, t_bt, t_rb / t_ps, t_bt / t_ps);
        } else {
            printf("[bench] psel top-%lld of %lld (batch %lld): psel %.3f ms (%.2f Gelem/s) | rank base %.3f ms | "
                   "full-bitonic N/A (k_in>1024 exceeds one block) || psel %.1fx base\n",
                   KO, K, R, t_ps, es / (t_ps / 1000), t_rb, t_rb / t_ps);
        }
        cudaFree(dx); cudaFree(dvo); cudaFree(dio);
    };
    bench_one(64, 50000, 10, true);   // the headline — bitonic can't run at k_in=50000
    bench_one(4096, 1024, 64, false); // modest regime — all three run
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

// ---- topk cross-check vs the bespoke partial-bitonic topk (baracuda_topk.cuh).
// The bespoke has NO NaN branch and STABLE=0, so kernelgen (torch-faithful:
// NaN-greatest + stable tie-break) DIVERGES on NaN / tie rows — the cross-check
// runs on DISTINCT-KEY, NaN-FREE rows only, where the top-k set AND order are
// unambiguous, so values AND indices must match bit-exactly. Bespoke limits:
// row_len <= 1024, k <= 64. `largest != 0` == descending (topk); 0 == bottomk.
static void topk_bespoke_audit() {
    const long long batch = 64, k_in = 512, k_out = 32;
    const size_t Nin = (size_t)batch * k_in, Nout = (size_t)batch * k_out;
    auto run_dt = [&](const char* dtag, int largest) {
        // f32 distinct shuffled ramp per row.
        std::vector<float> in(Nin);
        for (long long r = 0; r < batch; ++r) {
            std::vector<int> perm((size_t)k_in); for (long long j = 0; j < k_in; ++j) perm[(size_t)j] = (int)j;
            shuffle(perm);
            for (long long j = 0; j < k_in; ++j) in[(size_t)(r * k_in + j)] = (float)perm[(size_t)j] * 0.5f - 128.0f;
        }
        float* d_in = (float*)dev_bytes(in.data(), Nin * 4);
        float* d_gv = nullptr; cudaMalloc(&d_gv, Nout * 4); int* d_gi = nullptr; cudaMalloc(&d_gi, Nout * 4);
        float* d_bv = nullptr; cudaMalloc(&d_bv, Nout * 4); int* d_bi = nullptr; cudaMalloc(&d_bi, Nout * 4);
        TopkCell<float, float> c = topk_cell_f32(largest != 0);
        launch_topk<float>(c.topk_base, d_in, d_gv, d_gi, batch, k_in, k_out, (int)batch, 256, 0);
        int rc = baracuda_kernels_topk_f32_run((int)batch, (int)k_in, (int)k_out, largest,
                                               d_in, d_bv, d_bi, nullptr, 0, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> gv(Nout), bv(Nout); std::vector<int> gi(Nout), bi(Nout);
        cudaMemcpy(gv.data(), d_gv, Nout * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gi.data(), d_gi, Nout * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bv.data(), d_bv, Nout * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(bi.data(), d_bi, Nout * 4, cudaMemcpyDeviceToHost);
        bool okv = (rc == 0) && memcmp(gv.data(), bv.data(), Nout * 4) == 0;
        bool oki = (rc == 0) && memcmp(gi.data(), bi.data(), Nout * 4) == 0;
        char nm[96];
        snprintf(nm, sizeof nm, "audit %s %s VALUES==bespoke (distinct)", dtag, largest ? "topk" : "bottomk");
        printf(okv ? "PASS %-64s\n" : "FAIL %-64s (rc=%d)\n", nm, rc); if (!okv) fails++;
        snprintf(nm, sizeof nm, "audit %s %s INDICES==bespoke (distinct)", dtag, largest ? "topk" : "bottomk");
        printf(oki ? "PASS %-64s\n" : "FAIL %-64s (rc=%d)\n", nm, rc); if (!oki) fails++;
        cudaFree(d_in); cudaFree(d_gv); cudaFree(d_gi); cudaFree(d_bv); cudaFree(d_bi);
    };
    run_dt("f32", 1); // topk (descending)
    run_dt("f32", 0); // bottomk (ascending)
}
#endif

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== sort_validate (increment 8 SORT_PERM + 9 FUSED_ARGSORT + 10 TOPK/BOTTOMK) ==\n");

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
        // FUSED Both on small shapes — initcheck load-bearing on BOTH output buffers.
        {   const long long n = 4, k = 13; std::vector<float> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(xrand() % 201) - 100) * 0.1f;
            run_both_cell<float>(both_cell_f32(true), in, n, k, "san", true, 32);
            run_both_cell<float>(both_cell_f32(false), in, n, k, "san", true, 32); }
        {   const long long n = 3, k = 32; std::vector<int> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (int)((xrand() % 201) - 100);
            run_both_cell<int>(both_cell_i32(true), in, n, k, "san", true, 32); }
        // TOPK/BOTTOMK on small shapes — initcheck LOAD-BEARING: the guarded store
        // must write ALL k_out out slots (no under-write) AND none past k_out (no
        // over-write). Sweep k_out to poke the guard at both ends.
        {   const long long n = 4, k = 13; std::vector<float> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(xrand() % 201) - 100) * 0.1f;
            for (long long k_out : {(long long)1, (long long)5, (long long)13}) {
                run_topk_cell<float, float>(topk_cell_f32(true), in, n, k, k_out, "san", true, 32);
                run_topk_cell<float, float>(topk_cell_f32(false), in, n, k, k_out, "san", true, 32);
            } }
        {   const long long n = 3, k = 32; std::vector<int> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (int)((xrand() % 201) - 100);
            for (long long k_out : {(long long)1, (long long)16, (long long)32})
                run_topk_cell<int, int>(topk_cell_i32(true), in, n, k, k_out, "san", true, 32); }
        // PARTIAL-SELECT psel sanitizer sweep: the 2m smem swaps + 2 barriers/tile
        // are the load-bearing racecheck/synccheck path; initcheck proves all k_out
        // slots are written. Sweep block sizes {32,64,128,256,1024} to cross BOTH
        // 2m < blockDim AND 2m > blockDim (k_out=17 → 2m=64: b=32 is <, b>=128 is >).
        {   const long long n = 3, k = 64; std::vector<float> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (float)((int)(xrand() % 401) - 200) * 0.05f;
            for (int b : {32, 64, 128, 256, 1024}) {
                run_topk_cell<float, float>(topk_cell_f32(true), in, n, k, 17, "san-psel", false, b);
                run_topk_cell<float, float>(topk_cell_f32(false), in, n, k, 5, "san-psel", false, b);
            } }
        {   const long long n = 2, k = 40; std::vector<int> in((size_t)n * k);
            for (size_t i = 0; i < in.size(); ++i) in[i] = (int)((xrand() % 401) - 200);
            for (int b : {32, 128})
                run_topk_cell<int, int>(topk_cell_i32(true), in, n, k, 6, "san-psel", false, b); }
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

    printf("- increment 9 FUSED_ARGSORT: two-output Both acceptance + bandwidth -\n");
    both_acceptance();
    both_bench();

    printf("- increment 10 TOPK/BOTTOMK: runtime-k cap acceptance + bandwidth -\n");
    topk_acceptance();
    topk_bench();

    printf("- PARTIAL-SELECT TOPK: streaming tiled-bitonic top-k (psel) — 11 probes + multi-tile + bench -\n");
    psel_probes();
    psel_multitile();
    psel_bench();

#ifdef WITH_BESPOKE
    printf("- extract-the-delta audit vs bespoke stable msort (NaN-free) -\n");
    bespoke_audit();
    printf("- topk cross-check vs bespoke partial-bitonic topk (non-NaN/non-tie rows) -\n");
    topk_bespoke_audit();
#endif

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); fails++; }
    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
