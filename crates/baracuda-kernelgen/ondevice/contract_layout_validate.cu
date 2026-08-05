// On-device numeric proof for the sub-spec A contraction LAYOUT classes (this
// session's NEW layout code, previously only CPU-oracle-validated — Tasks 5/8
// emit the address math, Tasks 6/9 proved it against the CPU oracle):
//   - bmm_transposed — batched [B,M,K]·[B,K,N], rhs physically stored [B,N,K]
//     (K inner per slice): the SDPA Q·Kᵀ core. Proves the TRANSPOSED binding
//     `in1[b*k*n + col*k + kk]` (NOT the canonical `b*k*n + kk*n + col`).
//   - bmm_gqa — batched [B,M,K]·[B,K,N], rhs BROADCAST over batch (stride 0):
//     GQA broadcast-KV, one KV slice shared by every batch/head group. Proves
//     the batch term is DROPPED — `in1[kk*n + col]`, no `b*k*n` prefix.
//   - bmm_gqa_t — both at once (broadcast-batch AND transposed rhs), the
//     combined GQA+Kᵀ cell: `in1[col*k + kk]` (no batch term, transposed form).
// The plain canonical batched path is already RTX-4070-proven by
// contract_bias_batched_validate.cu (B14); this harness proves the TRANSPOSED
// and BROADCAST address bindings specifically.
//
// Method: diff the generated kernels vs a host f64 reference (same two-regime
// protocol as contract_bias_batched_validate.cu).
//   - Exactly-representable integer inputs (small K, |partial sums| < 2^24) ⇒
//     the kernel's f32 accumulation is BIT-EXACT to (float)(f64 reference);
//     asserted with `==` (max abs diff 0). This is the load/stride correctness
//     proof: a mis-transposed or mis-broadcast rhs read changes the exact value.
//   - Hashed pseudo-random inputs, large K ⇒ f32 rounding genuinely diverges
//     from f64; asserted within a small relative tolerance (< 1e-4).
//
// NEGATIVE CONTROLS (prove the harness bites — see the two hand-written
// `_neg_*` kernels below, NOT generated, deliberately WRONG bindings):
//   - bmm_transposed_neg_canonical reads the SAME transposed-stored rhs with
//     the CANONICAL (non-transposed) binding — must diverge from the correct
//     reference.
//   - bmm_gqa_neg_realstride reads rhs with a REAL per-batch stride (as if the
//     broadcast axis were mishandled and kept a live per-batch term) over a
//     buffer where batches 1..B-1 hold DISTINCT decoy data a correct broadcast
//     kernel must never read — must diverge for those batches, while the real
//     generated `bmm_gqa` kernel (which never reads the decoy data at all)
//     still passes on the identical buffer.
//
// Compile (from a VS dev shell so nvcc finds cl.exe; no cuBLAS needed):
//   nvcc -O3 -arch=sm_89 contract_layout_validate.cu -o contract_layout_validate
// Run:  ./contract_layout_validate           (correctness, all shapes)
//       ./contract_layout_validate san       (small shapes only, no tolerance regime)
//       compute-sanitizer --tool memcheck  ./contract_layout_validate san
//       compute-sanitizer --tool initcheck ./contract_layout_validate san
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "baracuda_gen_bmm_transposed_f32_contract_tll.cu"
#include "baracuda_gen_bmm_gqa_f32_contract_tll.cu"
#include "baracuda_gen_bmm_gqa_t_f32_contract_tll.cu"

static int g_fails = 0;

// Deterministic hash → f32 in [-1, 1). Products/sums of these are generally NOT
// exactly representable, so f32 accumulation genuinely diverges from f64 (a real
// non-zero-ULP tolerance case), unlike the small-dyadic integer inputs.
static inline float frand(long long i) {
    uint32_t x = (uint32_t)((uint64_t)i * 2654435761u + 1013904223u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return (float)((double)x / 4294967296.0 * 2.0 - 1.0);
}

// Monotonic total ordering of IEEE-754 f32 bit patterns → ULP distance.
static inline long long total_order(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (long long)u;
}
static inline long long ulp_diff(float a, float b) {
    if (a == b) return 0;
    long long d = total_order(a) - total_order(b);
    return d < 0 ? -d : d;
}

// ---------------------------------------------------------------------------
// Host f64 references. All three share the same lhs [B,M,K] canonical layout;
// they differ only in how the rhs storage maps to the logical (b,kk,ni) triple.
// ---------------------------------------------------------------------------

// out[b,mi,ni] = Σ_k lhs[b,mi,kk] · rhs_t[b,ni,kk] — rhs physically stored
// [B,N,K] (K inner per slice): the transposed-rhs binding under the hood.
static std::vector<float> ref_bmm_transposed(const std::vector<float>& lhs,
                                             const std::vector<float>& rhs_t,
                                             long long B, long long M, long long N, long long K) {
    std::vector<float> out(B * M * N);
    for (long long b = 0; b < B; ++b)
        for (long long mi = 0; mi < M; ++mi)
            for (long long ni = 0; ni < N; ++ni) {
                double acc = 0.0;
                for (long long kk = 0; kk < K; ++kk)
                    acc += (double)lhs[b * M * K + mi * K + kk] *
                           (double)rhs_t[b * N * K + ni * K + kk];
                out[b * M * N + mi * N + ni] = (float)acc;
            }
    return out;
}

// out[b,mi,ni] = Σ_k lhs[b,mi,kk] · rhs_slice[kk,ni] — the SAME [K,N] slice
// for every batch (GQA broadcast-KV: the KV projection is shared across the
// batch/head group).
static std::vector<float> ref_bmm_gqa(const std::vector<float>& lhs,
                                      const std::vector<float>& rhs_slice,
                                      long long B, long long M, long long N, long long K) {
    std::vector<float> out(B * M * N);
    for (long long b = 0; b < B; ++b)
        for (long long mi = 0; mi < M; ++mi)
            for (long long ni = 0; ni < N; ++ni) {
                double acc = 0.0;
                for (long long kk = 0; kk < K; ++kk)
                    acc += (double)lhs[b * M * K + mi * K + kk] *
                           (double)rhs_slice[kk * N + ni];
                out[b * M * N + mi * N + ni] = (float)acc;
            }
    return out;
}

// out[b,mi,ni] = Σ_k lhs[b,mi,kk] · rhs_slice_t[ni,kk] — the shared slice,
// stored TRANSPOSED [N,K] (K inner): the combined GQA + Kᵀ cell.
static std::vector<float> ref_bmm_gqa_t(const std::vector<float>& lhs,
                                        const std::vector<float>& rhs_slice_t,
                                        long long B, long long M, long long N, long long K) {
    std::vector<float> out(B * M * N);
    for (long long b = 0; b < B; ++b)
        for (long long mi = 0; mi < M; ++mi)
            for (long long ni = 0; ni < N; ++ni) {
                double acc = 0.0;
                for (long long kk = 0; kk < K; ++kk)
                    acc += (double)lhs[b * M * K + mi * K + kk] *
                           (double)rhs_slice_t[ni * K + kk];
                out[b * M * N + mi * N + ni] = (float)acc;
            }
    return out;
}

// Compare device output vs host f64 reference. bitexact ⇒ require `==` (0 ULP).
static void check(const char* label, const std::vector<float>& got,
                  const std::vector<float>& ref, bool bitexact, double tol) {
    double max_abs = 0, max_rel = 0;
    long long max_ulp = 0;
    long long n = (long long)got.size();
    for (long long i = 0; i < n; ++i) {
        double a = got[i], r = ref[i];
        double d = fabs(a - r);
        double denom = fabs(r) > 1.0 ? fabs(r) : 1.0;
        if (d > max_abs) max_abs = d;
        if (d / denom > max_rel) max_rel = d / denom;
        long long u = ulp_diff(got[i], ref[i]);
        if (u > max_ulp) max_ulp = u;
    }
    bool ok = bitexact ? (max_abs == 0.0) : (max_rel < tol);
    printf("%s %-38s  maxabs %.3e | maxrel %.3e | maxULP %lld  (%lld vals)\n",
           ok ? "PASS" : "FAIL", label, max_abs, max_rel, max_ulp, n);
    if (!ok) g_fails++;
}

// Negative-control comparator: the OPPOSITE assertion of `check` — a
// hand-mutated wrong-binding kernel MUST diverge from the correct reference,
// so `bites` requires max_abs >= min_abs (not < tol). A negative control that
// does NOT diverge means the diff above would not have caught this bug class
// — that failure mode counts toward g_fails, same as a real mismatch would.
static void check_neg(const char* label, const std::vector<float>& got,
                      const std::vector<float>& ref, double min_abs) {
    double max_abs = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > max_abs) max_abs = d;
    }
    bool bites = max_abs >= min_abs;
    printf("%s %-38s  maxabs %.3e  (expected to diverge; harness %s)\n",
           bites ? "PASS" : "FAIL", label, max_abs, bites ? "bites" : "BLIND");
    if (!bites) g_fails++;
}

// ---------------------------------------------------------------------------
// NEGATIVE CONTROL kernels (hand-written, NOT generated). Structurally
// identical to the generated skinny-SIMT schedule (`accs[8]`, grid-stride over
// N, `blockIdx.z` batch), but each intentionally uses the WRONG rhs binding —
// proving `check_neg` above genuinely detects a mis-bound address, not a
// trivially-passing comparison.
// ---------------------------------------------------------------------------

// bmm_transposed's WRONG sibling: reads the transposed-STORED rhs with the
// CANONICAL (non-transposed) binding `b*k*n + kk*n + col` instead of the
// correct `b*k*n + col*k + kk` — as if the transposed storage were misread as
// canonical [K,N] row-major.
extern "C" __global__ void bmm_transposed_neg_canonical(
    const float* __restrict__ in0, const float* __restrict__ in1,
    float* __restrict__ out, long long m, long long n, long long k) {
    long long b = (long long)blockIdx.z;
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; col < n; col += step) {
        float accs[8];
        #pragma unroll
        for (int mm = 0; mm < 8; ++mm) accs[mm] = 0.0f;
        for (long long kk = 0; kk < k; ++kk) {
            float w = in1[b * k * n + kk * n + col]; // WRONG: canonical binding
            #pragma unroll
            for (int mm = 0; mm < 8; ++mm)
                if (mm < m) accs[mm] += in0[b * m * k + mm * k + kk] * w;
        }
        #pragma unroll
        for (int mm = 0; mm < 8; ++mm)
            if (mm < m) out[b * m * n + mm * n + col] = accs[mm];
    }
}

// bmm_gqa's WRONG sibling: reads rhs with a REAL per-batch stride
// `b*k*n + kk*n + col` instead of the correct broadcast-dropped `kk*n + col`
// — as if the broadcast axis were mishandled and kept a live per-batch term.
extern "C" __global__ void bmm_gqa_neg_realstride(
    const float* __restrict__ in0, const float* __restrict__ in1,
    float* __restrict__ out, long long m, long long n, long long k) {
    long long b = (long long)blockIdx.z;
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; col < n; col += step) {
        float accs[8];
        #pragma unroll
        for (int mm = 0; mm < 8; ++mm) accs[mm] = 0.0f;
        for (long long kk = 0; kk < k; ++kk) {
            float w = in1[b * k * n + kk * n + col]; // WRONG: real per-batch stride
            #pragma unroll
            for (int mm = 0; mm < 8; ++mm)
                if (mm < m) accs[mm] += in0[b * m * k + mm * k + kk] * w;
        }
        #pragma unroll
        for (int mm = 0; mm < 8; ++mm)
            if (mm < m) out[b * m * n + mm * n + col] = accs[mm];
    }
}

// ----- bmm_transposed: rhs stored [B,N,K] (K inner), distinct data per batch. -----
static void run_transposed(long long B, long long M, long long N, long long K,
                           bool integers, const char* tag) {
    std::vector<float> lhs(B * M * K), rhs_t(B * N * K);
    for (long long b = 0; b < B; ++b) {
        for (long long i = 0; i < M * K; ++i)
            lhs[b * M * K + i] = integers ? (float)(((i + 2 * b) % 7) - 3)
                                          : frand(b * 1000003 + i);
        for (long long i = 0; i < N * K; ++i)
            rhs_t[b * N * K + i] = integers ? (float)(((i + 3 * b) % 5) - 2)
                                            : frand(b * 7000019 + i + 777);
    }
    float *dl, *dr, *dout;
    cudaMalloc(&dl, B * M * K * 4); cudaMalloc(&dr, B * N * K * 4);
    cudaMalloc(&dout, B * M * N * 4);
    cudaMemcpy(dl, lhs.data(), B * M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs_t.data(), B * N * K * 4, cudaMemcpyHostToDevice);

    const int block = 256;
    dim3 g((unsigned)((N + block - 1) / block), 1u, (unsigned)B);
    std::vector<float> got(B * M * N);
    std::vector<float> ref = ref_bmm_transposed(lhs, rhs_t, B, M, N, K);

    baracuda_gen_bmm_transposed_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bmm_transposed      ") + tag).c_str(), got, ref, integers, 1e-4);

    // NEGATIVE CONTROL: same transposed-stored buffer, WRONG (canonical) bind.
    bmm_transposed_neg_canonical<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check_neg((std::string("NEGCTRL canonical-bind ") + tag).c_str(), got, ref, 1e-2);

    cudaFree(dl); cudaFree(dr); cudaFree(dout);
}

// ----- bmm_gqa: rhs [B,K,N] full buffer; batch 0 = real shared slice, -----
// ----- batches 1..B-1 = DECOY distinct data a correct broadcast kernel  -----
// ----- must never read.                                                 -----
static void run_gqa(long long B, long long M, long long N, long long K,
                    bool integers, const char* tag) {
    std::vector<float> lhs(B * M * K);
    std::vector<float> rhs_full(B * K * N);
    for (long long b = 0; b < B; ++b)
        for (long long i = 0; i < M * K; ++i)
            lhs[b * M * K + i] = integers ? (float)(((i + 2 * b) % 7) - 3)
                                          : frand(b * 1000003 + i);
    // Batch 0 = the REAL shared KV slice (the reference reads this ONLY).
    for (long long i = 0; i < K * N; ++i)
        rhs_full[i] = integers ? (float)((i % 5) - 2) : frand(i + 777);
    // Batches 1..B-1 = DECOY distinct data; a correct broadcast kernel drops
    // the batch term entirely and never touches this region.
    for (long long b = 1; b < B; ++b)
        for (long long i = 0; i < K * N; ++i)
            rhs_full[b * K * N + i] = integers ? (float)(((i + 11 * b) % 5) - 2)
                                                : frand(b * 9001007 + i + 321);

    float *dl, *dr, *dout;
    cudaMalloc(&dl, B * M * K * 4); cudaMalloc(&dr, B * K * N * 4);
    cudaMalloc(&dout, B * M * N * 4);
    cudaMemcpy(dl, lhs.data(), B * M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs_full.data(), B * K * N * 4, cudaMemcpyHostToDevice);

    const int block = 256;
    dim3 g((unsigned)((N + block - 1) / block), 1u, (unsigned)B);
    std::vector<float> got(B * M * N);
    std::vector<float> rhs_slice(rhs_full.begin(), rhs_full.begin() + K * N);
    std::vector<float> ref = ref_bmm_gqa(lhs, rhs_slice, B, M, N, K);

    // The correct kernel drops the batch term entirely, so its output must
    // match `ref` (the single-slice broadcast reference) EVEN THOUGH the
    // buffer holds distinct decoy data for b>=1 — proof it never reads it.
    baracuda_gen_bmm_gqa_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bmm_gqa       ") + tag).c_str(), got, ref, integers, 1e-4);

    // NEGATIVE CONTROL: same buffer, WRONG (real per-batch stride) bind — must
    // diverge for batches 1..B-1 (reads the decoy data instead of batch 0's).
    bmm_gqa_neg_realstride<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check_neg((std::string("NEGCTRL real-stride ") + tag).c_str(), got, ref, 1e-2);

    cudaFree(dl); cudaFree(dr); cudaFree(dout);
}

// ----- bmm_gqa_t: rhs = ONE [N,K] slice (shared, transposed), no negative control. -----
static void run_gqa_t(long long B, long long M, long long N, long long K,
                      bool integers, const char* tag) {
    std::vector<float> lhs(B * M * K), rhs_slice_t(N * K);
    for (long long b = 0; b < B; ++b)
        for (long long i = 0; i < M * K; ++i)
            lhs[b * M * K + i] = integers ? (float)(((i + 2 * b) % 7) - 3)
                                          : frand(b * 1000003 + i);
    for (long long i = 0; i < N * K; ++i)
        rhs_slice_t[i] = integers ? (float)((i % 5) - 2) : frand(i + 777);

    float *dl, *dr, *dout;
    cudaMalloc(&dl, B * M * K * 4); cudaMalloc(&dr, N * K * 4);
    cudaMalloc(&dout, B * M * N * 4);
    cudaMemcpy(dl, lhs.data(), B * M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs_slice_t.data(), N * K * 4, cudaMemcpyHostToDevice);

    const int block = 256;
    dim3 g((unsigned)((N + block - 1) / block), 1u, (unsigned)B);
    std::vector<float> got(B * M * N);
    std::vector<float> ref = ref_bmm_gqa_t(lhs, rhs_slice_t, B, M, N, K);

    baracuda_gen_bmm_gqa_t_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bmm_gqa_t     ") + tag).c_str(), got, ref, integers, 1e-4);

    cudaFree(dl); cudaFree(dr); cudaFree(dout);
}

int main(int argc, char** argv) {
    bool san = argc > 1 && strcmp(argv[1], "san") == 0;

    printf("== bmm_transposed: batched matmul, transposed rhs (Q.K^T core) ==\n");
    run_transposed(3, 2, 3, 4, true, "bitexact [3;2x3x4]");
    run_transposed(4, 8, 33, 16, true, "bitexact [4;8x33x16]"); // M at Tiny ceiling
    run_transposed(8, 1, 5, 8, true, "bitexact [8;1x5x8]");     // many batches
    if (!san) {
        run_transposed(8, 8, 1024, 1024, false, "tol [8;8x1024x1024]");
        run_transposed(16, 4, 512, 512, false, "tol [16;4x512x512]");
    }

    printf("== bmm_gqa: batched matmul, broadcast-batch rhs (GQA KV) ==\n");
    run_gqa(3, 2, 3, 4, true, "bitexact [3;2x3x4]");
    run_gqa(4, 8, 33, 16, true, "bitexact [4;8x33x16]");
    run_gqa(8, 1, 5, 8, true, "bitexact [8;1x5x8]");
    if (!san) {
        run_gqa(8, 8, 1024, 1024, false, "tol [8;8x1024x1024]");
        run_gqa(16, 4, 512, 512, false, "tol [16;4x512x512]");
    }

    printf("== bmm_gqa_t: broadcast-batch AND transposed rhs (combined GQA+K^T) ==\n");
    run_gqa_t(3, 2, 3, 4, true, "bitexact [3;2x3x4]");
    run_gqa_t(4, 8, 33, 16, true, "bitexact [4;8x33x16]");
    if (!san) {
        run_gqa_t(8, 8, 1024, 1024, false, "tol [8;8x1024x1024]");
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(g_fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", g_fails);
    return g_fails ? 1 : 0;
}
