// On-device numeric proof for the B13/B14 contraction growth (this session's
// NEW contraction code, previously only CPU-oracle-validated):
//   B13 — fused matmul + per-column [N] bias/activation epilogue (the `in2[col]`
//         bias load in emit_contraction).
//   B14 — batched [B,M,K]·[B,K,N] (the `b = blockIdx.z` batch offset + the
//         per-operand batch strides `b*m*k` / `b*k*n` / `b*m*n`).
// The plain + fused-relu (no-bias) path is already RTX-4070-proven by
// contract_validate.cu (item 10); this harness proves the BIAS LOAD and the
// BATCH STRIDES specifically.
//
// Method: diff the generated kernels vs a host f64 reference.
//   - Exactly-representable integer inputs (small K, |partial sums| < 2^24) ⇒ the
//     kernel's f32 accumulation is BIT-EXACT to (float)(f64 reference); asserted
//     with `==` (max abs diff 0). This is the load/stride correctness proof: a
//     mis-indexed bias column or a wrong batch slice changes the exact value.
//   - Fractional inputs at large K ⇒ f32 rounding differs from f64; asserted with
//     a small relative tolerance, max abs/rel/ULP reported.
// Batched cases fill every batch with DISTINCT data, so a batch-stride bug (e.g.
// reading batch 0 for all b) is caught, and cross-check vs cublasSgemmStridedBatched.
//
// Compile (from a VS dev shell so nvcc finds cl.exe):
//   nvcc -O3 -arch=sm_89 contract_bias_batched_validate.cu -lcublas \
//        -o contract_bias_batched_validate
// Run:  ./contract_bias_batched_validate           (correctness + cuBLAS + bench)
//       ./contract_bias_batched_validate san       (small shapes only, no cuBLAS)
//       compute-sanitizer --tool memcheck  ./contract_bias_batched_validate san
//       compute-sanitizer --tool initcheck ./contract_bias_batched_validate san
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "baracuda_gen_matmul_bias_f32_contract_tll.cu"
#include "baracuda_gen_matmul_bias_relu_f32_contract_tll.cu"
#include "baracuda_gen_bmm_f32_contract_tll.cu"
#include "baracuda_gen_bmm_relu_f32_contract_tll.cu"

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

// Host f64 reference: out[b,mi,ni] = epi( Σ_k lhs[b,mi,kk]·rhs[b,kk,ni] [+ bias[ni]] ).
// bias==nullptr ⇒ no bias; relu==true ⇒ clamp the (possibly biased) result at 0.
static std::vector<float> ref_contract(const std::vector<float>& lhs,
                                       const std::vector<float>& rhs,
                                       const float* bias, bool relu,
                                       long long B, long long M, long long N, long long K) {
    std::vector<float> out(B * M * N);
    for (long long b = 0; b < B; ++b)
        for (long long mi = 0; mi < M; ++mi)
            for (long long ni = 0; ni < N; ++ni) {
                double acc = 0.0;
                for (long long kk = 0; kk < K; ++kk)
                    acc += (double)lhs[b * M * K + mi * K + kk] *
                           (double)rhs[b * K * N + kk * N + ni];
                if (bias) acc += (double)bias[ni];
                if (relu && acc < 0.0) acc = 0.0;
                out[b * M * N + ni + mi * N] = (float)acc;
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

template <class F>
static float timed(F launch, int iters = 100) {
    for (int i = 0; i < 10; ++i) launch();
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

// ----- B13: fused bias (+/- relu), rank-2. -----
static void run_bias(long long M, long long N, long long K, bool integers,
                     const char* tag) {
    std::vector<float> lhs(M * K), rhs(K * N), bias(N);
    for (long long i = 0; i < M * K; ++i)
        lhs[i] = integers ? (float)((i % 7) - 3) : frand(i);
    for (long long i = 0; i < K * N; ++i)
        rhs[i] = integers ? (float)((i % 5) - 2) : frand(i + 777);
    // Column-dependent bias, some negative — so relu actually clamps and a
    // column-index bug in the bias load changes the result.
    for (long long j = 0; j < N; ++j)
        bias[j] = integers ? (float)((j % 11) - 7) * 4.0f : frand(j + 12345) * 8.0f;

    float *dl, *dr, *db, *dout;
    cudaMalloc(&dl, M * K * 4); cudaMalloc(&dr, K * N * 4);
    cudaMalloc(&db, N * 4);     cudaMalloc(&dout, M * N * 4);
    cudaMemcpy(dl, lhs.data(), M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs.data(), K * N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), N * 4, cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid = (int)((N + block - 1) / block);
    std::vector<float> got(M * N);

    // matmul_bias (no activation): isolates the bias LOAD.
    baracuda_gen_matmul_bias_f32_contract_tll<<<grid, block>>>(dl, dr, db, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bias      ") + tag).c_str(), got,
          ref_contract(lhs, rhs, bias.data(), false, 1, M, N, K), integers, 1e-4);

    // matmul_bias_relu: fused bias + activation.
    baracuda_gen_matmul_bias_relu_f32_contract_tll<<<grid, block>>>(dl, dr, db, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bias_relu ") + tag).c_str(), got,
          ref_contract(lhs, rhs, bias.data(), true, 1, M, N, K), integers, 1e-4);

    cudaFree(dl); cudaFree(dr); cudaFree(db); cudaFree(dout);
}

// ----- B14: batched [B,M,K]·[B,K,N], distinct data per batch. -----
static void run_batched(long long B, long long M, long long N, long long K,
                        bool integers, bool do_cublas, const char* tag) {
    std::vector<float> lhs(B * M * K), rhs(B * K * N);
    for (long long b = 0; b < B; ++b) {
        for (long long i = 0; i < M * K; ++i)
            lhs[b * M * K + i] = integers ? (float)(((i + 2 * b) % 7) - 3)
                                          : frand(b * 1000003 + i);
        for (long long i = 0; i < K * N; ++i)
            rhs[b * K * N + i] = integers ? (float)(((i + 3 * b) % 5) - 2)
                                          : frand(b * 7000019 + i + 777);
    }
    float *dl, *dr, *dout;
    cudaMalloc(&dl, B * M * K * 4); cudaMalloc(&dr, B * K * N * 4);
    cudaMalloc(&dout, B * M * N * 4);
    cudaMemcpy(dl, lhs.data(), B * M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs.data(), B * K * N * 4, cudaMemcpyHostToDevice);

    const int block = 256;
    dim3 g((unsigned)((N + block - 1) / block), 1u, (unsigned)B);
    std::vector<float> got(B * M * N);

    // bmm (identity epilogue): isolates the batch STRIDES.
    baracuda_gen_bmm_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bmm       ") + tag).c_str(), got,
          ref_contract(lhs, rhs, nullptr, false, B, M, N, K), integers, 1e-4);

    // bmm_relu: batch strides + relu epilogue.
    baracuda_gen_bmm_relu_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
    check((std::string("bmm_relu  ") + tag).c_str(), got,
          ref_contract(lhs, rhs, nullptr, true, B, M, N, K), integers, 1e-4);

    // Cross-check bmm vs cublasSgemmStridedBatched (row-major C=A·B per batch via
    // column-major C^T = rhs^T·lhs^T, strided over the batch dimension).
    if (do_cublas) {
        cublasHandle_t h; cublasCreate(&h);
        float *dc; cudaMalloc(&dc, B * M * N * 4);
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemmStridedBatched(
            h, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
            &alpha, dr, (int)N, K * N, dl, (int)K, M * K,
            &beta, dc, (int)N, M * N, (int)B);
        cudaDeviceSynchronize();
        std::vector<float> c(B * M * N);
        cudaMemcpy(c.data(), dc, B * M * N * 4, cudaMemcpyDeviceToHost);
        // Recompute bmm into `got` (dout was overwritten by bmm_relu above).
        baracuda_gen_bmm_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), dout, B * M * N * 4, cudaMemcpyDeviceToHost);
        check((std::string("bmm vs cuBLAS ") + tag).c_str(), got, c, false, 1e-4);
        cudaFree(dc); cublasDestroy(h);
    }
    cudaFree(dl); cudaFree(dr); cudaFree(dout);
}

int main(int argc, char** argv) {
    bool san = argc > 1 && strcmp(argv[1], "san") == 0;

    printf("== B13 fused bias / activation epilogue ==\n");
    // Bit-exact (integer inputs, small K): the load/index correctness proof.
    run_bias(3, 5, 4, true, "bitexact [3x5x4]");
    run_bias(8, 37, 16, true, "bitexact [8x37x16]");   // M at the Tiny ceiling
    run_bias(1, 4096, 64, true, "bitexact [1x4096x64]"); // grid-stride many cols
    if (!san) {
        // Tolerance (fractional inputs, large K): realistic decode/flat-GEMM shape.
        run_bias(8, 4096, 4096, false, "tol [8x4096x4096]");
        run_bias(1, 4096, 4096, false, "tol [1x4096x4096]");
    }

    printf("== B14 batched [B,M,K].[B,K,N] ==\n");
    run_batched(3, 2, 3, 4, true, false, "bitexact [3;2x3x4]");
    run_batched(4, 8, 33, 16, true, false, "bitexact [4;8x33x16]"); // M at ceiling
    run_batched(8, 1, 5, 8, true, false, "bitexact [8;1x5x8]");     // many batches
    if (!san) {
        run_batched(8, 8, 1024, 1024, false, true, "tol [8;8x1024x1024]");
        run_batched(16, 4, 512, 512, false, true, "tol [16;4x512x512]");
    }

    // Bench (skipped under sanitizer): batched vs cublasSgemmStridedBatched.
    if (!san) {
        printf("== bench: bmm vs cublasSgemmStridedBatched ==\n");
        const long long B = 8, M = 8, N = 4096, K = 4096;
        std::vector<float> lhs(B * M * K), rhs(B * K * N);
        for (long long i = 0; i < B * M * K; ++i) lhs[i] = ((i % 23) - 11) * 0.0625f;
        for (long long i = 0; i < B * K * N; ++i) rhs[i] = ((i % 19) - 9) * 0.03125f;
        float *dl, *dr, *dout, *dc;
        cudaMalloc(&dl, B * M * K * 4); cudaMalloc(&dr, B * K * N * 4);
        cudaMalloc(&dout, B * M * N * 4); cudaMalloc(&dc, B * M * N * 4);
        cudaMemcpy(dl, lhs.data(), B * M * K * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dr, rhs.data(), B * K * N * 4, cudaMemcpyHostToDevice);
        const int block = 256;
        dim3 g((unsigned)((N + block - 1) / block), 1u, (unsigned)B);
        cublasHandle_t h; cublasCreate(&h);
        const float alpha = 1.0f, beta = 0.0f;
        auto gen = [&] { baracuda_gen_bmm_f32_contract_tll<<<g, block>>>(dl, dr, dout, M, N, K); };
        auto blas = [&] {
            cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                                      &alpha, dr, (int)N, K * N, dl, (int)K, M * K,
                                      &beta, dc, (int)N, M * N, (int)B);
        };
        float t_gen = timed(gen), t_blas = timed(blas);
        double gb = (double)(B * (M * K + K * N + M * N)) * 4.0 / 1e9;
        printf("  bmm %8.3f ms (%6.1f GB/s) | cuBLAS strided %8.3f ms (%6.1f GB/s) | gen vs cuBLAS %.2fx\n",
               t_gen, gb / (t_gen / 1000), t_blas, gb / (t_blas / 1000), t_blas / t_gen);
        cublasDestroy(h);
        cudaFree(dl); cudaFree(dr); cudaFree(dout); cudaFree(dc);
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(g_fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", g_fails);
    return g_fails ? 1 : 0;
}
