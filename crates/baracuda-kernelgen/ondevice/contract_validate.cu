// Go/no-go for the item-10 skinny contraction cell: numeric validation vs a
// CPU f64 oracle AND cuBLAS, then the long-tail bench — generated skinny SIMT
// vs cublasSgemm at M in {1, 8} (K = N = 4096, f32, row-major).
// Compile: nvcc -O3 -arch=sm_89 contract_validate.cu -lcublas
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "baracuda_gen_matmul_f32_contract_tll.cu"

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

int main() {
    const long long K = 4096, N = 4096;
    cublasHandle_t h;
    cublasCreate(&h);
    int fails = 0;

    for (long long M : {1LL, 8LL}) {
        std::vector<float> hl(M * K), hr(K * N);
        for (long long i = 0; i < M * K; ++i) hl[i] = ((i % 23) - 11) * 0.0625f;
        for (long long i = 0; i < K * N; ++i) hr[i] = ((i % 19) - 9) * 0.03125f;

        float *dl, *dr, *dg, *dc;
        cudaMalloc((void**)&dl, M * K * 4);
        cudaMalloc((void**)&dr, K * N * 4);
        cudaMalloc((void**)&dg, M * N * 4);
        cudaMalloc((void**)&dc, M * N * 4);
        cudaMemcpy(dl, hl.data(), M * K * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dr, hr.data(), K * N * 4, cudaMemcpyHostToDevice);

        const int block = 256;
        const int grid = (int)((N + block - 1) / block);
        auto gen = [&] {
            baracuda_gen_matmul_f32_contract_tll<<<grid, block>>>(dl, dr, dg, M, N, K);
        };
        // Row-major C = A·B via column-major cuBLAS: C^T = B^T · A^T.
        const float alpha = 1.0f, beta = 0.0f;
        auto blas = [&] {
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)M, (int)K,
                        &alpha, dr, (int)N, dl, (int)K, &beta, dc, (int)N);
        };

        // ---- correctness: generated vs f64 oracle (sample) and vs cuBLAS ----
        gen(); blas();
        cudaDeviceSynchronize();
        std::vector<float> g(M * N), c(M * N);
        cudaMemcpy(g.data(), dg, M * N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), dc, M * N * 4, cudaMemcpyDeviceToHost);
        double maxrel_o = 0, maxrel_b = 0;
        for (long long m = 0; m < M; ++m) {
            for (long long n = 0; n < N; n += 97) { // sampled oracle columns
                double acc = 0;
                for (long long kk = 0; kk < K; ++kk)
                    acc += (double)hl[m * K + kk] * (double)hr[kk * N + n];
                double denom = fabs(acc) > 1.0 ? fabs(acc) : 1.0;
                maxrel_o = fmax(maxrel_o, fabs((double)g[m * N + n] - acc) / denom);
            }
        }
        for (long long i = 0; i < M * N; ++i) {
            double denom = fabs((double)c[i]) > 1.0 ? fabs((double)c[i]) : 1.0;
            maxrel_b = fmax(maxrel_b, fabs((double)g[i] - (double)c[i]) / denom);
        }
        bool ok = maxrel_o < 1e-4 && maxrel_b < 1e-4;
        printf(ok ? "PASS M=%lld  vs f64 oracle %.2e | vs cuBLAS %.2e\n"
                  : "FAIL M=%lld  vs f64 oracle %.2e | vs cuBLAS %.2e\n",
               M, maxrel_o, maxrel_b);
        if (!ok) fails++;

        // ---- the long-tail bench ----
        float t_gen = timed(gen);
        float t_blas = timed(blas);
        const double gb = (M * K + K * N + M * N) * 4.0 / 1e9; // streamed bytes
        printf("  M=%lld: generated %8.3f ms (%6.1f GB/s) | cuBLAS %8.3f ms (%6.1f GB/s) | speedup %.2fx\n",
               M, t_gen, gb / (t_gen / 1000), t_blas, gb / (t_blas / 1000), t_blas / t_gen);

        cudaFree(dl); cudaFree(dr); cudaFree(dg); cudaFree(dc);
    }
    cublasDestroy(h);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
