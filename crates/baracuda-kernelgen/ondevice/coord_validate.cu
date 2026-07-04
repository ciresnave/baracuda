// Increment 0d (Coord leaf) on-device validation.
//
// Proof vehicle: the generated coord-triu mask
//   out[i,j] = in[i,j] * (j >= i + diagonal ? 1 : 0)
// expressed as  input(0) * (coord(1) >= coord(0) + konst(diagonal))
// must be BIT-EXACT to the bespoke triu kernel (baracuda_triu_tril.cuh:
//   output[i,j] = j >= i + diagonal ? input[i,j] : 0),
// across square/non-square/degenerate/large-axis shapes and f32/f64.
// Plus: iota (out = coord(1)) and alibi ((coord(1)-coord(0))*slope) vs a
// CPU double reference.
//
// Generated kernels are produced by the regeneration snippet in
// ondevice/README.md ("coord ops (increment 0d)").
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "baracuda_gen_triu_mask_f32_strided_r2.cu"
#include "baracuda_gen_triu_mask_km1_f32_strided_r2.cu"
#include "baracuda_gen_triu_mask_k2_f32_strided_r2.cu"
#include "baracuda_gen_triu_mask_f64_strided_r2.cu"
#include "baracuda_gen_iota1_f32_strided_r2.cu"
#include "baracuda_gen_alibi_f32_strided_r2.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_triu_tril.cuh"

// Instantiate the bespoke triu FFI launchers for f32 + f64.
BARACUDA_KERNELS_TRIU_INSTANTIATE(f32, float)
BARACUDA_KERNELS_TRIU_INSTANTIATE(f64, double)

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int g_fail = 0;

// Contiguous rank-2 [M,N]: row-major strides {N,1}.
static void run_triu_case(long long M, long long N, int diagonal) {
    long long n = M * N;
    float* hin = (float*)malloc(n * 4);
    float* hgen = (float*)malloc(n * 4);
    float* hbes = (float*)malloc(n * 4);
    srand(1234 + (int)M * 7 + diagonal);
    for (long long t = 0; t < n; ++t)
        hin[t] = (float)((rand() % 4001) - 2000) * 0.01f;

    float *din, *dgen, *dbes;
    CHECK(cudaMalloc((void**)&din, n * 4));
    CHECK(cudaMalloc((void**)&dgen, n * 4));
    CHECK(cudaMalloc((void**)&dbes, n * 4));
    CHECK(cudaMemcpy(din, hin, n * 4, cudaMemcpyHostToDevice));

    int blocks = (int)((n + 255) / 256); if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    // Generated: pick the kernel matching the baked diagonal.
    CHECK(cudaMemset(dgen, 0, n * 4));
    if (diagonal == 0)
        baracuda_gen_triu_mask_f32_strided_r2<<<blocks,256>>>(din, dgen, M, N, N, 1, N, 1, n);
    else if (diagonal == -1)
        baracuda_gen_triu_mask_km1_f32_strided_r2<<<blocks,256>>>(din, dgen, M, N, N, 1, N, 1, n);
    else if (diagonal == 2)
        baracuda_gen_triu_mask_k2_f32_strided_r2<<<blocks,256>>>(din, dgen, M, N, N, 1, N, 1, n);
    else { printf("no generated kernel for diag %d\n", diagonal); g_fail = 1; return; }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hgen, dgen, n * 4, cudaMemcpyDeviceToHost));

    // Bespoke.
    CHECK(cudaMemset(dbes, 0, n * 4));
    int32_t shape[2] = { (int32_t)M, (int32_t)N };
    int rc = baracuda_kernels_triu_f32_run(din, dbes, shape, 2, diagonal, nullptr);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("bespoke triu rc=%d\n", rc); g_fail = 1; return; }
    CHECK(cudaMemcpy(hbes, dbes, n * 4, cudaMemcpyDeviceToHost));

    // The generated kernel is a mask-MULTIPLY (in * (cond?1:0)); bespoke is a
    // SELECT (cond?in:0). These are VALUE-equal but differ in the sign of zero
    // on masked-out NEGATIVE entries: negative*0.0f = -0.0, select stores +0.0.
    // So: require value-equality (float ==, which treats -0==+0), and account
    // every bit-difference as exactly that signed-zero case (masked negative).
    long long val_bad = 0, bit_diff = 0, signed_zero_diff = 0, def_bad = 0;
    for (long long i = 0; i < M; ++i)
        for (long long j = 0; j < N; ++j) {
            float g = hgen[i*N+j], b = hbes[i*N+j];
            if (g != b) val_bad++;                 // float !=  (NaN aside; none here)
            if (memcmp(&g, &b, 4) != 0) {
                bit_diff++;
                bool masked_neg = (j < i + diagonal) && (hin[i*N+j] < 0.0f);
                if (g == 0.0f && b == 0.0f && masked_neg) signed_zero_diff++;
            }
            float want = (j >= i + diagonal) ? hin[i*N+j] : 0.0f;
            if (g != want) def_bad++;              // value-equal to the definition
        }
    // PASS iff value-equal to bespoke AND to the definition, and every bit
    // difference is accounted for as the signed-zero-of-masked-negative case.
    int ok = (val_bad == 0) && (def_bad == 0) && (bit_diff == signed_zero_diff);
    if (!ok) g_fail = 1;
    printf("[%s] triu f32 [%4lldx%-4lld] diag %+d : val==bespoke %s, val==def %s, "
           "bitdiff %lld (all -0/+0 masked-neg: %s)\n",
           ok ? " ok " : "FAIL", M, N, diagonal,
           val_bad ? "NO" : "yes", def_bad ? "NO" : "yes",
           bit_diff, (bit_diff == signed_zero_diff) ? "yes" : "NO");
    cudaFree(din); cudaFree(dgen); cudaFree(dbes);
    free(hin); free(hgen); free(hbes);
}

static void run_triu_f64_case(long long M, long long N) {
    long long n = M * N;
    double* hin = (double*)malloc(n * 8);
    double* hgen = (double*)malloc(n * 8);
    double* hbes = (double*)malloc(n * 8);
    srand(77 + (int)N);
    for (long long t = 0; t < n; ++t)
        hin[t] = (double)((rand() % 4001) - 2000) * 0.01;
    double *din, *dgen, *dbes;
    CHECK(cudaMalloc((void**)&din, n * 8));
    CHECK(cudaMalloc((void**)&dgen, n * 8));
    CHECK(cudaMalloc((void**)&dbes, n * 8));
    CHECK(cudaMemcpy(din, hin, n * 8, cudaMemcpyHostToDevice));
    int blocks = (int)((n + 255) / 256); if (blocks < 1) blocks = 1;
    CHECK(cudaMemset(dgen, 0, n * 8));
    baracuda_gen_triu_mask_f64_strided_r2<<<blocks,256>>>(din, dgen, M, N, N, 1, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hgen, dgen, n * 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemset(dbes, 0, n * 8));
    int32_t shape[2] = { (int32_t)M, (int32_t)N };
    baracuda_kernels_triu_f64_run(din, dbes, shape, 2, 0, nullptr);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hbes, dbes, n * 8, cudaMemcpyDeviceToHost));
    // f64 checks the mathematical DEFINITION independently (not only bespoke),
    // matching the f32 path — so gen==bespoke AND gen==def both hold.
    long long val_bad = 0, bit_diff = 0, sz = 0, def_bad = 0;
    for (long long i = 0; i < M; ++i)
        for (long long j = 0; j < N; ++j) {
            double g = hgen[i*N+j], b = hbes[i*N+j];
            if (g != b) val_bad++;
            if (memcmp(&g, &b, 8) != 0) {
                bit_diff++;
                if (g == 0.0 && b == 0.0 && j < i && hin[i*N+j] < 0.0) sz++;
            }
            double want = (j >= i) ? hin[i*N+j] : 0.0;  // diagonal 0
            if (g != want) def_bad++;
        }
    int ok = (val_bad == 0) && (def_bad == 0) && (bit_diff == sz);
    if (!ok) g_fail = 1;
    printf("[%s] triu f64 [%4lldx%-4lld] diag  +0 : val==bespoke %s, val==def %s, bitdiff %lld (all -0: %s)\n",
           ok ? " ok " : "FAIL", M, N, val_bad ? "NO" : "yes", def_bad ? "NO" : "yes",
           bit_diff, (bit_diff == sz) ? "yes" : "NO");
    cudaFree(din); cudaFree(dgen); cudaFree(dbes);
    free(hin); free(hgen); free(hbes);
}

static void run_iota_case(long long M, long long N) {
    long long n = M * N;
    float* hgen = (float*)malloc(n * 4);
    float* dgen;
    CHECK(cudaMalloc((void**)&dgen, n * 4));
    int blocks = (int)((n + 255) / 256); if (blocks < 1) blocks = 1;
    baracuda_gen_iota1_f32_strided_r2<<<blocks,256>>>(dgen, M, N, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hgen, dgen, n * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (long long i = 0; i < M && !bad; ++i)
        for (long long j = 0; j < N; ++j) {
            float want = (float)j;  // coord(1) = column index
            if (memcmp(&hgen[i*N+j], &want, 4) != 0) { bad = 1; break; }
        }
    if (bad) g_fail = 1;
    printf("[%s] iota coord(1) f32 [%4lldx%-4lld] : gen==def %s\n",
           bad ? "FAIL" : " ok ", M, N, bad ? "NO" : "yes");
    cudaFree(dgen); free(hgen);
}

static void run_alibi_case(long long M, long long N, float slope) {
    long long n = M * N;
    float* hgen = (float*)malloc(n * 4);
    float* dgen;
    CHECK(cudaMalloc((void**)&dgen, n * 4));
    int blocks = (int)((n + 255) / 256); if (blocks < 1) blocks = 1;
    baracuda_gen_alibi_f32_strided_r2<<<blocks,256>>>(dgen, M, N, N, 1, n, slope);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hgen, dgen, n * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (long long i = 0; i < M && !bad; ++i)
        for (long long j = 0; j < N; ++j) {
            float want = ((float)j - (float)i) * slope;
            if (memcmp(&hgen[i*N+j], &want, 4) != 0) { bad = 1; break; }
        }
    if (bad) g_fail = 1;
    printf("[%s] alibi (c1-c0)*p f32 [%4lldx%-4lld] slope %.3f : gen==def %s\n",
           bad ? "FAIL" : " ok ", M, N, slope, bad ? "NO" : "yes");
    cudaFree(dgen); free(hgen);
}

int main() {
    // triu masks vs bespoke, diag 0/-1/2, across shapes incl. non-square,
    // degenerate, and a coordinate axis > 2^11 (5000 > 2048, still exact in f32).
    for (int diag : {0, -1, 2}) {
        run_triu_case(37, 53, diag);
        run_triu_case(128, 128, diag);
        run_triu_case(1, 1, diag);
        run_triu_case(5000, 33, diag);   // >2^11 row-coord axis
        run_triu_case(33, 5000, diag);   // >2^11 col-coord axis
    }
    run_triu_f64_case(128, 128);
    run_triu_f64_case(37, 53);
    run_triu_f64_case(5000, 33);   // >2^11 row-coord axis, f64 (exact to 2^53)
    run_triu_f64_case(33, 5000);   // >2^11 col-coord axis, f64
    run_iota_case(64, 4096);   // col coord up to 4095 (>2^11), exact in f32
    run_alibi_case(128, 128, 0.125f);
    run_alibi_case(37, 53, -0.5f);
    printf(g_fail ? "\n== FAILURES ==\n" : "\n== ALL PASSED ==\n");
    return g_fail;
}
