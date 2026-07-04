// Item 01 (layout/shape views) on-device validation + generated-vs-bespoke bench.
//
// Proof vehicle: a FUSED transpose-elementwise kernel reads an input THROUGH a
// Permute view in ONE pass (the §1 memory-optimal win — no materialized
// contiguize/transpose copy):
//
//   relu_t : out[i,j] = relu(x[j,i])                 (input 0 via Permute{[1,0]})
//   addb_t : out[i,j] = x[j,i] + b[j]                (in0 transposed; in1 a
//                                                      per-column bias broadcast
//                                                      over the row axis)
//
// x is the PRODUCER buffer, physically [N,M] row-major contiguous (strides
// [M,1]); the output is [M,N]. The generated strided kernel folds the transpose
// into address math: iteration axis d reads producer stride perm[d], so
//   o0 = c0*s0_1 + c1*s0_0   (SWAPPED strides — the item-01 remap).
//
// Two checks per shape:
//   1. BIT-EXACT vs a CPU double reference AND vs the bespoke materialize-then-op
//      path = baracuda::contiguize(x^T) [baracuda_contiguize.cuh] THEN relu — two
//      kernels + a materialized transpose buffer + an extra DRAM round-trip.
//   2. TIMED: the generated ONE fused pass vs the bespoke TWO-kernel path.
//      Expected: the generator WINS (no materialized copy, half the DRAM traffic).
//
// Generated kernels are produced by the regeneration snippet in ondevice/README.md
// ("view_validate.cu — layout/shape views (item 01)").
//
// Compile (bespoke contiguize header wants the MSVC conforming preprocessor):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <kernels/include> view_validate.cu -o view_validate
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "baracuda_gen_relu_t_f32_strided_r2.cu"
#include "baracuda_gen_addb_t_f32_strided_r2.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_contiguize.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int g_fail = 0;

// Bespoke "second step": a plain contiguous relu over the materialized x^T.
__global__ void relu_contig(const float* __restrict__ in, float* __restrict__ out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) out[i] = in[i] < 0.0f ? 0.0f : in[i];
}

static int grid_for(long long n) {
    long long b = (n + 255) / 256;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

// out[i,j] = relu(x[j,i]).  x is producer [N,M] contiguous; output [M,N].
static void run_relu_t(long long M, long long N, const char* tag) {
    long long n = M * N;
    size_t bytes = (size_t)n * 4;
    float* hx  = (float*)malloc(bytes);   // x stored [N][M] row-major
    float* hg  = (float*)malloc(bytes);   // generated out [M][N]
    float* hb  = (float*)malloc(bytes);   // bespoke   out [M][N]
    srand(4321 + (int)(M * 31 + N));
    for (long long t = 0; t < n; ++t) hx[t] = (float)((rand() % 4001) - 2000) * 0.01f;

    float *dx, *dg, *db, *dxt;
    CHECK(cudaMalloc((void**)&dx,  bytes));
    CHECK(cudaMalloc((void**)&dg,  bytes));
    CHECK(cudaMalloc((void**)&db,  bytes));
    CHECK(cudaMalloc((void**)&dxt, bytes));   // materialized x^T (bespoke only)
    CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
    int blocks = grid_for(n);

    // ---- generated: ONE fused strided pass ----
    // ABI: (in0, out, shape0=M, shape1=N, s0_0=M, s0_1=1, so_0=N, so_1=1, n)
    baracuda_gen_relu_t_f32_strided_r2<<<blocks, 256>>>(dx, dg, M, N, M, 1, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, bytes, cudaMemcpyDeviceToHost));

    // ---- bespoke: contiguize(x^T) THEN relu (two kernels + materialized buffer) ----
    // dest [M,N] reads source x with strides [1, M] (dest axis 0 -> x axis 1, etc.)
    int32_t cshape[2] = { (int32_t)M, (int32_t)N };
    int64_t csx[2]    = { 1, M };
    int rc = baracuda::contiguize::launch_contiguize<4>(dx, dxt, cshape, csx, 0, 2, 0);
    if (rc != 0) { printf("contiguize rc=%d\n", rc); g_fail = 1; }
    relu_contig<<<blocks, 256>>>(dxt, db, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hb, db, bytes, cudaMemcpyDeviceToHost));

    // ---- correctness: gen == bespoke == CPU double reference, BIT-EXACT ----
    long long bit_bad = 0, def_bad = 0;
    for (long long i = 0; i < M; ++i)
        for (long long j = 0; j < N; ++j) {
            float g = hg[i*N+j], b = hb[i*N+j];
            double xr = (double)hx[j*M+i];
            float want = (float)(xr < 0.0 ? 0.0 : xr);
            if (memcmp(&g, &b, 4) != 0) bit_bad++;
            if (memcmp(&g, &want, 4) != 0) def_bad++;
        }

    // ---- timing: averaged over reps, after a warmup ----
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    const int reps = 50;
    // generated
    baracuda_gen_relu_t_f32_strided_r2<<<blocks, 256>>>(dx, dg, M, N, M, 1, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r)
        baracuda_gen_relu_t_f32_strided_r2<<<blocks, 256>>>(dx, dg, M, N, M, 1, N, 1, n);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float t_gen = 0; CHECK(cudaEventElapsedTime(&t_gen, e0, e1)); t_gen /= reps;
    // bespoke (contiguize + relu)
    baracuda::contiguize::launch_contiguize<4>(dx, dxt, cshape, csx, 0, 2, 0);
    relu_contig<<<blocks, 256>>>(dxt, db, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) {
        baracuda::contiguize::launch_contiguize<4>(dx, dxt, cshape, csx, 0, 2, 0);
        relu_contig<<<blocks, 256>>>(dxt, db, n);
    }
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float t_bes = 0; CHECK(cudaEventElapsedTime(&t_bes, e0, e1)); t_bes /= reps;

    int ok = (bit_bad == 0) && (def_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] relu_t %-9s [%5lldx%-5lld] gen==bespoke %s  gen==ref %s | "
           "gen %.3f ms  bespoke(contig+relu) %.3f ms  speedup %.2fx\n",
           ok ? " ok " : "FAIL", tag, M, N,
           bit_bad ? "NO" : "yes", def_bad ? "NO" : "yes",
           t_gen, t_bes, t_bes / t_gen);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(dxt);
    free(hx); free(hg); free(hb);
}

// out[i,j] = x[j,i] + b[j].  x producer [N,M]; b a per-column [N] bias (index j).
static void run_addb_t(long long M, long long N, const char* tag) {
    long long n = M * N;
    size_t bytes = (size_t)n * 4;
    float* hx = (float*)malloc(bytes);
    float* hbias = (float*)malloc((size_t)N * 4);
    float* hg = (float*)malloc(bytes);
    srand(99 + (int)(M + N));
    for (long long t = 0; t < n; ++t) hx[t] = (float)((rand() % 4001) - 2000) * 0.01f;
    for (long long j = 0; j < N; ++j) hbias[j] = (float)((rand() % 2001) - 1000) * 0.01f;

    float *dx, *dbias, *dg;
    CHECK(cudaMalloc((void**)&dx, bytes));
    CHECK(cudaMalloc((void**)&dbias, (size_t)N * 4));
    CHECK(cudaMalloc((void**)&dg, bytes));
    CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dbias, hbias, (size_t)N * 4, cudaMemcpyHostToDevice));
    int blocks = grid_for(n);

    // ABI: (in0, in1, out, shape0=M, shape1=N, s0_0=M, s0_1=1, s1_0=0, s1_1=1,
    //       so_0=N, so_1=1, n).  in1 is broadcast over axis 0 (stride 0), varies
    //       along axis 1 with stride 1 -> b[j].
    baracuda_gen_addb_t_f32_strided_r2<<<blocks, 256>>>(dx, dbias, dg, M, N, M, 1, 0, 1, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, bytes, cudaMemcpyDeviceToHost));

    long long def_bad = 0;
    for (long long i = 0; i < M; ++i)
        for (long long j = 0; j < N; ++j) {
            float want = (float)((double)hx[j*M+i] + (double)hbias[j]);
            if (memcmp(&hg[i*N+j], &want, 4) != 0) def_bad++;
        }
    int ok = (def_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] addb_t %-9s [%5lldx%-5lld] gen==ref %s (transpose + per-column bias)\n",
           ok ? " ok " : "FAIL", tag, M, N, def_bad ? "NO" : "yes");

    cudaFree(dx); cudaFree(dbias); cudaFree(dg);
    free(hx); free(hbias); free(hg);
}

int main() {
    printf("== item 01 layout-view validation (RTX 4070 sm_89) ==\n");
    // square, non-square (both orientations), degenerate, and a large cell.
    run_relu_t(512, 512, "square");
    run_relu_t(384, 1024, "wide");
    run_relu_t(1024, 384, "tall");
    run_relu_t(1, 4096, "row");
    run_relu_t(4096, 4096, "large");
    run_addb_t(512, 512, "square");
    run_addb_t(384, 1024, "wide");
    run_addb_t(4096, 4096, "large");
    printf(g_fail ? "\n== FAILURES ==\n" : "\n== ALL PASSED ==\n");
    return g_fail;
}
