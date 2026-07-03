// On-device validation + bench of the split-K outer-axis reduction VARIANT
// (phase 2, first bench-gated variant) against the single-pass baseline.
//
// Checks:
//  1. numeric: baseline AND split-K (ragged chunks) vs a CPU f64 oracle;
//  2. degenerate n_chunks=1: split-K association == the baseline's sequential
//     fold per column -> outputs must be memcmp-IDENTICAL;
//  3. determinism: two identical split-K runs are memcmp-identical;
//  4. bench: [8192,8192] f32 sum over axis 0 — baseline (1 thread/col, the
//     measured 118 GB/s occupancy gap) vs split-K (n_chunks partial folds).
// Compile: nvcc -O3 -arch=sm_89 splitk_validate.cu
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "baracuda_gen_sum_f32_reduce_sum_ax1.cu"          // baseline (general path)
#include "baracuda_gen_sum_f32_reduce_sum_ax1_splitk_partial.cu"
#include "baracuda_gen_sum_f32_reduce_sum_ax1_splitk_combine.cu"

static int fails = 0;

template <class F>
static float timed(F launch, int iters = 50) {
    for (int i = 0; i < 5; ++i) launch();
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

// Launch the split-K pair as one logical op.
static void run_splitk(const float* d_in, float* d_ws, float* d_out,
                       long long rows, long long cols, long long n_chunks) {
    const int block = 256;
    long long chunk_rows = (rows + n_chunks - 1) / n_chunks;
    dim3 g1((unsigned)((cols + block - 1) / block), (unsigned)n_chunks);
    baracuda_gen_sum_f32_reduce_sum_ax1_splitk_partial<<<g1, block>>>(
        d_in, d_ws, rows, cols, chunk_rows);
    baracuda_gen_sum_f32_reduce_sum_ax1_splitk_combine<<<(unsigned)((cols + block - 1) / block), block>>>(
        d_ws, d_out, cols, n_chunks);
}

int main() {
    // ---- correctness: rows=100 cols=64, ragged chunks (48,48,4) ----
    {
        const long long rows = 100, cols = 64, n_chunks = 3;
        std::vector<float> in(rows * cols);
        for (long long i = 0; i < rows * cols; ++i) in[i] = (float)((i % 37) - 18) * 0.25f;
        // Column 0: all -0.0 — its exact total is -0.0. The combine seeds from
        // the first partial (not 0 + w, which would flip -0 to +0), so split-K
        // must preserve the sign exactly as the baseline fold does.
        for (long long r = 0; r < rows; ++r) in[r * cols] = -0.0f;
        std::vector<double> oracle(cols, 0.0);
        for (long long r = 0; r < rows; ++r)
            for (long long c = 0; c < cols; ++c) oracle[c] += (double)in[r * cols + c];

        float *d_in, *d_ws, *d_out, *d_base;
        cudaMalloc((void**)&d_in, rows * cols * 4);
        cudaMalloc((void**)&d_ws, n_chunks * cols * 4);
        cudaMalloc((void**)&d_out, cols * 4);
        cudaMalloc((void**)&d_base, cols * 4);
        cudaMemcpy(d_in, in.data(), rows * cols * 4, cudaMemcpyHostToDevice);

        // Baseline (general path: shape/strides on device).
        long long shape[2] = {rows, cols}, s0[2] = {cols, 1}, so[1] = {1};
        long long *d_sh, *d_s0, *d_so;
        cudaMalloc((void**)&d_sh, sizeof(shape)); cudaMemcpy(d_sh, shape, sizeof(shape), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_s0, sizeof(s0)); cudaMemcpy(d_s0, s0, sizeof(s0), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_so, sizeof(so)); cudaMemcpy(d_so, so, sizeof(so), cudaMemcpyHostToDevice);
        baracuda_gen_sum_f32_reduce_sum_ax1<<<32, 128>>>(d_in, d_base, d_sh, d_s0, d_so, cols);

        run_splitk(d_in, d_ws, d_out, rows, cols, n_chunks);
        cudaDeviceSynchronize();

        std::vector<float> got(cols), base(cols);
        cudaMemcpy(got.data(), d_out, cols * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(base.data(), d_base, cols * 4, cudaMemcpyDeviceToHost);
        double maxrel = 0, maxrel_b = 0;
        for (long long c = 0; c < cols; ++c) {
            double denom = fabs(oracle[c]) > 1.0 ? fabs(oracle[c]) : 1.0;
            maxrel = fmax(maxrel, fabs((double)got[c] - oracle[c]) / denom);
            maxrel_b = fmax(maxrel_b, fabs((double)base[c] - oracle[c]) / denom);
        }
        bool ok = maxrel < 1e-5 && maxrel_b < 1e-5;
        printf(ok ? "PASS %-26s splitk relerr %.2e | base relerr %.2e\n"
                  : "FAIL %-26s splitk relerr %.2e | base relerr %.2e\n",
               "sum_ax0_ragged_vs_oracle", maxrel, maxrel_b);
        if (!ok) fails++;

        // Zero-sign semantics: the baseline's 0.0f-SEEDED fold maps an
        // all-(-0.0) column to +0 (its first add is 0 + (-0) = +0). The variant
        // must reproduce that fold semantic bit-for-bit — same seed in the
        // partial, first-partial seed in the combine — NOT the unseeded exact
        // sum (-0). Assert the two kernels AGREE.
        unsigned int gb, bb;
        memcpy(&gb, &got[0], 4);
        memcpy(&bb, &base[0], 4);
        bool nz = gb == bb && gb == 0x00000000u;
        printf(nz ? "PASS %-26s (both 0x%08x: zero-seeded fold semantic)\n"
                  : "FAIL %-26s (splitk 0x%08x, base 0x%08x)\n",
               "neg_zero_column_agree", gb, bb);
        if (!nz) fails++;

        // ---- degenerate n_chunks=1: association == baseline -> bit-identical ----
        run_splitk(d_in, d_ws, d_out, rows, cols, 1);
        cudaDeviceSynchronize();
        std::vector<float> one(cols);
        cudaMemcpy(one.data(), d_out, cols * 4, cudaMemcpyDeviceToHost);
        bool bit = memcmp(one.data(), base.data(), cols * 4) == 0;
        printf(bit ? "PASS %-26s (n_chunks=1 == baseline, memcmp)\n"
                   : "FAIL %-26s (n_chunks=1 != baseline)\n",
               "splitk_degenerate_bitexact");
        if (!bit) fails++;

        // ---- determinism: two identical runs are bit-identical ----
        run_splitk(d_in, d_ws, d_out, rows, cols, 7);
        cudaDeviceSynchronize();
        std::vector<float> a(cols);
        cudaMemcpy(a.data(), d_out, cols * 4, cudaMemcpyDeviceToHost);
        cudaMemset(d_out, 0xAA, cols * 4);
        run_splitk(d_in, d_ws, d_out, rows, cols, 7);
        cudaDeviceSynchronize();
        std::vector<float> b(cols);
        cudaMemcpy(b.data(), d_out, cols * 4, cudaMemcpyDeviceToHost);
        bool det = memcmp(a.data(), b.data(), cols * 4) == 0;
        printf(det ? "PASS %-26s (run-to-run memcmp)\n" : "FAIL %-26s\n", "splitk_deterministic");
        if (!det) fails++;

        cudaFree(d_in); cudaFree(d_ws); cudaFree(d_out); cudaFree(d_base);
        cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }

    // ---- bench sweep: fixed ~256MB read, cols from starved to saturated ----
    // The baseline is 1 thread per column: at small cols it cannot fill the GPU
    // (cols=512 -> 2 blocks); split-K restores parallelism via row chunks.
    {
        const long long total = 64LL * 1024 * 1024; // 256 MB f32
        const double read_gb = total * 4.0 / 1e9;
        printf("\nf32 sum over axis 0, %.2f GB read per launch (rows*cols fixed)\n\n", read_gb);
        printf("%8s %9s | %9s %9s | %9s %9s %9s | %8s\n",
               "cols", "rows", "base ms", "base GB/s", "sk ms", "sk GB/s", "chunks", "speedup");
        for (long long cols = 256; cols <= 65536; cols <<= 2) {
            long long rows = total / cols;
            // Enough chunks to fill the GPU: target >= 1024 blocks in kernel 1.
            const int block = 256;
            long long col_tiles = (cols + block - 1) / block;
            long long n_chunks = 1024 / col_tiles;
            if (n_chunks < 1) n_chunks = 1;
            if (n_chunks > rows) n_chunks = rows;

            float *d_in, *d_ws, *d_out;
            cudaMalloc((void**)&d_in, total * 4);
            cudaMalloc((void**)&d_ws, n_chunks * cols * 4);
            cudaMalloc((void**)&d_out, cols * 4);
            cudaMemset(d_in, 0x3C, total * 4);

            long long shape[2] = {rows, cols}, s0[2] = {cols, 1}, so[1] = {1};
            long long *d_sh, *d_s0, *d_so;
            cudaMalloc((void**)&d_sh, sizeof(shape)); cudaMemcpy(d_sh, shape, sizeof(shape), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_s0, sizeof(s0)); cudaMemcpy(d_s0, s0, sizeof(s0), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_so, sizeof(so)); cudaMemcpy(d_so, so, sizeof(so), cudaMemcpyHostToDevice);

            int grid_base = (int)col_tiles;
            float t_base = timed([&] {
                baracuda_gen_sum_f32_reduce_sum_ax1<<<grid_base, block>>>(d_in, d_out, d_sh, d_s0, d_so, cols);
            });
            float t_sk = timed([&] { run_splitk(d_in, d_ws, d_out, rows, cols, n_chunks); });

            printf("%8lld %9lld | %9.3f %9.1f | %9.3f %9.1f %9lld | %7.2fx\n",
                   cols, rows, t_base, read_gb / (t_base / 1000),
                   t_sk, read_gb / (t_sk / 1000), n_chunks, t_base / t_sk);

            cudaFree(d_in); cudaFree(d_ws); cudaFree(d_out);
            cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
        }
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
