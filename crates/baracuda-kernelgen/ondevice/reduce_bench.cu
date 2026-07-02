// Perf benchmark of the item-03 reduction kernels on the RTX 4070.
// Compares the fast-path (last-axis, one thread per row) vs the general path
// (outer-axis, one thread per column) on a large [N,M] f32 tensor, against a
// copy-bandwidth reference. Reductions are memory-bound, so GB/s vs. peak is
// the figure of merit. Compile: nvcc -arch=sm_89 reduce_bench.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "baracuda_gen_mean_f32_reduce_mean.cu"    // fast path: reduce LAST axis
#include "baracuda_gen_sum_f32_reduce_sum_ax1.cu"  // general: reduce axis 0 (outer)

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += s) out[i] = in[i];
}

template <class F>
static float timed(F launch, int iters = 50) {
    for (int i = 0; i < 5; ++i) launch(); // warmup
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

int main() {
    const long long N = 8192, M = 8192; // [N, M]
    const long long total = N * M;
    const size_t bytes = total * sizeof(float);
    const double read_gb = (double)bytes / 1e9;

    std::vector<float> h(total);
    for (long long i = 0; i < total; ++i) h[i] = (float)(i % 1000) * 0.001f;
    float *d_in = nullptr, *d_out = nullptr, *d_copy = nullptr;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, N * sizeof(float));
    cudaMalloc((void**)&d_copy, bytes);
    cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice);

    long long shape[2] = {N, M}, s0[2] = {M, 1}, so[1] = {1};
    long long *d_shape, *d_s0, *d_so;
    cudaMalloc((void**)&d_shape, sizeof(shape)); cudaMemcpy(d_shape, shape, sizeof(shape), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_s0, sizeof(s0)); cudaMemcpy(d_s0, s0, sizeof(s0), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_so, sizeof(so)); cudaMemcpy(d_so, so, sizeof(so), cudaMemcpyHostToDevice);

    const int block = 256;
    const int grid_r = (int)((N + block - 1) / block);   // one thread per output row/col
    const int grid_c = 512;                              // copy: grid-stride

    // Copy reference (read + write => 2x traffic) -> peak achievable bandwidth.
    float t_copy = timed([&] { copy_kernel<<<grid_c, block>>>(d_in, d_copy, total); });
    // InnerContig last-axis: now BLOCK-per-row (one block/row, coalesced + block reduce).
    float t_last = timed([&] { baracuda_gen_mean_f32_reduce_mean<<<(int)N, block>>>(d_in, d_out, N, M); });
    // General path: reduce axis 0 (one thread per column; warp reads coalesced).
    float t_outer = timed([&] { baracuda_gen_sum_f32_reduce_sum_ax1<<<grid_r, block>>>(d_in, d_out, d_shape, d_s0, d_so, N); });

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }

    printf("tensor [%lld,%lld] f32 = %.0f MB read per reduction\n\n", N, M, read_gb * 1000);
    printf("%-34s %8s %10s\n", "kernel", "ms", "GB/s");
    printf("%-34s %8.3f %10.1f  (read+write, ~peak)\n", "copy (bandwidth ref)", t_copy, 2 * read_gb / (t_copy / 1000));
    printf("%-34s %8.3f %10.1f\n", "reduce LAST axis (block-per-row)", t_last, read_gb / (t_last / 1000));
    printf("%-34s %8.3f %10.1f\n", "reduce axis 0 (general/outer)", t_outer, read_gb / (t_outer / 1000));
    return 0;
}
