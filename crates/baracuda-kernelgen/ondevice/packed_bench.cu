// Bandwidth benchmark: packed f16 elementwise (128-bit half2 accesses) vs the
// scalar-fallback kernel (2-byte accesses), against a copy reference.
// Compile: nvcc -O3 -arch=sm_89 packed_bench.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "baracuda_gen_add_f16_co_v8.cu"
#include "baracuda_gen_add_f16_scalar.cu"

__global__ void copy_kernel(const float4* __restrict__ in, float4* __restrict__ out, long long n4) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long s = (long long)gridDim.x * blockDim.x;
    for (; i < n4; i += s) out[i] = in[i];
}

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

int main() {
    const long long nmax = 1LL << 26;
    unsigned short *d0, *d1, *dout;
    cudaMalloc((void**)&d0, nmax * 2);
    cudaMalloc((void**)&d1, nmax * 2);
    cudaMalloc((void**)&dout, nmax * 2);
    cudaMemset(d0, 0x3C, nmax * 2); // ~1.0-ish halves
    cudaMemset(d1, 0x38, nmax * 2);
    const int block = 256;

    printf("f16 add: scalar (2B accesses) vs packed v8 (128-bit), RTX 4070 (36MB L2)\n\n");
    printf("%10s %10s | %9s %9s %8s | %9s %9s %8s\n",
           "n(halves)", "MB moved", "scal ms", "scal GB/s", "", "pack ms", "pack GB/s", "speedup");
    for (long long n = 1LL << 16; n <= nmax; n <<= 2) {
        const double moved_gb = 3.0 * n * 2 / 1e9;
        // Each kernel gets the config a real launcher computes from its OWN
        // element count: ceil(count/block), capped for grid-stride.
        auto grid_for = [&](long long count) {
            long long g = (count + block - 1) / block;
            if (g < 1) g = 1;
            return (int)std::min<long long>(2048, g);
        };
        int grid_scal = grid_for(n), grid_pack = grid_for(n / 8);
        int iters = n >= (1LL << 24) ? 50 : 400;
        float t_scal = timed([&] {
            baracuda_gen_add_f16_scalar<<<grid_scal, block>>>((const __half*)d0, (const __half*)d1, (__half*)dout, n);
        }, iters);
        float t_pack = timed([&] {
            baracuda_gen_add_f16_co_v8<<<grid_pack, block>>>(
                (const baracuda_gen_add_f16_co_v8_vec*)d0, (const baracuda_gen_add_f16_co_v8_vec*)d1,
                (baracuda_gen_add_f16_co_v8_vec*)dout, n / 8);
        }, iters);
        printf("%10lld %10.1f | %9.4f %9.1f %8s | %9.4f %9.1f %7.2fx\n",
               n, n * 2.0 / 1e6, t_scal, moved_gb / (t_scal / 1000), "",
               t_pack, moved_gb / (t_pack / 1000), t_scal / t_pack);
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    return 0;
}
