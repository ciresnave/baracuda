// On-device bit-exactness validation of the item-09 packed f16/bf16 path.
// Runs each PACKED kernel (half2/bf162 pairs, 128-bit accesses) and its SCALAR
// sibling (the oracle) over a corpus where input0 sweeps EVERY 16-bit pattern —
// all NaN payloads, +/-Inf, +/-0, every subnormal, max-finite — and requires the
// raw u16 output buffers to be memcmp-identical. The packed path claims to be a
// pure win: text changes, bits must not.
// Compile: nvcc -O3 -arch=sm_89 packed_validate.cu
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_gen_add_f16_co_v8.cu"
#include "baracuda_gen_add_f16_scalar.cu"
#include "baracuda_gen_relu_add_f16_co_v8.cu"
#include "baracuda_gen_relu_add_f16_scalar.cu"
#include "baracuda_gen_add_bf16_co_v8.cu"
#include "baracuda_gen_add_bf16_scalar.cu"
#include "baracuda_gen_relu_add_bf16_co_v8.cu"
#include "baracuda_gen_relu_add_bf16_scalar.cu"

static int fails = 0;

// Compare two device u16 output buffers bit-for-bit.
static void diff(const char* name, const unsigned short* d_a, const unsigned short* d_b, long long n) {
    std::vector<unsigned short> a(n), b(n);
    cudaMemcpy(a.data(), d_a, n * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, n * 2, cudaMemcpyDeviceToHost);
    for (long long i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("FAIL %-24s first mismatch at %lld: packed 0x%04x vs scalar 0x%04x\n",
                   name, i, a[i], b[i]);
            fails++;
            return;
        }
    }
    printf("PASS %-24s bit-identical over %lld elems (full 16-bit sweep)\n", name, n);
}

int main() {
    const long long n = 1 << 16; // input0 sweeps every 16-bit pattern; %8 == 0
    std::vector<unsigned short> h0(n), h1(n);
    for (long long i = 0; i < n; ++i) {
        h0[i] = (unsigned short)i;                          // every f16/bf16 bit pattern
        h1[i] = (unsigned short)((i * 2654435761u) >> 13);  // mixed partner incl. specials
    }
    unsigned short *d0, *d1, *dp, *ds;
    cudaMalloc((void**)&d0, n * 2);
    cudaMalloc((void**)&d1, n * 2);
    cudaMalloc((void**)&dp, n * 2); // packed output
    cudaMalloc((void**)&ds, n * 2); // scalar-oracle output
    cudaMemcpy(d0, h0.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d1, h1.data(), n * 2, cudaMemcpyHostToDevice);
    const long long nv = n / 8;

    // ---- f16 add: Tier A (native __half2 operator+) ----
    cudaMemset(dp, 0xAA, n * 2); cudaMemset(ds, 0x55, n * 2);
    baracuda_gen_add_f16_co_v8<<<64, 256>>>(
        (const baracuda_gen_add_f16_co_v8_vec*)d0, (const baracuda_gen_add_f16_co_v8_vec*)d1,
        (baracuda_gen_add_f16_co_v8_vec*)dp, nv);
    baracuda_gen_add_f16_scalar<<<64, 256>>>((const __half*)d0, (const __half*)d1, (__half*)ds, n);
    cudaDeviceSynchronize();
    diff("add_f16", dp, ds, n);

    // ---- f16 relu(a+b): Tier A add + Tier B relu (pair-scalarized float) ----
    cudaMemset(dp, 0xAA, n * 2); cudaMemset(ds, 0x55, n * 2);
    baracuda_gen_relu_add_f16_co_v8<<<64, 256>>>(
        (const baracuda_gen_relu_add_f16_co_v8_vec*)d0, (const baracuda_gen_relu_add_f16_co_v8_vec*)d1,
        (baracuda_gen_relu_add_f16_co_v8_vec*)dp, nv);
    baracuda_gen_relu_add_f16_scalar<<<64, 256>>>((const __half*)d0, (const __half*)d1, (__half*)ds, n);
    cudaDeviceSynchronize();
    diff("relu_add_f16", dp, ds, n);

    // ---- bf16 add ----
    cudaMemset(dp, 0xAA, n * 2); cudaMemset(ds, 0x55, n * 2);
    baracuda_gen_add_bf16_co_v8<<<64, 256>>>(
        (const baracuda_gen_add_bf16_co_v8_vec*)d0, (const baracuda_gen_add_bf16_co_v8_vec*)d1,
        (baracuda_gen_add_bf16_co_v8_vec*)dp, nv);
    baracuda_gen_add_bf16_scalar<<<64, 256>>>(
        (const __nv_bfloat16*)d0, (const __nv_bfloat16*)d1, (__nv_bfloat16*)ds, n);
    cudaDeviceSynchronize();
    diff("add_bf16", dp, ds, n);

    // ---- bf16 relu(a+b) ----
    cudaMemset(dp, 0xAA, n * 2); cudaMemset(ds, 0x55, n * 2);
    baracuda_gen_relu_add_bf16_co_v8<<<64, 256>>>(
        (const baracuda_gen_relu_add_bf16_co_v8_vec*)d0, (const baracuda_gen_relu_add_bf16_co_v8_vec*)d1,
        (baracuda_gen_relu_add_bf16_co_v8_vec*)dp, nv);
    baracuda_gen_relu_add_bf16_scalar<<<64, 256>>>(
        (const __nv_bfloat16*)d0, (const __nv_bfloat16*)d1, (__nv_bfloat16*)ds, n);
    cudaDeviceSynchronize();
    diff("relu_add_bf16", dp, ds, n);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
