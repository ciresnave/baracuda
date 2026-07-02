// On-device numeric validation of the item-02 shared-interior DAG emitter.
// Launches the *generated* diamond kernels (out = g/(g+1), g = a*b — the product
// hoisted to one `tmp`, computed once) and diffs against a host oracle. The oracle
// mirrors the emitted math exactly: g in float, then `g/(g+1.0)` in double (the
// `1.0` literal promotes), narrowed to float — so dedup must be a no-op on values.
// Compile: nvcc -O3 -arch=sm_89 dag_validate.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "baracuda_gen_diamond_f32_scalar.cu"  // scalar hoist:  float tmp0 = (in0[i]*in1[i]);
#include "baracuda_gen_diamond_f32_co_v4.cu"   // per-lane hoist: { float tmp0 = (v0.x*v1.x); ... }

static int check(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    float maxerr = 0.0f; long long badi = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float e = fabsf(got[i] - ref[i]);
        if (e > maxerr) { maxerr = e; badi = (long long)i; }
    }
    bool ok = maxerr <= 1e-6f;
    if (ok) printf("PASS %-22s maxerr %g\n", name, maxerr);
    else printf("FAIL %-22s maxerr %g at %lld (got %g want %g)\n", name, maxerr, badi, got[badi], ref[badi]);
    return ok ? 0 : 1;
}

int main() {
    const long long n = 4096; // multiple of 4 for the vectorized path
    std::vector<float> a(n), b(n), ref(n);
    for (long long i = 0; i < n; ++i) {
        a[i] = (float)((i % 21) - 10) * 0.5f;  // spans negatives and zero
        b[i] = (float)((i % 13) - 6) * 0.25f;
        float g = a[i] * b[i];
        double d = (double)g / ((double)g + 1.0); // matches the emitted double-promote
        ref[i] = (float)d;
    }
    float *d0 = nullptr, *d1 = nullptr, *dout = nullptr;
    cudaMalloc((void**)&d0, n * sizeof(float));
    cudaMalloc((void**)&d1, n * sizeof(float));
    cudaMalloc((void**)&dout, n * sizeof(float));
    cudaMemcpy(d0, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d1, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int fails = 0;
    std::vector<float> got(n);

    cudaMemset(dout, 0, n * sizeof(float));
    baracuda_gen_diamond_f32_scalar<<<64, 128>>>(d0, d1, dout, n);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    fails += check("diamond_scalar", got, ref);

    cudaMemset(dout, 0, n * sizeof(float));
    baracuda_gen_diamond_f32_co_v4<<<64, 128>>>((const float4*)d0, (const float4*)d1, (float4*)dout, n / 4);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    fails += check("diamond_vectorized", got, ref);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
