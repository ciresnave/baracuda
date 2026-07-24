// Lift round-trip DEVICE differential: prove the CUDA -> IR `lift` frontend's
// re-emission is numerically FAITHFUL. The re-emitted kernel (`lift_gen.cu`,
// from `generate()` of the lifted IR) must produce BIT-IDENTICAL output to the
// original hand-written kernel (`lift_orig.cu`, the SAME per-element body),
// element for element, over a large buffer of random f32 inputs.
//
// Phase 4 of the IR-translation-hub roadmap (docs/design/ir-translation-hub.md):
// a lifted-then-re-emitted elementwise kernel is the SAME integer/float
// arithmetic as the source (source -> IR -> re-emit is a faithful identity on
// the recognized idiom), so the validation is exhaustive BIT-exactness (memcmp),
// not a tolerance. Both kernels compute `out[i] = in0[i]*in1[i] + in0[i]` as a
// single `a*b+c` expression, so nvcc contracts both to the same fma -> identical.
//
//   LIFT_OUT=<work> cargo test -p baracuda-kernelgen --lib \
//       lift::tests::dump_lift_roundtrip -- --ignored --nocapture
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//        -I <work> \
//        crates/baracuda-kernelgen/ondevice/lift_roundtrip_validate.cu \
//        -o <work>/lift_roundtrip_validate
//   <work>/lift_roundtrip_validate
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "lift_gen.cu"    // generated: extern "C" baracuda_gen_lift_rt_f32_scalar(...)
#include "lift_orig.cu"   // original:  __global__ void lift_orig(...) — SAME body

// The generated kernel's symbol carries a cell suffix. The dump test prints it
// as `GENERATED KERNEL NAME:` — confirm the generated kernel name from the dump
// test output and update this if it changes (override with
// `-DLIFT_GEN_KERNEL=<name>` on the nvcc line).
#ifndef LIFT_GEN_KERNEL
#define LIFT_GEN_KERNEL baracuda_gen_lift_rt_f32_scalar
#endif

#define CHECK(x) do { cudaError_t e_=(x); if(e_){ \
  printf("CUDA err %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2);} } while(0)

static int fails = 0;

int main() {
    const long long n = 1LL << 22;           // ~4.19M elements
    const size_t bytes = (size_t)n * sizeof(float);
    printf("== lift round-trip device differential: %lld random f32 elements ==\n", n);

    std::vector<float> h_in0(n), h_in1(n);
    srand(20260710u);
    auto rf = []() { return ((float)rand() / (float)RAND_MAX - 0.5f) * 8.0f; };  // ~[-4,4]
    for (long long i = 0; i < n; ++i) { h_in0[i] = rf(); h_in1[i] = rf(); }
    // Seed a handful of edge values — signed zero, tiny, huge, and an
    // Inf/NaN-producing pair — so the bit-compare also covers special FP.
    h_in0[0] = 0.0f;         h_in1[0] = 3.14159f;
    h_in0[1] = -0.0f;        h_in1[1] = -2.5f;
    h_in0[2] = 1e30f;        h_in1[2] = 1e30f;     // overflow -> +Inf
    h_in0[3] = -1e30f;       h_in1[3] = 1e30f;     // -> -Inf
    h_in0[4] = 1e-30f;       h_in1[4] = 1e-30f;    // underflow-ish
    h_in0[5] = INFINITY;     h_in1[5] = 0.0f;      // Inf*0 -> NaN

    float *d_in0, *d_in1, *d_gen, *d_orig;
    CHECK(cudaMalloc(&d_in0, bytes));
    CHECK(cudaMalloc(&d_in1, bytes));
    CHECK(cudaMalloc(&d_gen, bytes));
    CHECK(cudaMalloc(&d_orig, bytes));
    CHECK(cudaMemcpy(d_in0, h_in0.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_in1, h_in1.data(), bytes, cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (int)((n + block - 1) / block);

    LIFT_GEN_KERNEL<<<grid, block>>>(d_in0, d_in1, d_gen, n);
    CHECK(cudaGetLastError());
    lift_orig<<<grid, block>>>(d_in0, d_in1, d_orig, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gen(n), h_orig(n);
    CHECK(cudaMemcpy(h_gen.data(),  d_gen,  bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_orig.data(), d_orig, bytes, cudaMemcpyDeviceToHost));

    // BIT-identical: raw-byte memcmp of the whole buffer.
    int mism = memcmp(h_gen.data(), h_orig.data(), bytes) == 0 ? 0 : 1;
    long long first_bad = -1, count_bad = 0;
    for (long long i = 0; i < n; ++i) {
        uint32_t g, o;
        memcpy(&g, &h_gen[i], 4);
        memcpy(&o, &h_orig[i], 4);
        if (g != o) { if (first_bad < 0) first_bad = i; count_bad++; }
    }
    if (mism || count_bad) {
        fails++;
        printf("lift round-trip: FAIL — %lld/%lld elements differ (first @ %lld: "
               "gen=%.9g orig=%.9g)\n",
               count_bad, n, first_bad,
               first_bad >= 0 ? h_gen[first_bad] : 0.0f,
               first_bad >= 0 ? h_orig[first_bad] : 0.0f);
    } else {
        printf("lift round-trip: ok — all %lld elements BIT-IDENTICAL "
               "(re-emitted == original)\n", n);
    }

    cudaFree(d_in0); cudaFree(d_in1); cudaFree(d_gen); cudaFree(d_orig);
    printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
