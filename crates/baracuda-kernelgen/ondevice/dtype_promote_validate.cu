// Exhaustive dtype-promote validation: EVERY f16 and EVERY bf16 bit pattern
// (all 65536 each) through the GENERATED helper (baracuda::dtype::gen::*) vs the
// hand-written baracuda_dtype_promote.cuh (baracuda::load_as_f32<T> /
// store_from_f32<T>) and the lossless round-trip identity.
//
// Phase 2c of the IR-translation-hub roadmap (docs/design/ir-translation-hub.md):
// dtype_promote is a DE-DUPLICATION win (both sides compile to the same single
// intrinsic — no speedup), so the validation is exhaustive bit-exactness.
//
//   DTYPE_OUT=<work> cargo test -p baracuda-kernelgen dump_dtype_promote_helper -- --ignored
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <work> -I crates/baracuda-kernels-sys/kernels/include \
//        crates/baracuda-kernelgen/ondevice/dtype_promote_validate.cu -o <work>/dtype_promote_validate
//   <work>/dtype_promote_validate
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "baracuda_dtype_promote.cuh"   // hand-written: baracuda::load_as_f32<T> / store_from_f32<T>
#include "dtype_promote_gen.cuh"        // generated:   baracuda::dtype::gen::load_as_f32_{f16,bf16}

#define CHECK(x) do { cudaError_t e_=(x); if(e_){ \
  printf("CUDA err %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2);} } while(0)

static int fails = 0;

// One thread per f16 code: generated + hand-written promotion, and the round-trip.
__global__ void sweep_f16(uint32_t* gen_load, uint32_t* hand_load, uint16_t* gen_rt, uint16_t* hand_rt) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 65536) return;
    __half h = __ushort_as_half((uint16_t)t);
    float g = baracuda::dtype::gen::load_as_f32_f16(h);
    float w = baracuda::load_as_f32<__half>(h);
    gen_load[t]  = __float_as_uint(g);
    hand_load[t] = __float_as_uint(w);
    gen_rt[t]  = __half_as_ushort(baracuda::dtype::gen::store_from_f32_f16(g));
    hand_rt[t] = __half_as_ushort(baracuda::store_from_f32<__half>(w));
}

__global__ void sweep_bf16(uint32_t* gen_load, uint32_t* hand_load, uint16_t* gen_rt, uint16_t* hand_rt) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 65536) return;
    __nv_bfloat16 h = __ushort_as_bfloat16((uint16_t)t);
    float g = baracuda::dtype::gen::load_as_f32_bf16(h);
    float w = baracuda::load_as_f32<__nv_bfloat16>(h);
    gen_load[t]  = __float_as_uint(g);
    hand_load[t] = __float_as_uint(w);
    gen_rt[t]  = __bfloat16_as_ushort(baracuda::dtype::gen::store_from_f32_bf16(g));
    hand_rt[t] = __bfloat16_as_ushort(baracuda::store_from_f32<__nv_bfloat16>(w));
}

// NaN? exponent all-ones AND mantissa nonzero. f16: exp bits 10..14, mant 10.
// bf16: exp bits 7..14, mant 7.
static bool is_nan_f16(uint16_t c)  { return ((c & 0x7C00) == 0x7C00) && (c & 0x03FF); }
static bool is_nan_bf16(uint16_t c) { return ((c & 0x7F80) == 0x7F80) && (c & 0x007F); }

template <class Launch>
static void run(const char* name, bool (*is_nan)(uint16_t), Launch launch) {
    uint32_t *dgl, *dhl; uint16_t *dgr, *dhr;
    CHECK(cudaMalloc(&dgl, 65536*4)); CHECK(cudaMalloc(&dhl, 65536*4));
    CHECK(cudaMalloc(&dgr, 65536*2)); CHECK(cudaMalloc(&dhr, 65536*2));
    launch(dgl, dhl, dgr, dhr);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> gl(65536), hl(65536);
    std::vector<uint16_t> gr(65536), hr(65536);
    CHECK(cudaMemcpy(gl.data(), dgl, 65536*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hl.data(), dhl, 65536*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(gr.data(), dgr, 65536*2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hr.data(), dhr, 65536*2, cudaMemcpyDeviceToHost));
    int load_mism = 0, rt_mism = 0, rt_identity = 0;
    for (int c = 0; c < 65536; ++c) {
        if (gl[c] != hl[c]) load_mism++;                                  // gen load == hand load
        if (gr[c] != hr[c]) rt_mism++;                                    // gen round-trip == hand round-trip
        if (!is_nan((uint16_t)c) && gr[c] != (uint16_t)c) rt_identity++;  // lossless round-trip (non-NaN)
    }
    printf("%-5s  load gen==hand: %-4s (%d)  |  round-trip gen==hand: %-4s (%d)  |  lossless-identity[non-NaN]: %-4s (%d)\n",
           name, load_mism ? "FAIL" : "ok", load_mism,
           rt_mism ? "FAIL" : "ok", rt_mism, rt_identity ? "FAIL" : "ok", rt_identity);
    if (load_mism || rt_mism || rt_identity) fails++;
    cudaFree(dgl); cudaFree(dhl); cudaFree(dgr); cudaFree(dhr);
}

int main() {
    printf("== exhaustive dtype-promote: all 65536 f16 + all 65536 bf16 codes ==\n");
    run("f16",  is_nan_f16,  [](uint32_t* a, uint32_t* b, uint16_t* c, uint16_t* d){ sweep_f16<<<256,256>>>(a,b,c,d); });
    run("bf16", is_nan_bf16, [](uint32_t* a, uint32_t* b, uint16_t* c, uint16_t* d){ sweep_bf16<<<256,256>>>(a,b,c,d); });
    printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "PASSED", fails, fails==1 ? "" : "s");
    return fails ? 1 : 0;
}
