// Thorough elementwise dtype-cast validation: the GENERATED cast helper
// (baracuda::cast::gen::cast_<sin>_<sout>, from emit_cast_helper) vs the
// hand-written baracuda_cast.cuh (baracuda::cast::cast_value<TIn, TOut>), across
// the full 8×8 dtype matrix, plus an independent CPU static_cast reference for
// the arithmetic (non-half) pairs.
//
// Phase 2c of the IR-translation-hub roadmap (docs/design/ir-translation-hub.md).
// This is the CAST pivot of the fp_bits/cast helper migration: fp_bits does NOT
// factor (the generator emits none of baracuda_fp_bits.cuh's mantissa/exponent/
// sign/TF32 logic — its only FP-bit use is inf/NaN sentinels, already single-
// sourced), so we migrate `cast` instead. Like dtype-promote this is a
// DE-DUPLICATION win (each generated cast resolves to the same static_cast /
// half-intrinsic the hand-written cast_value does — no speedup), so the proof is
// bit-exactness: gen == hand for every pair, exhaustive over every f16 + bf16 +
// i8 + u8 source code and a curated sample of the wider dtypes.
//
//   CAST_OUT=<work> cargo test -p baracuda-kernelgen dump_cast_helper -- --ignored
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <work> -I crates/baracuda-kernels-sys/kernels/include \
//        crates/baracuda-kernelgen/ondevice/cast_validate.cu -o <work>/cast_validate
//   <work>/cast_validate
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "baracuda_cast.cuh"   // hand-written: baracuda::cast::cast_value<TIn, TOut>
#include "cast_gen.cuh"        // generated:    baracuda::cast::gen::cast_<sin>_<sout>

#define CHECK(x) do { cudaError_t e_=(x); if(e_){ \
  printf("CUDA err %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2);} } while(0)

static int fails = 0;

// ---------------------------------------------------------------------------
// Reinterpret any cast result as a fixed-width (zero-extended) bit pattern so
// gen and hand outputs can be compared exactly regardless of destination type.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t bits(__half h)         { return (uint64_t)__half_as_ushort(h); }
__device__ __forceinline__ uint64_t bits(__nv_bfloat16 b)  { return (uint64_t)__bfloat16_as_ushort(b); }
__device__ __forceinline__ uint64_t bits(float f)          { return (uint64_t)__float_as_uint(f); }
__device__ __forceinline__ uint64_t bits(double d)         { return (uint64_t)__double_as_longlong(d); }
__device__ __forceinline__ uint64_t bits(int i)            { return (uint64_t)(uint32_t)i; }
__device__ __forceinline__ uint64_t bits(long long l)      { return (uint64_t)l; }
__device__ __forceinline__ uint64_t bits(signed char c)    { return (uint64_t)(uint8_t)c; }
__device__ __forceinline__ uint64_t bits(unsigned char c)  { return (uint64_t)c; }

// One probe per (source, destination) dtype: cast every input through the
// generated helper AND the hand-written cast_value, recording both bit patterns.
// The host launcher wrapper gives a uniform function-pointer signature per source
// type so a single templated runner can drive all destinations.
#define PROBE(SN, DN, STY, DTY)                                                          \
  __global__ void k_##SN##_##DN(const STY* in, uint64_t* g, uint64_t* h, int n) {        \
    int t = blockIdx.x * blockDim.x + threadIdx.x;                                        \
    if (t >= n) return;                                                                   \
    STY x = in[t];                                                                        \
    g[t] = bits(baracuda::cast::gen::cast_##SN##_##DN(x));                                \
    h[t] = bits(baracuda::cast::cast_value<STY, DTY>(x));                                 \
  }                                                                                       \
  static void launch_##SN##_##DN(const STY* in, uint64_t* g, uint64_t* h, int n) {        \
    k_##SN##_##DN<<<(n + 255) / 256, 256>>>(in, g, h, n);                                 \
  }

// The full 8×8 matrix. STY / DTY are the C++ types; SN / DN the short tags used
// in the generated symbol names.
#define PROBES_FROM(SN, STY)                     \
  PROBE(SN, f16,  STY, __half)                    \
  PROBE(SN, bf16, STY, __nv_bfloat16)             \
  PROBE(SN, f32,  STY, float)                     \
  PROBE(SN, f64,  STY, double)                    \
  PROBE(SN, i32,  STY, int)                       \
  PROBE(SN, i64,  STY, long long)                 \
  PROBE(SN, i8,   STY, signed char)               \
  PROBE(SN, u8,   STY, unsigned char)

PROBES_FROM(f16,  __half)
PROBES_FROM(bf16, __nv_bfloat16)
PROBES_FROM(f32,  float)
PROBES_FROM(f64,  double)
PROBES_FROM(i32,  int)
PROBES_FROM(i64,  long long)
PROBES_FROM(i8,   signed char)
PROBES_FROM(u8,   unsigned char)

template <class STY>
using probe_fn = void (*)(const STY*, uint64_t*, uint64_t*, int);

template <class STY>
struct DestProbe { const char* name; probe_fn<STY> fn; };

// Run every destination probe for one source dtype, comparing gen vs hand bit
// patterns over `in`. Prints one line per (source, dest) with the mismatch count.
template <class STY>
static void run_source(const char* sname, const std::vector<STY>& in,
                       const std::vector<DestProbe<STY>>& dests) {
    int n = (int)in.size();
    STY* din;
    CHECK(cudaMalloc(&din, (size_t)n * sizeof(STY)));
    CHECK(cudaMemcpy(din, in.data(), (size_t)n * sizeof(STY), cudaMemcpyHostToDevice));
    uint64_t *dg, *dh;
    CHECK(cudaMalloc(&dg, (size_t)n * 8));
    CHECK(cudaMalloc(&dh, (size_t)n * 8));
    std::vector<uint64_t> g(n), h(n);
    printf("== source %-4s (%d inputs) ==\n", sname, n);
    for (const auto& d : dests) {
        d.fn(din, dg, dh, n);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(g.data(), dg, (size_t)n * 8, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h.data(), dh, (size_t)n * 8, cudaMemcpyDeviceToHost));
        int mism = 0;
        for (int i = 0; i < n; ++i) {
            if (g[i] != h[i]) mism++;
        }
        printf("   %-4s -> %-4s   gen==hand: %-4s (%d)\n", sname, d.name,
               mism ? "FAIL" : "ok", mism);
        if (mism) fails++;
    }
    cudaFree(din);
    cudaFree(dg);
    cudaFree(dh);
}

#define DEST(SN, DN) DestProbe<STY_T>{#DN, launch_##SN##_##DN}

// Build the destination-probe list for a given source SN (macro binds STY_T).
#define DESTS(SN) std::vector<DestProbe<STY_T>>{ \
    DEST(SN, f16), DEST(SN, bf16), DEST(SN, f32), DEST(SN, f64), \
    DEST(SN, i32), DEST(SN, i64), DEST(SN, i8), DEST(SN, u8) }

// ---------------------------------------------------------------------------
// Independent CPU reference for the arithmetic (non-half) casts: a plain host
// static_cast, reinterpreted to the same bit pattern. Confirms gen isn't merely
// agreeing with a matching-but-wrong hand kernel. Half endpoints are excluded
// (host has no guaranteed __half rounding); those are covered exhaustively by the
// gen==hand sweep over every 16-bit code.
// ---------------------------------------------------------------------------
static uint64_t hbits_f32(float f)     { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint64_t hbits_f64(double d)    { uint64_t u; std::memcpy(&u, &d, 8); return u; }
static uint64_t hbits_i32(int i)       { return (uint64_t)(uint32_t)i; }
static uint64_t hbits_i64(long long l) { return (uint64_t)l; }

static uint64_t ref_i8_i32_h(signed char x)  { return hbits_i32((int)x); }
static uint64_t ref_i8_f32_h(signed char x)  { return hbits_f32((float)x); }
static uint64_t ref_u8_f32_h(unsigned char x) { return hbits_f32((float)x); }
static uint64_t ref_i32_i64_h(int x)         { return hbits_i64((long long)x); }
static uint64_t ref_i32_f64_h(int x)         { return hbits_f64((double)x); }
static uint64_t ref_f64_f32_h(double x)      { return hbits_f32((float)x); }

// gen-only launchers for the reference pairs (write only the generated result).
#define GENONLY(SN, DN, STY)                                                         \
  __global__ void kg_##SN##_##DN(const STY* in, uint64_t* g, int n) {                \
    int t = blockIdx.x * blockDim.x + threadIdx.x;                                    \
    if (t >= n) return;                                                               \
    g[t] = bits(baracuda::cast::gen::cast_##SN##_##DN(in[t]));                        \
  }                                                                                   \
  static void rung_##SN##_##DN(const STY* in, uint64_t* g, int n) {                   \
    kg_##SN##_##DN<<<(n + 255) / 256, 256>>>(in, g, n);                               \
  }

GENONLY(i8,  i32, signed char)
GENONLY(i8,  f32, signed char)
GENONLY(u8,  f32, unsigned char)
GENONLY(i32, i64, int)
GENONLY(i32, f64, int)
GENONLY(f64, f32, double)

// Reference-check one arithmetic pair on the host: device GENERATED result vs the
// host static_cast (reinterpreted to the same bit pattern).
template <class STY>
static void ref_check(const char* pair, const std::vector<STY>& in,
                      void (*run_gen)(const STY*, uint64_t*, int),
                      uint64_t (*host_ref)(STY)) {
    int n = (int)in.size();
    STY* din;
    CHECK(cudaMalloc(&din, (size_t)n * sizeof(STY)));
    CHECK(cudaMemcpy(din, in.data(), (size_t)n * sizeof(STY), cudaMemcpyHostToDevice));
    uint64_t* dg;
    CHECK(cudaMalloc(&dg, (size_t)n * 8));
    run_gen(din, dg, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> g(n);
    CHECK(cudaMemcpy(g.data(), dg, (size_t)n * 8, cudaMemcpyDeviceToHost));
    int mism = 0;
    for (int i = 0; i < n; ++i) {
        if (g[i] != host_ref(in[i])) mism++;
    }
    printf("   ref %-12s gen==cpu:  %-4s (%d)\n", pair, mism ? "FAIL" : "ok", mism);
    if (mism) fails++;
    cudaFree(din);
    cudaFree(dg);
}

int main() {
    printf("== generated cast helper vs hand-written baracuda_cast.cuh (8x8 matrix) ==\n\n");

    // Exhaustive 16-bit source sweeps (every half / bfloat16 code).
    {
        using STY_T = __half;
        std::vector<__half> in(65536);
        for (int c = 0; c < 65536; ++c) { uint16_t u = (uint16_t)c; std::memcpy(&in[c], &u, 2); }
        run_source<__half>("f16", in, DESTS(f16));
    }
    {
        using STY_T = __nv_bfloat16;
        std::vector<__nv_bfloat16> in(65536);
        for (int c = 0; c < 65536; ++c) { uint16_t u = (uint16_t)c; std::memcpy(&in[c], &u, 2); }
        run_source<__nv_bfloat16>("bf16", in, DESTS(bf16));
    }
    // Exhaustive 8-bit integer source sweeps (every i8 / u8 value).
    {
        using STY_T = signed char;
        std::vector<signed char> in(256);
        for (int c = 0; c < 256; ++c) in[c] = (signed char)c;
        run_source<signed char>("i8", in, DESTS(i8));
    }
    {
        using STY_T = unsigned char;
        std::vector<unsigned char> in(256);
        for (int c = 0; c < 256; ++c) in[c] = (unsigned char)c;
        run_source<unsigned char>("u8", in, DESTS(u8));
    }
    // Curated samples for the wider dtypes (specials + int/float boundaries).
    {
        using STY_T = float;
        std::vector<float> in = {
            0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 1.5f, -2.5f, 3.4028235e38f, -3.4028235e38f,
            1.1754944e-38f, 123.456f, -123.456f, 16777216.0f, 16777217.0f, 65504.0f, -65504.0f,
            1e30f, -1e30f, 100.0f, -100.0f, 255.0f, 256.0f, 127.0f, 128.0f, 2147483520.0f,
            0.9999f, -0.9999f, 42.0f };
        run_source<float>("f32", in, DESTS(f32));
    }
    {
        using STY_T = double;
        std::vector<double> in = {
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -2.5, 1.7976931348623157e308,
            -1.7976931348623157e308, 2.2250738585072014e-308, 123.456, -123.456,
            9007199254740992.0, 9007199254740993.0, 65504.0, -65504.0, 3.5e38, -3.5e38,
            100.0, -100.0, 255.0, 256.0, 2147483647.0, -2147483648.0, 42.0 };
        run_source<double>("f64", in, DESTS(f64));
    }
    {
        using STY_T = int;
        std::vector<int> in = {
            0, 1, -1, 2, -2, 127, -128, 255, 256, 1000, -1000, 65535, 65536,
            16777216, 16777217, 2147483647, -2147483648, 100, -100, 1 << 30 };
        run_source<int>("i32", in, DESTS(i32));
    }
    {
        using STY_T = long long;
        std::vector<long long> in = {
            0LL, 1LL, -1LL, 2LL, -2LL, 127LL, -128LL, 255LL, 256LL,
            16777216LL, 16777217LL, 9007199254740992LL, 9007199254740993LL,
            2147483647LL, -2147483648LL, 9223372036854775807LL,
            (-9223372036854775807LL - 1LL), 1000000000000LL, -1000000000000LL };
        run_source<long long>("i64", in, DESTS(i64));
    }

    // Independent CPU static_cast reference for well-defined arithmetic pairs.
    printf("\n== CPU static_cast reference (arithmetic pairs) ==\n");
    {
        std::vector<signed char> i8v(256);
        for (int c = 0; c < 256; ++c) i8v[c] = (signed char)c;
        ref_check<signed char>("i8->i32", i8v, rung_i8_i32, ref_i8_i32_h);
        ref_check<signed char>("i8->f32", i8v, rung_i8_f32, ref_i8_f32_h);
        std::vector<unsigned char> u8v(256);
        for (int c = 0; c < 256; ++c) u8v[c] = (unsigned char)c;
        ref_check<unsigned char>("u8->f32", u8v, rung_u8_f32, ref_u8_f32_h);
    }
    {
        std::vector<int> i32v = { 0, 1, -1, 127, -128, 255, 1000, -1000, 65536,
                                  16777216, 2147483647, -2147483648, 1 << 30 };
        ref_check<int>("i32->i64", i32v, rung_i32_i64, ref_i32_i64_h);
        ref_check<int>("i32->f64", i32v, rung_i32_f64, ref_i32_f64_h);
    }
    {
        // f64->f32 narrowing over finite, in-range doubles (round-to-nearest is
        // identical on host and device for IEEE-754 round-half-even).
        std::vector<double> f64v = { 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -2.5, 123.456,
                                     -123.456, 100.0, 255.0, 3.5e38, -3.5e38, 1e-40, 42.0 };
        ref_check<double>("f64->f32", f64v, rung_f64_f32, ref_f64_f32_h);
    }

    printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "PASSED", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
