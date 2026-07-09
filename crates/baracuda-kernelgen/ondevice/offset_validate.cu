// On-device validation of the BASE_OFFSET SLICE operand (baracuda-kernelgen
// post-ramp increment): a per-operand runtime base ELEMENT offset added to an
// operand's base pointer at kernel entry (`long long off{i}` launch arg, element
// units) — a runtime launch-arg slice, NOT a stride View, NOT an index gather.
//
// Cases (device-launched EXCEPT case 7, equivalence-covered by case 10, and
// case 9, a host-side compile-time-backed symbol assertion — counted separately
// in the tally, the sort_validate labeling precedent):
//   1. Offset-0 byte-identity : the `_off1` rope-even kernel launched with off1=0
//      memcmp'd against the no-offset rope-even kernel on the same data (the
//      additive/identity guarantee — off=0 on the Strided kernel is byte-exact).
//   2. Strided + offset vs CPU oracle : addoff (out[i] = in0[off0+i] + in1[i]),
//      f32 and f64, small offsets {0,1,3,17} + high-edge.
//   3. Offset on a fully-broadcast (hoisted) operand : bcastoff (out[i] =
//      in0[off0] + in1[i]) — the bump lands on the hoisted `in0[0]`.
//   4. Offset + Permute view : permoff (out[r,c] = x[off0 + r + c*R]) — the offset
//      is the view ORIGIN, applied before the swapped-stride walk.
//   5. Offset + gather (Clamp) : gathoff (offset on the gathered DATA operand) and
//      gathoffi (offset on the INDEX operand) — dual-pointer independence.
//   6. Offset + scatter (Assign, Skip) : scatoff (offsets on the VALUE input and
//      the OUTPUT; unique target indices — the documented Assign precondition).
//   7. Interleave pair (rope precursor) : equivalence-covered by case 10 — the two
//      stride-2 launches (out origins 0 and 1) fully cover the zero-initialized y
//      buffer, memcmp'd whole-buffer vs the bespoke single-launch output (any gap
//      or overlap would memcmp-fail against the fully-written bespoke buffer).
//   8. High-edge memcheck : off + extent == buffer end (run under compute-sanitizer
//      memcheck/racecheck/synccheck/initcheck; the load-bearing OOB proof).
//   9. Gate-accept (HOST-SIDE, compile-time-backed — not a device cell): the
//      `_strided_..._off0` addoff symbol is stringified from the SAME token whose
//      address must link in this TU, proving the schedule gate chose Strided+offset
//      for a cell that would otherwise be `_co_v4`. The runtime strstr cannot fail
//      unless the naming scheme changes; the load-bearing gate pin is the Rust-side
//      G4 test (`runtime_offset_forces_strided_on_a_would_be_vectorized_cell`).
//  10. Rope E2E (the acceptance gate) : the two-launch rope pair decomposition
//      (even `_off1` + odd `_off1o`) memcmp'd BIT-EXACT against the bespoke
//      `launch_rope_apply_fp<float>` (`baracuda_attention.cuh`) on probe-headed x
//      (the house 25 special f32 encodings injected at the head, then uniform
//      random), plus a GB/s bench.
//  11. Multi-offset frozen ABI : add3off (`_off02o` — Runtime inputs 0 AND 2 plus a
//      Runtime output; out[offo+i] = in0[off0+i] + in1[i] + in2[off2+i]) vs CPU
//      oracle, ascending off args, three bumps; the untouched out head [0,offo)
//      stays zero (the output bump proof). Chained ADDs — never FMA-contracted, so
//      the CPU oracle is bit-deterministic.
//
// OOB safety is a CALLER PRECONDITION (the k/n_out trust model, like gather/scatter
// gext/sext): off + max-declared-extent address must be in-bounds — no bound is
// emitted, no address is formed before the bump. Case 8 proves it holds at the edge.
// Only off >= 0 is validated (negative offsets are not forbidden by the emitted
// code but are unvalidated — see the kernelgen de-scopes).
//
// GPU/toolchain: RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc.
//
// Regeneration (from a VS dev shell) — dump the generated oracle .cu, then build:
//   OFFSET_OUT=<outdir> cargo test -p baracuda-kernelgen dump_offset_sources -- --ignored --nocapture
//   OFFSET_OUT=<outdir> cargo test -p baracuda-kernelgen dump_rope_pair_sources -- --ignored --nocapture
//   cp crates/baracuda-kernelgen/ondevice/offset_validate.cu <outdir>/
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//        -I C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include \
//        -Xcompiler "/Zc:preprocessor /std:c++17" \
//        <outdir>/offset_validate.cu -o offset_validate
//   ./offset_validate         (full)   |   ./offset_validate san   (small, sanitizers)
//   compute-sanitizer --tool memcheck   ./offset_validate san
//   compute-sanitizer --tool racecheck  ./offset_validate san
//   compute-sanitizer --tool initcheck  ./offset_validate san
//   compute-sanitizer --tool synccheck  ./offset_validate san

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// GENERATED offset kernels (the semantics oracles).
// ABI (frozen order, all `long long`): input pointers, out pointer, shape{d},
// per-input strides s{i}_{d}, output strides so_{d}, [gext | sext], off{i} for
// each Runtime input ascending, offo if the output is Runtime, n, then float
// params. `off*` are element units, bumped onto the base pointer at kernel entry.
#include "baracuda_gen_addoff_f32_strided_r1_off0.cu"
#include "baracuda_gen_addoff_f64_strided_r1_off0.cu"
#include "baracuda_gen_add3off_f32_strided_r1_off02o.cu"
#include "baracuda_gen_bcastoff_f32_strided_r1_off0.cu"
#include "baracuda_gen_permoff_f32_strided_r2_off0.cu"
#include "baracuda_gen_gathoff_f32_i32_strided_r1_off0.cu"
#include "baracuda_gen_gathoffi_f32_i32_strided_r1_off1.cu"
#include "baracuda_gen_scatoff_f32_i32_strided_r1_off0o.cu"
// GENERATED rope pair decomposition (even/odd + the no-offset even baseline).
#include "baracuda_gen_rope_even_f32_strided_r3_off1.cu"
#include "baracuda_gen_rope_even_f32_strided_r3.cu"
#include "baracuda_gen_rope_odd_f32_strided_r3_off1o.cu"

// BESPOKE rope apply (templated __host__ launcher launch_rope_apply_fp<float>).
#include "baracuda_attention.cuh"

static int fails = 0;
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fails++; } } while (0)

static void pass(const char* label) { printf("PASS %-52s\n", label); }
static void fail(const char* label, const char* why) { printf("FAIL %-52s %s\n", label, why); fails++; }

// memcmp two device buffers of `bytes` bytes; report the first mismatch element.
static bool diff_bytes(const char* label, const void* d_a, const void* d_b, size_t bytes, size_t esz) {
    std::vector<uint8_t> a(bytes), b(bytes);
    CHECK(cudaMemcpy(a.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    if (memcmp(a.data(), b.data(), bytes) == 0) { pass(label); return true; }
    for (size_t i = 0; i < bytes; ++i) if (a[i] != b[i]) {
        printf("FAIL %-52s first mismatch elem %zu (byte %zu): 0x%02x vs 0x%02x\n",
               label, i / esz, i, a[i], b[i]);
        break;
    }
    fails++;
    return false;
}

static void* dev(const void* h, size_t bytes) { void* d = nullptr; CHECK(cudaMalloc(&d, bytes)); CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice)); return d; }

static uint64_t xs = 0x9e3779b97f4a7c15ull;
static uint32_t xrand() { xs ^= xs << 13; xs ^= xs >> 7; xs ^= xs << 17; return (uint32_t)(xs >> 32); }
static float frand() { return (float)((int)(xrand() % 20001) - 10000) * 0.001f; }

// The house f32 probe set (relu_propagating_validate.cu precedent): 25 special
// encodings — signed zeros/ones/infs, qNaN/sNaN payloads, subnormal extremes,
// normal extremes, ±pi. Injected at the HEAD of rope's x so the bit-exactness
// memcmp covers non-finite propagation through the contraction-audited FMA path,
// not just the finite frand() range.
static const uint32_t kProbeU32[] = {
    0x00000000u, 0x80000000u,              // +0, -0
    0x3f800000u, 0xbf800000u,              // +1, -1
    0x7f800000u, 0xff800000u,              // +inf, -inf
    0x7fc00000u, 0xffc00000u,              // qNaN, -qNaN
    0x7fc00001u, 0x7fdead01u, 0x7ff00000u, // qNaN payloads (three)
    0x7f800001u, 0x7fbfffffu,              // sNaN payloads (two)
    0xffc0beefu, 0xffbfffffu,              // negative NaN payloads (two)
    0x00000001u, 0x80000001u,              // +/- smallest subnormal
    0x007fffffu, 0x807fffffu,              // +/- largest subnormal
    0x00800000u, 0x80800000u,              // +/- smallest normal
    0x7f7fffffu, 0xff7fffffu,              // +/- largest finite
    0x40490fdbu, 0xc0490fdbu,              // +/- pi
};
static void probe_head(std::vector<float>& v) {
    size_t k = sizeof kProbeU32 / sizeof kProbeU32[0];
    if (k > v.size()) k = v.size();
    memcpy(v.data(), kProbeU32, k * sizeof(float));
}

static int grid_for(long long n) { long long b = (n + 255) / 256; if (b > 65535) b = 65535; if (b < 1) b = 1; return (int)b; }

// -------------------------------------------------------------------------
// Case 2: strided + offset vs CPU oracle (out[i] = in0[off0 + i] + in1[i]).
// -------------------------------------------------------------------------
template <typename T, typename Launch>
static void case_addoff(const char* tag, Launch launch, long long N, long long off0) {
    // in0 needs N + off0 elements (the offset window is a caller precondition).
    std::vector<T> in0((size_t)(N + off0)), in1((size_t)N), gy((size_t)N, (T)0);
    for (auto& v : in0) v = (T)frand();
    for (auto& v : in1) v = (T)frand();
    T* d0 = (T*)dev(in0.data(), in0.size() * sizeof(T));
    T* d1 = (T*)dev(in1.data(), in1.size() * sizeof(T));
    T* dy = (T*)dev(gy.data(), gy.size() * sizeof(T));
    launch(d0, d1, dy, /*shape0*/ N, /*s0_0*/ (long long)1, /*s1_0*/ (long long)1,
           /*so_0*/ (long long)1, off0, /*n*/ N);
    CHECK(cudaDeviceSynchronize());
    std::vector<T> got((size_t)N);
    CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(T), cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long i = 0; i < N; ++i) {
        T oracle = (T)(in0[(size_t)(off0 + i)] + in1[(size_t)i]);
        if (memcmp(&got[(size_t)i], &oracle, sizeof(T)) != 0) { bad = i; break; }
    }
    char lbl[96]; snprintf(lbl, sizeof lbl, "%s off=%lld", tag, off0);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(d0); cudaFree(d1); cudaFree(dy);
}

// -------------------------------------------------------------------------
// Case 11: MULTI-offset frozen ABI (`_off02o` — Runtime inputs 0 and 2 + Runtime
// output). out[offo+i] = in0[off0+i] + in1[i] + in2[off2+i]; ascending off args
// (off0, off2, offo) per the frozen ABI; three entry bumps. Chained ADDs (both
// sides associate (in0+in1)+in2) — FADD is never FMA-contracted, so the CPU
// oracle is bit-deterministic. The untouched out head [0, offo) must stay zero
// (the output-bump proof: a mis-bumped out would write there / skip the tail).
// -------------------------------------------------------------------------
static void case_add3off(long long N, long long off0, long long off2, long long offo) {
    std::vector<float> in0((size_t)(N + off0)), in1((size_t)N), in2((size_t)(N + off2));
    std::vector<float> gy((size_t)(N + offo), 0.0f);
    for (auto& v : in0) v = frand();
    for (auto& v : in1) v = frand();
    for (auto& v : in2) v = frand();
    float* d0 = (float*)dev(in0.data(), in0.size() * 4);
    float* d1 = (float*)dev(in1.data(), in1.size() * 4);
    float* d2 = (float*)dev(in2.data(), in2.size() * 4);
    float* dy = (float*)dev(gy.data(), gy.size() * 4);
    baracuda_gen_add3off_f32_strided_r1_off02o<<<grid_for(N), 256>>>(
        d0, d1, d2, dy, /*shape0*/ N, /*s0_0*/ 1, /*s1_0*/ 1, /*s2_0*/ 1,
        /*so_0*/ 1, off0, off2, offo, N);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)(N + offo));
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long i = 0; i < offo && bad < 0; ++i)
        if (got[(size_t)i] != 0.0f) bad = i; // head must be untouched
    for (long long i = 0; i < N && bad < 0; ++i) {
        float o = (in0[(size_t)(off0 + i)] + in1[(size_t)i]) + in2[(size_t)(off2 + i)];
        if (memcmp(&got[(size_t)(offo + i)], &o, 4) != 0) bad = offo + i;
    }
    char lbl[96]; snprintf(lbl, sizeof lbl, "add3off multi off0=%lld off2=%lld offo=%lld", off0, off2, offo);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(d0); cudaFree(d1); cudaFree(d2); cudaFree(dy);
}

// -------------------------------------------------------------------------
// Case 3: offset on a fully-broadcast operand (out[i] = in0[off0] + in1[i]).
// The generated kernel hoists `h0 = in0[0]` AFTER the bump ⇒ reads element off0.
// -------------------------------------------------------------------------
static void case_bcastoff(long long N, long long off0) {
    std::vector<float> in0((size_t)(off0 + 1)), in1((size_t)N), gy((size_t)N, 0.0f);
    for (auto& v : in0) v = frand();
    for (auto& v : in1) v = frand();
    float* d0 = (float*)dev(in0.data(), in0.size() * 4);
    float* d1 = (float*)dev(in1.data(), in1.size() * 4);
    float* dy = (float*)dev(gy.data(), gy.size() * 4);
    // in0 fully broadcast ⇒ stride 0; the bump makes the hoisted load read in0[off0].
    baracuda_gen_bcastoff_f32_strided_r1_off0<<<grid_for(N), 256>>>(
        d0, d1, dy, /*shape0*/ N, /*s0_0*/ 0, /*s1_0*/ 1, /*so_0*/ 1, off0, N);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)N);
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long i = 0; i < N; ++i) { float o = in0[(size_t)off0] + in1[(size_t)i]; if (got[(size_t)i] != o) { bad = i; break; } }
    char lbl[64]; snprintf(lbl, sizeof lbl, "bcast-hoist off=%lld", off0);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(d0); cudaFree(d1); cudaFree(dy);
}

// -------------------------------------------------------------------------
// Rope pair decomposition launchers (generated even/odd) over [BH,S,P], P=D/2.
// Strides: x/y stride-2 innermost (S*D, D, 2); cos/sin broadcast axis 0 (0, P, 1).
// -------------------------------------------------------------------------
static void launch_rope_gen(const float* x, const float* cos_t, const float* sin_t, float* y,
                            long long BH, long long S, long long D, bool zero_first) {
    const long long P = D / 2, SD = S * D, n = BH * S * P;
    if (zero_first) CHECK(cudaMemset(y, 0, (size_t)BH * SD * 4));
    // even: in0=x_e(off0), in1=x_o(off1=1), in2=cos, in3=sin, out=y_e(off0). off1=1.
    baracuda_gen_rope_even_f32_strided_r3_off1<<<grid_for(n), 256>>>(
        x, x, cos_t, sin_t, y,
        BH, S, P,
        SD, D, 2,   // s0 (x_e)
        SD, D, 2,   // s1 (x_o)
        0, P, 1,    // s2 (cos, broadcast axis0)
        0, P, 1,    // s3 (sin)
        SD, D, 2,   // so (y_e)
        /*off1*/ 1, n);
    // odd: body x_o*cos + x_e*sin; in1=x_o(off1=1), out=y_o(offo=1).
    baracuda_gen_rope_odd_f32_strided_r3_off1o<<<grid_for(n), 256>>>(
        x, x, cos_t, sin_t, y,
        BH, S, P,
        SD, D, 2, SD, D, 2, 0, P, 1, 0, P, 1, SD, D, 2,
        /*off1*/ 1, /*offo*/ 1, n);
}

// Rope E2E: generated pair vs bespoke launch_rope_apply_fp<float>, bit-exact.
static void case_rope_e2e(const char* tag, long long B, long long H, long long S, long long D) {
    const long long BH = B * H, P = D / 2, SD = S * D, total = BH * SD;
    std::vector<float> x((size_t)total), cos_t((size_t)(S * P)), sin_t((size_t)(S * P));
    for (auto& v : x) v = frand();
    probe_head(x); // 25 special encodings at the head (NaN/inf/subnormal coverage)
    // Realistic cos/sin: theta = pos * base^(-2*pair/D), base=10000 (shared table).
    for (long long s = 0; s < S; ++s) for (long long p = 0; p < P; ++p) {
        double freq = pow(10000.0, -(double)(2 * p) / (double)D);
        double th = (double)s * freq;
        cos_t[(size_t)(s * P + p)] = (float)cos(th);
        sin_t[(size_t)(s * P + p)] = (float)sin(th);
    }
    float* dx  = (float*)dev(x.data(), x.size() * 4);
    float* dc  = (float*)dev(cos_t.data(), cos_t.size() * 4);
    float* ds  = (float*)dev(sin_t.data(), sin_t.size() * 4);
    float* dyg = nullptr; CHECK(cudaMalloc(&dyg, (size_t)total * 4)); // generated
    float* dyb = nullptr; CHECK(cudaMalloc(&dyb, (size_t)total * 4)); // bespoke
    // Generated pair (two launches).
    launch_rope_gen(dx, dc, ds, dyg, BH, S, D, /*zero_first*/ true);
    CHECK(cudaDeviceSynchronize());
    // Bespoke apply (single launch): td = S*D, d = D, stride_b = 0 (shared table).
    baracuda::attention::launch_rope_apply_fp<float>(dx, dc, ds, dyb,
        (int32_t)BH, (int32_t)SD, (int32_t)D, /*stride_b*/ 0, 0);
    CHECK(cudaDeviceSynchronize());
    char lbl[96]; snprintf(lbl, sizeof lbl, "rope E2E %s (B%lld H%lld S%lld D%lld)", tag, B, H, S, D);
    diff_bytes(lbl, dyg, dyb, (size_t)total * 4, 4);
    cudaFree(dx); cudaFree(dc); cudaFree(ds); cudaFree(dyg); cudaFree(dyb);
}

static void case_rope_bench(long long B, long long H, long long S, long long D) {
    const long long BH = B * H, SD = S * D, total = BH * SD;
    std::vector<float> x((size_t)total), cos_t((size_t)(S * (D / 2))), sin_t((size_t)(S * (D / 2)));
    for (auto& v : x) v = frand();
    for (auto& v : cos_t) v = frand();
    for (auto& v : sin_t) v = frand();
    float* dx = (float*)dev(x.data(), x.size() * 4);
    float* dc = (float*)dev(cos_t.data(), cos_t.size() * 4);
    float* ds = (float*)dev(sin_t.data(), sin_t.size() * 4);
    float* dyg = nullptr; CHECK(cudaMalloc(&dyg, (size_t)total * 4));
    float* dyb = nullptr; CHECK(cudaMalloc(&dyb, (size_t)total * 4));
    auto timeit = [&](auto fn) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e); return ms / 20;
    };
    float tg = timeit([&] { launch_rope_gen(dx, dc, ds, dyg, BH, S, D, false); });
    float tb = timeit([&] { baracuda::attention::launch_rope_apply_fp<float>(
        dx, dc, ds, dyb, (int32_t)BH, (int32_t)SD, (int32_t)D, 0, 0); });
    double gb = (double)total * 4.0 * 2 / 1e9;     // one read + one write per element
    double ge = (double)total / 1e9;
    printf("[bench] rope apply B%lld H%lld S%lld D%lld (%lld elems): "
           "generated(2 launches) %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
           "bespoke apply %.3f ms (%.2f Gelem/s, %.1f GB/s) | generated %.2fx bespoke\n",
           B, H, S, D, total, tg, ge / (tg / 1000), gb / (tg / 1000),
           tb, ge / (tb / 1000), gb / (tb / 1000), tb / tg);
    cudaFree(dx); cudaFree(dc); cudaFree(ds); cudaFree(dyg); cudaFree(dyb);
}

// -------------------------------------------------------------------------
// Case 4: offset + Permute view. Iteration [R,C]; in0 read through Permute{1,0}
// with declared strides (s0_0=R, s0_1=1) => o0 = c0*s0_1 + c1*s0_0 = r + c*R.
// Oracle: out[r*C + c] = x[off0 + r + c*R]; x sized off0 + R*C.
// -------------------------------------------------------------------------
static void case_permoff(long long R, long long C, long long off0) {
    const long long n = R * C;
    std::vector<float> x((size_t)(off0 + n)), gy((size_t)n, 0.0f);
    for (auto& v : x) v = frand();
    float* dx = (float*)dev(x.data(), x.size() * 4);
    float* dy = (float*)dev(gy.data(), gy.size() * 4);
    baracuda_gen_permoff_f32_strided_r2_off0<<<grid_for(n), 256>>>(
        dx, dy, /*shape0*/ R, /*shape1*/ C, /*s0_0*/ R, /*s0_1*/ 1,
        /*so_0*/ C, /*so_1*/ 1, off0, n);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)n);
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long r = 0; r < R && bad < 0; ++r)
        for (long long c = 0; c < C; ++c) {
            float o = x[(size_t)(off0 + r + c * R)];
            if (got[(size_t)(r * C + c)] != o) { bad = r * C + c; break; }
        }
    char lbl[64]; snprintf(lbl, sizeof lbl, "permute-view off=%lld", off0);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(dx); cudaFree(dy);
}

// -------------------------------------------------------------------------
// Case 5: offset + gather (Clamp). gathoff: offset on the gathered DATA operand
// (out[i] = data[off0 + clamp(idx[i])]); gathoffi: offset on the INDEX operand
// (out[i] = data[clamp(idx[off1 + i])]). Dual-pointer independence.
// -------------------------------------------------------------------------
static void case_gathoff(long long N, long long gext, long long off) {
    std::vector<float> data((size_t)(off + gext));
    std::vector<int> idx((size_t)N);
    std::vector<float> gy((size_t)N, 0.0f);
    for (auto& v : data) v = frand();
    for (long long i = 0; i < N; ++i) idx[(size_t)i] = (int)(xrand() % (uint32_t)(gext + 4)) - 2; // includes OOB
    float* dd = (float*)dev(data.data(), data.size() * 4);
    int* di = (int*)dev(idx.data(), idx.size() * 4);
    float* dy = (float*)dev(gy.data(), gy.size() * 4);
    baracuda_gen_gathoff_f32_i32_strided_r1_off0<<<grid_for(N), 256>>>(
        dd, di, dy, /*shape0*/ N, /*s0_0*/ 1, /*s1_0*/ 1, /*so_0*/ 1, gext, off, N);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)N);
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long i = 0; i < N; ++i) {
        long long q = idx[(size_t)i]; q = q < 0 ? 0 : (q >= gext ? gext - 1 : q);
        if (got[(size_t)i] != data[(size_t)(off + q)]) { bad = i; break; }
    }
    char lbl[64]; snprintf(lbl, sizeof lbl, "gather+data-off off=%lld", off);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(dd); cudaFree(di); cudaFree(dy);
}

static void case_gathoffi(long long N, long long gext, long long off) {
    std::vector<float> data((size_t)gext);
    std::vector<int> idx((size_t)(off + N));
    std::vector<float> gy((size_t)N, 0.0f);
    for (auto& v : data) v = frand();
    for (auto& v : idx) v = (int)(xrand() % (uint32_t)(gext + 4)) - 2;
    float* dd = (float*)dev(data.data(), data.size() * 4);
    int* di = (int*)dev(idx.data(), idx.size() * 4);
    float* dy = (float*)dev(gy.data(), gy.size() * 4);
    baracuda_gen_gathoffi_f32_i32_strided_r1_off1<<<grid_for(N), 256>>>(
        dd, di, dy, /*shape0*/ N, /*s0_0*/ 1, /*s1_0*/ 1, /*so_0*/ 1, gext, off, N);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)N);
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    long long bad = -1;
    for (long long i = 0; i < N; ++i) {
        long long q = idx[(size_t)(off + i)]; q = q < 0 ? 0 : (q >= gext ? gext - 1 : q);
        if (got[(size_t)i] != data[(size_t)q]) { bad = i; break; }
    }
    char lbl[64]; snprintf(lbl, sizeof lbl, "gather+index-off off=%lld", off);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(dd); cudaFree(di); cudaFree(dy);
}

// -------------------------------------------------------------------------
// Case 6: offset + scatter (Assign, Skip). out[offo + idx[i]] = updates[off0 + i]
// for in-range idx; OOB targets are skipped. Unique target indices (a rotated
// permutation) — the documented Assign determinism precondition.
// -------------------------------------------------------------------------
static void case_scatoff(long long N, long long sext, long long off0, long long offo) {
    std::vector<float> upd((size_t)(off0 + N));
    std::vector<int> idx((size_t)N);
    std::vector<float> init((size_t)(offo + sext));
    for (auto& v : upd) v = frand();
    for (auto& v : init) v = -1234.5f;
    // Unique targets: a rotated permutation with a few OOB stubs (skipped).
    for (long long i = 0; i < N; ++i) idx[(size_t)i] = (int)((i * 7 + 3) % sext);
    if (N >= 2) { idx[0] = -1; idx[1] = (int)sext; } // OOB: skipped
    float* du = (float*)dev(upd.data(), upd.size() * 4);
    int* di = (int*)dev(idx.data(), idx.size() * 4);
    float* dy = (float*)dev(init.data(), init.size() * 4);
    baracuda_gen_scatoff_f32_i32_strided_r1_off0o<<<grid_for(N), 256>>>(
        du, di, dy, /*shape0*/ N, /*s0_0*/ 1, /*s1_0*/ 1, /*so_0*/ 1, sext, off0, offo, N);
    CHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)(offo + sext));
    CHECK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    // CPU oracle over the initialized destination.
    std::vector<float> want = init;
    for (long long i = 0; i < N; ++i) {
        int q = idx[(size_t)i];
        if (q >= 0 && q < sext) want[(size_t)(offo + q)] = upd[(size_t)(off0 + i)];
    }
    long long bad = -1;
    for (long long i = 0; i < offo + sext; ++i)
        if (got[(size_t)i] != want[(size_t)i]) { bad = i; break; }
    char lbl[72]; snprintf(lbl, sizeof lbl, "scatter+off in=%lld out=%lld", off0, offo);
    if (bad < 0) pass(lbl); else { char why[64]; snprintf(why, sizeof why, "elem %lld", bad); fail(lbl, why); }
    cudaFree(du); cudaFree(di); cudaFree(dy);
}

// Case 1: offset-0 byte-identity of the rope-even `_off1` kernel (off1=0) vs the
// no-offset rope-even kernel, on the same data.
static void case_offset0_identity(long long BH, long long S, long long D) {
    const long long P = D / 2, SD = S * D, n = BH * S * P;
    std::vector<float> x((size_t)BH * SD), cs((size_t)(S * P));
    for (auto& v : x) v = frand();
    for (auto& v : cs) v = frand();
    float* dx = (float*)dev(x.data(), x.size() * 4);
    float* dc = (float*)dev(cs.data(), cs.size() * 4);
    float* dyA = nullptr; CHECK(cudaMalloc(&dyA, (size_t)BH * SD * 4)); CHECK(cudaMemset(dyA, 0, (size_t)BH * SD * 4));
    float* dyB = nullptr; CHECK(cudaMalloc(&dyB, (size_t)BH * SD * 4)); CHECK(cudaMemset(dyB, 0, (size_t)BH * SD * 4));
    // A: the `_off1` kernel with off1 = 0 (identity — no shift, x_o reads x_e's lane).
    baracuda_gen_rope_even_f32_strided_r3_off1<<<grid_for(n), 256>>>(
        dx, dx, dc, dc, dyA, BH, S, P, SD, D, 2, SD, D, 2, 0, P, 1, 0, P, 1, SD, D, 2, /*off1*/ 0, n);
    // B: the no-offset kernel (x_o == x_e implicitly, off1 absent).
    baracuda_gen_rope_even_f32_strided_r3<<<grid_for(n), 256>>>(
        dx, dx, dc, dc, dyB, BH, S, P, SD, D, 2, SD, D, 2, 0, P, 1, 0, P, 1, SD, D, 2, n);
    CHECK(cudaDeviceSynchronize());
    diff_bytes("offset-0 identity (rope-even off1=0 == no-off)", dyA, dyB, (size_t)BH * SD * 4, 4);
    cudaFree(dx); cudaFree(dc); cudaFree(dyA); cudaFree(dyB);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== offset_validate (BASE_OFFSET SLICE) ==\n");

    // Case 9 (gate-accept — HOST-SIDE, compile-time-backed; NOT a device cell,
    // counted separately in the tally): the string is STRINGIFIED from the same
    // token whose address must link below, so it cannot go stale relative to the
    // generated symbol; the compile itself proves the schedule gate emitted
    // `_strided_..._off0` for a cell that would otherwise be `_co_v4`. The runtime
    // strstr can only fail if the generated NAMING SCHEME changes — the
    // load-bearing gate pin is the Rust-side G4 test
    // (`runtime_offset_forces_strided_on_a_would_be_vectorized_cell`).
    {
#define GATE_STR1(x) #x
#define GATE_STR(x) GATE_STR1(x)
#define GATE_SYM baracuda_gen_addoff_f32_strided_r1_off0
        auto* gate_fp = &GATE_SYM; (void)gate_fp;   // must link: the symbol exists
        const char* sym = GATE_STR(GATE_SYM);       // stringified from the SAME token
        bool ok = strstr(sym, "_strided_") && strstr(sym, "_off");
        if (ok) pass("gate-accept [host-side]: offset forces _strided_..._off symbol");
        else fail("gate-accept [host-side]", "symbol lacks _strided/_off");
    }

    if (san) {
        // Small shapes for compute-sanitizer (memcheck/racecheck/synccheck/initcheck).
        case_addoff<float>("addoff-f32", [](float* d0, float* d1, float* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
            baracuda_gen_addoff_f32_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 4096, 17);
        case_addoff<double>("addoff-f64", [](double* d0, double* d1, double* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
            baracuda_gen_addoff_f64_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 4096, 3);
        // Case 8 high-edge: off + N == buffer end (in0 sized exactly N+off).
        case_addoff<float>("addoff-f32 HIGH-EDGE", [](float* d0, float* d1, float* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
            baracuda_gen_addoff_f32_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 1024, 7);
        case_add3off(2048, 7, 3, 5);
        case_bcastoff(2048, 5);
        case_permoff(16, 32, 7);
        case_gathoff(1024, 256, 17);
        case_gathoffi(1024, 256, 9);
        case_scatoff(512, 640, 11, 5);
        case_offset0_identity(2, 4, 16);
        case_rope_e2e("san", 1, 2, 8, 16);
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    // Case 1: offset-0 identity.
    case_offset0_identity(32, 64, 128);
    // Case 2: strided + offset CPU oracle, small offsets + high-edge.
    for (long long off : {0LL, 1LL, 3LL, 17LL}) {
        case_addoff<float>("addoff-f32", [](float* d0, float* d1, float* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
            baracuda_gen_addoff_f32_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 1 << 20, off);
        case_addoff<double>("addoff-f64", [](double* d0, double* d1, double* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
            baracuda_gen_addoff_f64_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 1 << 20, off);
    }
    // Case 8: high-edge (off + N == buffer end) at a realistic size.
    case_addoff<float>("addoff-f32 HIGH-EDGE", [](float* d0, float* d1, float* dy, long long sh, long long a, long long b, long long c, long long o, long long n) {
        baracuda_gen_addoff_f32_strided_r1_off0<<<grid_for(n), 256>>>(d0, d1, dy, sh, a, b, c, o, n); }, 1 << 20, 12345);
    // Case 11: multi-offset frozen ABI (`_off02o`, ascending args, three bumps).
    case_add3off(1 << 20, 12345, 777, 4097);
    // Case 3: broadcast-hoist offset.
    case_bcastoff(1 << 20, 9999);
    // Case 4: offset + Permute view.
    case_permoff(128, 256, 4097);
    // Case 5: offset + gather (data-side, then index-side).
    case_gathoff(1 << 16, 4096, 12345);
    case_gathoffi(1 << 16, 4096, 777);
    // Case 6: offset + scatter (Assign; unique targets + OOB stubs skipped).
    case_scatoff(1 << 14, 1 << 15, 333, 21);
    // Case 10: rope E2E (bit-exact) — a power-of-two shape + a non-power shape.
    case_rope_e2e("pow2", 4, 8, 1024, 128);
    case_rope_e2e("nonpow", 3, 5, 257, 64);
    // Bench (GB/s) generated pair vs bespoke apply.
    case_rope_bench(4, 8, 1024, 128);

    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
