// F64 HETERO MULTI-OUTPUT (dropout-class) on-device validation — the F64-PARAM-
// CHANNEL increment (increment 12). The scalar-param launch channel is now
// dtype-aware: the two runtime params (keep_prob=p0, scale=p1) are declared
// `double p0, double p1` (not `float`), and the fused dropout kernel computes in
// IEEE double:
//   y[i]    = x[i] * (rand[i] < keep_prob ? scale : 0)     (F64 value output)
//   mask[i] = (rand[i] < keep_prob)                         (U8 keep-mask output)
// written by ONE kernel from a shared body-DAG (the `rand < keep_prob` compare
// hoisted ONCE into a `double tmp0`, consumed by BOTH stores).
//
// THE ACCEPTANCE GATE (cell 1): PRIMARY oracle = a CPU f64 CLOSED-FORM reference
// (kernel-independent, exact IEEE double), whole-buffer memcmp bit_diff(y)==0 AND
// bit_diff(mask)==0 with the SAME host-filled `double` rand buffer and the SAME
// `double` keep_prob/scale. The generated kernel does `in1[i] < p0` (double<double)
// and one LONE `in0[i] * mult` multiply — the CPU oracle mirrors that arithmetic
// exactly: a lone correctly-rounded double multiply has no FMA contraction (no
// add), so host↔device is bit-identical, `-0.0` from `x*(+0.0)` and NaN-through-
// multiply preserved on both sides.
//
// SECONDARY cross-check (cell 3, -DWITH_BESPOKE) = the bespoke
// `baracuda_kernels_dropout_f64_run` under a documented WIDENING PROTOCOL. The
// bespoke device kernel (`baracuda_random.cuh` dropout_fw_kernel<double,double>)
// hardwires `rand` as `const float*` and `keep_prob` as `float` (= 1.0f - p) EVEN
// at T=double — while the generated kernel loads rand as `double` and compares
// double<double. A naive memcmp would diverge on rows where the float-vs-double
// compare flips near threshold (the parallel to topk's bespoke NaN/tie carve-out).
// To make the memcmp EXACT: fill the generator's f64 rand with the EXACT WIDENING
// of the bespoke's float rand (float→double is exact and order-preserving, so
// `(double)a < (double)b  <=>  a < b`), and pass the generator keep_prob as
// `(double)(1.0f - p)` — widen the FLOAT keep_prob, NOT `1.0 - (double)p` (which
// can flip mask bits near threshold). `scale` is a pure double passthrough in the
// bespoke ABI, so both sides get the identical double scale. Then masks and values
// match bit-for-bit.
//
// Cell 6 is the byte-identical-f32-dropout regression: the shipped f32 kernel
// source is UNCHANGED by this increment (scalar_ctype(F32)=="float" by
// construction), so its device behavior is unchanged — checked here against a CPU
// f32 oracle so the f64 harness is self-contained.
//
// Source-shape / cross-body-CSE facts are pinned by the Rust golden
// `cuda::dropout_hetero_tests::dropout_scalar_hetero_golden_f64`. Generated kernels
// come from the `dump_dropout_sources` test — see the ondevice README
// "dropout f64 (F64-PARAM-CHANNEL increment)" section for the exact regen + compile
// lines. Build with -DWITH_BESPOKE (adds the bespoke f64 instantiation +
// `-I <kernels>/include -Xcompiler "/Zc:preprocessor /std:c++17"`); without it
// cells 1/2/6 run against the CPU oracle only (still the acceptance gate).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// GENERATED f64 dropout kernels (this build's dump_dropout_sources output).
#include "baracuda_gen_dropout_f64_mo2_scalar.cu"
#include "baracuda_gen_dropout_f64_mo2_strided_r2.cu"
// GENERATED f32 dropout kernel (the byte-identical-f32 regression cell).
#include "baracuda_gen_dropout_f32_mo2_scalar.cu"

// BESPOKE dropout_fw — the SECONDARY widening-protocol cross-check target. The f64
// instantiation is SCALE_T=double (which is WHY this increment needs double params);
// rand stays `const float*` and keep_prob stays `float` inside the macro.
#ifdef WITH_BESPOKE
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_random.cuh"
BARACUDA_KERNELS_DROPOUT_INSTANTIATE(f64, double, double)
BARACUDA_KERNELS_DROPOUT_INSTANTIATE(f32, float, float)
#endif

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int fails = 0;

// Double probe classes (the f64 analog of the f32 harness set): +/-0, +/-1,
// +/-inf, qNaN payloads, sNaN payloads, negative-NaN payloads, subnormals, min
// normal, max finite, +/-pi. Tiled across `x` so each class lands in BOTH kept and
// dropped positions of every p in the sweep.
static std::vector<uint64_t> probe_u64() {
    return {
        0x0000000000000000ull, 0x8000000000000000ull, // +0, -0
        0x3ff0000000000000ull, 0xbff0000000000000ull, // +1, -1 (dropped negatives -> -0.0)
        0x7ff0000000000000ull, 0xfff0000000000000ull, // +inf, -inf
        0x7ff8000000000000ull, 0xfff8000000000000ull, // qNaN, -qNaN
        0x7ff8000000000001ull, 0x7ffdead000000001ull, 0x7fff000000000000ull, // qNaN payloads
        0x7ff0000000000001ull, 0x7ff7ffffffffffffull, // sNaN payloads
        0xfff8beef00000000ull, 0xfff7ffffffffffffull, // negative NaN payloads
        0x0000000000000001ull, 0x8000000000000001ull, // +/- smallest subnormal
        0x000fffffffffffffull, 0x800fffffffffffffull, // +/- largest subnormal
        0x0010000000000000ull, 0x8010000000000000ull, // +/- smallest normal
        0x7fefffffffffffffull, 0xffefffffffffffffull, // +/- largest finite
        0x400921fb54442d18ull, 0xc00921fb54442d18ull, // +/-pi
    };
}

static uint64_t g_seed = 0x9e3779b97f4a7c15ull;
static uint64_t xrand() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return g_seed; }
static void reseed() { g_seed = 0x9e3779b97f4a7c15ull; }

// Fill `x` words (double bit patterns) with the tiled probe set followed by an
// xorshift bit sweep — every IEEE class exercised, dense values in between.
static void fill_words(std::vector<uint64_t>& h) {
    auto probes = probe_u64();
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (i < probes.size()) ? probes[i]
             : (i % 3 == 0) ? probes[(xrand() % probes.size())]
                            : xrand();
}
// Fill a DOUBLE rand buffer uniform in [0,1) (never NaN) — the PRIMARY oracle noise.
static void fill_rand_f64(std::vector<double>& h) {
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (double)(xrand() & 0x1FFFFFFFFFFFFFull) / (double)(1ull << 53); // [0,1)
}
// Fill a FLOAT rand buffer uniform in [0,1) (the bespoke's noise; the generator's
// double rand for the secondary cell is the EXACT widening of this).
static void fill_rand_f32(std::vector<float>& h) {
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (float)((uint32_t)(xrand() & 0xFFFFFFu)) / 16777216.0f; // [0,1)
}

static int launch_blocks(long long n) {
    long long b = (n + 255) / 256; if (b < 1) b = 1; if (b > 65535) b = 65535;
    return (int)b;
}

// ============ cell 1: THE ACCEPTANCE GATE — gen vs CPU f64 closed-form ============
static void cell1_primary(bool small_only) {
    struct Shape { long long n; const char* name; };
    const Shape shapes_full[] = { {1,"1"}, {37*53,"37x53"}, {4096,"4096"}, {1000003,"1000003(prime)"} };
    const Shape shapes_san[]  = { {37*53,"37x53"}, {1,"1"} };
    const Shape* shapes = small_only ? shapes_san : shapes_full;
    const int nshapes = small_only ? 2 : 4;
    // keep_prob sweep: 0 (drop all), near-threshold, 0.5, 0.9, 1 (keep all).
    const double ps[] = { 0.0, 0.1, 0.5, 0.9 };

    for (int s = 0; s < nshapes; ++s) {
        long long n = shapes[s].n;
        reseed();
        std::vector<uint64_t> hx((size_t)n);
        std::vector<double> hrand((size_t)n);
        fill_words(hx);
        fill_rand_f64(hrand);

        double *dx, *drand, *dy_gen;
        uint8_t *dm_gen;
        CHECK(cudaMalloc(&dx, n*8));      CHECK(cudaMalloc(&drand, n*8));
        CHECK(cudaMalloc(&dy_gen, n*8));  CHECK(cudaMalloc(&dm_gen, n));
        CHECK(cudaMemcpy(dx, hx.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(drand, hrand.data(), n*8, cudaMemcpyHostToDevice));

        for (double p : ps) {
            const double keep_prob = 1.0 - p;
            const double scale = (p < 1.0) ? 1.0 / (1.0 - p) : 0.0;
            int blocks = launch_blocks(n);
            CHECK(cudaMemset(dy_gen, 0x5B, n*8)); // poison (initcheck + real memcmp target)
            CHECK(cudaMemset(dm_gen, 0x33, n));

            baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>(
                (const double*)dx, (const double*)drand, dy_gen, dm_gen, n, keep_prob, scale);
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            std::vector<uint64_t> ygen((size_t)n);
            std::vector<uint8_t> mgen((size_t)n);
            CHECK(cudaMemcpy(ygen.data(), dy_gen, n*8, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(mgen.data(), dm_gen, n, cudaMemcpyDeviceToHost));

            long long ybd = 0, mbd = 0, negzero = 0;
            for (long long t = 0; t < n; ++t) {
                double x; memcpy(&x, &hx[(size_t)t], 8);
                // CPU f64 closed-form oracle (mirrors the generated kernel exactly).
                bool keep = hrand[(size_t)t] < keep_prob;
                double mult = keep ? scale : 0.0;
                double y = x * mult;              // lone multiply — no FMA, correctly rounded
                uint64_t yb; memcpy(&yb, &y, 8);
                uint8_t mb = keep ? 1u : 0u;
                if (ygen[(size_t)t] != yb) {
                    ybd++;
                    bool in_neg = (hx[(size_t)t] & 0x8000000000000000ull) != 0;
                    if (!keep && in_neg && ((ygen[(size_t)t] | yb) & 0x7fffffffffffffffull) == 0) negzero++;
                }
                if (mgen[(size_t)t] != mb) mbd++;
            }
            bool ok = (ybd == 0) && (mbd == 0);
            if (!ok) fails++;
            printf("[%s] dropout f64 [%12s] p=%.2f : bit_diff(y)=%lld (neg-zero class %lld) bit_diff(mask)=%lld\n",
                   ok ? " ok " : "FAIL", shapes[s].name, p, ybd, negzero, mbd);
        }
        cudaFree(dx); cudaFree(drand); cudaFree(dy_gen); cudaFree(dm_gen);
    }
}

// ============ cell 2: strided address math (both outputs) == scalar ============
static void cell2_strided(long long M, long long N) {
    const long long n = M * N;
    reseed();
    std::vector<uint64_t> hx((size_t)n);
    std::vector<double> hrand((size_t)n);
    fill_words(hx); fill_rand_f64(hrand);
    const double keep_prob = 0.5, scale = 2.0; // p = 0.5

    double *dx, *drand, *dy_sc, *dy_st;
    uint8_t *dm_sc, *dm_st;
    CHECK(cudaMalloc(&dx, n*8)); CHECK(cudaMalloc(&drand, n*8));
    CHECK(cudaMalloc(&dy_sc, n*8)); CHECK(cudaMalloc(&dy_st, n*8));
    CHECK(cudaMalloc(&dm_sc, n)); CHECK(cudaMalloc(&dm_st, n));
    CHECK(cudaMemcpy(dx, hx.data(), n*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand, hrand.data(), n*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dy_sc, 0x5B, n*8)); CHECK(cudaMemset(dy_st, 0x3C, n*8));
    CHECK(cudaMemset(dm_sc, 0x11, n)); CHECK(cudaMemset(dm_st, 0x22, n));
    int blocks = launch_blocks(n);

    // Scalar reference: contiguous [M,N] row-major, both outputs.
    baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>(
        (const double*)dx, (const double*)drand, dy_sc, dm_sc, n, keep_prob, scale);

    // Strided: read x,rand row-major (strides [N,1]); WRITE outputs column-major
    // (strides [1,M]) so out[i,j] lands at j*M + i.
    baracuda_gen_dropout_f64_mo2_strided_r2<<<blocks,256>>>(
        (const double*)dx, (const double*)drand, dy_st, dm_st,
        M, N,           // shape0, shape1
        N, 1,           // s0 (in0 = x): row-major
        N, 1,           // s1 (in1 = rand): row-major
        1, M,           // so0 (out0 = y): column-major
        1, M,           // so1 (out1 = mask): column-major
        n, keep_prob, scale);
    CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> ysc((size_t)n), yst((size_t)n);
    std::vector<uint8_t> msc((size_t)n), mst((size_t)n);
    CHECK(cudaMemcpy(ysc.data(), dy_sc, n*8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(yst.data(), dy_st, n*8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(msc.data(), dm_sc, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mst.data(), dm_st, n, cudaMemcpyDeviceToHost));

    long long ybd = 0, mbd = 0, mdef = 0;
    for (long long i = 0; i < M; ++i)
        for (long long j = 0; j < N; ++j) {
            size_t rc = (size_t)(i * N + j);       // row-major (scalar output)
            size_t cc = (size_t)(j * M + i);       // column-major (strided output)
            if (ysc[rc] != yst[cc]) ybd++;
            if (msc[rc] != mst[cc]) mbd++;
            uint8_t want = (hrand[rc] < keep_prob) ? 1u : 0u;
            if (msc[rc] != want) mdef++;
        }
    bool ok = (ybd == 0) && (mbd == 0) && (mdef == 0);
    if (!ok) fails++;
    printf("[%s] dropout f64 strided==scalar [%lldx%lld] : bit_diff(y)=%lld bit_diff(mask)=%lld mask_def_bad=%lld\n",
           ok ? " ok " : "FAIL", M, N, ybd, mbd, mdef);
    cudaFree(dx); cudaFree(drand); cudaFree(dy_sc); cudaFree(dy_st); cudaFree(dm_sc); cudaFree(dm_st);
}

// ==== cell 3: SECONDARY cross-check — bespoke f64 under the widening protocol ====
#ifdef WITH_BESPOKE
static void cell3_bespoke(bool small_only) {
    struct Shape { long long n; const char* name; };
    const Shape shapes_full[] = { {37*53,"37x53"}, {4096,"4096"}, {1000003,"1000003(prime)"} };
    const Shape shapes_san[]  = { {37*53,"37x53"} };
    const Shape* shapes = small_only ? shapes_san : shapes_full;
    const int nshapes = small_only ? 1 : 3;
    const float ps[] = { 0.1f, 0.5f, 0.9f };

    for (int s = 0; s < nshapes; ++s) {
        long long n = shapes[s].n;
        reseed();
        std::vector<uint64_t> hx((size_t)n);
        std::vector<float> hrand_f32((size_t)n);
        fill_words(hx);
        fill_rand_f32(hrand_f32);
        // The generator's f64 rand = EXACT widening of the bespoke's float rand.
        std::vector<double> hrand_f64((size_t)n);
        for (long long t = 0; t < n; ++t) hrand_f64[(size_t)t] = (double)hrand_f32[(size_t)t];

        double *dx, *drand_gen; float *drand_bes;
        double *dy_gen, *dy_bes; uint8_t *dm_gen, *dm_bes;
        CHECK(cudaMalloc(&dx, n*8));
        CHECK(cudaMalloc(&drand_gen, n*8)); CHECK(cudaMalloc(&drand_bes, n*4));
        CHECK(cudaMalloc(&dy_gen, n*8));    CHECK(cudaMalloc(&dy_bes, n*8));
        CHECK(cudaMalloc(&dm_gen, n));      CHECK(cudaMalloc(&dm_bes, n));
        CHECK(cudaMemcpy(dx, hx.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(drand_gen, hrand_f64.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(drand_bes, hrand_f32.data(), n*4, cudaMemcpyHostToDevice));

        for (float p : ps) {
            // Widening protocol: gen keep_prob = (double)(1.0f - p) — widen the
            // FLOAT keep_prob the bespoke computes internally (NOT 1.0 - (double)p).
            const float keep_prob_f32 = 1.0f - p;
            const double keep_prob_gen = (double)keep_prob_f32;
            const double scale = 1.0 / (1.0 - (double)p); // pure double passthrough (both sides)
            int blocks = launch_blocks(n);
            CHECK(cudaMemset(dy_gen, 0x5B, n*8)); CHECK(cudaMemset(dy_bes, 0xA1, n*8));
            CHECK(cudaMemset(dm_gen, 0x33, n));   CHECK(cudaMemset(dm_bes, 0xCC, n));

            baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>(
                (const double*)dx, (const double*)drand_gen, dy_gen, dm_gen, n, keep_prob_gen, scale);
            CHECK(cudaGetLastError());
            int rc = baracuda_kernels_dropout_f64_run(
                n, p, scale, dx, drand_bes, dy_bes, dm_bes, nullptr, 0, nullptr);
            CHECK(cudaDeviceSynchronize());
            if (rc != 0) { printf("FAIL dropout f64 [%s] p=%.2f: bespoke rc=%d\n", shapes[s].name, p, rc); fails++; }

            std::vector<uint64_t> ygen((size_t)n), ybes((size_t)n);
            std::vector<uint8_t> mgen((size_t)n), mbes((size_t)n);
            CHECK(cudaMemcpy(ygen.data(), dy_gen, n*8, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(ybes.data(), dy_bes, n*8, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(mgen.data(), dm_gen, n, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(mbes.data(), dm_bes, n, cudaMemcpyDeviceToHost));

            long long ybd = 0, mbd = 0;
            for (long long t = 0; t < n; ++t) {
                if (ygen[(size_t)t] != ybes[(size_t)t]) ybd++;
                if (mgen[(size_t)t] != mbes[(size_t)t]) mbd++;
            }
            bool ok = (ybd == 0) && (mbd == 0);
            if (!ok) fails++;
            printf("[%s] dropout f64 gen-vs-bespoke(widened) [%12s] p=%.2f : bit_diff(y)=%lld bit_diff(mask)=%lld\n",
                   ok ? " ok " : "FAIL", shapes[s].name, p, ybd, mbd);
        }
        cudaFree(dx); cudaFree(drand_gen); cudaFree(drand_bes);
        cudaFree(dy_gen); cudaFree(dy_bes); cudaFree(dm_gen); cudaFree(dm_bes);
    }
}
#endif

// ============ cell 4: run-to-run determinism (both outputs) ============
static void cell4_determinism() {
    const long long n = 1000003;
    reseed();
    std::vector<uint64_t> hx((size_t)n);
    std::vector<double> hrand((size_t)n);
    fill_words(hx); fill_rand_f64(hrand);
    const double keep_prob = 0.3, scale = 1.0 / 0.3;
    double *dx, *drand, *dy1, *dy2; uint8_t *dm1, *dm2;
    CHECK(cudaMalloc(&dx, n*8)); CHECK(cudaMalloc(&drand, n*8));
    CHECK(cudaMalloc(&dy1, n*8)); CHECK(cudaMalloc(&dy2, n*8));
    CHECK(cudaMalloc(&dm1, n)); CHECK(cudaMalloc(&dm2, n));
    CHECK(cudaMemcpy(dx, hx.data(), n*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand, hrand.data(), n*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dy1, 0x5B, n*8)); CHECK(cudaMemset(dy2, 0x3C, n*8));
    CHECK(cudaMemset(dm1, 0x44, n)); CHECK(cudaMemset(dm2, 0x77, n));
    int blocks = launch_blocks(n);
    baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>((const double*)dx,(const double*)drand,dy1,dm1,n,keep_prob,scale);
    baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>((const double*)dx,(const double*)drand,dy2,dm2,n,keep_prob,scale);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint64_t> y1((size_t)n), y2((size_t)n);
    std::vector<uint8_t> m1((size_t)n), m2((size_t)n);
    CHECK(cudaMemcpy(y1.data(), dy1, n*8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y2.data(), dy2, n*8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m1.data(), dm1, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m2.data(), dm2, n, cudaMemcpyDeviceToHost));
    bool ok = (memcmp(y1.data(), y2.data(), (size_t)n*8) == 0) && (memcmp(m1.data(), m2.data(), (size_t)n) == 0);
    if (!ok) fails++;
    printf("[%s] run-to-run determinism (two f64 dropout launches, memcmp y AND mask) over %lld elems\n",
           ok ? " ok " : "FAIL", n);
    cudaFree(dx); cudaFree(drand); cudaFree(dy1); cudaFree(dy2); cudaFree(dm1); cudaFree(dm2);
}

// ==== cell 6: byte-identical-f32-dropout regression (the shipped f32 vehicle) ====
// The f32 kernel source is UNCHANGED by the F64-param increment (scalar_ctype(F32)
// =="float" by construction), so its device behavior is unchanged; confirm against
// a CPU f32 closed-form oracle. IMPORTANT NaN asymmetry (empirically confirmed): the
// GPU CANONICALIZES an f32 NaN result of a multiply to qNaN (0x7fc00000), while the
// x86 host PRESERVES the operand NaN payload — a known platform difference, NOT a
// regression — so for a NaN input x we require gen output to be NaN (payload-agnostic)
// rather than bit-identical. (This is exactly why the f64 PRIMARY oracle in cell 1 is
// exact: the GPU PRESERVES f64 NaN payloads, matching x86, so cell 1 passes bit_diff=0
// even on NaN — and why the f32 acceptance historically uses device-to-device bespoke.)
static void cell6_f32_regression() {
    const long long n = 1000003;
    reseed();
    std::vector<uint32_t> hx((size_t)n);
    std::vector<float> hrand((size_t)n);
    { auto probes = std::vector<uint32_t>{
        0x00000000u,0x80000000u,0x3f800000u,0xbf800000u,0x7f800000u,0xff800000u,
        0x7fc00000u,0xffc00000u,0x7fc00001u,0x7f800001u,0x00000001u,0x007fffffu,
        0x00800000u,0x7f7fffffu,0x40490fdbu,0xc0490fdbu };
      for (size_t i=0;i<hx.size();++i) hx[i] = (i<probes.size())?probes[i]:(uint32_t)xrand(); }
    for (size_t i=0;i<hrand.size();++i) hrand[i] = (float)((uint32_t)(xrand()&0xFFFFFFu))/16777216.0f;
    const float ps[] = { 0.0f, 0.5f, 0.9f };
    float *dx,*drand,*dy; uint8_t *dm;
    CHECK(cudaMalloc(&dx,n*4)); CHECK(cudaMalloc(&drand,n*4));
    CHECK(cudaMalloc(&dy,n*4)); CHECK(cudaMalloc(&dm,n));
    CHECK(cudaMemcpy(dx,hx.data(),n*4,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand,hrand.data(),n*4,cudaMemcpyHostToDevice));
    for (float p : ps) {
        const float keep_prob = 1.0f - p;
        const float scale = (p < 1.0f) ? 1.0f/(1.0f-p) : 0.0f;
        int blocks = launch_blocks(n);
        CHECK(cudaMemset(dy,0x5B,n*4)); CHECK(cudaMemset(dm,0x33,n));
        baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>((const float*)dx,(const float*)drand,dy,dm,n,keep_prob,scale);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> ygen((size_t)n); std::vector<uint8_t> mgen((size_t)n);
        CHECK(cudaMemcpy(ygen.data(),dy,n*4,cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(mgen.data(),dm,n,cudaMemcpyDeviceToHost));
        long long ybd=0, mbd=0;
        for (long long t=0;t<n;++t) {
            float x; memcpy(&x,&hx[(size_t)t],4);
            bool keep = hrand[(size_t)t] < keep_prob;
            float y = x * (keep ? scale : 0.0f);
            float ygen_f; memcpy(&ygen_f,&ygen[(size_t)t],4);
            if (y != y) {
                // NaN RESULT (NaN input, or inf*0 for a dropped +/-inf): the GPU
                // canonicalizes the f32 NaN payload, x86 preserves it — require gen
                // NaN (payload-agnostic), not bit-identical.
                if (ygen_f == ygen_f) ybd++;
            } else {
                uint32_t yb; memcpy(&yb,&y,4);
                if (ygen[(size_t)t]!=yb) ybd++;
            }
            if (mgen[(size_t)t]!=(keep?1u:0u)) mbd++;
        }
        bool ok = (ybd==0)&&(mbd==0);
        if (!ok) fails++;
        printf("[%s] dropout f32 REGRESSION vs CPU oracle p=%.2f : bit_diff(y)=%lld bit_diff(mask)=%lld\n",
               ok?" ok ":"FAIL", p, ybd, mbd);
    }
    cudaFree(dx); cudaFree(drand); cudaFree(dy); cudaFree(dm);
}

// ==== cell 7: GB/s bench vs bespoke f64 ====
#ifdef WITH_BESPOKE
static void cell7_bench() {
    const long long M = 4096, N = 4096, n = M * N;
    reseed();
    std::vector<uint64_t> hx((size_t)n);
    std::vector<double> hrand((size_t)n);
    fill_words(hx); fill_rand_f64(hrand);
    std::vector<float> hrand_f32((size_t)n);
    for (long long t=0;t<n;++t) hrand_f32[(size_t)t] = (float)hrand[(size_t)t];
    const float p = 0.5f; const double keep_prob = 0.5, scale = 2.0;
    double *dx,*drand,*dy; float *drand_f32; uint8_t *dm;
    CHECK(cudaMalloc(&dx,n*8)); CHECK(cudaMalloc(&drand,n*8)); CHECK(cudaMalloc(&drand_f32,n*4));
    CHECK(cudaMalloc(&dy,n*8)); CHECK(cudaMalloc(&dm,n));
    CHECK(cudaMemcpy(dx,hx.data(),n*8,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand,hrand.data(),n*8,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand_f32,hrand_f32.data(),n*4,cudaMemcpyHostToDevice));
    int blocks = launch_blocks(n);
    auto timeit = [&](auto fn) {
        cudaEvent_t a,e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i=0;i<3;++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i=0;i<20;++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms,a,e); cudaEventDestroy(a); cudaEventDestroy(e); return ms/20;
    };
    float t_gen = timeit([&]{ baracuda_gen_dropout_f64_mo2_scalar<<<blocks,256>>>((const double*)dx,(const double*)drand,dy,dm,n,keep_prob,scale); });
    float t_bes = timeit([&]{ baracuda_kernels_dropout_f64_run(n,p,scale,dx,drand_f32,dy,dm,nullptr,0,nullptr); });
    // Traffic: read x+rand (2*8B) + write y (8B) + mask (1B) = 25 B/elem.
    double gb = (double)n * 25.0 / 1e9; double ge = (double)n / 1e9;
    printf("[bench] dropout f64 %lldx%lld: generated fused %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
           "bespoke %.3f ms (%.2f Gelem/s, %.1f GB/s) | gen/bespoke %.2fx\n",
           M, N, t_gen, ge/(t_gen/1000), gb/(t_gen/1000), t_bes, ge/(t_bes/1000), gb/(t_bes/1000), t_bes/t_gen);
    cudaFree(dx); cudaFree(drand); cudaFree(drand_f32); cudaFree(dy); cudaFree(dm);
}
#endif

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== dropout_f64_validate (F64-PARAM-CHANNEL increment) ==\n");
#ifdef WITH_BESPOKE
    printf("   (WITH_BESPOKE: cell 3 is the widening-protocol cross-check vs the bespoke f64)\n");
#else
    printf("   (no bespoke: cells 1/2/6 run the CPU f64/f32 oracle — still the acceptance gate)\n");
#endif

    if (san) {
        // Cell 5: small shapes for compute-sanitizer (memcheck/racecheck/synccheck/
        // initcheck) — every kernel family launched once. initcheck matters: the U8
        // mask and the f64 value buffers must be FULLY written, no over-read.
        cell1_primary(true);
        cell2_strided(37, 53);
        cell4_determinism();
        cell6_f32_regression();
#ifdef WITH_BESPOKE
        cell3_bespoke(true);
#endif
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    printf("- cell 1 (ACCEPTANCE GATE): generated f64 dropout vs CPU f64 closed-form oracle, bit_diff(y)==0 AND bit_diff(mask)==0 -\n");
    cell1_primary(false);
    printf("- cell 2: strided address math (transposed output) == scalar, both outputs -\n");
    cell2_strided(128, 96);
    cell2_strided(37, 53);
    printf("- cell 4: run-to-run determinism (both outputs) -\n");
    cell4_determinism();
    printf("- cell 6: byte-identical-f32-dropout regression vs CPU f32 oracle -\n");
    cell6_f32_regression();
#ifdef WITH_BESPOKE
    printf("- cell 3 (SECONDARY): gen vs bespoke f64 under the widening protocol -\n");
    cell3_bespoke(false);
    printf("- cell 7: bench vs bespoke f64 -\n");
    cell7_bench();
#endif
    // cell (source-level cross-body CSE) is the Rust golden dropout_scalar_hetero_golden_f64.

    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
