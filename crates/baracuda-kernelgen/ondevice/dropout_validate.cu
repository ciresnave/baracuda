// HETERO MULTI-OUTPUT (dropout-class) on-device validation.
//
// THE ACCEPTANCE GATE (cell 1): the generated fused dropout kernel
//   y[i]    = x[i] * (rand[i] < keep_prob ? scale : 0)     (F32 value output)
//   mask[i] = (rand[i] < keep_prob)                         (U8 keep-mask output)
// written by ONE kernel from a shared body-DAG (the `rand < keep_prob` comparison
// hoisted ONCE, consumed by BOTH outputs), must be BIT-IDENTICAL — whole-buffer
// memcmp, `bit_diff(y)==0` AND `bit_diff(mask)==0` — to the bespoke
// `baracuda_kernels_dropout_f32_run` across a p × shape matrix, with the SAME
// host-filled `rand` buffer and the SAME host `keep_prob = 1-p` / `scale = 1/(1-p)`.
// Probe-seeded `x` (dropped negatives → -0.0, dropped NaN/Inf propagated, kept
// x·scale) exercises every IEEE class in both the KEPT and DROPPED partitions
// (uniform `rand` × the p sweep drives each x position above/below keep_prob).
// The value output is a genuine MULTIPLY (not a select-of-value), so it has no
// triu-style signed-zero hazard: both generated and bespoke compute `x·mult`
// (mult ∈ {scale, 0}) with the identical single IEEE multiply, so the memcmp is
// bit_diff==0 outright.
//
// Cross-body CSE (cell 3) is a SOURCE-level fact proven by the Rust golden
// `cuda::dropout_hetero_tests::dropout_scalar_hetero_golden_f32` (the comparison
// hoists to ONE `float tmp0 = (...)` referenced by both stores). This harness
// carries the NUMERIC cells: gate (1), strided (2), determinism (4), a `san` mode
// for compute-sanitizer (5), and a GB/s bench (6).
//
// Generated kernels are produced by the `dump_dropout_sources` test — see the
// ondevice README "dropout (HETERO MULTI-OUTPUT increment)" section for the exact
// regeneration + compile lines. Build the acceptance gate with -DWITH_BESPOKE
// (adds the bespoke header + `-I <kernels>/include -Xcompiler "/Zc:preprocessor
// /std:c++17"`); without it cells 1/2 fall back to a CPU oracle.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// GENERATED dropout kernels (this build's dump_dropout_sources output).
#include "baracuda_gen_dropout_f32_mo2_scalar.cu"
#include "baracuda_gen_dropout_f32_mo2_strided_r2.cu"

// BESPOKE dropout_fw (the memcmp target for the acceptance gate).
#ifdef WITH_BESPOKE
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_random.cuh"
BARACUDA_KERNELS_DROPOUT_INSTANTIATE(f32, float, float)
#endif

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int fails = 0;

// Probe classes (house standard, relu_propagating_validate.cu): +/-0, +/-1,
// +/-inf, qNaN payloads, sNaN payloads, negative-NaN payloads, subnormals, min
// normal, max finite, +/-pi. Tiled across `x` so each class lands in BOTH kept
// and dropped positions of every p in the sweep.
static std::vector<uint32_t> probe_u32() {
    return {
        0x00000000u, 0x80000000u,              // +0, -0
        0x3f800000u, 0xbf800000u,              // +1, -1 (dropped negatives -> -0.0)
        0x7f800000u, 0xff800000u,              // +inf, -inf
        0x7fc00000u, 0xffc00000u,              // qNaN, -qNaN
        0x7fc00001u, 0x7fdead01u, 0x7ff00000u, // qNaN payloads (three)
        0x7f800001u, 0x7fbfffffu,              // sNaN payloads (two)
        0xffc0beefu, 0xffbfffffu,              // negative NaN payloads (two)
        0x00000001u, 0x80000001u,              // +/- smallest subnormal
        0x007fffffu, 0x807fffffu,              // +/- largest subnormal
        0x00800000u, 0x80800000u,              // +/- smallest normal
        0x7f7fffffu, 0xff7fffffu,              // +/- largest finite
        0x40490fdbu, 0xc0490fdbu,              // +/-pi
    };
}

static uint64_t g_seed = 0x9e3779b97f4a7c15ull;
static uint64_t xrand() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return g_seed; }
static void reseed() { g_seed = 0x9e3779b97f4a7c15ull; }

// Fill `x` words with the tiled probe set followed by an xorshift bit sweep.
static void fill_words(std::vector<uint32_t>& h) {
    auto probes = probe_u32();
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (i < probes.size()) ? probes[i]
             : (i % 3 == 0) ? probes[(xrand() % probes.size())]
                            : (uint32_t)xrand();
}
// Fill `rand` with uniform values in [0, 1) (never NaN) — the SHARED noise the
// generated + bespoke kernels both read, so both see the same kept/dropped split.
static void fill_rand(std::vector<float>& h) {
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (float)((uint32_t)(xrand() & 0xFFFFFFu)) / 16777216.0f; // [0,1)
}

static int launch_blocks(long long n) {
    long long b = (n + 255) / 256; if (b < 1) b = 1; if (b > 65535) b = 65535;
    return (int)b;
}

// ============================ cell 1: the gate ============================
// Generated fused dropout vs bespoke dropout_fw: whole-buffer memcmp on BOTH the
// F32 value output and the U8 mask output, bit_diff == 0. A third check compares
// the mask against its raw definition (rand < keep_prob) so a both-wrong regression
// cannot pass.
static void cell1_dropout(bool small_only) {
    struct Shape { long long n; const char* name; };
    const Shape shapes_full[] = { {1,"1"}, {37*53,"37x53"}, {4096,"4096"}, {1000003,"1000003(prime)"} };
    const Shape shapes_san[]  = { {37*53,"37x53"}, {1,"1"} };
    const Shape* shapes = small_only ? shapes_san : shapes_full;
    const int nshapes = small_only ? 2 : 4;
    const float ps[] = { 0.0f, 0.1f, 0.5f, 0.9f };

    for (int s = 0; s < nshapes; ++s) {
        long long n = shapes[s].n;
        reseed();
        std::vector<uint32_t> hx((size_t)n);
        std::vector<float> hrand((size_t)n);
        fill_words(hx);
        fill_rand(hrand);

        float *dx, *drand, *dy_gen, *dy_bes;
        uint8_t *dm_gen, *dm_bes;
        CHECK(cudaMalloc(&dx, n*4));      CHECK(cudaMalloc(&drand, n*4));
        CHECK(cudaMalloc(&dy_gen, n*4));  CHECK(cudaMalloc(&dy_bes, n*4));
        CHECK(cudaMalloc(&dm_gen, n));    CHECK(cudaMalloc(&dm_bes, n));
        CHECK(cudaMemcpy(dx, hx.data(), n*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(drand, hrand.data(), n*4, cudaMemcpyHostToDevice));

        for (float p : ps) {
            const float keep_prob = 1.0f - p;
            const float scale = 1.0f / (1.0f - p);
            int blocks = launch_blocks(n);
            // Poison the outputs (initcheck + a real memcmp target).
            CHECK(cudaMemset(dy_gen, 0x5B, n*4)); CHECK(cudaMemset(dy_bes, 0xA1, n*4));
            CHECK(cudaMemset(dm_gen, 0x33, n));   CHECK(cudaMemset(dm_bes, 0xCC, n));

            baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>(
                (const float*)dx, (const float*)drand, dy_gen, dm_gen, n, keep_prob, scale);
            CHECK(cudaGetLastError());

            long long ybd = 0, mbd = 0, mdef_bad = 0, negzero = 0;
            std::vector<uint32_t> ygen((size_t)n), ybes((size_t)n);
            std::vector<uint8_t> mgen((size_t)n), mbes((size_t)n);
#ifdef WITH_BESPOKE
            int rc = baracuda_kernels_dropout_f32_run(
                n, p, scale, dx, drand, dy_bes, dm_bes, nullptr, 0, nullptr);
            CHECK(cudaDeviceSynchronize());
            if (rc != 0) { printf("FAIL dropout [%s] p=%.2f: bespoke rc=%d\n", shapes[s].name, p, rc); fails++; }
            CHECK(cudaMemcpy(ybes.data(), dy_bes, n*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(mbes.data(), dm_bes, n, cudaMemcpyDeviceToHost));
#else
            CHECK(cudaDeviceSynchronize());
#endif
            CHECK(cudaMemcpy(ygen.data(), dy_gen, n*4, cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(mgen.data(), dm_gen, n, cudaMemcpyDeviceToHost));

            for (long long t = 0; t < n; ++t) {
                bool keep = hrand[(size_t)t] < keep_prob;
                // mask vs its raw definition (exact 0/1).
                uint8_t want_m = keep ? 1u : 0u;
                if (mgen[(size_t)t] != want_m) mdef_bad++;
#ifdef WITH_BESPOKE
                if (ygen[(size_t)t] != ybes[(size_t)t]) {
                    ybd++;
                    // classify a dropped-negative -0.0 vs +0.0 disagreement (must be 0).
                    bool in_neg = (hx[(size_t)t] & 0x80000000u) != 0;
                    if (!keep && in_neg && ((ygen[(size_t)t] | ybes[(size_t)t]) & 0x7fffffffu) == 0) negzero++;
                }
                if (mgen[(size_t)t] != mbes[(size_t)t]) mbd++;
#endif
            }
#ifdef WITH_BESPOKE
            bool ok = (ybd == 0) && (mbd == 0) && (mdef_bad == 0);
            if (!ok) fails++;
            printf("[%s] dropout f32 [%12s] p=%.2f : bit_diff(y)=%lld (neg-zero class %lld) bit_diff(mask)=%lld mask_def_bad=%lld\n",
                   ok ? " ok " : "FAIL", shapes[s].name, p, ybd, negzero, mbd, mdef_bad);
#else
            bool ok = (mdef_bad == 0);
            if (!ok) fails++;
            printf("[%s] dropout f32 [%12s] p=%.2f : mask vs def bad=%lld (build -DWITH_BESPOKE for the value memcmp gate)\n",
                   ok ? " ok " : "FAIL", shapes[s].name, p, mdef_bad);
#endif
        }
        cudaFree(dx); cudaFree(drand); cudaFree(dy_gen); cudaFree(dy_bes); cudaFree(dm_gen); cudaFree(dm_bes);
    }
}

// ================= cell 2: strided address math (both outputs) =================
// The generated strided kernel writes a TRANSPOSED [M,N] output (column-major
// output strides, row-major input reads) — a shape the contig-only bespoke cannot
// serve without a materialization. Un-transpose on host and memcmp BOTH outputs
// against the scalar (contiguous) kernel: same x/rand/keep_prob/scale ⇒ the same
// single IEEE multiply ⇒ bit-identical, proving oo0/oo1 index correctly.
static void cell2_strided(long long M, long long N) {
    const long long n = M * N;
    reseed();
    std::vector<uint32_t> hx((size_t)n);
    std::vector<float> hrand((size_t)n);
    fill_words(hx); fill_rand(hrand);
    const float keep_prob = 0.5f, scale = 2.0f; // p = 0.5

    float *dx, *drand, *dy_sc, *dy_st;
    uint8_t *dm_sc, *dm_st;
    CHECK(cudaMalloc(&dx, n*4)); CHECK(cudaMalloc(&drand, n*4));
    CHECK(cudaMalloc(&dy_sc, n*4)); CHECK(cudaMalloc(&dy_st, n*4));
    CHECK(cudaMalloc(&dm_sc, n)); CHECK(cudaMalloc(&dm_st, n));
    CHECK(cudaMemcpy(dx, hx.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand, hrand.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dy_sc, 0x5B, n*4)); CHECK(cudaMemset(dy_st, 0x3C, n*4));
    CHECK(cudaMemset(dm_sc, 0x11, n)); CHECK(cudaMemset(dm_st, 0x22, n));
    int blocks = launch_blocks(n);

    // Scalar reference: contiguous [M,N] row-major, both outputs.
    baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>(
        (const float*)dx, (const float*)drand, dy_sc, dm_sc, n, keep_prob, scale);

    // Strided: read x,rand row-major (strides [N,1]); WRITE outputs column-major
    // (strides [1,M]) so out[i,j] lands at j*M + i.
    baracuda_gen_dropout_f32_mo2_strided_r2<<<blocks,256>>>(
        (const float*)dx, (const float*)drand, dy_st, dm_st,
        M, N,           // shape0, shape1
        N, 1,           // s0 (in0 = x): row-major
        N, 1,           // s1 (in1 = rand): row-major
        1, M,           // so0 (out0 = y): column-major
        1, M,           // so1 (out1 = mask): column-major
        n, keep_prob, scale);
    CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> ysc((size_t)n), yst((size_t)n);
    std::vector<uint8_t> msc((size_t)n), mst((size_t)n);
    CHECK(cudaMemcpy(ysc.data(), dy_sc, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(yst.data(), dy_st, n*4, cudaMemcpyDeviceToHost));
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
    printf("[%s] dropout strided==scalar [%lldx%lld] : bit_diff(y)=%lld bit_diff(mask)=%lld mask_def_bad=%lld\n",
           ok ? " ok " : "FAIL", M, N, ybd, mbd, mdef);
    cudaFree(dx); cudaFree(drand); cudaFree(dy_sc); cudaFree(dy_st); cudaFree(dm_sc); cudaFree(dm_st);
}

// ================= cell 4: run-to-run determinism (both outputs) =================
static void cell4_determinism() {
    const long long n = 1000003;
    reseed();
    std::vector<uint32_t> hx((size_t)n);
    std::vector<float> hrand((size_t)n);
    fill_words(hx); fill_rand(hrand);
    const float keep_prob = 0.3f, scale = 1.0f / 0.3f;
    float *dx, *drand, *dy1, *dy2; uint8_t *dm1, *dm2;
    CHECK(cudaMalloc(&dx, n*4)); CHECK(cudaMalloc(&drand, n*4));
    CHECK(cudaMalloc(&dy1, n*4)); CHECK(cudaMalloc(&dy2, n*4));
    CHECK(cudaMalloc(&dm1, n)); CHECK(cudaMalloc(&dm2, n));
    CHECK(cudaMemcpy(dx, hx.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand, hrand.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dy1, 0x5B, n*4)); CHECK(cudaMemset(dy2, 0x3C, n*4));
    CHECK(cudaMemset(dm1, 0x44, n)); CHECK(cudaMemset(dm2, 0x77, n));
    int blocks = launch_blocks(n);
    baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>((const float*)dx,(const float*)drand,dy1,dm1,n,keep_prob,scale);
    baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>((const float*)dx,(const float*)drand,dy2,dm2,n,keep_prob,scale);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> y1((size_t)n), y2((size_t)n);
    std::vector<uint8_t> m1((size_t)n), m2((size_t)n);
    CHECK(cudaMemcpy(y1.data(), dy1, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(y2.data(), dy2, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m1.data(), dm1, n, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m2.data(), dm2, n, cudaMemcpyDeviceToHost));
    bool ok = (memcmp(y1.data(), y2.data(), (size_t)n*4) == 0) && (memcmp(m1.data(), m2.data(), (size_t)n) == 0);
    if (!ok) fails++;
    printf("[%s] run-to-run determinism (two dropout launches, memcmp y AND mask) over %lld elems\n",
           ok ? " ok " : "FAIL", n);
    cudaFree(dx); cudaFree(drand); cudaFree(dy1); cudaFree(dy2); cudaFree(dm1); cudaFree(dm2);
}

// ================= cell 6: GB/s bench vs bespoke =================
#ifdef WITH_BESPOKE
static void cell6_bench() {
    const long long M = 4096, N = 4096, n = M * N;
    reseed();
    std::vector<uint32_t> hx((size_t)n);
    std::vector<float> hrand((size_t)n);
    fill_words(hx); fill_rand(hrand);
    const float p = 0.5f, keep_prob = 0.5f, scale = 2.0f;
    float *dx, *drand, *dy; uint8_t *dm;
    CHECK(cudaMalloc(&dx, n*4)); CHECK(cudaMalloc(&drand, n*4));
    CHECK(cudaMalloc(&dy, n*4)); CHECK(cudaMalloc(&dm, n));
    CHECK(cudaMemcpy(dx, hx.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(drand, hrand.data(), n*4, cudaMemcpyHostToDevice));
    int blocks = launch_blocks(n);
    auto timeit = [&](auto fn) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e);
        cudaEventDestroy(a); cudaEventDestroy(e); return ms / 20;
    };
    float t_gen = timeit([&] {
        baracuda_gen_dropout_f32_mo2_scalar<<<blocks,256>>>((const float*)dx,(const float*)drand,dy,dm,n,keep_prob,scale);
    });
    float t_bes = timeit([&] {
        baracuda_kernels_dropout_f32_run(n, p, scale, dx, drand, dy, dm, nullptr, 0, nullptr);
    });
    // Traffic: read x+rand (2*4B) + write y (4B) + mask (1B) = 13 B/elem.
    double gb = (double)n * 13.0 / 1e9;
    double ge = (double)n / 1e9;
    printf("[bench] dropout f32 %lldx%lld: generated fused %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
           "bespoke %.3f ms (%.2f Gelem/s, %.1f GB/s) | gen/bespoke %.2fx\n",
           M, N, t_gen, ge / (t_gen / 1000), gb / (t_gen / 1000),
           t_bes, ge / (t_bes / 1000), gb / (t_bes / 1000), t_bes / t_gen);
    cudaFree(dx); cudaFree(drand); cudaFree(dy); cudaFree(dm);
}
#endif

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== dropout_validate (HETERO MULTI-OUTPUT increment) ==\n");
#ifdef WITH_BESPOKE
    printf("   (WITH_BESPOKE: cell 1 is the bit-exact acceptance gate)\n");
#else
    printf("   (no bespoke: cell 1 checks mask vs its definition only)\n");
#endif

    if (san) {
        // Cell 5: small shapes for compute-sanitizer (memcheck/racecheck/
        // synccheck/initcheck) — every kernel family launched once. initcheck
        // matters: the U8 mask buffer must be FULLY written.
        cell1_dropout(true);
        cell2_strided(37, 53);
        cell4_determinism();
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    printf("- cell 1 (ACCEPTANCE GATE): generated fused dropout vs bespoke, memcmp bit_diff(y)==0 AND bit_diff(mask)==0 -\n");
    cell1_dropout(false);
    printf("- cell 2: strided address math (transposed output) == scalar, both outputs -\n");
    cell2_strided(128, 96);
    cell2_strided(37, 53);
    printf("- cell 4: run-to-run determinism (both outputs) -\n");
    cell4_determinism();
#ifdef WITH_BESPOKE
    printf("- cell 6: bench -\n");
    cell6_bench();
#endif
    // cell 3 (cross-body CSE, source-level) is the Rust golden
    // dropout_scalar_hetero_golden_f32 (single hoisted `tmp0`).

    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
