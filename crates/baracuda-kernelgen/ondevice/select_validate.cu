// WHERE/SELECT increment on-device validation.
//
// THE ACCEPTANCE GATE (cell 1): the generated select-triu/tril
//   out[i,j] = (j >= i + k) ? in[i,j] : +0.0        (tril: j <= i + k)
// expressed as  coord(1).binary(CmpGe, coord(0) + konst(k)).select(input(0), konst(0.0))
// must be BIT-IDENTICAL (whole-buffer memcmp, bit_diff == 0) to the bespoke
// baracuda_kernels_{triu,tril}_{f32,f64}_run kernels across the 0d shape
// matrix, with probe-seeded inputs (masked NEGATIVES — the 0d signed-zero
// finding; masked NaNs — the 0d latent value gap; kept sNaN payloads — the
// quieting gap). The 0d `signed_zero_diff` accounting must COLLAPSE TO ZERO:
// a select moves the chosen arm's exact bits, so there is nothing left to
// account for. (0d measured 84,489 -0.0-on-masked-negative bit diffs at
// 5000x33 under the mask-multiply idiom; this harness requires 0.)
//
// Plus: fused-cmp select vs a CPU raw-bit oracle (cell 2), cross-check vs the
// bespoke u8-cond `where` (cell 3), f16/bf16 raw-pick with NaN payloads in
// both arms (cell 4), vectorized-vs-scalar (cell 5), run-to-run determinism
// (cell 6), a `san` mode for compute-sanitizer (cell 7), and a GB/s bench vs
// bespoke triu (cell 8).
//
// Generated kernels are produced by the `dump_select_sources` test — see the
// ondevice README "select (WHERE/SELECT increment)" section for the exact
// regeneration + compile lines.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GENERATED select kernels (this build's dump_select_sources output).
#include "baracuda_gen_triu_sel_f32_strided_r2.cu"
#include "baracuda_gen_triu_sel_km1_f32_strided_r2.cu"
#include "baracuda_gen_triu_sel_k2_f32_strided_r2.cu"
#include "baracuda_gen_tril_sel_f32_strided_r2.cu"
#include "baracuda_gen_tril_sel_km1_f32_strided_r2.cu"
#include "baracuda_gen_tril_sel_k2_f32_strided_r2.cu"
#include "baracuda_gen_triu_sel_f64_strided_r2.cu"
#include "baracuda_gen_triu_sel_km1_f64_strided_r2.cu"
#include "baracuda_gen_triu_sel_k2_f64_strided_r2.cu"
#include "baracuda_gen_tril_sel_f64_strided_r2.cu"
#include "baracuda_gen_tril_sel_km1_f64_strided_r2.cu"
#include "baracuda_gen_tril_sel_k2_f64_strided_r2.cu"
#include "baracuda_gen_sel_cmp_f32_scalar.cu"
#include "baracuda_gen_sel_cmp_f64_scalar.cu"
#include "baracuda_gen_sel_and_f32_scalar.cu"
#include "baracuda_gen_mask_ge_f32_scalar.cu"
#include "baracuda_gen_sel_f16_scalar.cu"
#include "baracuda_gen_sel_bf16_scalar.cu"
#include "baracuda_gen_sel_f32_scalar.cu"
#include "baracuda_gen_sel_f32_co_v4.cu"

// BESPOKE triu/tril (the memcmp targets) + the u8-cond where family.
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_triu_tril.cuh"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/where_fp.cu"

BARACUDA_KERNELS_TRIU_INSTANTIATE(f32, float)
BARACUDA_KERNELS_TRIU_INSTANTIATE(f64, double)
BARACUDA_KERNELS_TRIL_INSTANTIATE(f32, float)
BARACUDA_KERNELS_TRIL_INSTANTIATE(f64, double)

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int fails = 0;

// Probe classes (house standard, relu_propagating_validate.cu): +/-0, +/-1,
// +/-inf, qNaN payloads, sNaN payloads, negative-NaN payloads, subnormals,
// min normal, max finite, +/-pi. Tiled across every buffer, so each class
// lands in BOTH kept and masked positions of every triu/tril shape.
static std::vector<uint32_t> probe_u32() {
    return {
        0x00000000u, 0x80000000u,              // +0, -0
        0x3f800000u, 0xbf800000u,              // +1, -1 (masked negatives!)
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
static std::vector<uint64_t> probe_u64() {
    return {
        0x0000000000000000ull, 0x8000000000000000ull, // +0,-0
        0x3ff0000000000000ull, 0xbff0000000000000ull, // +1,-1
        0x7ff0000000000000ull, 0xfff0000000000000ull, // +inf,-inf
        0x7ff8000000000000ull, 0xfff8000000000000ull, // qNaN,-qNaN
        0x7ff8000000000001ull, 0x7ffdeadbeef00001ull, // qNaN payloads
        0x7ffc0ffee0000000ull,                        // qNaN payload #3
        0x7ff0000000000001ull, 0x7ff7ffffffffffffull, // sNaN payloads
        0xfff8badc0ffee000ull, 0xfff4baddcafe0001ull, // negative NaN payloads
        0x0000000000000001ull, 0x8000000000000001ull, // +/- min subnormal
        0x000fffffffffffffull, 0x800fffffffffffffull, // +/- max subnormal
        0x0010000000000000ull, 0x8010000000000000ull, // +/- min normal
        0x7fefffffffffffffull, 0xffefffffffffffffull, // +/- max finite
        0x400921fb54442d18ull, 0xc00921fb54442d18ull, // +/-pi
    };
}

static uint64_t g_seed = 0x9e3779b97f4a7c15ull;
static uint64_t xrand() { g_seed ^= g_seed << 13; g_seed ^= g_seed >> 7; g_seed ^= g_seed << 17; return g_seed; }

// Fill n words with the tiled probe set followed by an xorshift bit sweep.
template <typename U>
static void probe_fill(std::vector<U>& h, const std::vector<U>& probes) {
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (i < probes.size()) ? probes[i]
             : (i % 3 == 0) ? probes[(xrand() % probes.size())] // re-tile classes deep
                            : (U)xrand();
}

static int launch_blocks(long long n) {
    long long b = (n + 255) / 256; if (b < 1) b = 1; if (b > 65535) b = 65535;
    return (int)b;
}

// probe_fill with the right probe set per word width.
static void fill_words(std::vector<uint32_t>& h) { probe_fill(h, probe_u32()); }
static void fill_words(std::vector<uint64_t>& h) { probe_fill(h, probe_u64()); }

// ============================ cell 1: the gate ============================
// Generated select-triu/tril vs bespoke: whole-buffer memcmp, bit_diff == 0.
// The 0d signed_zero_diff accounting is retained and must COLLAPSE: with a
// select there are no bit diffs to account for. A third check compares
// against the raw-bit mathematical definition (keep ? in : +0.0), so a
// both-wrong-the-same-way regression cannot pass.

template <typename T, typename U, typename GenF>
static void masked_case(const char* fam, long long M, long long N, int diag,
                        GenF gen_launch,
                        int32_t (*bes)(const void*, void*, const int32_t*, int32_t, int32_t, void*),
                        bool keep_ge /* triu: j >= i+diag; tril: j <= i+diag */) {
    const long long n = M * N;
    const size_t esz = sizeof(T);
    std::vector<U> hin((size_t)n), hgen((size_t)n), hbes((size_t)n);
    fill_words(hin);

    T *din, *dgen, *dbes;
    CHECK(cudaMalloc(&din, n * esz)); CHECK(cudaMalloc(&dgen, n * esz)); CHECK(cudaMalloc(&dbes, n * esz));
    CHECK(cudaMemcpy(din, hin.data(), n * esz, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dgen, 0x5B, n * esz)); CHECK(cudaMemset(dbes, 0xA1, n * esz));

    gen_launch(din, dgen, M, N);
    CHECK(cudaGetLastError());
    int32_t shape[2] = { (int32_t)M, (int32_t)N };
    int rc = bes(din, dbes, shape, 2, diag, nullptr);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("FAIL %s [%lldx%lld] diag %+d: bespoke rc=%d\n", fam, M, N, diag, rc); fails++; goto done; }
    CHECK(cudaMemcpy(hgen.data(), dgen, n * esz, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hbes.data(), dbes, n * esz, cudaMemcpyDeviceToHost));

    {
        // THE GATE: whole-buffer memcmp — zero bit diffs allowed.
        long long bit_diff = 0, signed_zero_diff = 0, def_bit_bad = 0;
        for (long long i = 0; i < M; ++i)
            for (long long j = 0; j < N; ++j) {
                size_t t = (size_t)(i * N + j);
                bool keep = keep_ge ? (j >= i + diag) : (j <= i + diag);
                if (hgen[t] != hbes[t]) {
                    bit_diff++;
                    // The 0d accounting class: masked negative stored as -0/+0.
                    // (Retained so the printout shows the class COLLAPSED, not
                    // merely untested.)
                    U sign_mask = (U)1 << (esz * 8 - 1);
                    bool in_neg = (hin[t] & sign_mask) != 0;
                    if (!keep && in_neg && ((hgen[t] | hbes[t]) & ~sign_mask) == 0)
                        signed_zero_diff++;
                }
                U want = keep ? hin[t] : (U)0; // literal +0.0 bits, bespoke zero_of<T>
                if (hgen[t] != want) def_bit_bad++;
            }
        bool ok = (bit_diff == 0) && (def_bit_bad == 0);
        if (!ok) fails++;
        printf("[%s] %s %s [%4lldx%-4lld] diag %+d : bit_diff %lld (signed-zero class: %lld) def_bit_bad %lld\n",
               ok ? " ok " : "FAIL", fam, sizeof(U) == 4 ? "f32" : "f64", M, N, diag,
               bit_diff, signed_zero_diff, def_bit_bad);
    }
done:
    cudaFree(din); cudaFree(dgen); cudaFree(dbes);
}

static void cell1_triu_tril(bool small_only) {
    struct Shape { long long m, n; };
    const Shape shapes_full[] = { {128,128}, {37,53}, {1,1}, {5000,33}, {33,5000} };
    const Shape shapes_san[]  = { {37,53}, {1,1} };
    const Shape* shapes = small_only ? shapes_san : shapes_full;
    const int nshapes = small_only ? 2 : 5;

    for (int s = 0; s < nshapes; ++s) {
        long long M = shapes[s].m, N = shapes[s].n, n = M * N;
        int blocks = launch_blocks(n);
        for (int diag : {0, -1, 2}) {
            // f32 triu + tril (generated kernel is compiled per baked diagonal).
            auto g32 = [&](int d, bool triu) {
                return [=](float* din, float* dgen, long long m, long long nn) {
                    if (triu) {
                        if (d == 0)  baracuda_gen_triu_sel_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == -1) baracuda_gen_triu_sel_km1_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == 2)  baracuda_gen_triu_sel_k2_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                    } else {
                        if (d == 0)  baracuda_gen_tril_sel_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == -1) baracuda_gen_tril_sel_km1_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == 2)  baracuda_gen_tril_sel_k2_f32_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                    }
                };
            };
            auto g64 = [&](int d, bool triu) {
                return [=](double* din, double* dgen, long long m, long long nn) {
                    if (triu) {
                        if (d == 0)  baracuda_gen_triu_sel_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == -1) baracuda_gen_triu_sel_km1_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == 2)  baracuda_gen_triu_sel_k2_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                    } else {
                        if (d == 0)  baracuda_gen_tril_sel_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == -1) baracuda_gen_tril_sel_km1_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                        if (d == 2)  baracuda_gen_tril_sel_k2_f64_strided_r2<<<blocks,256>>>(din, dgen, m, nn, nn, 1, nn, 1, m*nn);
                    }
                };
            };
            masked_case<float, uint32_t>("triu", M, N, diag, g32(diag, true),
                                         baracuda_kernels_triu_f32_run, true);
            masked_case<float, uint32_t>("tril", M, N, diag, g32(diag, false),
                                         baracuda_kernels_tril_f32_run, false);
            masked_case<double, uint64_t>("triu", M, N, diag, g64(diag, true),
                                          baracuda_kernels_triu_f64_run, true);
            masked_case<double, uint64_t>("tril", M, N, diag, g64(diag, false),
                                          baracuda_kernels_tril_f64_run, false);
        }
    }
}

// ================= cell 2: fused-cmp select vs CPU raw-bit oracle =================
static void cell2_fused_cmp(long long n) {
    // f32: sel_cmp = (a >= b) ? x : y  (a NaN compare OPERAND makes `>=`
    //      unordered -> cond value 0.0 -> y; a cond whose VALUE is NaN would be
    //      nonzero-TRUE and pick x — that is cell 4, not this cell);
    //      sel_and = ((a >= b) * (a != 0)) != 0 ? x : y — a composed cond.
    {
        std::vector<uint32_t> a((size_t)n), b((size_t)n), x((size_t)n), y((size_t)n), got((size_t)n);
        fill_words(a); fill_words(b); fill_words(x); fill_words(y);
        uint32_t *da, *db, *dx, *dy, *dout;
        CHECK(cudaMalloc(&da, n*4)); CHECK(cudaMalloc(&db, n*4)); CHECK(cudaMalloc(&dx, n*4));
        CHECK(cudaMalloc(&dy, n*4)); CHECK(cudaMalloc(&dout, n*4));
        CHECK(cudaMemcpy(da, a.data(), n*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db, b.data(), n*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, x.data(), n*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, y.data(), n*4, cudaMemcpyHostToDevice));
        int blocks = launch_blocks(n);

        CHECK(cudaMemset(dout, 0x5B, n*4));
        baracuda_gen_sel_cmp_f32_scalar<<<blocks,256>>>((const float*)da, (const float*)db,
                                                        (const float*)dx, (const float*)dy,
                                                        (float*)dout, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), dout, n*4, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (size_t t = 0; t < (size_t)n; ++t) {
            float fa, fb; memcpy(&fa, &a[t], 4); memcpy(&fb, &b[t], 4);
            uint32_t want = (fa >= fb) ? x[t] : y[t]; // raw-bit pick
            if (got[t] != want) bad++;
        }
        if (bad) fails++;
        printf("[%s] sel_cmp f32 (a>=b ? x : y) vs raw-bit oracle over %lld elems: %lld diffs\n",
               bad ? "FAIL" : " ok ", n, bad);

        CHECK(cudaMemset(dout, 0x3C, n*4));
        baracuda_gen_sel_and_f32_scalar<<<blocks,256>>>((const float*)da, (const float*)db,
                                                        (const float*)dx, (const float*)dy,
                                                        (float*)dout, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), dout, n*4, cudaMemcpyDeviceToHost));
        bad = 0;
        for (size_t t = 0; t < (size_t)n; ++t) {
            float fa, fb; memcpy(&fa, &a[t], 4); memcpy(&fb, &b[t], 4);
            bool cond = (fa >= fb) && (fa != 0.0f); // (1*1 != 0) — the composed cond
            uint32_t want = cond ? x[t] : y[t];
            if (got[t] != want) bad++;
        }
        if (bad) fails++;
        printf("[%s] sel_and f32 ((a>=b)*(a!=0) ? x : y) composed cond over %lld elems: %lld diffs\n",
               bad ? "FAIL" : " ok ", n, bad);
        cudaFree(da); cudaFree(db); cudaFree(dx); cudaFree(dy); cudaFree(dout);
    }
    // f64 sel_cmp.
    {
        std::vector<uint64_t> a((size_t)n), b((size_t)n), x((size_t)n), y((size_t)n), got((size_t)n);
        fill_words(a); fill_words(b); fill_words(x); fill_words(y);
        uint64_t *da, *db, *dx, *dy, *dout;
        CHECK(cudaMalloc(&da, n*8)); CHECK(cudaMalloc(&db, n*8)); CHECK(cudaMalloc(&dx, n*8));
        CHECK(cudaMalloc(&dy, n*8)); CHECK(cudaMalloc(&dout, n*8));
        CHECK(cudaMemcpy(da, a.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db, b.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, x.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, y.data(), n*8, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dout, 0x5B, n*8));
        baracuda_gen_sel_cmp_f64_scalar<<<launch_blocks(n),256>>>((const double*)da, (const double*)db,
                                                                  (const double*)dx, (const double*)dy,
                                                                  (double*)dout, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), dout, n*8, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (size_t t = 0; t < (size_t)n; ++t) {
            double fa, fb; memcpy(&fa, &a[t], 8); memcpy(&fb, &b[t], 8);
            uint64_t want = (fa >= fb) ? x[t] : y[t];
            if (got[t] != want) bad++;
        }
        if (bad) fails++;
        printf("[%s] sel_cmp f64 (a>=b ? x : y) vs raw-bit oracle over %lld elems: %lld diffs\n",
               bad ? "FAIL" : " ok ", n, bad);
        cudaFree(da); cudaFree(db); cudaFree(dx); cudaFree(dy); cudaFree(dout);
    }
}

// ============ cell 3: generated fused select vs bespoke u8-cond where ============
// Device-compute the u8 mask with the generated 0b pred kernel, feed it to the
// bespoke where launcher, and memcmp against the generated one-kernel fused
// select — both are verbatim picks of the same lanes. (Scope note for the
// README: bespoke where's U8 cond OPERAND is the [U8,T,T] tuple — inexpressible
// under v1 uniform keying; this cell proves value/bit EQUIVALENCE of the fused
// form, not a bespoke-where bind.)
static void cell3_vs_bespoke_where(long long n) {
    std::vector<uint32_t> a((size_t)n), b((size_t)n), x((size_t)n), y((size_t)n);
    fill_words(a); fill_words(b); fill_words(x); fill_words(y);
    uint32_t *da, *db, *dx, *dy, *dgen, *dbes;
    uint8_t* dmask;
    CHECK(cudaMalloc(&da, n*4)); CHECK(cudaMalloc(&db, n*4)); CHECK(cudaMalloc(&dx, n*4));
    CHECK(cudaMalloc(&dy, n*4)); CHECK(cudaMalloc(&dgen, n*4)); CHECK(cudaMalloc(&dbes, n*4));
    CHECK(cudaMalloc(&dmask, n));
    CHECK(cudaMemcpy(da, a.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, x.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy, y.data(), n*4, cudaMemcpyHostToDevice));
    int blocks = launch_blocks(n);

    // Device mask: mask = (a >= b) as u8 (generated 0b pred kernel).
    baracuda_gen_mask_ge_f32_scalar<<<blocks,256>>>((const float*)da, (const float*)db, dmask, n);
    // Bespoke where on (mask, x, y) — the strided launcher, contiguous strides
    // (this device-launches the strided symbol per the brief's cell list).
    int32_t shape[1] = { (int32_t)n };
    int64_t s1[1] = { 1 };
    CHECK(cudaMemset(dbes, 0xA1, n*4));
    int rc = baracuda_kernels_where_f32_strided_run(n, 1, shape, s1, s1, s1, s1,
                                                    dmask, dx, dy, dbes, nullptr, 0, nullptr);
    // Generated fused select on (a, b, x, y).
    CHECK(cudaMemset(dgen, 0x5B, n*4));
    baracuda_gen_sel_cmp_f32_scalar<<<blocks,256>>>((const float*)da, (const float*)db,
                                                    (const float*)dx, (const float*)dy,
                                                    (float*)dgen, n);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("FAIL bespoke where rc=%d\n", rc); fails++; }
    std::vector<uint32_t> hgen((size_t)n), hbes((size_t)n);
    CHECK(cudaMemcpy(hgen.data(), dgen, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hbes.data(), dbes, n*4, cudaMemcpyDeviceToHost));
    long long bad = memcmp(hgen.data(), hbes.data(), (size_t)n*4) == 0 ? 0 : -1;
    if (bad) { bad = 0; for (size_t t = 0; t < (size_t)n; ++t) if (hgen[t] != hbes[t]) bad++; }
    if (bad) fails++;
    printf("[%s] fused select == bespoke where(dev-mask(a>=b), x, y) over %lld elems: %lld diffs\n",
           bad ? "FAIL" : " ok ", n, bad);
    cudaFree(da); cudaFree(db); cudaFree(dx); cudaFree(dy);
    cudaFree(dgen); cudaFree(dbes); cudaFree(dmask);
}

// ================= cell 4: f16 / bf16 raw-pick (NaN payloads in arms) =================
// cond sweeps EVERY 16-bit pattern; arms carry random 16-bit patterns
// (including NaN payload space by construction). Oracle: a raw u16 lane copy —
// picked iff (cond & 0x7fff) != 0 (both zero signs false; NaN/inf/subnormals
// true — the same rule for f16 and bf16). Proves no promote-demote touched an
// arm (M7's on-device dual).
static void cell4_half_raw_pick(long long n_arg, bool full_sweep) {
    const long long n = full_sweep ? (1 << 16) : n_arg;
    for (int is_bf16 = 0; is_bf16 < 2; ++is_bf16) {
        std::vector<uint16_t> c((size_t)n), x((size_t)n), y((size_t)n), got((size_t)n);
        for (long long i = 0; i < n; ++i) {
            c[i] = full_sweep ? (uint16_t)i : (uint16_t)xrand();
            x[i] = (uint16_t)xrand();
            y[i] = (uint16_t)xrand();
        }
        // Seed explicit NaN payloads (incl. signaling-class low-mantissa ones)
        // into both arms at fixed positions.
        const uint16_t f16_payloads[]  = { 0x7c01, 0x7e01, 0xfc01, 0x7fff, 0xfe0f };
        const uint16_t bf16_payloads[] = { 0x7f81, 0x7fc1, 0xff81, 0x7fff, 0xffc3 };
        const uint16_t* pl = is_bf16 ? bf16_payloads : f16_payloads;
        for (int t = 0; t < 5 && (long long)t * 7 + 3 < n; ++t) {
            x[(size_t)t * 7 + 1] = pl[t];
            y[(size_t)t * 7 + 3] = pl[t];
        }
        uint16_t *dc, *dx, *dy, *dout;
        CHECK(cudaMalloc(&dc, n*2)); CHECK(cudaMalloc(&dx, n*2));
        CHECK(cudaMalloc(&dy, n*2)); CHECK(cudaMalloc(&dout, n*2));
        CHECK(cudaMemcpy(dc, c.data(), n*2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dx, x.data(), n*2, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, y.data(), n*2, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(dout, 0x5B, n*2));
        if (is_bf16)
            baracuda_gen_sel_bf16_scalar<<<launch_blocks(n),256>>>(
                (const __nv_bfloat16*)dc, (const __nv_bfloat16*)dx,
                (const __nv_bfloat16*)dy, (__nv_bfloat16*)dout, n);
        else
            baracuda_gen_sel_f16_scalar<<<launch_blocks(n),256>>>(
                (const __half*)dc, (const __half*)dx, (const __half*)dy, (__half*)dout, n);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(got.data(), dout, n*2, cudaMemcpyDeviceToHost));
        long long bad = 0;
        for (size_t t = 0; t < (size_t)n; ++t) {
            uint16_t want = ((c[t] & 0x7fff) != 0) ? x[t] : y[t];
            if (got[t] != want) bad++;
        }
        if (bad) fails++;
        printf("[%s] sel %s raw u16 pick (%s cond) over %lld elems: %lld diffs\n",
               bad ? "FAIL" : " ok ", is_bf16 ? "bf16" : "f16",
               full_sweep ? "full 16-bit sweep" : "random", n, bad);
        cudaFree(dc); cudaFree(dx); cudaFree(dy); cudaFree(dout);
    }
}

// ================= cell 5: vectorized per-lane vs scalar sibling =================
static void cell5_vectorized(long long n_in) {
    long long n = (n_in + 3) & ~3ll; // multiple of 4
    std::vector<uint32_t> c((size_t)n), x((size_t)n), y((size_t)n);
    fill_words(c); fill_words(x); fill_words(y);
    uint32_t *dc, *dx, *dy, *ds, *dv;
    CHECK(cudaMalloc(&dc, n*4)); CHECK(cudaMalloc(&dx, n*4)); CHECK(cudaMalloc(&dy, n*4));
    CHECK(cudaMalloc(&ds, n*4)); CHECK(cudaMalloc(&dv, n*4));
    CHECK(cudaMemcpy(dc, c.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, x.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dy, y.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(ds, 0x5B, n*4)); CHECK(cudaMemset(dv, 0x3C, n*4));
    baracuda_gen_sel_f32_scalar<<<launch_blocks(n),256>>>(
        (const float*)dc, (const float*)dx, (const float*)dy, (float*)ds, n);
    baracuda_gen_sel_f32_co_v4<<<launch_blocks(n/4),256>>>(
        (const float4*)dc, (const float4*)dx, (const float4*)dy, (float4*)dv, n/4);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> hs((size_t)n), hv((size_t)n);
    CHECK(cudaMemcpy(hs.data(), ds, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hv.data(), dv, n*4, cudaMemcpyDeviceToHost));
    long long bad = 0;
    for (size_t t = 0; t < (size_t)n; ++t) if (hs[t] != hv[t]) bad++;
    if (bad) fails++;
    printf("[%s] sel f32 co_v4 == scalar over %lld elems: %lld diffs\n",
           bad ? "FAIL" : " ok ", n, bad);
    cudaFree(dc); cudaFree(dx); cudaFree(dy); cudaFree(ds); cudaFree(dv);
}

// ================= cell 6: run-to-run determinism =================
static void cell6_determinism() {
    const long long M = 5000, N = 33, n = M * N;
    std::vector<uint32_t> hin((size_t)n);
    fill_words(hin);
    uint32_t *din, *d1, *d2;
    CHECK(cudaMalloc(&din, n*4)); CHECK(cudaMalloc(&d1, n*4)); CHECK(cudaMalloc(&d2, n*4));
    CHECK(cudaMemcpy(din, hin.data(), n*4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d1, 0x5B, n*4)); CHECK(cudaMemset(d2, 0x3C, n*4));
    int blocks = launch_blocks(n);
    baracuda_gen_triu_sel_f32_strided_r2<<<blocks,256>>>((const float*)din, (float*)d1, M, N, N, 1, N, 1, n);
    baracuda_gen_triu_sel_f32_strided_r2<<<blocks,256>>>((const float*)din, (float*)d2, M, N, N, 1, N, 1, n);
    CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> h1((size_t)n), h2((size_t)n);
    CHECK(cudaMemcpy(h1.data(), d1, n*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h2.data(), d2, n*4, cudaMemcpyDeviceToHost));
    bool ok = memcmp(h1.data(), h2.data(), (size_t)n*4) == 0;
    if (!ok) fails++;
    printf("[%s] run-to-run determinism (two triu_sel launches, memcmp) over %lld elems\n",
           ok ? " ok " : "FAIL", n);
    cudaFree(din); cudaFree(d1); cudaFree(d2);
}

// ================= cell 8: GB/s bench vs bespoke triu =================
static void cell8_bench() {
    const long long M = 4096, N = 4096, n = M * N;
    std::vector<uint32_t> hin((size_t)n);
    fill_words(hin);
    uint32_t *din, *dout;
    CHECK(cudaMalloc(&din, n*4)); CHECK(cudaMalloc(&dout, n*4));
    CHECK(cudaMemcpy(din, hin.data(), n*4, cudaMemcpyHostToDevice));
    int blocks = launch_blocks(n);
    int32_t shape[2] = { (int32_t)M, (int32_t)N };
    auto timeit = [&](auto fn) {
        cudaEvent_t a, e; cudaEventCreate(&a); cudaEventCreate(&e);
        for (int i = 0; i < 3; ++i) fn(); cudaDeviceSynchronize(); cudaEventRecord(a);
        for (int i = 0; i < 20; ++i) fn(); cudaEventRecord(e); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, a, e);
        cudaEventDestroy(a); cudaEventDestroy(e); return ms / 20;
    };
    float t_gen = timeit([&] {
        baracuda_gen_triu_sel_f32_strided_r2<<<blocks,256>>>((const float*)din, (float*)dout, M, N, N, 1, N, 1, n);
    });
    float t_bes = timeit([&] {
        baracuda_kernels_triu_f32_run(din, dout, shape, 2, 0, nullptr);
    });
    double gb = (double)n * 4.0 * 2 / 1e9; // one read + one write
    double ge = (double)n / 1e9;
    printf("[bench] triu f32 %lldx%lld: generated select %.3f ms (%.2f Gelem/s, %.1f GB/s) | "
           "bespoke %.3f ms (%.2f Gelem/s, %.1f GB/s) | gen/bespoke %.2fx\n",
           M, N, t_gen, ge / (t_gen / 1000), gb / (t_gen / 1000),
           t_bes, ge / (t_bes / 1000), gb / (t_bes / 1000), t_bes / t_gen);
    cudaFree(din); cudaFree(dout);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== select_validate (WHERE/SELECT increment) ==\n");

    if (san) {
        // Cell 7: small shapes for compute-sanitizer (memcheck/racecheck/
        // synccheck/initcheck) — every kernel family launched once.
        cell1_triu_tril(true);
        cell2_fused_cmp(1024);
        cell3_vs_bespoke_where(1024);
        cell4_half_raw_pick(1024, false);
        cell5_vectorized(1024);
        printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
        return fails ? 1 : 0;
    }

    printf("- cell 1 (ACCEPTANCE GATE): generated select-triu/tril vs bespoke, memcmp bit_diff == 0 -\n");
    cell1_triu_tril(false);
    printf("- cell 2: fused-cmp select vs CPU raw-bit oracle (probe + random sweep) -\n");
    cell2_fused_cmp(4ll * 1024 * 1024);
    printf("- cell 3: fused select vs bespoke u8-cond where (device-computed mask) -\n");
    cell3_vs_bespoke_where(4ll * 1024 * 1024);
    printf("- cell 4: f16/bf16 raw-pick (full 16-bit cond sweep, NaN payload arms) -\n");
    cell4_half_raw_pick(0, true);
    printf("- cell 5: vectorized per-lane vs scalar -\n");
    cell5_vectorized(16ll * 1024 * 1024);
    printf("- cell 6: run-to-run determinism -\n");
    cell6_determinism();
    printf("- cell 8: bench -\n");
    cell8_bench();

    printf(fails ? "\n%d case(s) FAILED\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
