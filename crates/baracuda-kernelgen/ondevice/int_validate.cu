// Increment-0c on-device validation (RTX 4070, sm_89):
//
//   1. All five bitwise/shift ops × i32, generated-vs-BESPOKE OUTPUT BIT-EXACT
//      on the full probe set — including the out-of-range shift amounts
//      (0/31/32/33/-1/64/-32): both kernels emit the same raw C `<<`/`>>`
//      compiled by the same nvcc for the same arch, and the bit-exact match IS
//      the semantics-match proof (the bespoke kernels document out-of-range as
//      architecture-inherited, so no host-side expectation exists there).
//      CPU two's-complement reference additionally checks the defined subset
//      (and/or/xor everywhere; shifts at b in [0,31]).
//   2. The three logical ops × u8, exhaustive 256×256 (unnormalized bytes:
//      2 && 4 == 1, the normalization probe), generated-vs-bespoke bit-exact
//      AND vs the CPU (a!=0 OP b!=0) reference.
//   3. Wrapping u8/i8 add/mul, exhaustive 256×256 vs a CPU wrapping reference
//      (no bespoke elementwise int add/mul exists — CPU is the oracle).
//   4. 8-bit shifts vs the documented promotion model (promote to int, C
//      shift, store-truncate) at b in [0,31]; observed rows printed for the
//      out-of-range amounts (no bespoke 8-bit sibling, so observe-only).
//
// Build: generate the int kernels into <outdir> — they are NOT yet emitted by
// the bin/kernelgen.rs catalog; the README's "int ops (increment 0c)" section
// carries the exact generation spec — then copy this harness beside them (the
// standard ondevice workflow) and, from a VS dev shell:
//   nvcc -O3 -arch=sm_89 int_validate.cu -o int_validate.exe
//   ./int_validate.exe

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ---- generated kernels (place this harness beside them; see the README's
// "int ops (increment 0c)" section for the generation spec) -------------------
#include "baracuda_gen_band_i32_scalar.cu"
#include "baracuda_gen_bor_i32_scalar.cu"
#include "baracuda_gen_bxor_i32_scalar.cu"
#include "baracuda_gen_shl_i32_scalar.cu"
#include "baracuda_gen_shr_i32_scalar.cu"
#include "baracuda_gen_land_u8_scalar.cu"
#include "baracuda_gen_lor_u8_scalar.cu"
#include "baracuda_gen_lxor_u8_scalar.cu"
#include "baracuda_gen_addw_u8_scalar.cu"
#include "baracuda_gen_mulw_u8_scalar.cu"
#include "baracuda_gen_addw_i8_scalar.cu"
#include "baracuda_gen_mulw_i8_scalar.cu"
#include "baracuda_gen_shl_u8_scalar.cu"
#include "baracuda_gen_shr_u8_scalar.cu"
#include "baracuda_gen_shr_i8_scalar.cu"

// ---- bespoke kernels (the semantics reference) ------------------------------
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_bitwise_and_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_bitwise_or_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_bitwise_xor_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_bitwise_left_shift_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_bitwise_right_shift_int.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_logical_and_bool.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_logical_or_bool.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/binary_logical_xor_bool.cu"

static int g_failures = 0;

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__),      \
                   __FILE__, __LINE__);                                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define LAUNCH(kern, da, db, dy, n)                                            \
    do {                                                                       \
        long long n__ = (long long)(n);                                        \
        int grid__ = (int)std::min<long long>((n__ + 255) / 256, 4096);        \
        if (grid__ < 1) grid__ = 1;                                            \
        kern<<<grid__, 256>>>((da), (db), (dy), n__);                          \
        CUDA_OK(cudaGetLastError());                                           \
        CUDA_OK(cudaDeviceSynchronize());                                      \
    } while (0)

// ---- CPU two's-complement reference models ---------------------------------
static int32_t cpu_and32(int32_t a, int32_t b) { return a & b; }
static int32_t cpu_or32(int32_t a, int32_t b) { return a | b; }
static int32_t cpu_xor32(int32_t a, int32_t b) { return a ^ b; }
// Defined-subset shift models (b in [0,31] only): pure bit shifts on the
// two's-complement pattern, arithmetic fill for signed >> — matching the
// bespoke kernels' documented semantics without host UB.
static int32_t cpu_shl32(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a << (uint32_t)b);
}
static int32_t cpu_shr32(int32_t a, int32_t b) {
    uint32_t ua = (uint32_t)a;
    uint32_t r = (a < 0) ? ~(~ua >> (uint32_t)b) : (ua >> (uint32_t)b);
    return (int32_t)r;
}

// ---- i32 bitwise/shift: generated vs bespoke (bit-exact) + CPU subset ------
typedef int32_t (*bespoke_run_t)(int64_t, const void*, const void*, void*, void*, size_t, void*);

struct I32Case {
    const char* name;
    void (*gen)(const int*, const int*, int*, long long);
    bespoke_run_t bespoke;
    int32_t (*cpu)(int32_t, int32_t);
    bool shift; // CPU model valid only for b in [0,31]
};

static void run_i32_case(const I32Case& c) {
    // Edge cross: every a-pattern × every b-pattern. Shift b set carries the
    // charter probes 0/31/32/33/-1 plus 64/-32/1/5/7/15/24/30.
    const int32_t edge_a[] = {0, 1, -1, INT32_MIN, INT32_MAX, 0x5A5A5A5A,
                              (int32_t)0xA5A5A5A5, 123456789, -987654321, 2, -2, 256};
    const int32_t shift_b[] = {0, 31, 32, 33, -1, 64, -32, 1, 5, 7, 15, 24, 30};
    const int32_t patt_b[] = {0, 1, -1, INT32_MIN, INT32_MAX, 0x0F0F0F0F,
                              (int32_t)0xF0F0F0F0, 0x5A5A5A5A, 7, -8, 1024, -4096};
    std::vector<int32_t> ha, hb;
    const int32_t* bset = c.shift ? shift_b : patt_b;
    size_t nb = c.shift ? sizeof(shift_b) / 4 : sizeof(patt_b) / 4;
    for (int32_t a : edge_a) {
        for (size_t j = 0; j < nb; ++j) {
            ha.push_back(a);
            hb.push_back(bset[j]);
        }
    }
    // + 65536 randoms; for shifts, half in-range b, half arbitrary (bespoke-
    // comparison only).
    srand(13);
    for (int i = 0; i < 65536; ++i) {
        int32_t a = (rand() << 17) ^ (rand() << 5) ^ rand();
        int32_t b = (rand() << 17) ^ (rand() << 5) ^ rand();
        if (c.shift && (i & 1)) b = ((b % 32) + 32) % 32; // in-range half
        ha.push_back(a);
        hb.push_back(b);
    }
    size_t n = ha.size();
    int *da, *db, *dg, *dbp;
    CUDA_OK(cudaMalloc(&da, n * 4));
    CUDA_OK(cudaMalloc(&db, n * 4));
    CUDA_OK(cudaMalloc(&dg, n * 4));
    CUDA_OK(cudaMalloc(&dbp, n * 4));
    CUDA_OK(cudaMemcpy(da, ha.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), n * 4, cudaMemcpyHostToDevice));
    LAUNCH(c.gen, da, db, dg, n);
    int32_t st = c.bespoke((int64_t)n, da, db, dbp, nullptr, 0, nullptr);
    if (st != 0) {
        printf("[FAIL] %s: bespoke launcher status %d\n", c.name, st);
        ++g_failures;
        return;
    }
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> hg(n), hbp(n);
    CUDA_OK(cudaMemcpy(hg.data(), dg, n * 4, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(hbp.data(), dbp, n * 4, cudaMemcpyDeviceToHost));

    size_t mism = 0, first = SIZE_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (hg[i] != hbp[i]) {
            if (first == SIZE_MAX) first = i;
            ++mism;
        }
    }
    if (mism == 0) {
        printf("[PASS] %-9s i32 generated == bespoke BIT-EXACT  (%zu probes incl. shift edges)\n",
               c.name, n);
    } else {
        printf("[FAIL] %-9s i32 generated != bespoke: %zu/%zu mismatches; first a=%d b=%d gen=%d bespoke=%d\n",
               c.name, mism, n, ha[first], hb[first], hg[first], hbp[first]);
        ++g_failures;
    }
    // CPU reference on the defined subset.
    size_t cpu_checked = 0, cpu_bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (c.shift && (hb[i] < 0 || hb[i] > 31)) continue;
        ++cpu_checked;
        int32_t want = c.cpu(ha[i], hb[i]);
        if (hg[i] != want) {
            if (cpu_bad == 0)
                printf("       first CPU mismatch: a=%d b=%d gen=%d cpu=%d\n", ha[i], hb[i],
                       hg[i], want);
            ++cpu_bad;
        }
    }
    if (cpu_bad == 0) {
        printf("[PASS] %-9s i32 generated == CPU reference       (%zu defined-subset probes)\n",
               c.name, cpu_checked);
    } else {
        printf("[FAIL] %-9s i32 generated != CPU reference: %zu/%zu\n", c.name, cpu_bad,
               cpu_checked);
        ++g_failures;
    }
    // Observed (architecture-inherited) edge behavior, informational: what
    // BOTH kernels computed for the out-of-range amounts.
    if (c.shift) {
        printf("       observed (gen==bespoke) edges: ");
        for (int32_t be : {0, 31, 32, 33, -1}) {
            for (size_t i = 0; i < n; ++i) {
                if (ha[i] == 1 && hb[i] == be) {
                    printf("1%s%d=%d  ", c.name[2] == 'l' ? "<<" : ">>", be, hg[i]);
                    break;
                }
            }
        }
        printf("\n");
    }
    cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dbp);
}

// ---- u8 exhaustive helpers ---------------------------------------------------
static void build_exhaustive_u8(std::vector<uint8_t>& a, std::vector<uint8_t>& b) {
    a.resize(65536);
    b.resize(65536);
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j) {
            a[i * 256 + j] = (uint8_t)i;
            b[i * 256 + j] = (uint8_t)j;
        }
}

struct U8Case {
    const char* name;
    void (*gen)(const unsigned char*, const unsigned char*, unsigned char*, long long);
    bespoke_run_t bespoke; // may be null (no bespoke sibling)
    uint8_t (*cpu)(uint8_t, uint8_t);
};

static uint8_t cpu_land(uint8_t a, uint8_t b) { return (a != 0 && b != 0) ? 1 : 0; }
static uint8_t cpu_lor(uint8_t a, uint8_t b) { return (a != 0 || b != 0) ? 1 : 0; }
static uint8_t cpu_lxor(uint8_t a, uint8_t b) { return ((a != 0) != (b != 0)) ? 1 : 0; }
static uint8_t cpu_addu8(uint8_t a, uint8_t b) { return (uint8_t)(a + b); }
static uint8_t cpu_mulu8(uint8_t a, uint8_t b) { return (uint8_t)(a * b); }

static void run_u8_case(const U8Case& c) {
    std::vector<uint8_t> ha, hb;
    build_exhaustive_u8(ha, hb);
    size_t n = ha.size();
    unsigned char *da, *db, *dg, *dbp;
    CUDA_OK(cudaMalloc(&da, n));
    CUDA_OK(cudaMalloc(&db, n));
    CUDA_OK(cudaMalloc(&dg, n));
    CUDA_OK(cudaMalloc(&dbp, n));
    CUDA_OK(cudaMemcpy(da, ha.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), n, cudaMemcpyHostToDevice));
    LAUNCH(c.gen, da, db, dg, n);
    std::vector<uint8_t> hg(n);
    CUDA_OK(cudaMemcpy(hg.data(), dg, n, cudaMemcpyDeviceToHost));

    if (c.bespoke) {
        int32_t st = c.bespoke((int64_t)n, da, db, dbp, nullptr, 0, nullptr);
        if (st != 0) {
            printf("[FAIL] %s: bespoke launcher status %d\n", c.name, st);
            ++g_failures;
        } else {
            CUDA_OK(cudaDeviceSynchronize());
            std::vector<uint8_t> hbp(n);
            CUDA_OK(cudaMemcpy(hbp.data(), dbp, n, cudaMemcpyDeviceToHost));
            if (memcmp(hg.data(), hbp.data(), n) == 0) {
                printf("[PASS] %-9s u8 generated == bespoke BIT-EXACT   (65536 exhaustive)\n",
                       c.name);
            } else {
                size_t i = 0;
                while (hg[i] == hbp[i]) ++i;
                printf("[FAIL] %-9s u8 generated != bespoke; first a=%u b=%u gen=%u bespoke=%u\n",
                       c.name, ha[i], hb[i], hg[i], hbp[i]);
                ++g_failures;
            }
        }
    }
    size_t bad = 0, firsti = 0;
    for (size_t i = 0; i < n; ++i) {
        if (hg[i] != c.cpu(ha[i], hb[i])) {
            if (bad == 0) firsti = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[PASS] %-9s u8 generated == CPU reference        (65536 exhaustive)\n", c.name);
    } else {
        printf("[FAIL] %-9s u8 generated != CPU: %zu/65536; first a=%u b=%u gen=%u cpu=%u\n",
               c.name, bad, ha[firsti], hb[firsti], hg[firsti], c.cpu(ha[firsti], hb[firsti]));
        ++g_failures;
    }
    cudaFree(da); cudaFree(db); cudaFree(dg); cudaFree(dbp);
}

// ---- i8 exhaustive (bytes reinterpreted as signed) ---------------------------
struct I8Case {
    const char* name;
    void (*gen)(const signed char*, const signed char*, signed char*, long long);
    int8_t (*cpu)(int8_t, int8_t);
};

static int8_t cpu_addi8(int8_t a, int8_t b) { return (int8_t)(uint8_t)((uint8_t)a + (uint8_t)b); }
static int8_t cpu_muli8(int8_t a, int8_t b) {
    // promote to int (exact), truncate to 8 bits = wrapping two's-complement
    return (int8_t)(uint8_t)((uint32_t)((int)a * (int)b) & 0xFF);
}

static void run_i8_case(const I8Case& c) {
    std::vector<uint8_t> ha, hb;
    build_exhaustive_u8(ha, hb);
    size_t n = ha.size();
    signed char *da, *db, *dg;
    CUDA_OK(cudaMalloc(&da, n));
    CUDA_OK(cudaMalloc(&db, n));
    CUDA_OK(cudaMalloc(&dg, n));
    CUDA_OK(cudaMemcpy(da, ha.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), n, cudaMemcpyHostToDevice));
    LAUNCH(c.gen, da, db, dg, n);
    std::vector<uint8_t> hg(n);
    CUDA_OK(cudaMemcpy(hg.data(), dg, n, cudaMemcpyDeviceToHost));
    size_t bad = 0, firsti = 0;
    for (size_t i = 0; i < n; ++i) {
        int8_t want = c.cpu((int8_t)ha[i], (int8_t)hb[i]);
        if ((int8_t)hg[i] != want) {
            if (bad == 0) firsti = i;
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[PASS] %-9s i8 generated == CPU wrapping ref     (65536 exhaustive)\n", c.name);
    } else {
        printf("[FAIL] %-9s i8 generated != CPU: %zu/65536; first a=%d b=%d gen=%d\n", c.name,
               bad, (int8_t)ha[firsti], (int8_t)hb[firsti], (int8_t)hg[firsti]);
        ++g_failures;
    }
    cudaFree(da); cudaFree(db); cudaFree(dg);
}

// ---- 8-bit shifts vs the promotion model (b in [0,31] asserted) -------------
static void run_u8_shift(const char* name,
                         void (*gen)(const unsigned char*, const unsigned char*, unsigned char*,
                                     long long),
                         bool left) {
    std::vector<uint8_t> ha, hb;
    for (int a = 0; a < 256; ++a)
        for (int b : {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 24, 31, 32, 63, 255}) {
            ha.push_back((uint8_t)a);
            hb.push_back((uint8_t)b);
        }
    size_t n = ha.size();
    unsigned char *da, *db, *dg;
    CUDA_OK(cudaMalloc(&da, n));
    CUDA_OK(cudaMalloc(&db, n));
    CUDA_OK(cudaMalloc(&dg, n));
    CUDA_OK(cudaMemcpy(da, ha.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), n, cudaMemcpyHostToDevice));
    LAUNCH(gen, da, db, dg, n);
    std::vector<uint8_t> hg(n);
    CUDA_OK(cudaMemcpy(hg.data(), dg, n, cudaMemcpyDeviceToHost));
    size_t bad = 0, checked = 0;
    for (size_t i = 0; i < n; ++i) {
        if (hb[i] > 31) continue; // out-of-promoted-range: observe-only
        ++checked;
        // promotion model: zero-extend to 32-bit, shift, truncate mod 2^8.
        uint32_t wide = left ? ((uint32_t)ha[i] << hb[i]) : ((uint32_t)ha[i] >> hb[i]);
        uint8_t want = (uint8_t)wide;
        if (hg[i] != want) {
            if (bad == 0)
                printf("       first mismatch: a=%u b=%u gen=%u model=%u\n", ha[i], hb[i], hg[i],
                       want);
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[PASS] %-9s u8 generated == promotion model      (%zu probes, b in [0,31])\n",
               name, checked);
    } else {
        printf("[FAIL] %-9s u8 vs promotion model: %zu/%zu\n", name, bad, checked);
        ++g_failures;
    }
    // Observed out-of-range rows (no bespoke 8-bit sibling — informational).
    for (int be : {32, 255}) {
        for (size_t i = 0; i < n; ++i)
            if (ha[i] == 1 && hb[i] == be) {
                printf("       observed: 1 %s %d = %u (architecture-inherited)\n",
                       left ? "<<" : ">>", be, hg[i]);
                break;
            }
    }
    cudaFree(da); cudaFree(db); cudaFree(dg);
}

static void run_i8_shr() {
    std::vector<uint8_t> ha, hb;
    for (int a = 0; a < 256; ++a)
        for (int b = 0; b <= 31; ++b) {
            ha.push_back((uint8_t)a);
            hb.push_back((uint8_t)b);
        }
    size_t n = ha.size();
    signed char *da, *db, *dg;
    CUDA_OK(cudaMalloc(&da, n));
    CUDA_OK(cudaMalloc(&db, n));
    CUDA_OK(cudaMalloc(&dg, n));
    CUDA_OK(cudaMemcpy(da, ha.data(), n, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), n, cudaMemcpyHostToDevice));
    LAUNCH(baracuda_gen_shr_i8_scalar, da, db, dg, n);
    std::vector<uint8_t> hg(n);
    CUDA_OK(cudaMemcpy(hg.data(), dg, n, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        int8_t a = (int8_t)ha[i];
        int b = hb[i];
        // arithmetic shift of the sign-extended 32-bit value, truncated.
        uint32_t ua = (uint32_t)(int32_t)a;
        uint32_t r = (a < 0) ? ~(~ua >> b) : (ua >> b);
        int8_t want = (int8_t)(uint8_t)r;
        if ((int8_t)hg[i] != want) {
            if (bad == 0)
                printf("       first mismatch: a=%d b=%d gen=%d model=%d\n", a, b, (int8_t)hg[i],
                       want);
            ++bad;
        }
    }
    if (bad == 0) {
        printf("[PASS] shr       i8 ARITHMETIC (sign-replicating)  (%zu probes; -128>>7 == -1)\n",
               n);
    } else {
        printf("[FAIL] shr i8 vs arithmetic model: %zu/%zu\n", bad, n);
        ++g_failures;
    }
    cudaFree(da); cudaFree(db); cudaFree(dg);
}

int main() {
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s (sm_%d%d)\n\n", p.name, p.major, p.minor);

    // 1. i32 bitwise/shift: generated vs bespoke bit-exact + CPU subset.
    const I32Case i32cases[] = {
        {"band", baracuda_gen_band_i32_scalar, baracuda_kernels_binary_bitwise_and_i32_run,
         cpu_and32, false},
        {"bor", baracuda_gen_bor_i32_scalar, baracuda_kernels_binary_bitwise_or_i32_run, cpu_or32,
         false},
        {"bxor", baracuda_gen_bxor_i32_scalar, baracuda_kernels_binary_bitwise_xor_i32_run,
         cpu_xor32, false},
        {"shl", baracuda_gen_shl_i32_scalar, baracuda_kernels_binary_bitwise_left_shift_i32_run,
         cpu_shl32, true},
        {"shr", baracuda_gen_shr_i32_scalar, baracuda_kernels_binary_bitwise_right_shift_i32_run,
         cpu_shr32, true},
    };
    for (const auto& c : i32cases) run_i32_case(c);
    printf("\n");

    // 2. logical u8: generated vs bespoke bit-exact + CPU, exhaustive.
    const U8Case lcases[] = {
        {"land", baracuda_gen_land_u8_scalar, baracuda_kernels_binary_logical_and_bool_run,
         cpu_land},
        {"lor", baracuda_gen_lor_u8_scalar, baracuda_kernels_binary_logical_or_bool_run, cpu_lor},
        {"lxor", baracuda_gen_lxor_u8_scalar, baracuda_kernels_binary_logical_xor_bool_run,
         cpu_lxor},
    };
    for (const auto& c : lcases) run_u8_case(c);
    // The explicit normalization probe the charter asks for: 2 && 4 == 1.
    {
        uint8_t a2 = 2, b4 = 4, out = 0xEE;
        unsigned char *da, *db, *dy;
        CUDA_OK(cudaMalloc(&da, 1));
        CUDA_OK(cudaMalloc(&db, 1));
        CUDA_OK(cudaMalloc(&dy, 1));
        CUDA_OK(cudaMemcpy(da, &a2, 1, cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(db, &b4, 1, cudaMemcpyHostToDevice));
        LAUNCH(baracuda_gen_land_u8_scalar, da, db, dy, 1);
        CUDA_OK(cudaMemcpy(&out, dy, 1, cudaMemcpyDeviceToHost));
        if (out == 1) {
            printf("[PASS] land      normalization probe: 2 && 4 == 1 (not 2 & 4 == 0)\n");
        } else {
            printf("[FAIL] land normalization probe: 2 && 4 == %u\n", out);
            ++g_failures;
        }
        cudaFree(da); cudaFree(db); cudaFree(dy);
    }
    printf("\n");

    // 3. wrapping 8-bit arithmetic, exhaustive vs CPU.
    run_u8_case({"addw", baracuda_gen_addw_u8_scalar, nullptr, cpu_addu8});
    run_u8_case({"mulw", baracuda_gen_mulw_u8_scalar, nullptr, cpu_mulu8});
    run_i8_case({"addw", baracuda_gen_addw_i8_scalar, cpu_addi8});
    run_i8_case({"mulw", baracuda_gen_mulw_i8_scalar, cpu_muli8});
    printf("\n");

    // 4. 8-bit shifts vs the promotion model.
    run_u8_shift("shl", baracuda_gen_shl_u8_scalar, true);
    run_u8_shift("shr", baracuda_gen_shr_u8_scalar, false);
    run_i8_shr();

    printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
