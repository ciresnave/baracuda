// On-device validation of the increment-11 IM2COL kernels (2-D im2col / unfold —
// the conv-lowering expanding gather). The generated kernel is a PURE RAW-BIT
// gather-or-zero (one read-or-zero + one store per output cell, ZERO arithmetic),
// so it is VariantFidelity::BitIdentical and byte-identical to the bespoke
// `im2col_2d_fw_kernel` across the ENTIRE probe space (NaN payloads, +/-inf, +/-0,
// subnormals) with NO carve-out — there is no NaN/tie convention to diverge on.
//
// Acceptance (per output cell), whole-buffer memcmp `bit_diff == 0`:
//   1. y_gen (the generated baracuda_gen_*_im2col kernel)
//   2. y_bespoke (baracuda::im2col::launch_im2col_2d<T>, the shipped oracle)
//   3. y_cpu (an INDEPENDENT CPU closed-form byte oracle — catches a self-consistent
//      WRONG column order / axis swap that the bespoke restatement alone could not).
// Same layout [N, C*kh*kw, oH*oW], same strides ⇒ a flat memcmp is exact.
//
// The generated .cu kernels are #included by name (they must sit in this dir — the
// `dump_im2col_sources` test regenerates them; see ondevice/README.md). Kernel
// signature is uniform:
//   (const T* in0, T* out, long long N, long long C, long long H_in, long long W_in,
//    long long oH, long long oW)
// The conv geometry (kernel/stride/pad/dilation) is baked into each kernel as
// compile-time literals — the op_name encodes it so distinct geometries emit
// distinct symbols (the case table below mirrors dump_im2col_sources).
//
// Build (needs the bespoke header include dir):
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//        -I C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include \
//        im2col_validate.cu -o im2col_validate && ./im2col_validate
// Sanitizers (small shapes via the `san` argv). Which layer catches what: the
// 0xAA-poison + memcmp proves write-completeness (an unwritten slot stays 0xAA != the
// gather/zero -> diverges from the oracle); memcheck catches a true out-of-allocation
// source read (a corner tap indexing past the whole input); initcheck guards any read
// of genuinely-uninitialized global memory. (initcheck can NOT see an under-written
// slot here -- the output is poison-INITIALIZED, not uninitialized.)
//   compute-sanitizer --tool memcheck  ./im2col_validate san
//   compute-sanitizer --tool racecheck ./im2col_validate san
//   compute-sanitizer --tool synccheck ./im2col_validate san
//   compute-sanitizer --tool initcheck ./im2col_validate san
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- the bespoke oracle (template launcher, extends to i32/i64) ----
#include "baracuda_im2col.cuh"

// ---- generated im2col kernels (mirrors dump_im2col_sources: 6 geometries x 6
// dtypes = 36 cells). Symbol = baracuda_gen_{geometry}_{dtag}_im2col. ----
#include "baracuda_gen_k1_f32_im2col.cu"
#include "baracuda_gen_k3p1_f32_im2col.cu"
#include "baracuda_gen_k3s2_f32_im2col.cu"
#include "baracuda_gen_k35_f32_im2col.cu"
#include "baracuda_gen_s3k2_f32_im2col.cu"
#include "baracuda_gen_d2_f32_im2col.cu"
#include "baracuda_gen_k1_f64_im2col.cu"
#include "baracuda_gen_k3p1_f64_im2col.cu"
#include "baracuda_gen_k3s2_f64_im2col.cu"
#include "baracuda_gen_k35_f64_im2col.cu"
#include "baracuda_gen_s3k2_f64_im2col.cu"
#include "baracuda_gen_d2_f64_im2col.cu"
#include "baracuda_gen_k1_f16_im2col.cu"
#include "baracuda_gen_k3p1_f16_im2col.cu"
#include "baracuda_gen_k3s2_f16_im2col.cu"
#include "baracuda_gen_k35_f16_im2col.cu"
#include "baracuda_gen_s3k2_f16_im2col.cu"
#include "baracuda_gen_d2_f16_im2col.cu"
#include "baracuda_gen_k1_bf16_im2col.cu"
#include "baracuda_gen_k3p1_bf16_im2col.cu"
#include "baracuda_gen_k3s2_bf16_im2col.cu"
#include "baracuda_gen_k35_bf16_im2col.cu"
#include "baracuda_gen_s3k2_bf16_im2col.cu"
#include "baracuda_gen_d2_bf16_im2col.cu"
#include "baracuda_gen_k1_i32_im2col.cu"
#include "baracuda_gen_k3p1_i32_im2col.cu"
#include "baracuda_gen_k3s2_i32_im2col.cu"
#include "baracuda_gen_k35_i32_im2col.cu"
#include "baracuda_gen_s3k2_i32_im2col.cu"
#include "baracuda_gen_d2_i32_im2col.cu"
#include "baracuda_gen_k1_i64_im2col.cu"
#include "baracuda_gen_k3p1_i64_im2col.cu"
#include "baracuda_gen_k3s2_i64_im2col.cu"
#include "baracuda_gen_k35_i64_im2col.cu"
#include "baracuda_gen_s3k2_i64_im2col.cu"
#include "baracuda_gen_d2_i64_im2col.cu"

static int fails = 0;
static long long cells = 0;

// Geometry: (kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w).
struct Geo {
    int kh, kw, sh, sw, ph, pw, dh, dw;
};

// h_out = (H_in + 2*pad_h - dilation_h*(kh-1) - 1)/stride_h + 1   (bespoke formula)
static long long out_dim(long long in, int k, int s, int p, int d) {
    return (in + 2 * p - (long long)d * (k - 1) - 1) / s + 1;
}

// A generated-kernel function pointer (the uniform 6-extent ABI).
template <typename T>
using GenKernel = void (*)(const T*, T*, long long, long long, long long, long long,
                           long long, long long);

static int grid_for(long long total) {
    long long b = (total + 255) / 256;
    if (b < 1) b = 1;
    return (int)(b > 65535 ? 65535 : b);
}

// Probe-seed a byte buffer: a byte LCG covers the FULL bit space (every element is
// an arbitrary bit pattern — NaN payloads / subnormals / +/-inf / +/-0 all appear),
// with a few explicit special f32/f16 patterns planted up front. Since im2col is a
// raw-bit gather, EVERY bit pattern must be preserved verbatim.
static void seed_bytes(uint8_t* p, size_t n) {
    uint32_t s = 0x9E3779B9u;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = (uint8_t)(s >> 24);
    }
    // Plant explicit specials (4-byte aligned front region).
    const uint32_t specials[] = {
        0x7FC00001u, // qNaN with payload
        0x7F800001u, // sNaN with payload
        0xFF800000u, // -inf
        0x7F800000u, // +inf
        0x80000000u, // -0.0
        0x00000000u, // +0.0
        0x00000001u, // smallest positive subnormal
        0x807FFFFFu, // negative subnormal
    };
    size_t words = n / 4;
    for (size_t w = 0; w < words && w < sizeof(specials) / 4; ++w) {
        memcpy(p + w * 4, &specials[w], 4);
    }
}

// The independent CPU closed-form BYTE oracle (Layout A). For each output cell it
// copies `sz` source bytes when in-bounds, else zeroes them (the typed zero is
// all-zero bits for every dtype). Kernel-independent — catches a wrong column order.
static void cpu_im2col(const uint8_t* x, uint8_t* y, int sz, long long N, long long C,
                       long long H, long long W, long long oH, long long oW, const Geo& g) {
    long long spatial = oH * oW;
    long long col_rows = C * g.kh * g.kw;
    long long total = N * col_rows * spatial;
    for (long long tid = 0; tid < total; ++tid) {
        long long col = tid % spatial;
        long long row_full = tid / spatial;
        long long oh = col / oW;
        long long ow = col - oh * oW;
        long long patch = row_full % col_rows;
        long long n = row_full / col_rows;
        long long c = patch / (g.kh * g.kw);
        long long kij = patch - c * (g.kh * g.kw);
        long long ki = kij / g.kw;
        long long kj = kij - ki * g.kw;
        long long in_h = oh * g.sh - g.ph + ki * g.dh;
        long long in_w = ow * g.sw - g.pw + kj * g.dw;
        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
            long long src = (((n * C + c) * H + in_h) * W + in_w);
            memcpy(y + tid * sz, x + src * sz, sz);
        } else {
            memset(y + tid * sz, 0, sz);
        }
    }
}

static long long bit_diff(const uint8_t* a, const uint8_t* b, size_t n) {
    long long d = 0;
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) d++;
    return d;
}

// Run one cell: seed input, launch gen + bespoke, compute CPU oracle, memcmp all 3.
template <typename T>
static void run_cell(const char* nm, GenKernel<T> gen, const Geo& g, long long N,
                     long long C, long long H, long long W) {
    const int sz = (int)sizeof(T);
    long long oH = out_dim(H, g.kh, g.sh, g.ph, g.dh);
    long long oW = out_dim(W, g.kw, g.sw, g.pw, g.dw);
    if (oH <= 0 || oW <= 0) {
        printf("SKIP %-30s (degenerate oH=%lld oW=%lld)\n", nm, oH, oW);
        return;
    }
    cells++;
    long long in_n = N * C * H * W;
    long long out_n = N * (long long)C * g.kh * g.kw * oH * oW;

    std::vector<uint8_t> h_in((size_t)in_n * sz);
    seed_bytes(h_in.data(), h_in.size());

    T *d_in = nullptr, *d_gen = nullptr, *d_bsp = nullptr;
    cudaMalloc(&d_in, (size_t)in_n * sz);
    cudaMalloc(&d_gen, (size_t)out_n * sz);
    cudaMalloc(&d_bsp, (size_t)out_n * sz);
    cudaMemcpy(d_in, h_in.data(), (size_t)in_n * sz, cudaMemcpyHostToDevice);
    // Poison both outputs with 0xAA so an UNWRITTEN slot (a total-count bug) diverges
    // from the oracle (0xAA != the raw-bit gather / zero) — on top of initcheck.
    cudaMemset(d_gen, 0xAA, (size_t)out_n * sz);
    cudaMemset(d_bsp, 0xAA, (size_t)out_n * sz);

    gen<<<grid_for(out_n), 256>>>(d_in, d_gen, N, C, H, W, oH, oW);
    int rc = baracuda::im2col::launch_im2col_2d<T>(
        (int)N, (int)C, (int)H, (int)W, (int)oH, (int)oW, g.kh, g.kw, g.sh, g.sw,
        g.ph, g.pw, g.dh, g.dw, d_in, d_bsp, 0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    std::vector<uint8_t> h_gen((size_t)out_n * sz), h_bsp((size_t)out_n * sz),
        h_ora((size_t)out_n * sz);
    cudaMemcpy(h_gen.data(), d_gen, (size_t)out_n * sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bsp.data(), d_bsp, (size_t)out_n * sz, cudaMemcpyDeviceToHost);
    cpu_im2col(h_in.data(), h_ora.data(), sz, N, C, H, W, oH, oW, g);

    long long d_bsp_diff = bit_diff(h_gen.data(), h_bsp.data(), h_gen.size());
    long long d_ora_diff = bit_diff(h_gen.data(), h_ora.data(), h_gen.size());
    bool ok = (rc == 0) && (err == cudaSuccess) && d_bsp_diff == 0 && d_ora_diff == 0;
    if (ok)
        printf("PASS %-30s [N%lld C%lld %lldx%lld -> %lldx%lld] bit_diff bespoke=0 cpu=0\n",
               nm, N, C, H, W, oH, oW);
    else {
        printf("FAIL %-30s bespoke_diff=%lld cpu_diff=%lld rc=%d err=%s\n", nm,
               d_bsp_diff, d_ora_diff, rc, cudaGetErrorString(err));
        fails++;
    }
    cudaFree(d_in);
    cudaFree(d_gen);
    cudaFree(d_bsp);
}

// Determinism: two gen launches on the same input must be byte-identical.
template <typename T>
static void determinism(const char* nm, GenKernel<T> gen, const Geo& g, long long N,
                        long long C, long long H, long long W) {
    const int sz = (int)sizeof(T);
    long long oH = out_dim(H, g.kh, g.sh, g.ph, g.dh);
    long long oW = out_dim(W, g.kw, g.sw, g.pw, g.dw);
    long long in_n = N * C * H * W, out_n = N * (long long)C * g.kh * g.kw * oH * oW;
    std::vector<uint8_t> h_in((size_t)in_n * sz);
    seed_bytes(h_in.data(), h_in.size());
    T *d_in = nullptr, *d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&d_in, (size_t)in_n * sz);
    cudaMalloc(&d_a, (size_t)out_n * sz);
    cudaMalloc(&d_b, (size_t)out_n * sz);
    cudaMemcpy(d_in, h_in.data(), (size_t)in_n * sz, cudaMemcpyHostToDevice);
    gen<<<grid_for(out_n), 256>>>(d_in, d_a, N, C, H, W, oH, oW);
    gen<<<grid_for(out_n), 256>>>(d_in, d_b, N, C, H, W, oH, oW);
    cudaDeviceSynchronize();
    std::vector<uint8_t> a((size_t)out_n * sz), b((size_t)out_n * sz);
    cudaMemcpy(a.data(), d_a, a.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, b.size(), cudaMemcpyDeviceToHost);
    bool ok = bit_diff(a.data(), b.data(), a.size()) == 0;
    printf(ok ? "PASS %-30s (run-to-run deterministic)\n"
              : "FAIL %-30s (non-deterministic)\n", nm);
    if (!ok) fails++;
    cudaFree(d_in);
    cudaFree(d_a);
    cudaFree(d_b);
}

// GB/s bench (pure memory traffic) vs the bespoke.
static void bench() {
    Geo g{3, 3, 1, 1, 1, 1, 1, 1};
    long long N = 8, C = 64, H = 56, W = 56;
    long long oH = out_dim(H, g.kh, g.sh, g.ph, g.dh);
    long long oW = out_dim(W, g.kw, g.sw, g.pw, g.dw);
    long long in_n = N * C * H * W, out_n = N * C * g.kh * g.kw * oH * oW;
    std::vector<float> h_in((size_t)in_n, 1.0f);
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, (size_t)in_n * 4);
    cudaMalloc(&d_out, (size_t)out_n * 4);
    cudaMemcpy(d_in, h_in.data(), (size_t)in_n * 4, cudaMemcpyHostToDevice);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    baracuda_gen_k3p1_f32_im2col<<<grid_for(out_n), 256>>>(d_in, d_out, N, C, H, W, oH, oW);
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int it = 0; it < 30; ++it)
        baracuda_gen_k3p1_f32_im2col<<<grid_for(out_n), 256>>>(d_in, d_out, N, C, H, W, oH, oW);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float gms = 0;
    cudaEventElapsedTime(&gms, s, e);
    gms /= 30.0f;
    cudaEventRecord(s);
    for (int it = 0; it < 30; ++it)
        baracuda::im2col::launch_im2col_2d<float>((int)N, (int)C, (int)H, (int)W, (int)oH,
                                                  (int)oW, g.kh, g.kw, g.sh, g.sw, g.ph,
                                                  g.pw, g.dh, g.dw, d_in, d_out, 0);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float bms = 0;
    cudaEventElapsedTime(&bms, s, e);
    bms /= 30.0f;
    double bytes = (double)(in_n + out_n) * 4.0; // approx read+write
    printf("PERF im2col 3x3 [N%lld C%lld %lldx%lld]: gen %.2f ms (%.0f GB/s), bespoke %.2f ms (%.0f GB/s)\n",
           N, C, H, W, gms, bytes / (gms * 1e-3) / 1e9, bms, bytes / (bms * 1e-3) / 1e9);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
}

// One (dtype-specialized) sweep of all 6 geometries over the non-square batched
// shape N=2,C=3,H=5,W=7 (H!=W catches oh/ow + H/W axis swaps; N>1 catches the batch
// stride; the geometries cover 1x1, 3x3+pad OOB, stride>kernel, kh!=kw, dilation>1).
#define SWEEP(T, DTAG)                                                                    \
    do {                                                                                  \
        run_cell<T>("k1_" #DTAG, baracuda_gen_k1_##DTAG##_im2col, gk1, N, C, H, W);       \
        run_cell<T>("k3p1_" #DTAG, baracuda_gen_k3p1_##DTAG##_im2col, gk3p1, N, C, H, W); \
        run_cell<T>("k3s2_" #DTAG, baracuda_gen_k3s2_##DTAG##_im2col, gk3s2, N, C, H, W); \
        run_cell<T>("k35_" #DTAG, baracuda_gen_k35_##DTAG##_im2col, gk35, N, C, H, W);    \
        run_cell<T>("s3k2_" #DTAG, baracuda_gen_s3k2_##DTAG##_im2col, gs3k2, N, C, H, W); \
        run_cell<T>("d2_" #DTAG, baracuda_gen_d2_##DTAG##_im2col, gd2, N, C, H, W);       \
    } while (0)

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);

    // Geometry table (mirrors dump_im2col_sources).
    Geo gk1{1, 1, 1, 1, 0, 0, 1, 1};
    Geo gk3p1{3, 3, 1, 1, 1, 1, 1, 1};
    Geo gk3s2{3, 3, 2, 2, 1, 1, 1, 1};
    Geo gk35{3, 5, 1, 1, 1, 2, 1, 1};
    Geo gs3k2{2, 2, 3, 3, 0, 0, 1, 1};
    Geo gd2{3, 3, 1, 1, 2, 2, 2, 2};

    // Non-square batched base shape (small under `san` for the sanitizer sweep).
    long long N = san ? 2 : 2, C = san ? 2 : 3, H = san ? 5 : 5, W = san ? 7 : 7;

    SWEEP(float, f32);
    SWEEP(double, f64);
    SWEEP(__half, f16);
    SWEEP(__nv_bfloat16, bf16);
    SWEEP(int32_t, i32);
    SWEEP(int64_t, i64);

    // Extra ABI-order shape cells (the 6-extent ABI order is silently wrong on a
    // swapped H/W or oH/oW): minimal 1x1x1x1, a square in-bounds no-pad case, and a
    // larger non-square batched shape — all through the k1 (identity) + k3p1 kernels.
    if (!san) {
        run_cell<float>("k1 minimal 1x1x1x1", baracuda_gen_k1_f32_im2col, gk1, 1, 1, 1, 1);
        run_cell<float>("k1 square 8x8", baracuda_gen_k1_f32_im2col, gk1, 3, 4, 8, 8);
        run_cell<float>("k3p1 nonsq 9x11", baracuda_gen_k3p1_f32_im2col, gk3p1, 3, 4, 9, 11);
        run_cell<double>("k3p1 nonsq 11x9 f64", baracuda_gen_k3p1_f64_im2col, gk3p1, 2, 3, 11, 9);
        run_cell<__half>("k35 nonsq 13x7 f16", baracuda_gen_k35_f16_im2col, gk35, 2, 3, 13, 7);
    }

    determinism<float>("k3p1 f32 determinism", baracuda_gen_k3p1_f32_im2col, gk3p1, N, C, H, W);

    if (!san) bench();

    printf(fails ? "\n%lld cells checked; %d CHECK(S) FAILED\n"
                 : "\n%lld cells checked; ALL PASSED\n", cells, fails);
    return fails ? 1 : 0;
}
