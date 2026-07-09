// On-device BIT-IDENTITY validation of the bespoke NaN-PROPAGATING relu kernel
// (baracuda-kernels-sys `unary_relu_propagating_fp.cu`, the family Fuel rebinds
// OpKind::ReluElementwise to) against the baracuda-kernelgen GENERATED relu (the
// semantics oracle — cuda.rs UnaryOp::Relu). Requires raw output bytes to be
// memcmp-identical between:
//   - the bespoke extern-C launcher (`baracuda_kernels_unary_relu_propagating_*_run`)
//   - the generated SCALAR relu kernel (per-element oracle)
//   - the generated VECTORIZED contig relu kernel (the path Fuel adopts)
// AND validates the bespoke STRIDED launcher (`*_strided_run`) and all 8
// `*_can_implement` / `*_strided_can_implement` host gates are device-run/called.
//
// Coverage:
//   f16 / bf16 : FULL 16-bit sweep (all 65536 patterns — every NaN payload,
//                +/-Inf, +/-0, every subnormal, max-finite).
//   f32 / f64  : a MATCHED targeted probe set (all NaN payload classes, sNaN,
//                +/-0, +/-Inf, min/max subnormal, min normal, max finite,
//                boundary negatives) PLUS a large pseudo-random bit-pattern sweep.
//   strided    : a transposed 2-D view per dtype, diffed against the (already
//                oracle-validated) contig bespoke output — the `*_strided_run`
//                launcher device-launched over probes + random patterns.
//   gates      : all 8 `*_can_implement` / `*_strided_can_implement` called.
//
// Regeneration (from a VS dev shell) — first dump the generated oracle .cu, then
// build the harness beside them:
//   RELU_OUT=<outdir> cargo test -p baracuda-kernelgen dump_relu_sources -- --ignored --nocapture
//   cp crates/baracuda-kernelgen/ondevice/relu_propagating_validate.cu <outdir>/
//   nvcc -O3 -arch=sm_89 -std=c++17 \
//        -I C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include \
//        <outdir>/relu_propagating_validate.cu -o relu_propagating_validate
//   ./relu_propagating_validate
//   compute-sanitizer --tool memcheck ./relu_propagating_validate

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GENERATED relu (the semantics oracle): scalar per-element + vectorized contig.
#include "baracuda_gen_relu_f32_scalar.cu"
#include "baracuda_gen_relu_f32_co_v4.cu"
#include "baracuda_gen_relu_f64_scalar.cu"
#include "baracuda_gen_relu_f64_co_v2.cu"
#include "baracuda_gen_relu_f16_scalar.cu"
#include "baracuda_gen_relu_f16_co_v8.cu"
#include "baracuda_gen_relu_bf16_scalar.cu"
#include "baracuda_gen_relu_bf16_co_v8.cu"

// BESPOKE NaN-propagating relu (extern-C launchers), from the kernels-sys source.
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/elementwise/unary_relu_propagating_fp.cu"

static int fails = 0;

// memcmp two device buffers of `bytes` bytes; report the first mismatch.
static bool diff_bytes(const char* label, const void* d_a, const void* d_b, size_t bytes, size_t esz) {
    std::vector<uint8_t> a(bytes), b(bytes);
    cudaMemcpy(a.data(), d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_b, bytes, cudaMemcpyDeviceToHost);
    if (memcmp(a.data(), b.data(), bytes) == 0) return true;
    for (size_t i = 0; i < bytes; ++i) {
        if (a[i] != b[i]) {
            size_t elem = i / esz;
            printf("FAIL %-44s first mismatch at elem %zu (byte %zu): 0x%02x vs 0x%02x\n",
                   label, elem, i, a[i], b[i]);
            break;
        }
    }
    fails++;
    return false;
}

static void ok(const char* label, long long n) {
    printf("PASS %-44s bit-identical over %lld elems\n", label, n);
}

// Host builder for the f32 probe set (returned as u32 bit patterns).
static std::vector<uint32_t> probe_u32() {
    return {
        0x00000000u, 0x80000000u,             // +0, -0
        0x3f800000u, 0xbf800000u,             // +1, -1
        0x7f800000u, 0xff800000u,             // +inf, -inf
        0x7fc00000u, 0xffc00000u,             // qNaN, -qNaN
        0x7fc00001u, 0x7fdead01u, 0x7ff00000u,// qNaN payloads (three)
        0x7f800001u, 0x7fbfffffu,             // sNaN payloads (two)
        0xffc0beefu, 0xffbfffffu,             // negative NaN payloads (two)
        0x00000001u, 0x80000001u,             // +/- smallest subnormal
        0x007fffffu, 0x807fffffu,             // +/- largest subnormal
        0x00800000u, 0x80800000u,             // +/- smallest normal
        0x7f7fffffu, 0xff7fffffu,             // +/- largest finite
        0x40490fdbu, 0xc0490fdbu,             // +/-pi
    };
}
// f64 probe set — MATCHED class-for-class to probe_u32() (finding 15): three
// qNaN payloads, two sNaN payloads, two negative-NaN payloads. 25 encodings.
static std::vector<uint64_t> probe_u64() {
    return {
        0x0000000000000000ull, 0x8000000000000000ull, // +0,-0
        0x3ff0000000000000ull, 0xbff0000000000000ull, // +1,-1
        0x7ff0000000000000ull, 0xfff0000000000000ull, // +inf,-inf
        0x7ff8000000000000ull, 0xfff8000000000000ull, // qNaN,-qNaN
        0x7ff8000000000001ull, 0x7ffdeadbeef00001ull, // qNaN payloads #1,#2
        0x7ffc0ffee0000000ull,                        // qNaN payload #3 (was missing)
        0x7ff0000000000001ull, 0x7ff7ffffffffffffull, // sNaN payloads (two)
        0xfff8badc0ffee000ull, 0xfff4baddcafe0001ull, // negative NaN payloads #1,#2 (#2 was missing)
        0x0000000000000001ull, 0x8000000000000001ull, // +/- min subnormal
        0x000fffffffffffffull, 0x800fffffffffffffull, // +/- max subnormal
        0x0010000000000000ull, 0x8010000000000000ull, // +/- min normal
        0x7fefffffffffffffull, 0xffefffffffffffffull, // +/- max finite
        0x400921fb54442d18ull, 0xc00921fb54442d18ull, // +/-pi
    };
}

// extern-C launcher function-pointer types (match the UNARY_POINTWISE macros).
typedef int32_t (*contig_run_t)(int64_t, const void*, void*, void*, size_t, void*);
typedef int32_t (*strided_run_t)(int64_t, int32_t, const int32_t*, const int64_t*,
                                 const int64_t*, const void*, void*, void*, size_t, void*);

// Validate the bespoke `*_strided_run` launcher through a TRANSPOSED 2-D view:
// the transposed-read / contiguous-write output must byte-match the contig
// bespoke output on the same logical values (which the full sweeps already prove
// bit-identical to the generated oracle). `T` is the raw-bit element type.
template <typename T>
static void strided_transpose_check(const char* label, contig_run_t crun, strided_run_t srun,
                                    const std::vector<T>& src, int R, int C) {
    const long long n = (long long)R * C;
    // x = transpose(src) so a transposed-read strided run reproduces relu(src):
    //   y[r*C+c] = relu(x[c*R + r]); want == relu(src[r*C+c]) => x[c*R+r] = src[r*C+c].
    std::vector<T> x((size_t)n);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            x[(size_t)c * R + r] = src[(size_t)r * C + c];

    T *d_src, *d_x, *d_ref, *d_y;
    cudaMalloc(&d_src, n * sizeof(T)); cudaMalloc(&d_x, n * sizeof(T));
    cudaMalloc(&d_ref, n * sizeof(T)); cudaMalloc(&d_y, n * sizeof(T));
    cudaMemcpy(d_src, src.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_ref, 0x5B, n * sizeof(T)); cudaMemset(d_y, 0x3C, n * sizeof(T));

    // Reference: contig bespoke relu on src (== the oracle-validated output).
    crun(n, d_src, d_ref, nullptr, 0, nullptr);
    // Strided: transposed read (stride [1, R]), contiguous write (stride [C, 1]).
    int32_t shape[2] = { R, C };
    int64_t sx[2] = { 1, (int64_t)R };
    int64_t sy[2] = { (int64_t)C, 1 };
    int32_t rc = srun(n, 2, shape, sx, sy, d_x, d_y, nullptr, 0, nullptr);
    cudaDeviceSynchronize();
    if (rc != 0) { printf("FAIL %-44s strided_run returned %d\n", label, rc); fails++; }
    else if (diff_bytes(label, d_ref, d_y, n * sizeof(T), sizeof(T))) ok(label, n);
    cudaFree(d_src); cudaFree(d_x); cudaFree(d_ref); cudaFree(d_y);
}

int main() {
    // ================= f16 / bf16 : full 16-bit sweep =================
    {
        const long long n = 1 << 16;      // every u16 pattern; multiple of 8
        const long long nv8 = n / 8;
        std::vector<uint16_t> h(n);
        for (long long i = 0; i < n; ++i) h[i] = (uint16_t)i;
        uint16_t *d_in, *d_be, *d_gs, *d_gv;
        cudaMalloc(&d_in, n*2); cudaMalloc(&d_be, n*2); cudaMalloc(&d_gs, n*2); cudaMalloc(&d_gv, n*2);
        cudaMemcpy(d_in, h.data(), n*2, cudaMemcpyHostToDevice);

        // f16
        cudaMemset(d_be,0xA1,n*2); cudaMemset(d_gs,0x5B,n*2); cudaMemset(d_gv,0x3C,n*2);
        baracuda_kernels_unary_relu_propagating_f16_run(n, d_in, d_be, nullptr, 0, nullptr);
        baracuda_gen_relu_f16_scalar<<<128,256>>>((const __half*)d_in, (__half*)d_gs, n);
        baracuda_gen_relu_f16_co_v8<<<128,256>>>((const baracuda_gen_relu_f16_co_v8_vec*)d_in,
                                                 (baracuda_gen_relu_f16_co_v8_vec*)d_gv, nv8);
        cudaDeviceSynchronize();
        if (diff_bytes("f16 bespoke==gen_scalar (full sweep)", d_be, d_gs, n*2, 2)) ok("f16 bespoke==gen_scalar (full sweep)", n);
        if (diff_bytes("f16 bespoke==gen_vec_v8 (full sweep)", d_be, d_gv, n*2, 2)) ok("f16 bespoke==gen_vec_v8 (full sweep)", n);

        // bf16
        cudaMemset(d_be,0xA1,n*2); cudaMemset(d_gs,0x5B,n*2); cudaMemset(d_gv,0x3C,n*2);
        baracuda_kernels_unary_relu_propagating_bf16_run(n, d_in, d_be, nullptr, 0, nullptr);
        baracuda_gen_relu_bf16_scalar<<<128,256>>>((const __nv_bfloat16*)d_in, (__nv_bfloat16*)d_gs, n);
        baracuda_gen_relu_bf16_co_v8<<<128,256>>>((const baracuda_gen_relu_bf16_co_v8_vec*)d_in,
                                                  (baracuda_gen_relu_bf16_co_v8_vec*)d_gv, nv8);
        cudaDeviceSynchronize();
        if (diff_bytes("bf16 bespoke==gen_scalar (full sweep)", d_be, d_gs, n*2, 2)) ok("bf16 bespoke==gen_scalar (full sweep)", n);
        if (diff_bytes("bf16 bespoke==gen_vec_v8 (full sweep)", d_be, d_gv, n*2, 2)) ok("bf16 bespoke==gen_vec_v8 (full sweep)", n);

        cudaFree(d_in); cudaFree(d_be); cudaFree(d_gs); cudaFree(d_gv);
    }

    // ================= f32 : probe set + random sweep =================
    {
        auto probes = probe_u32();
        const long long nrand = 64ll * 1024 * 1024;                 // 64M random patterns
        long long n = ((long long)probes.size() + nrand + 3) & ~3ll; // pad to multiple of 4
        std::vector<uint32_t> h(n);
        for (size_t i = 0; i < probes.size(); ++i) h[i] = probes[i];
        uint64_t s = 0x9e3779b97f4a7c15ull;
        for (long long i = probes.size(); i < n; ++i) { s ^= s<<13; s ^= s>>7; s ^= s<<17; h[i] = (uint32_t)s; }
        uint32_t *d_in,*d_be,*d_gs,*d_gv;
        cudaMalloc(&d_in,n*4); cudaMalloc(&d_be,n*4); cudaMalloc(&d_gs,n*4); cudaMalloc(&d_gv,n*4);
        cudaMemcpy(d_in,h.data(),n*4,cudaMemcpyHostToDevice);
        cudaMemset(d_be,0xA1,n*4); cudaMemset(d_gs,0x5B,n*4); cudaMemset(d_gv,0x3C,n*4);
        baracuda_kernels_unary_relu_propagating_f32_run(n, d_in, d_be, nullptr, 0, nullptr);
        baracuda_gen_relu_f32_scalar<<<256,256>>>((const float*)d_in,(float*)d_gs,n);
        baracuda_gen_relu_f32_co_v4<<<256,256>>>((const float4*)d_in,(float4*)d_gv,n/4);
        cudaDeviceSynchronize();
        if (diff_bytes("f32 bespoke==gen_scalar (probe+64M rand)", d_be, d_gs, n*4, 4)) ok("f32 bespoke==gen_scalar (probe+64M rand)", n);
        if (diff_bytes("f32 bespoke==gen_vec_v4 (probe+64M rand)", d_be, d_gv, n*4, 4)) ok("f32 bespoke==gen_vec_v4 (probe+64M rand)", n);
        cudaFree(d_in);cudaFree(d_be);cudaFree(d_gs);cudaFree(d_gv);
    }

    // ================= f64 : probe set + random sweep =================
    {
        auto probes = probe_u64();
        const long long nrand = 32ll * 1024 * 1024;                 // 32M random patterns
        long long n = ((long long)probes.size() + nrand + 1) & ~1ll; // pad to multiple of 2
        std::vector<uint64_t> h(n);
        for (size_t i = 0; i < probes.size(); ++i) h[i] = probes[i];
        uint64_t s = 0xda3e39cb94b95bdbull;
        for (long long i = probes.size(); i < n; ++i) { s ^= s<<13; s ^= s>>7; s ^= s<<17; h[i] = s; }
        uint64_t *d_in,*d_be,*d_gs,*d_gv;
        cudaMalloc(&d_in,n*8); cudaMalloc(&d_be,n*8); cudaMalloc(&d_gs,n*8); cudaMalloc(&d_gv,n*8);
        cudaMemcpy(d_in,h.data(),n*8,cudaMemcpyHostToDevice);
        cudaMemset(d_be,0xA1,n*8); cudaMemset(d_gs,0x5B,n*8); cudaMemset(d_gv,0x3C,n*8);
        baracuda_kernels_unary_relu_propagating_f64_run(n, d_in, d_be, nullptr, 0, nullptr);
        baracuda_gen_relu_f64_scalar<<<256,256>>>((const double*)d_in,(double*)d_gs,n);
        baracuda_gen_relu_f64_co_v2<<<256,256>>>((const double2*)d_in,(double2*)d_gv,n/2);
        cudaDeviceSynchronize();
        if (diff_bytes("f64 bespoke==gen_scalar (probe+32M rand)", d_be, d_gs, n*8, 8)) ok("f64 bespoke==gen_scalar (probe+32M rand)", n);
        if (diff_bytes("f64 bespoke==gen_vec_v2 (probe+32M rand)", d_be, d_gv, n*8, 8)) ok("f64 bespoke==gen_vec_v2 (probe+32M rand)", n);
        cudaFree(d_in);cudaFree(d_be);cudaFree(d_gs);cudaFree(d_gv);
    }

    // ================= strided launchers (transposed 2-D view) =================
    // Build a modest probe+random sweep per dtype, transpose it, and require the
    // bespoke `*_strided_run` output to byte-match the contig bespoke output. This
    // device-launches all 4 `*_strided_run` symbols (previously covered only by
    // shared-template construction).
    {
        const int R = 1024, C = 768;             // non-square ⇒ catches axis-swap bugs
        const long long n = (long long)R * C;
        // f32
        {
            auto p = probe_u32();
            std::vector<uint32_t> h((size_t)n);
            for (size_t i = 0; i < p.size(); ++i) h[i] = p[i];
            uint64_t s = 0x243f6a8885a308d3ull;
            for (long long i = p.size(); i < n; ++i) { s ^= s<<13; s ^= s>>7; s ^= s<<17; h[i] = (uint32_t)s; }
            strided_transpose_check<uint32_t>("f32 strided(transpose)==contig",
                baracuda_kernels_unary_relu_propagating_f32_run,
                baracuda_kernels_unary_relu_propagating_f32_strided_run, h, R, C);
        }
        // f64
        {
            auto p = probe_u64();
            std::vector<uint64_t> h((size_t)n);
            for (size_t i = 0; i < p.size(); ++i) h[i] = p[i];
            uint64_t s = 0x13198a2e03707344ull;
            for (long long i = p.size(); i < n; ++i) { s ^= s<<13; s ^= s>>7; s ^= s<<17; h[i] = s; }
            strided_transpose_check<uint64_t>("f64 strided(transpose)==contig",
                baracuda_kernels_unary_relu_propagating_f64_run,
                baracuda_kernels_unary_relu_propagating_f64_strided_run, h, R, C);
        }
        // f16 / bf16 (raw u16 patterns: first 65536 are the full sweep, rest random).
        for (int is_bf16 = 0; is_bf16 < 2; ++is_bf16) {
            std::vector<uint16_t> h((size_t)n);
            for (long long i = 0; i < n && i < (1 << 16); ++i) h[i] = (uint16_t)i;
            uint64_t s = is_bf16 ? 0xa4093822299f31d0ull : 0x082efa98ec4e6c89ull;
            for (long long i = (1 << 16); i < n; ++i) { s ^= s<<13; s ^= s>>7; s ^= s<<17; h[i] = (uint16_t)s; }
            if (is_bf16)
                strided_transpose_check<uint16_t>("bf16 strided(transpose)==contig",
                    baracuda_kernels_unary_relu_propagating_bf16_run,
                    baracuda_kernels_unary_relu_propagating_bf16_strided_run, h, R, C);
            else
                strided_transpose_check<uint16_t>("f16 strided(transpose)==contig",
                    baracuda_kernels_unary_relu_propagating_f16_run,
                    baracuda_kernels_unary_relu_propagating_f16_strided_run, h, R, C);
        }
    }

    // ================= can_implement gates (all 8) =================
    // Host validators: every `*_can_implement` / `*_strided_can_implement` must
    // ACCEPT a well-formed request (return 0). Previously never called.
    {
        int32_t shape[2] = { 32, 32 };
        int64_t sx[2] = { 1, 32 };
        int64_t sy[2] = { 32, 1 };
        const long long n = 1024;
        struct Gate { const char* name; int32_t rc; };
        Gate gates[8] = {
            {"f32  can_implement",          baracuda_kernels_unary_relu_propagating_f32_can_implement(n, nullptr, nullptr)},
            {"f32  strided_can_implement",  baracuda_kernels_unary_relu_propagating_f32_strided_can_implement(n, 2, shape, sx, sy, nullptr, nullptr)},
            {"f64  can_implement",          baracuda_kernels_unary_relu_propagating_f64_can_implement(n, nullptr, nullptr)},
            {"f64  strided_can_implement",  baracuda_kernels_unary_relu_propagating_f64_strided_can_implement(n, 2, shape, sx, sy, nullptr, nullptr)},
            {"f16  can_implement",          baracuda_kernels_unary_relu_propagating_f16_can_implement(n, nullptr, nullptr)},
            {"f16  strided_can_implement",  baracuda_kernels_unary_relu_propagating_f16_strided_can_implement(n, 2, shape, sx, sy, nullptr, nullptr)},
            {"bf16 can_implement",          baracuda_kernels_unary_relu_propagating_bf16_can_implement(n, nullptr, nullptr)},
            {"bf16 strided_can_implement",  baracuda_kernels_unary_relu_propagating_bf16_strided_can_implement(n, 2, shape, sx, sy, nullptr, nullptr)},
        };
        for (const auto& g : gates) {
            if (g.rc == 0) printf("PASS %-44s can_implement gate accepts (rc=0)\n", g.name);
            else { printf("FAIL %-44s can_implement gate rc=%d (expected 0)\n", g.name, g.rc); fails++; }
        }
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
