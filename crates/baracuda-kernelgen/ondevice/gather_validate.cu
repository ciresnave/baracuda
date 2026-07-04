// Increment 4 (GATHER) on-device validation + generated-vs-bespoke audit.
//
// Proof vehicle: the generated strided emitter reads an INPUT through a
// data-dependent index operand (the first access pattern whose address is a
// runtime tensor value). The gathered-axis coordinate `c{axis}` is replaced by a
// value loaded from an integer index tensor:
//
//   gather  : out[r,c] = src[index[r,c], c]      (axis 0, FULL-shape i32/i64 idx)
//   gatheru : out[r,c] = src[index[r,c], c]      (axis 0, FULL-shape U32 idx —
//                                                 the Fuel-facing Model-A variant,
//                                                 cross-checked vs the i32 kernel)
//   isel    : out[r,c] = src[idx[r], c]          (axis 0, 1-D idx; index_select)
//   emb     : out[n,d] = weight[ids[n], d]       (axis 0, 1-D idx; ZeroFill OOB)
//   gclamp  : out[r,c] = src[clamp(index[r,c]),c](axis 0, Clamp — generator-only)
//
// THE POINT is the OOB PROBES: every run feeds NEGATIVE and OUT-OF-RANGE indices
// and asserts the generated kernel matches the BESPOKE policy EXACTLY —
//   Skip     (gather / index_select): the OOB output cell is left UNWRITTEN
//            (bespoke `continue;`). We pre-fill both output buffers with the same
//            sentinel and require them bit-equal after the launch.
//   ZeroFill (embedding): the OOB / negative row is zeroed (bespoke embedding).
//   Clamp    (gclamp): no bespoke op clamps — diffed vs a CPU clamp reference
//            only; still an OOB probe (indices out of range, memcheck must be
//            clean).
// The generated kernel CLAMPS the load address in-bounds, so an OOB gather never
// issues an out-of-range read — compute-sanitizer memcheck must report 0 errors
// under EVERY policy (`memcheck` is the load-bearing check; see README).
//
// Correctness per run: generated == bespoke (bit-exact) AND generated == a CPU
// f64/exact reference. Expected perf: a TIE at the memory wall (both are plain
// gathers — coverage is the point, not a speedup).
//
// Bespoke: baracuda::indexing::launch_gather / launch_index_select
// (baracuda_indexing.cuh) + baracuda::embedding::launch_embedding
// (baracuda_embedding.cuh) — the templated launchers called directly.
//
// Generated kernels are produced by the regen snippet in ondevice/README.md
// ("gather_validate.cu — GATHER (increment 4)").
//
// Compile (the bespoke headers want the MSVC conforming preprocessor):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <kernels/include> gather_validate.cu -o gather_validate
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_gen_gather_f32_i32_strided_r2.cu"
#include "baracuda_gen_gather_f32_i64_strided_r2.cu"
#include "baracuda_gen_gather_f32_u32_strided_r2.cu"
#include "baracuda_gen_isel_f32_i32_strided_r2.cu"
#include "baracuda_gen_emb_f32_i32_strided_r2.cu"
#include "baracuda_gen_gclamp_f32_i32_strided_r2.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_embedding.cuh"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(2); } } while (0)

static int g_fail = 0;
static const float kSentinel = -987654.0f; // marks a "left unwritten" (Skip) cell

static int grid_for(long long n) {
    long long b = (n + 255) / 256;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535;
    return (int)b;
}

// Build an index array over [0,R*C) mixing valid, NEGATIVE, and OOB entries — the
// OOB probe. `V` is the source extent along the gathered axis.
static void fill_indices_i32(int32_t* idx, long long count, int V) {
    for (long long t = 0; t < count; ++t) {
        long long m = t % 7;
        if      (m == 3) idx[t] = -1;         // negative -> OOB (no from-end wrap)
        else if (m == 5) idx[t] = V + (int)(t % 3); // past the extent -> OOB
        else if (m == 6) idx[t] = -(V + 2);   // very negative -> OOB
        else             idx[t] = (int)(t * 2654435761u % (unsigned)V); // valid
    }
}

// ============================================================================
// torch-gather along axis 0, rank 2: out[r,c] = src[index[r,c], c]. FULL-shape
// index (varies on every axis). OOB policy = Skip.
// ============================================================================
static void run_gather(int V, int R, int C, const char* tag) {
    long long nout = (long long)R * C;
    size_t sbytes = (size_t)V * C * 4;
    size_t obytes = (size_t)nout * 4;
    float* hsrc = (float*)malloc(sbytes);
    int32_t* hidx = (int32_t*)malloc((size_t)nout * 4);
    float* hg = (float*)malloc(obytes);
    float* hb = (float*)malloc(obytes);
    srand(11 + V * 7 + R * 13 + C);
    for (long long t = 0; t < (long long)V * C; ++t) hsrc[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    fill_indices_i32(hidx, nout, V);

    float *dsrc, *dg, *db; int32_t* didx;
    CHECK(cudaMalloc((void**)&dsrc, sbytes));
    CHECK(cudaMalloc((void**)&didx, (size_t)nout * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dsrc, hsrc, sbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)nout * 4, cudaMemcpyHostToDevice));
    // Pre-fill BOTH outputs with the sentinel so a Skip (unwritten) cell matches.
    for (long long t = 0; t < nout; ++t) hg[t] = kSentinel;
    CHECK(cudaMemcpy(dg, hg, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hg, obytes, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    // Generated ABI: (in0=src, in1=idx, out, shape0=R, shape1=C, s0_0=C, s0_1=1,
    //                 s1_0=C, s1_1=1, so_0=C, so_1=1, gext=V, n).
    baracuda_gen_gather_f32_i32_strided_r2<<<blocks, 256>>>(
        dsrc, didx, dg, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    // Bespoke: launch_gather<float,int32_t>. out_numel, rank, gather_dim,
    // src_dim_size, out_shape, stride_src, stride_index, stride_out.
    int32_t oshape[2] = { R, C };
    int64_t ss[2] = { C, 1 }, si[2] = { C, 1 }, so[2] = { C, 1 };
    int rc = baracuda::indexing::launch_gather<float, int32_t>(
        dsrc, didx, db, nout, 2, 0, V, oshape, ss, si, so, 0);
    if (rc != 0) { printf("bespoke gather rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long bit_bad = 0, ref_bad = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            long long o = (long long)r * C + c;
            int idx = hidx[o];
            float want = (idx >= 0 && idx < V) ? hsrc[(long long)idx * C + c] : kSentinel;
            if (memcmp(&hg[o], &hb[o], 4) != 0) bit_bad++;
            if (memcmp(&hg[o], &want, 4) != 0) ref_bad++;
        }
    // ---- timing: generated ONE pass vs the bespoke gather (expect a TIE at the
    // memory wall — both are plain gathers; coverage is the point) ----
    cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
    const int reps = 50;
    baracuda_gen_gather_f32_i32_strided_r2<<<blocks, 256>>>(dsrc, didx, dg, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int rp = 0; rp < reps; ++rp)
        baracuda_gen_gather_f32_i32_strided_r2<<<blocks, 256>>>(dsrc, didx, dg, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float t_gen = 0; CHECK(cudaEventElapsedTime(&t_gen, e0, e1)); t_gen /= reps;
    baracuda::indexing::launch_gather<float, int32_t>(dsrc, didx, db, nout, 2, 0, V, oshape, ss, si, so, 0);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int rp = 0; rp < reps; ++rp)
        baracuda::indexing::launch_gather<float, int32_t>(dsrc, didx, db, nout, 2, 0, V, oshape, ss, si, so, 0);
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float t_bes = 0; CHECK(cudaEventElapsedTime(&t_bes, e0, e1)); t_bes /= reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    int ok = (bit_bad == 0) && (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] gather      %-8s V=%-5d [%4dx%-4d] gen==bespoke %s  gen==ref %s (Skip OOB) | "
           "gen %.4f ms  bespoke %.4f ms  %.2fx\n",
           ok ? " ok " : "FAIL", tag, V, R, C, bit_bad ? "NO" : "yes", ref_bad ? "NO" : "yes",
           t_gen, t_bes, t_bes / t_gen);

    cudaFree(dsrc); cudaFree(didx); cudaFree(dg); cudaFree(db);
    free(hsrc); free(hidx); free(hg); free(hb);
}

// i64-index gather — exercises the `long long` index-load path (distinct symbol).
static void run_gather_i64(int V, int R, int C, const char* tag) {
    long long nout = (long long)R * C;
    size_t sbytes = (size_t)V * C * 4;
    size_t obytes = (size_t)nout * 4;
    float* hsrc = (float*)malloc(sbytes);
    int64_t* hidx = (int64_t*)malloc((size_t)nout * 8);
    float* hg = (float*)malloc(obytes);
    float* hb = (float*)malloc(obytes);
    srand(29 + V + R + C);
    for (long long t = 0; t < (long long)V * C; ++t) hsrc[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    for (long long t = 0; t < nout; ++t) {
        long long m = t % 7;
        if (m == 3) hidx[t] = -1;
        else if (m == 5) hidx[t] = (int64_t)V + 4; // OOB
        else hidx[t] = (int64_t)((t * 2654435761u) % (unsigned)V);
    }

    float *dsrc, *dg, *db; int64_t* didx;
    CHECK(cudaMalloc((void**)&dsrc, sbytes));
    CHECK(cudaMalloc((void**)&didx, (size_t)nout * 8));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dsrc, hsrc, sbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)nout * 8, cudaMemcpyHostToDevice));
    for (long long t = 0; t < nout; ++t) hg[t] = kSentinel;
    CHECK(cudaMemcpy(dg, hg, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hg, obytes, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    baracuda_gen_gather_f32_i64_strided_r2<<<blocks, 256>>>(
        dsrc, didx, dg, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    int32_t oshape[2] = { R, C };
    int64_t ss[2] = { C, 1 }, si[2] = { C, 1 }, so[2] = { C, 1 };
    int rc = baracuda::indexing::launch_gather<float, int64_t>(
        dsrc, didx, db, nout, 2, 0, V, oshape, ss, si, so, 0);
    if (rc != 0) { printf("bespoke gather(i64) rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long bit_bad = 0, ref_bad = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            long long o = (long long)r * C + c;
            int64_t idx = hidx[o];
            float want = (idx >= 0 && idx < V) ? hsrc[idx * C + c] : kSentinel;
            if (memcmp(&hg[o], &hb[o], 4) != 0) bit_bad++;
            if (memcmp(&hg[o], &want, 4) != 0) ref_bad++;
        }
    int ok = (bit_bad == 0) && (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] gather(i64) %-8s V=%-5d [%4dx%-4d] gen==bespoke %s  gen==ref %s (Skip OOB)\n",
           ok ? " ok " : "FAIL", tag, V, R, C, bit_bad ? "NO" : "yes", ref_bad ? "NO" : "yes");

    cudaFree(dsrc); cudaFree(didx); cudaFree(dg); cudaFree(db);
    free(hsrc); free(hidx); free(hg); free(hb);
}

// u32-index gather (the FUEL-FACING index dtype — Model-A contract wiring). Fuel
// keys the gather index operand as a fixed U32 slot (`[T, U32, T]`), so u32 is the
// variant that carries the keyed contract. There is NO bespoke u32 sibling
// (bespoke launch_gather is i32/i64-templated), so this is diffed vs a CPU ref AND
// CROSS-CHECKED against the i32 kernel on the SAME non-negative index values (u32
// and i32 must be bit-identical there — u32 just can't express a negative index).
static void run_gather_u32(int V, int R, int C, const char* tag) {
    long long nout = (long long)R * C;
    size_t sbytes = (size_t)V * C * 4;
    size_t obytes = (size_t)nout * 4;
    float* hsrc = (float*)malloc(sbytes);
    uint32_t* hidx = (uint32_t*)malloc((size_t)nout * 4);
    int32_t* hidx32 = (int32_t*)malloc((size_t)nout * 4);
    float* hu = (float*)malloc(obytes); // u32-kernel result
    float* hi = (float*)malloc(obytes); // i32-kernel result (cross-check oracle)
    srand(211 + V * 7 + R * 13 + C);
    for (long long t = 0; t < (long long)V * C; ++t) hsrc[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    // NON-NEGATIVE indices (u32 has no sign): mix valid + OOB-past-extent. The
    // same values go into the i32 array so the two kernels are diffed head-to-head.
    for (long long t = 0; t < nout; ++t) {
        long long m = t % 5;
        uint32_t v = (m == 4) ? (uint32_t)(V + (int)(t % 3))        // OOB (>= V)
                              : (uint32_t)((t * 2654435761u) % (unsigned)V); // valid
        hidx[t] = v;
        hidx32[t] = (int32_t)v;
    }

    float *dsrc, *du, *di; uint32_t* didxu; int32_t* didxi;
    CHECK(cudaMalloc((void**)&dsrc, sbytes));
    CHECK(cudaMalloc((void**)&didxu, (size_t)nout * 4));
    CHECK(cudaMalloc((void**)&didxi, (size_t)nout * 4));
    CHECK(cudaMalloc((void**)&du, obytes));
    CHECK(cudaMalloc((void**)&di, obytes));
    CHECK(cudaMemcpy(dsrc, hsrc, sbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didxu, hidx, (size_t)nout * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didxi, hidx32, (size_t)nout * 4, cudaMemcpyHostToDevice));
    // Pre-fill both outputs with the sentinel (Skip leaves an OOB cell unwritten).
    for (long long t = 0; t < nout; ++t) hu[t] = kSentinel;
    CHECK(cudaMemcpy(du, hu, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(di, hu, obytes, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    // Identical ABI to the i32 kernel; only in1's pointer type differs (unsigned int).
    baracuda_gen_gather_f32_u32_strided_r2<<<blocks, 256>>>(
        dsrc, didxu, du, R, C, C, 1, C, 1, C, 1, V, nout);
    baracuda_gen_gather_f32_i32_strided_r2<<<blocks, 256>>>(
        dsrc, didxi, di, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hu, du, obytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hi, di, obytes, cudaMemcpyDeviceToHost));

    long long xchk_bad = 0, ref_bad = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            long long o = (long long)r * C + c;
            uint32_t idx = hidx[o];
            float want = (idx < (uint32_t)V) ? hsrc[(long long)idx * C + c] : kSentinel;
            if (memcmp(&hu[o], &hi[o], 4) != 0) xchk_bad++; // u32 == i32
            if (memcmp(&hu[o], &want, 4) != 0) ref_bad++;   // u32 == CPU ref
        }
    int ok = (xchk_bad == 0) && (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] gather(u32) %-8s V=%-5d [%4dx%-4d] u32==i32 %s  u32==ref %s (Skip OOB)\n",
           ok ? " ok " : "FAIL", tag, V, R, C, xchk_bad ? "NO" : "yes", ref_bad ? "NO" : "yes");

    cudaFree(dsrc); cudaFree(didxu); cudaFree(didxi); cudaFree(du); cudaFree(di);
    free(hsrc); free(hidx); free(hidx32); free(hu); free(hi);
}

// index_select along axis 0: out[r,c] = src[idx[r], c]. 1-D index (length R).
static void run_index_select(int V, int R, int C, const char* tag) {
    long long nout = (long long)R * C;
    size_t sbytes = (size_t)V * C * 4;
    size_t obytes = (size_t)nout * 4;
    float* hsrc = (float*)malloc(sbytes);
    int32_t* hidx = (int32_t*)malloc((size_t)R * 4);
    float* hg = (float*)malloc(obytes);
    float* hb = (float*)malloc(obytes);
    srand(7 + V + R * 5 + C);
    for (long long t = 0; t < (long long)V * C; ++t) hsrc[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    fill_indices_i32(hidx, R, V);

    float *dsrc, *dg, *db; int32_t* didx;
    CHECK(cudaMalloc((void**)&dsrc, sbytes));
    CHECK(cudaMalloc((void**)&didx, (size_t)R * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dsrc, hsrc, sbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)R * 4, cudaMemcpyHostToDevice));
    for (long long t = 0; t < nout; ++t) hg[t] = kSentinel;
    CHECK(cudaMemcpy(dg, hg, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hg, obytes, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    // Generated ABI: index is 1-D (bcast over axis 1) -> gidx_off = c0*s1_0. Pass
    // s1_0=1 (idx[r]), s1_1=0 (unused). (in0, in1, out, R, C, C,1, 1,0, C,1, V, n).
    baracuda_gen_isel_f32_i32_strided_r2<<<blocks, 256>>>(
        dsrc, didx, dg, R, C, C, 1, 1, 0, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    // Bespoke: launch_index_select (1-D idx read by coord[select_dim]).
    int32_t oshape[2] = { R, C };
    int64_t ss[2] = { C, 1 }, so[2] = { C, 1 };
    int rc = baracuda::indexing::launch_index_select<float, int32_t>(
        dsrc, didx, db, nout, 2, 0, V, oshape, ss, so, 0);
    if (rc != 0) { printf("bespoke index_select rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long bit_bad = 0, ref_bad = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            long long o = (long long)r * C + c;
            int idx = hidx[r];
            float want = (idx >= 0 && idx < V) ? hsrc[(long long)idx * C + c] : kSentinel;
            if (memcmp(&hg[o], &hb[o], 4) != 0) bit_bad++;
            if (memcmp(&hg[o], &want, 4) != 0) ref_bad++;
        }
    int ok = (bit_bad == 0) && (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] index_sel   %-8s V=%-5d [%4dx%-4d] gen==bespoke %s  gen==ref %s (Skip OOB)\n",
           ok ? " ok " : "FAIL", tag, V, R, C, bit_bad ? "NO" : "yes", ref_bad ? "NO" : "yes");

    cudaFree(dsrc); cudaFree(didx); cudaFree(dg); cudaFree(db);
    free(hsrc); free(hidx); free(hg); free(hb);
}

// embedding: out[n,d] = weight[ids[n], d]. 1-D ids (length N). ZeroFill OOB.
static void run_embedding(int V, int N, int D, const char* tag) {
    long long nout = (long long)N * D;
    size_t wbytes = (size_t)V * D * 4;
    size_t obytes = (size_t)nout * 4;
    float* hw = (float*)malloc(wbytes);
    int32_t* hids = (int32_t*)malloc((size_t)N * 4);
    float* hg = (float*)malloc(obytes);
    float* hb = (float*)malloc(obytes);
    srand(53 + V + N + D);
    for (long long t = 0; t < (long long)V * D; ++t) hw[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    fill_indices_i32(hids, N, V);

    float *dw, *dg, *db; int32_t* dids;
    CHECK(cudaMalloc((void**)&dw, wbytes));
    CHECK(cudaMalloc((void**)&dids, (size_t)N * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dw, hw, wbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dids, hids, (size_t)N * 4, cudaMemcpyHostToDevice));
    // ZeroFill always writes every cell, so no sentinel pre-fill needed; still
    // clear to a non-zero pattern to catch a missed store.
    for (long long t = 0; t < nout; ++t) hg[t] = kSentinel;
    CHECK(cudaMemcpy(dg, hg, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hg, obytes, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    // Generated ABI same shape as index_select (1-D index), ZeroFill store.
    baracuda_gen_emb_f32_i32_strided_r2<<<blocks, 256>>>(
        dw, dids, dg, N, D, D, 1, 1, 0, D, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    // Bespoke embedding: padding disabled (INT32_MIN sentinel) so only OOB/neg
    // (-> zero) is exercised. num_indices=N, num_embeddings=V, embedding_dim=D.
    int rc = baracuda::embedding::launch_embedding<float, int32_t>(
        dw, dids, db, N, V, D, (int64_t)baracuda::embedding::kPaddingDisabled, 0);
    if (rc != 0) { printf("bespoke embedding rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long bit_bad = 0, ref_bad = 0;
    for (int nn = 0; nn < N; ++nn)
        for (int d = 0; d < D; ++d) {
            long long o = (long long)nn * D + d;
            int idx = hids[nn];
            float want = (idx >= 0 && idx < V) ? hw[(long long)idx * D + d] : 0.0f;
            if (memcmp(&hg[o], &hb[o], 4) != 0) bit_bad++;
            if (memcmp(&hg[o], &want, 4) != 0) ref_bad++;
        }
    int ok = (bit_bad == 0) && (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] embedding   %-8s V=%-5d [%4dx%-4d] gen==bespoke %s  gen==ref %s (ZeroFill OOB)\n",
           ok ? " ok " : "FAIL", tag, V, N, D, bit_bad ? "NO" : "yes", ref_bad ? "NO" : "yes");

    cudaFree(dw); cudaFree(dids); cudaFree(dg); cudaFree(db);
    free(hw); free(hids); free(hg); free(hb);
}

// Clamp probe (generator-only policy; no bespoke clamps): out[r,c] =
// src[clamp(index[r,c], 0, V-1), c]. Diffed vs a CPU clamp reference. Still an
// OOB probe — the indices go negative + past V, and memcheck must be clean.
static void run_gather_clamp(int V, int R, int C, const char* tag) {
    long long nout = (long long)R * C;
    size_t sbytes = (size_t)V * C * 4;
    size_t obytes = (size_t)nout * 4;
    float* hsrc = (float*)malloc(sbytes);
    int32_t* hidx = (int32_t*)malloc((size_t)nout * 4);
    float* hg = (float*)malloc(obytes);
    srand(71 + V + R + C);
    for (long long t = 0; t < (long long)V * C; ++t) hsrc[t] = (float)((rand() % 8001) - 4000) * 0.01f;
    fill_indices_i32(hidx, nout, V);

    float *dsrc, *dg; int32_t* didx;
    CHECK(cudaMalloc((void**)&dsrc, sbytes));
    CHECK(cudaMalloc((void**)&didx, (size_t)nout * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMemcpy(dsrc, hsrc, sbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)nout * 4, cudaMemcpyHostToDevice));

    int blocks = grid_for(nout);
    baracuda_gen_gclamp_f32_i32_strided_r2<<<blocks, 256>>>(
        dsrc, didx, dg, R, C, C, 1, C, 1, C, 1, V, nout);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    long long ref_bad = 0;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            long long o = (long long)r * C + c;
            int idx = hidx[o];
            int cl = idx < 0 ? 0 : (idx >= V ? V - 1 : idx);
            float want = hsrc[(long long)cl * C + c];
            if (memcmp(&hg[o], &want, 4) != 0) ref_bad++;
        }
    int ok = (ref_bad == 0);
    if (!ok) g_fail = 1;
    printf("[%s] gather-clamp %-8s V=%-5d [%4dx%-4d] gen==ref %s (Clamp OOB; generator-only)\n",
           ok ? " ok " : "FAIL", tag, V, R, C, ref_bad ? "NO" : "yes");

    cudaFree(dsrc); cudaFree(didx); cudaFree(dg);
    free(hsrc); free(hidx); free(hg);
}

int main() {
    printf("== increment 4 GATHER validation (RTX 4070 sm_89) ==\n");
    printf("-- gather (full-shape index, Skip OOB) --\n");
    run_gather(6, 6, 4, "tiny");
    run_gather(128, 512, 64, "mid");
    run_gather(1000, 2048, 128, "large");
    run_gather(1, 64, 8, "V=1");        // degenerate source extent
    printf("-- gather i64 index --\n");
    run_gather_i64(128, 512, 64, "mid");
    printf("-- gather u32 index (Fuel-facing; cross-checked vs i32) --\n");
    run_gather_u32(6, 6, 4, "tiny");
    run_gather_u32(128, 512, 64, "mid");
    run_gather_u32(1000, 2048, 128, "large");
    run_gather_u32(1, 64, 8, "V=1");
    printf("-- index_select (1-D index, Skip OOB) --\n");
    run_index_select(6, 10, 4, "tiny");
    run_index_select(200, 1024, 96, "mid");
    printf("-- embedding (1-D ids, ZeroFill OOB) --\n");
    run_embedding(6, 10, 4, "tiny");
    run_embedding(512, 2048, 128, "mid");
    printf("-- gather clamp (generator-only OOB policy) --\n");
    run_gather_clamp(6, 6, 4, "tiny");
    run_gather_clamp(256, 1024, 64, "mid");
    printf(g_fail ? "\n== FAILURES ==\n" : "\n== ALL PASSED ==\n");
    return g_fail;
}
