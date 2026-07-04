// Increment 5 (SCATTER + ATOMIC_HISTOGRAM) on-device validation + audit.
//
// Proof vehicle: the generated strided emitter writes the OUTPUT through a
// data-dependent index operand (the write-side mirror of #4's gather). The
// scattered-axis coordinate `c{axis}` of the STORE is replaced by a value loaded
// from an integer index tensor, and the store becomes a WriteCombine op:
//
//   scatter      : out[idx[r,c], c]  = upd[r,c]         (Assign, full-shape idx)
//   scatter_add  : out[idx[r,c], c] += upd[r,c]         (AtomicAdd, full-shape)
//   bincount     : counts[x[i]]     += 1                (AtomicAdd Const(1), i32)
//
// THE CORE DISCIPLINE is DETERMINISM (see README):
//   * Assign (unique idx), INTEGER atomicAdd, bincount counts  → order-independent
//     result ⇒ generated == bespoke == CPU ref BIT-EXACT (deterministic).
//   * FP atomicAdd scatter_add is RUN-TO-RUN NON-DETERMINISTIC. The BASE lower()
//     is the deterministic GATHER-SUM (`_scatter_gathersum_`: one thread per output
//     cell scans the update domain, sums matching values in a fixed order, NO
//     atomics) — validated BIT-EXACT vs a CPU ref summed in the same order. The
//     ATOMIC scatter is the gated `Nondeterministic` variant (`_strided_`) —
//     validated VALUE-correct within an FP tolerance vs the CPU ref (never
//     bit-exact), never selected silently.
//
// OOB PROBES: every run feeds NEGATIVE and OUT-OF-RANGE indices; the generated
// kernel SKIPS them (bespoke `continue;`), leaving the target untouched. The write
// address is CLAMPED in-bounds so no OOB store ever occurs — compute-sanitizer
// memcheck must report 0 errors; racecheck must report 0 hazards on the
// DETERMINISTIC kernels (a real race there is a correctness bug) and treats the
// atomic variant's atomicAdd as legitimate.
//
// Bespoke: baracuda::indexing::launch_scatter / launch_scatter_add
// (baracuda_indexing.cuh) + baracuda::histogram::launch_bincount
// (baracuda_histogram.cuh) — the templated launchers called directly.
//
// Generated kernels: the regen snippet in ondevice/README.md.
//
// Compile (the bespoke headers want the MSVC conforming preprocessor):
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <kernels/include> scatter_validate.cu -o scatter_validate
// Sanitize:
//   compute-sanitizer --tool memcheck  ./scatter_validate san
//   compute-sanitizer --tool racecheck ./scatter_validate san
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_gen_scatter_f32_i32_strided_r2.cu"
#include "baracuda_gen_scatter_add_f32_i32_scatter_gathersum_r2.cu"
#include "baracuda_gen_scatter_add_f32_i32_strided_r2.cu"
#include "baracuda_gen_scatter_add_i32_i32_strided_r2.cu"
#include "baracuda_gen_bincount_i32_i32_strided_r1.cu"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_indexing.cuh"
#include "C:/Projects/baracuda/crates/baracuda-kernels-sys/kernels/include/baracuda_histogram.cuh"

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

// A per-column permutation of [0,M) for row r (unique targets), with a few cells
// replaced by NEGATIVE / OOB entries (skipped). Requires M >= R.
static void fill_unique_idx_with_oob(int32_t* idx, int R, int C, int M) {
    for (int c = 0; c < C; ++c) {
        // start from identity-ish permutation offset per column
        for (int r = 0; r < R; ++r) idx[r * C + c] = (r + c * 7) % M;
        // ensure uniqueness within the column: Fisher-Yates on [0,M) then take R
        // (identity+offset is already a valid injection for R<=M since %M of
        // distinct r+off are distinct while R<=M).
        // Now OOB-probe a few cells (skipped; targets stay unwritten).
        if (R > 3)  idx[3 * C + c] = -1;
        if (R > 7)  idx[7 * C + c] = M + (c % 3);
        if (R > 11) idx[11 * C + c] = -(M + 2);
    }
}

// Arbitrary indices in [0,M) with duplicates (real accumulation) + OOB probes.
static void fill_dup_idx_with_oob(int32_t* idx, long long count, int M) {
    for (long long t = 0; t < count; ++t) {
        long long m = t % 9;
        if      (m == 4) idx[t] = -1;
        else if (m == 6) idx[t] = M + (int)(t % 3);
        else if (m == 8) idx[t] = -(M + 5);
        else             idx[t] = (int)((t * 2654435761u) % (unsigned)M);
    }
}

static void report(const char* tag, bool ok, double detail) {
    printf("  %-42s %s   (%.3g)\n", tag, ok ? "PASS" : "FAIL", detail);
    if (!ok) g_fail = 1;
}

// ============================================================================
// scatter (Assign, unique idx): out[idx[r,c], c] = upd[r,c]. Deterministic ⇒
// generated == bespoke == CPU ref BIT-EXACT.
// ============================================================================
static void run_scatter_assign(int R, int C, int M) {
    long long nupd = (long long)R * C;
    size_t obytes = (size_t)M * C * 4;
    float* hupd = (float*)malloc((size_t)nupd * 4);
    int32_t* hidx = (int32_t*)malloc((size_t)nupd * 4);
    float* hg = (float*)malloc(obytes);
    float* hb = (float*)malloc(obytes);
    float* href = (float*)malloc(obytes);
    srand(3 + R * 5 + C * 7 + M);
    for (long long t = 0; t < nupd; ++t) hupd[t] = (float)((rand() % 8001) - 4000) * 0.013f;
    fill_unique_idx_with_oob(hidx, R, C, M);
    // CPU ref: sentinel, then write valid targets.
    for (long long t = 0; t < (long long)M * C; ++t) href[t] = kSentinel;
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) {
        int v = hidx[r * C + c];
        if (v >= 0 && v < M) href[(long long)v * C + c] = hupd[r * C + c];
    }

    float *dupd, *dg, *db; int32_t* didx;
    CHECK(cudaMalloc((void**)&dupd, (size_t)nupd * 4));
    CHECK(cudaMalloc((void**)&didx, (size_t)nupd * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dupd, hupd, (size_t)nupd * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)nupd * 4, cudaMemcpyHostToDevice));
    for (long long t = 0; t < (long long)M * C; ++t) hg[t] = kSentinel;
    CHECK(cudaMemcpy(dg, hg, obytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hg, obytes, cudaMemcpyHostToDevice));

    // Generated ABI: (upd, idx, out, shape0=R, shape1=C, s0_0=C,s0_1=1,
    //                 s1_0=C,s1_1=1, so_0=C,so_1=1, sext=M, n=R*C).
    baracuda_gen_scatter_f32_i32_strided_r2<<<grid_for(nupd), 256>>>(
        dupd, didx, dg, R, C, C, 1, C, 1, C, 1, M, nupd);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    int32_t ushape[2] = { R, C };
    int64_t su[2] = { C, 1 }, si[2] = { C, 1 }, so[2] = { C, 1 };
    int rc = baracuda::indexing::launch_scatter<float, int32_t>(
        dupd, didx, db, nupd, 2, 0, M, ushape, su, si, so, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("bespoke scatter rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long diff_ref = 0, diff_bes = 0;
    for (long long t = 0; t < (long long)M * C; ++t) {
        if (memcmp(&hg[t], &href[t], 4) != 0) diff_ref++;
        if (memcmp(&hg[t], &hb[t], 4) != 0) diff_bes++;
    }
    char tag[64]; snprintf(tag, 64, "scatter assign %dx%d M=%d gen==ref", R, C, M);
    report(tag, diff_ref == 0, (double)diff_ref);
    snprintf(tag, 64, "scatter assign %dx%d gen==bespoke (bit)", R, C);
    report(tag, diff_bes == 0, (double)diff_bes);

    cudaFree(dupd); cudaFree(didx); cudaFree(dg); cudaFree(db);
    free(hupd); free(hidx); free(hg); free(hb); free(href);
}

// ============================================================================
// integer scatter_add (i32): out[idx[r,c], c] += upd[r,c]. Integer atomicAdd ⇒
// order-independent ⇒ generated == bespoke == CPU ref BIT-EXACT.
// ============================================================================
static void run_scatter_add_int(int R, int C, int M) {
    long long nupd = (long long)R * C;
    size_t obytes = (size_t)M * C * 4;
    int32_t* hupd = (int32_t*)malloc((size_t)nupd * 4);
    int32_t* hidx = (int32_t*)malloc((size_t)nupd * 4);
    int32_t* hg = (int32_t*)malloc(obytes);
    int32_t* hb = (int32_t*)malloc(obytes);
    int32_t* href = (int32_t*)malloc(obytes);
    srand(17 + R + C + M);
    for (long long t = 0; t < nupd; ++t) hupd[t] = (rand() % 2001) - 1000;
    fill_dup_idx_with_oob(hidx, nupd, M);
    for (long long t = 0; t < (long long)M * C; ++t) href[t] = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) {
        int v = hidx[r * C + c];
        if (v >= 0 && v < M) href[(long long)v * C + c] += hupd[r * C + c];
    }

    int32_t *dupd, *dg, *db, *didx;
    CHECK(cudaMalloc((void**)&dupd, (size_t)nupd * 4));
    CHECK(cudaMalloc((void**)&didx, (size_t)nupd * 4));
    CHECK(cudaMalloc((void**)&dg, obytes));
    CHECK(cudaMalloc((void**)&db, obytes));
    CHECK(cudaMemcpy(dupd, hupd, (size_t)nupd * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, (size_t)nupd * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dg, 0, obytes));
    CHECK(cudaMemset(db, 0, obytes));

    baracuda_gen_scatter_add_i32_i32_strided_r2<<<grid_for(nupd), 256>>>(
        dupd, didx, dg, R, C, C, 1, C, 1, C, 1, M, nupd);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, obytes, cudaMemcpyDeviceToHost));

    int32_t ushape[2] = { R, C };
    int64_t su[2] = { C, 1 }, si[2] = { C, 1 }, so[2] = { C, 1 };
    int rc = baracuda::indexing::launch_scatter_add<int32_t, int32_t>(
        dupd, didx, db, nupd, 2, 0, M, ushape, su, si, so, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("bespoke int scatter_add rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaMemcpy(hb, db, obytes, cudaMemcpyDeviceToHost));

    long long dref = 0, dbes = 0;
    for (long long t = 0; t < (long long)M * C; ++t) {
        if (hg[t] != href[t]) dref++;
        if (hg[t] != hb[t]) dbes++;
    }
    char tag[64];
    snprintf(tag, 64, "int scatter_add %dx%d M=%d gen==ref (exact)", R, C, M);
    report(tag, dref == 0, (double)dref);
    snprintf(tag, 64, "int scatter_add %dx%d gen==bespoke (bit)", R, C);
    report(tag, dbes == 0, (double)dbes);

    cudaFree(dupd); cudaFree(didx); cudaFree(dg); cudaFree(db);
    free(hupd); free(hidx); free(hg); free(hb); free(href);
}

// ============================================================================
// bincount (i32): counts[x[i]] += 1. Integer count ⇒ deterministic ⇒
// generated == bespoke == CPU ref BIT-EXACT.
// ============================================================================
static void run_bincount(int N, int B) {
    int32_t* hx = (int32_t*)malloc((size_t)N * 4);
    int32_t* hg = (int32_t*)malloc((size_t)B * 4);
    int32_t* hb = (int32_t*)malloc((size_t)B * 4);
    int32_t* href = (int32_t*)calloc(B, 4);
    srand(29 + N + B);
    for (int i = 0; i < N; ++i) {
        int m = i % 11;
        if (m == 5) hx[i] = -1;         // OOB skip
        else if (m == 9) hx[i] = B + (i % 4);
        else hx[i] = (int)((i * 2654435761u) % (unsigned)B);
        if (hx[i] >= 0 && hx[i] < B) href[hx[i]]++;
    }
    int32_t *dx, *dg, *db;
    CHECK(cudaMalloc((void**)&dx, (size_t)N * 4));
    CHECK(cudaMalloc((void**)&dg, (size_t)B * 4));
    CHECK(cudaMalloc((void**)&db, (size_t)B * 4));
    CHECK(cudaMemcpy(dx, hx, (size_t)N * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dg, 0, (size_t)B * 4)); // generated kernel only atomicAdds

    // Generated ABI: (in0=x, out, shape0=N, s0_0=1, so_0=1, sext=B, n=N).
    baracuda_gen_bincount_i32_i32_strided_r1<<<grid_for(N), 256>>>(
        dx, dg, N, 1, 1, B, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, (size_t)B * 4, cudaMemcpyDeviceToHost));

    int rc = baracuda::histogram::launch_bincount<int32_t>(dx, db, N, B, 0);
    CHECK(cudaDeviceSynchronize());
    if (rc != 0) { printf("bespoke bincount rc=%d\n", rc); g_fail = 1; }
    CHECK(cudaMemcpy(hb, db, (size_t)B * 4, cudaMemcpyDeviceToHost));

    long long dref = 0, dbes = 0, total = 0;
    for (int i = 0; i < B; ++i) { if (hg[i] != href[i]) dref++; if (hg[i] != hb[i]) dbes++; total += hg[i]; }
    char tag[64];
    snprintf(tag, 64, "bincount N=%d B=%d gen==ref (exact)", N, B);
    report(tag, dref == 0, (double)dref);
    snprintf(tag, 64, "bincount N=%d B=%d gen==bespoke (bit)", N, B);
    report(tag, dbes == 0, (double)dbes);

    cudaFree(dx); cudaFree(dg); cudaFree(db);
    free(hx); free(hg); free(hb); free(href);
}

// ============================================================================
// FP scatter_add: the DETERMINISM SPLIT.
//   base   = gather-sum (deterministic): gen == CPU ref (j-order) BIT-EXACT.
//   variant= atomic (nondeterministic):  gen within FP tolerance of CPU ref;
//            two runs may differ run-to-run (documented).
// ============================================================================
static void run_scatter_add_fp(int R, int C, int M) {
    long long nupd = (long long)R * C;
    long long nout = (long long)M * C;
    size_t ub = (size_t)nupd * 4, ob = (size_t)nout * 4;
    float* hupd = (float*)malloc(ub);
    int32_t* hidx = (int32_t*)malloc(ub);
    double* hd = (double*)malloc((size_t)nout * 8);   // f64 ref
    float* href = (float*)malloc(ob);                 // f32 j-order ref (for gather-sum)
    float* hg = (float*)malloc(ob);
    srand(41 + R + C * 3 + M * 5);
    for (long long t = 0; t < nupd; ++t) hupd[t] = (float)((rand() % 20001) - 10000) * 0.001f;
    fill_dup_idx_with_oob(hidx, nupd, M);

    // f64 reference (order-independent, the "true" value) + f32 gather-sum ref
    // (sum matching updates in ascending update-index j — the kernel's fixed order).
    for (long long t = 0; t < nout; ++t) { hd[t] = 0.0; href[t] = 0.0f; }
    for (int m = 0; m < M; ++m) for (int c = 0; c < C; ++c) {
        float acc = 0.0f; double accd = 0.0;
        for (int r = 0; r < R; ++r) {
            int v = hidx[r * C + c];
            if (v == m) { acc += hupd[r * C + c]; accd += (double)hupd[r * C + c]; }
        }
        href[(long long)m * C + c] = acc;   // f32, j ascending == r ascending here
        hd[(long long)m * C + c] = accd;
    }

    float *dupd, *dg; int32_t* didx;
    CHECK(cudaMalloc((void**)&dupd, ub));
    CHECK(cudaMalloc((void**)&didx, ub));
    CHECK(cudaMalloc((void**)&dg, ob));
    CHECK(cudaMemcpy(dupd, hupd, ub, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(didx, hidx, ub, cudaMemcpyHostToDevice));

    // --- deterministic gather-sum base ---
    // ABI: (upd, idx, out, oshape0=M,oshape1=C, ushape0=R,ushape1=C,
    //       sv_0=C,sv_1=1, si_0=C,si_1=1, so_0=C,so_1=1, n_out=M*C, n_upd=R*C).
    CHECK(cudaMemset(dg, 0, ob));
    baracuda_gen_scatter_add_f32_i32_scatter_gathersum_r2<<<grid_for(nout), 256>>>(
        dupd, didx, dg, M, C, R, C, C, 1, C, 1, C, 1, nout, nupd);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hg, dg, ob, cudaMemcpyDeviceToHost));
    long long dbit = 0; double maxrel_base = 0;
    for (long long t = 0; t < nout; ++t) {
        if (memcmp(&hg[t], &href[t], 4) != 0) dbit++;
        double denom = fabs(hd[t]) + 1e-6;
        double rel = fabs((double)hg[t] - hd[t]) / denom;
        if (rel > maxrel_base) maxrel_base = rel;
    }
    char tag[80];
    snprintf(tag, 80, "fp scatter_add gather-sum BASE gen==ref (BIT)");
    report(tag, dbit == 0, (double)dbit);
    snprintf(tag, 80, "fp scatter_add gather-sum BASE vs f64 (rel)");
    report(tag, maxrel_base < 1e-4, maxrel_base);

    // --- nondeterministic atomic variant (value-correct within tolerance) ---
    // ABI: same as scatter_add atomic strided: (upd, idx, out, shape0=R,shape1=C,
    //      s0_0=C,s0_1=1, s1_0=C,s1_1=1, so_0=C,so_1=1, sext=M, n=R*C).
    CHECK(cudaMemset(dg, 0, ob));
    baracuda_gen_scatter_add_f32_i32_strided_r2<<<grid_for(nupd), 256>>>(
        dupd, didx, dg, R, C, C, 1, C, 1, C, 1, M, nupd);
    CHECK(cudaDeviceSynchronize());
    float* ha = (float*)malloc(ob);
    CHECK(cudaMemcpy(ha, dg, ob, cudaMemcpyDeviceToHost));
    double maxrel_at = 0;
    for (long long t = 0; t < nout; ++t) {
        double denom = fabs(hd[t]) + 1e-6;
        double rel = fabs((double)ha[t] - hd[t]) / denom;
        if (rel > maxrel_at) maxrel_at = rel;
    }
    snprintf(tag, 80, "fp scatter_add ATOMIC variant vs f64 (rel, ~nondet)");
    report(tag, maxrel_at < 5e-3, maxrel_at);

    // Run the atomic variant AGAIN and report run-to-run max abs diff (may be >0 —
    // documents the nondeterminism; the base is the reproducible route).
    CHECK(cudaMemset(dg, 0, ob));
    baracuda_gen_scatter_add_f32_i32_strided_r2<<<grid_for(nupd), 256>>>(
        dupd, didx, dg, R, C, C, 1, C, 1, C, 1, M, nupd);
    CHECK(cudaDeviceSynchronize());
    float* ha2 = (float*)malloc(ob);
    CHECK(cudaMemcpy(ha2, dg, ob, cudaMemcpyDeviceToHost));
    double maxdiff = 0; long long ndiff = 0;
    for (long long t = 0; t < nout; ++t) {
        double d = fabs((double)ha[t] - (double)ha2[t]);
        if (d > 0) ndiff++;
        if (d > maxdiff) maxdiff = d;
    }
    printf("  [note] fp atomic run-to-run: %lld/%lld cells differ, maxabs %.3g "
           "(nondeterministic by design; base is reproducible)\n", ndiff, nout, maxdiff);

    cudaFree(dupd); cudaFree(didx); cudaFree(dg);
    free(hupd); free(hidx); free(hd); free(href); free(hg); free(ha); free(ha2);
}

int main(int argc, char** argv) {
    bool san = (argc > 1 && strcmp(argv[1], "san") == 0);
    printf("== scatter_validate (increment 5) ==\n");
    if (san) {
        // Small shapes for compute-sanitizer (memcheck / racecheck).
        run_scatter_assign(8, 4, 16);
        run_scatter_add_int(8, 4, 6);
        run_bincount(64, 8);
        run_scatter_add_fp(8, 4, 6);
    } else {
        printf("- scatter (Assign, unique idx) -\n");
        run_scatter_assign(100, 32, 128);
        run_scatter_assign(64, 64, 100);
        printf("- integer scatter_add (deterministic atomicAdd) -\n");
        run_scatter_add_int(128, 64, 64);
        run_scatter_add_int(256, 16, 40);
        printf("- bincount (deterministic int counts) -\n");
        run_bincount(4096, 64);
        run_bincount(65536, 256);
        printf("- fp scatter_add: deterministic gather-sum base + nondet atomic variant -\n");
        run_scatter_add_fp(128, 32, 48);
        run_scatter_add_fp(64, 8, 16);
    }
    printf(g_fail ? "\nRESULT: FAIL\n" : "\nRESULT: ALL PASSED\n");
    return g_fail;
}
