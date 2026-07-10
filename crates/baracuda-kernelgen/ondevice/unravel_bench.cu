// On-device validation + micro-bench for the GENERATED coord-unravel helper
// (baracuda::coord::gen::unravel_offset_1_rN — per-rank UNROLLED) vs the
// hand-written baracuda::coord::unravel_offset_1 (runtime-rank loop).
//
// Phase 1 of the IR-translation-hub roadmap (docs/design/ir-translation-hub.md):
// proves the generator can emit a freestanding .cuh helper that is bit-identical
// to the hand-written one and at least as fast, from shared emitter logic.
//
// Generate the helper, then compile from an x64 Native Tools shell (or pass
// -ccbin to the MSVC host compiler):
//   UNRAVEL_OUT=<work> cargo test -p baracuda-kernelgen dump_coord_unravel_helper -- --ignored --nocapture
//   nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
//        -I <work> -I crates/baracuda-kernels-sys/kernels/include \
//        crates/baracuda-kernelgen/ondevice/unravel_bench.cu -o <work>/unravel_bench
//   <work>/unravel_bench
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "baracuda_coord_unravel.cuh"   // B: hand-written runtime-rank + DimsI32/DimsI64
#include "coord_unravel_gen.cuh"        // A: generated per-rank unrolled

using baracuda::coord::DimsI32;
using baracuda::coord::DimsI64;

#define CHECK(x) do { cudaError_t e_=(x); if(e_){ \
  printf("CUDA err %s @ %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(2);} } while(0)

static int fails = 0;

// CPU reference — bit-exact to unravel_offset_1.
static int64_t unravel_ref(int64_t linear, int rank, const int32_t* shape, const int64_t* stride) {
    int64_t off = 0;
    for (int d = rank - 1; d >= 0; --d) {
        int32_t s = shape[d];
        int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
        if (s != 0) linear /= (int64_t)s;
        off += c * stride[d];
    }
    return off;
}

// --- correctness kernels (one offset per thread) ---
__global__ void k_gen_r4(int64_t n, DimsI32 shape, DimsI64 stride, int64_t* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = baracuda::coord::gen::unravel_offset_1_r4(i, shape, stride);
}
__global__ void k_hand(int64_t n, int rank, DimsI32 shape, DimsI64 stride, int64_t* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = baracuda::coord::unravel_offset_1(i, rank, shape, stride);
}

// --- compute-bound bench kernels: REPEAT unravels accumulated, one write ---
#define REPEAT 64
__global__ void b_gen_r4(int64_t n, DimsI32 shape, DimsI64 stride, int64_t* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t acc = 0;
    for (int r = 0; r < REPEAT; ++r)
        acc += baracuda::coord::gen::unravel_offset_1_r4(i + r, shape, stride);
    out[i] = acc;
}
__global__ void b_hand(int64_t n, int rank, DimsI32 shape, DimsI64 stride, int64_t* out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t acc = 0;
    for (int r = 0; r < REPEAT; ++r)
        acc += baracuda::coord::unravel_offset_1(i + r, rank, shape, stride);
    out[i] = acc;
}

static DimsI32 mk_shape(std::vector<int32_t> v){ DimsI32 d{}; for(size_t i=0;i<v.size();++i) d.v[i]=v[i]; return d; }
static DimsI64 mk_stride(std::vector<int64_t> v){ DimsI64 d{}; for(size_t i=0;i<v.size();++i) d.v[i]=v[i]; return d; }

// One correctness case at rank 4: gen == hand == CPU ref, whole-buffer.
static void check_case(const char* name, int64_t n, std::vector<int32_t> shp, std::vector<int64_t> str) {
    DimsI32 shape = mk_shape(shp);
    DimsI64 stride = mk_stride(str);
    int64_t *dg, *dh;
    CHECK(cudaMalloc(&dg, n*sizeof(int64_t)));
    CHECK(cudaMalloc(&dh, n*sizeof(int64_t)));
    int blocks = (int)((n + 255) / 256);
    k_gen_r4<<<blocks,256>>>(n, shape, stride, dg);
    k_hand<<<blocks,256>>>(n, 4, shape, stride, dh);
    CHECK(cudaDeviceSynchronize());
    std::vector<int64_t> hg(n), hh(n);
    CHECK(cudaMemcpy(hg.data(), dg, n*sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hh.data(), dh, n*sizeof(int64_t), cudaMemcpyDeviceToHost));
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        int64_t ref = unravel_ref(i, 4, shp.data(), str.data());
        if (hg[i] != ref || hh[i] != ref) {
            if (bad < 3) printf("  [%s] i=%lld gen=%lld hand=%lld ref=%lld\n",
                                name, (long long)i, (long long)hg[i], (long long)hh[i], (long long)ref);
            bad++;
        }
    }
    printf("%-12s n=%-8lld %s\n", name, (long long)n, bad ? "FAIL" : "ok (gen==hand==ref)");
    if (bad) fails++;
    cudaFree(dg); cudaFree(dh);
}

// Empty-axis guard: shape has a 0 axis; a single index must not %0 and must match ref.
__global__ void k_gen_guard(DimsI32 shape, DimsI64 stride, int64_t linear, int64_t* out) {
    *out = baracuda::coord::gen::unravel_offset_1_r4(linear, shape, stride);
}
static void check_guard() {
    DimsI32 shape = mk_shape({3,0,4,2});      // axis 1 empty
    DimsI64 stride = mk_stride({100,50,10,1});
    int64_t linear = 123;
    int64_t *d; CHECK(cudaMalloc(&d, sizeof(int64_t)));
    k_gen_guard<<<1,1>>>(shape, stride, linear, d);
    CHECK(cudaDeviceSynchronize());
    int64_t got; CHECK(cudaMemcpy(&got, d, sizeof(int64_t), cudaMemcpyDeviceToHost));
    int32_t shp[4]={3,0,4,2}; int64_t str[4]={100,50,10,1};
    int64_t ref = unravel_ref(linear, 4, shp, str);
    printf("%-12s %s (empty-axis guard, no %%0; got=%lld ref=%lld)\n",
           "guard", got==ref ? "ok" : "FAIL", (long long)got, (long long)ref);
    if (got != ref) fails++;
    cudaFree(d);
}

// House cudaEvent timer: 5 warmup + iters timed, returns ms/iter.
template <class F>
static double timed(F launch, int iters=50) {
    for (int i=0;i<5;++i) launch();
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i=0;i<iters;++i) launch();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms/iters;
}

int main() {
    printf("== correctness (rank 4) ==\n");
    check_case("normal",    7*5*3*4, {7,5,3,4}, {60,12,4,1});
    check_case("broadcast", 7*5*3*4, {7,5,3,4}, {60,0,4,1});     // stride[1]=0
    check_case("negative",  7*5*3*4, {7,5,3,4}, {-60,12,4,1});   // stride[0]<0
    check_guard();

    printf("\n== bench (rank 4, REPEAT=%d unravels/elem, %d indices) ==\n", REPEAT, 4096*4096);
    int64_t n = (int64_t)4096*4096;
    DimsI32 shape = mk_shape({256,256,16,16});
    DimsI64 stride = mk_stride({65536,256,16,1});
    int64_t* d; CHECK(cudaMalloc(&d, n*sizeof(int64_t)));
    int blocks = (int)((n + 255)/256);
    double tg = timed([&]{ b_gen_r4<<<blocks,256>>>(n, shape, stride, d); });
    double th = timed([&]{ b_hand<<<blocks,256>>>(n, 4, shape, stride, d); });
    CHECK(cudaGetLastError());
    printf("gen  (unrolled r4)  : %.3f ms/iter\n", tg);
    printf("hand (runtime rank) : %.3f ms/iter\n", th);
    printf("speedup gen/hand    : %.2fx\n", th / tg);
    cudaFree(d);

    printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "PASSED", fails, fails==1?"":"s");
    return fails ? 1 : 0;
}
