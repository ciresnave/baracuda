// On-device numeric validation of the increment-1 MULTI_OUTPUT elementwise
// emitter: one kernel writing N outputs from a shared body-DAG (cross-body CSE —
// the shared `dy` load / an interior product emitted once, then N stores).
//
// Validates the GENERATED kernels (pasted verbatim below — headerless, exactly as
// `baracuda-kernelgen` emits them) against an f64 CPU oracle PER OUTPUT, on both a
// contiguous and a strided cell, and audits generated-vs-bespoke (the sibling
// `binary_mul_backward_fp.cu` / `binary_div_backward_fp.cu` math, inlined). Also
// a determinism check (two runs, bit-identical).
//
// Compile (from a VS dev shell so nvcc finds cl.exe):
//   nvcc -O3 -arch=sm_89 multi_output_validate.cu -o multi_output_validate
// Sanitizers (the multi-store to distinct buffers must have no races / OOB):
//   compute-sanitizer --tool memcheck  ./multi_output_validate
//   compute-sanitizer --tool racecheck ./multi_output_validate
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// ======================= GENERATED (verbatim, headerless) =======================

// mul_backward: da = dy*b, db = dy*a. dy (in0) loaded ONCE into tmp0.
extern "C" __global__ void baracuda_gen_mul_backward_f32_mo2_scalar(
    const float* __restrict__ in0,
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out0,
    float* __restrict__ out1,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        float tmp0 = in0[i];
        out0[i] = (tmp0 * in2[i]);
        out1[i] = (tmp0 * in1[i]);
    }
}

// mul_backward strided (rank 2): distinct per-output stride arrays + offsets.
extern "C" __global__ void baracuda_gen_mul_backward_f32_mo2_strided_r2(
    const float* __restrict__ in0,
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out0,
    float* __restrict__ out1,
    long long shape0, long long shape1,
    long long s0_0, long long s0_1,
    long long s1_0, long long s1_1,
    long long s2_0, long long s2_1,
    long long so0_0, long long so0_1,
    long long so1_0, long long so1_1,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        long long lin = i;
        long long c1 = lin % shape1; lin /= shape1;
        long long c0 = lin % shape0; lin /= shape0;
        long long o0 = c0*s0_0 + c1*s0_1;
        long long o1 = c0*s1_0 + c1*s1_1;
        long long o2 = c0*s2_0 + c1*s2_1;
        long long oo0 = c0*so0_0 + c1*so0_1;
        long long oo1 = c0*so1_0 + c1*so1_1;
        float tmp0 = in0[o0];
        out0[oo0] = (tmp0 * in2[o2]);
        out1[oo1] = (tmp0 * in1[o1]);
    }
}

// div_backward: da = dy/b; db = -((dy/b)*a/b). The dy/b interior (tmp1) and b
// (tmp0) each loaded/computed ONCE, shared across both stores.
extern "C" __global__ void baracuda_gen_div_backward_f32_mo2_scalar(
    const float* __restrict__ in0,
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out0,
    float* __restrict__ out1,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        float tmp0 = in2[i];
        float tmp1 = (in0[i] / tmp0);
        out0[i] = tmp1;
        out1[i] = (-((tmp1 * in1[i]) / tmp0));
    }
}

// fma_backward (3 outputs): da = dy*b, db = dy*a, dc = dy (a plain copy reusing
// the hoisted shared load).
extern "C" __global__ void baracuda_gen_fma_backward_f32_mo3_scalar(
    const float* __restrict__ in0,
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    float* __restrict__ out0,
    float* __restrict__ out1,
    float* __restrict__ out2,
    long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        float tmp0 = in0[i];
        out0[i] = (tmp0 * in2[i]);
        out1[i] = (tmp0 * in1[i]);
        out2[i] = tmp0;
    }
}

// ==================== BESPOKE-equivalent (sibling functor math) ====================
// binary_mul_backward_fp.cu MulBackwardFunctor: da = dy*b, db = dy*a.
extern "C" __global__ void bespoke_mul_backward(
    const float* dy, const float* a, const float* b, float* da, float* db, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) { da[i] = dy[i] * b[i]; db[i] = dy[i] * a[i]; }
}
// binary_div_backward_fp.cu DivBackwardFunctor: da = dy/b; db = -(dy*a)/(b*b).
extern "C" __global__ void bespoke_div_backward(
    const float* dy, const float* a, const float* b, float* da, float* db, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        da[i] = dy[i] / b[i];
        float prod_ab = dy[i] * a[i];
        float denom = b[i] * b[i];
        db[i] = (0.0f - prod_ab) / denom;
    }
}

// ================================ host oracle / check ================================
static int g_fails = 0;
// Relative-error check (a few f32 ULP tolerance; division/negation formulas differ
// between generated and bespoke db, both correct vs the f64 oracle).
static void check(const char* name, const std::vector<float>& got, const std::vector<double>& ref, double rtol) {
    double maxrel = 0.0; long long badi = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        double r = ref[i];
        double e = fabs((double)got[i] - r);
        double rel = e / (fabs(r) + 1e-30);
        if (rel > maxrel) { maxrel = rel; badi = (long long)i; }
    }
    bool ok = maxrel <= rtol;
    if (ok) printf("PASS %-32s maxrel %.3e\n", name, maxrel);
    else { printf("FAIL %-32s maxrel %.3e at %lld (got %g want %g)\n", name, maxrel, badi, got[badi], ref[badi]); g_fails++; }
}
static void bitcmp(const char* name, const std::vector<float>& x, const std::vector<float>& y) {
    bool ok = (x.size() == y.size()) && (memcmp(x.data(), y.data(), x.size()*sizeof(float)) == 0);
    printf("%s %-32s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_fails++;
}

int main() {
    const long long R = 64, C = 48, n = R * C; // 3072
    std::vector<float> dy(n), a(n), b(n);
    for (long long i = 0; i < n; ++i) {
        dy[i] = (float)((i % 21) - 10) * 0.5f;      // spans negatives / zero
        a[i]  = (float)((i % 13) - 6) * 0.25f;
        b[i]  = (float)((i % 7) + 1) * 0.5f;        // never zero (div contract)
    }
    float *ddy, *da_, *db_, *o0, *o1, *o2, *bo0, *bo1;
    cudaMalloc(&ddy, n*4); cudaMalloc(&da_, n*4); cudaMalloc(&db_, n*4);
    cudaMalloc(&o0, n*4); cudaMalloc(&o1, n*4); cudaMalloc(&o2, n*4);
    cudaMalloc(&bo0, n*4); cudaMalloc(&bo1, n*4);
    cudaMemcpy(ddy, dy.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(da_, a.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db_, b.data(), n*4, cudaMemcpyHostToDevice);

    std::vector<float> g0(n), g1(n), g2(n), c0(n), c1(n);
    std::vector<double> refA(n), refB(n), refC(n);

    // ---- mul_backward CONTIGUOUS: da=dy*b, db=dy*a ----
    for (long long i=0;i<n;++i){ refA[i]=(double)dy[i]*b[i]; refB[i]=(double)dy[i]*a[i]; }
    cudaMemset(o0,0,n*4); cudaMemset(o1,0,n*4);
    baracuda_gen_mul_backward_f32_mo2_scalar<<<64,128>>>(ddy,da_,db_,o0,o1,n);
    cudaDeviceSynchronize();
    cudaMemcpy(g0.data(),o0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),o1,n*4,cudaMemcpyDeviceToHost);
    check("mul_bw contig da (gen)", g0, refA, 1e-6);
    check("mul_bw contig db (gen)", g1, refB, 1e-6);
    // bespoke + generated-vs-bespoke (mul is a single multiply → bit-identical)
    bespoke_mul_backward<<<64,128>>>(ddy,da_,db_,bo0,bo1,n);
    cudaDeviceSynchronize();
    cudaMemcpy(c0.data(),bo0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(c1.data(),bo1,n*4,cudaMemcpyDeviceToHost);
    bitcmp("mul_bw da gen==bespoke (bit)", g0, c0);
    bitcmp("mul_bw db gen==bespoke (bit)", g1, c1);

    // ---- determinism: two runs bit-identical ----
    std::vector<float> r2(n);
    cudaMemset(o0,0,n*4);
    baracuda_gen_mul_backward_f32_mo2_scalar<<<64,128>>>(ddy,da_,db_,o0,o1,n);
    cudaDeviceSynchronize(); cudaMemcpy(r2.data(),o0,n*4,cudaMemcpyDeviceToHost);
    bitcmp("mul_bw determinism (2 runs)", g0, r2);

    // ---- div_backward CONTIGUOUS: da=dy/b, db=-dy*a/b^2 ----
    for (long long i=0;i<n;++i){ refA[i]=(double)dy[i]/b[i]; refB[i]=-((double)dy[i]*a[i])/((double)b[i]*b[i]); }
    cudaMemset(o0,0,n*4); cudaMemset(o1,0,n*4);
    baracuda_gen_div_backward_f32_mo2_scalar<<<64,128>>>(ddy,da_,db_,o0,o1,n);
    cudaDeviceSynchronize();
    cudaMemcpy(g0.data(),o0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),o1,n*4,cudaMemcpyDeviceToHost);
    check("div_bw contig da (gen)", g0, refA, 4e-6);
    check("div_bw contig db (gen)", g1, refB, 4e-6);
    bespoke_div_backward<<<64,128>>>(ddy,da_,db_,bo0,bo1,n);
    cudaDeviceSynchronize();
    cudaMemcpy(c0.data(),bo0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(c1.data(),bo1,n*4,cudaMemcpyDeviceToHost);
    check("div_bw contig da (bespoke)", c0, refA, 4e-6);
    check("div_bw contig db (bespoke)", c1, refB, 4e-6);
    // da is dy/b in both → bit-identical; db uses a different (interior-shared)
    // formula, so it matches the oracle but need not bit-match bespoke.
    bitcmp("div_bw da gen==bespoke (bit)", g0, c0);

    // ---- fma_backward (3 outputs): da=dy*b, db=dy*a, dc=dy ----
    for (long long i=0;i<n;++i){ refA[i]=(double)dy[i]*b[i]; refB[i]=(double)dy[i]*a[i]; refC[i]=dy[i]; }
    cudaMemset(o0,0,n*4); cudaMemset(o1,0,n*4); cudaMemset(o2,0,n*4);
    baracuda_gen_fma_backward_f32_mo3_scalar<<<64,128>>>(ddy,da_,db_,o0,o1,o2,n);
    cudaDeviceSynchronize();
    cudaMemcpy(g0.data(),o0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),o1,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(g2.data(),o2,n*4,cudaMemcpyDeviceToHost);
    check("fma_bw da (gen)", g0, refA, 1e-6);
    check("fma_bw db (gen)", g1, refB, 1e-6);
    check("fma_bw dc=dy copy (gen)", g2, refC, 0.0);

    // ---- mul_backward STRIDED cell (col-major inputs, row-major outputs) ----
    // logical [shape0=R, shape1=C]; input storage (r,c) at r + c*R (strides 1,R);
    // output storage (r,c) at r*C + c (strides C,1). Exercises distinct in/out
    // offsets for both outputs.
    for (long long r=0;r<R;++r) for (long long cc=0;cc<C;++cc) {
        long long inp = r + cc*R;      // col-major input element
        long long out = r*C + cc;      // row-major output element
        refA[out] = (double)dy[inp]*b[inp];
        refB[out] = (double)dy[inp]*a[inp];
    }
    cudaMemset(o0,0,n*4); cudaMemset(o1,0,n*4);
    baracuda_gen_mul_backward_f32_mo2_strided_r2<<<64,128>>>(
        ddy,da_,db_,o0,o1,
        R,C, 1,R, 1,R, 1,R, C,1, C,1, n);
    cudaDeviceSynchronize();
    cudaMemcpy(g0.data(),o0,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(),o1,n*4,cudaMemcpyDeviceToHost);
    check("mul_bw strided da (gen)", g0, refA, 1e-6);
    check("mul_bw strided db (gen)", g1, refB, 1e-6);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(g_fails ? "\n%d CHECK(S) FAILED\n" : "\nALL PASSED\n", g_fails);
    return g_fails ? 1 : 0;
}
